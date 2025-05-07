"""Train a BC model."""

from typing import Any, Iterable

import dataclasses
import jax
import jax.numpy as jnp
import jax.lax as lax
import optax
import orbax.checkpoint as ocp
import termcolor

from absl import logging
from dataclasses import dataclass, field
from flax import nnx, struct
from flax.training import train_state, checkpoints
from functools import partial
from jax import random
from omegaconf import OmegaConf

from seq_jax.dataloader import DataConfig, make_dataloader
from seq_jax.distributed import DistributedConfig, EnvConfig, make_mesh
from seq_jax.logging import LoggingConfig, make_meter
from seq_jax.actor import ModelConfig, make_model, Observation, Prediction
from seq_jax.optim import OptimConfig, make_optimizer, make_lr_scheduler
from seq_jax import util


@dataclass
class TrainConfig:
    exp_name: str = "fold"
    logdir: str = "/tmp/bc/"
    seed: int = 0
    platform: str = "gpu"

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@struct.dataclass
class TrainState:
    step: int
    params: nnx.State
    trainable_filter: Any
    model_def: nnx.GraphDef
    tx: optax.GradientTransformation
    opt_state: optax.OptState
    ema_alpha: float
    ema_params: Any


def make_train_state(
    cfg: TrainConfig,
    lr_scheduler: optax.Schedule,
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
):
    """Create initial training state."""
    # Create model and optimizer
    model = make_model(cfg.model, rngs=rngs)
    params = nnx.state(model)
    tx = make_optimizer(cfg.optim, lr_scheduler)

    # Filter frozen parameters
    trainable_filter = nnx.All(
        nnx.Param, nnx.Not(util.PathRegex(cfg.model.freeze_filter))
    )
    trainable_params = params.filter(trainable_filter)

    frozen_filter = nnx.All(nnx.Param, util.PathRegex(cfg.model.freeze_filter))
    frozen_params = params.filter(frozen_filter)
    print(
        "Frozen params: "
        + ", ".join(
            [
                ".".join([str(it) for it in path_tuple])
                for path_tuple in frozen_params.flat_state().paths
            ]
        ),
    )

    state = TrainState(
        step=0,
        params=params,
        trainable_filter=trainable_filter,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(trainable_params),
        ema_alpha=cfg.optim.ema_alpha,
        ema_params=params,
    )
    return state


def save_checkpoint(state: TrainState, outdir: str, step: int):
    step = int(state.step)
    logging.info(f"Saving checkpoint at step {step}")

    # replicated_state = jax.tree_util.tree_map(
    #     ocp.utils.fully_replicated_host_array_to_global_array, state
    # )
    # checkpoints.save_checkpoint_multiprocess(outdir, replicated_state, step, keep=3)


def train_step(
    cfg: TrainConfig,
    state: TrainState,
    rng: jax.Array,
    batch: Any,
    lr_scheduler: optax.Schedule,
):
    """Perform a training step."""
    lr = lr_scheduler(state.step)
    model = nnx.merge(state.model_def, state.params)
    model.train()

    vision, states, noisy_state, actions, masks = batch
    obs = Observation(
        img_rgb=None,
        img_feat=[vision[:, :, i, ...] for i in range(3)],
        img_pooled=None,
        state=states,
        prev_action=actions,
        mask=masks,
    )
    pred = Prediction(
        img_pooled=None,
        state=None,
        action=jnp.stack(
            [
                actions[:, i : i + cfg.model.pred_len, ...]
                for i in range(cfg.model.context_len)
            ],
            axis=1,
        ),
    )

    def loss_fn(model: nnx.Module, rngs: nnx.Rngs, obs: Observation, pred: Prediction):

        loss = model.compute_loss(obs, pred, deterministic=False, rngs=rngs)
        return loss

    # Compute loss and gradients
    diff_state = nnx.DiffState(0, state.trainable_filter)
    grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state)
    loss, grads = grad_fn(model, rng, obs, pred)

    params = state.params.filter(state.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )

    # Average gradients across devices
    # grads = lax.pmean(grads, axis_name="batch")
    # # Apply gradients
    # loss, new_model_state = aux
    # new_state = state.apply_gradients(
    #     grads=grads,
    #     batch_stats=lax.pmean(new_model_state["batch_stats"], axis_name="batch"),
    # )
    metrics = {"loss": loss}

    return new_state, metrics


def train(cfg: TrainConfig):

    logging.info(f"Jax process: {jax.process_index()} / {jax.process_count()}")
    logging.info(f"Jax devices: {jax.device_count()}")

    rng = jax.random.PRNGKey(cfg.seed)
    rngs = nnx.Rngs(rng)

    # Setup training components
    dataloader = make_dataloader(cfg.data, cfg.distributed.num_devices)
    lr_scheduler = make_lr_scheduler(cfg.optim)
    meter = make_meter(cfg.logging)
    mesh = make_mesh(cfg.distributed)
    state = make_train_state(cfg, lr_scheduler, mesh, rngs=rngs)

    ptrain_step = jax.jit(
        partial(
            train_step,
            lr_scheduler=lr_scheduler,
        ),
        # # Replicate state, split batch
        # in_axes=(None, 0),
        # # Replicate new_state, stack metrics
        # out_axes=(None, 0),
        # axis_name="batch",
    )

    # Training loop
    for step in range(state.step, cfg.optim.num_steps):
        data = next(dataloader)

        for d in data:
            print(d.shape)

        print(type(state), type(data))

        # Perform training step
        state, metrics = train_step(cfg, state, rng, data, lr_scheduler)
        # meter.update(metrics)
        print(metrics)

        # Save a checkpoint
        if (step + 1) % cfg.logging.save_freq == 0:
            save_checkpoint(state, cfg.logdir, step)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()


def get_cfg() -> TrainConfig:
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(TrainConfig())
    cfg = OmegaConf.merge(default_cfg, cli_args)
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    return cfg


if __name__ == "__main__":
    cfg: TrainConfig = get_cfg()
    train(cfg)
