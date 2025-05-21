"""Train a BC model."""

from typing import Any, Iterable, Tuple

import os
import dataclasses
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import optax
import orbax.checkpoint as ocp
import termcolor

from absl import logging
from dataclasses import dataclass, field
from flax import nnx, struct
from flax.training import train_state, checkpoints
from functools import partial
from jax import random
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf

from seq_jax.dataloader import DataConfig, make_dataloader
from seq_jax.distributed import DistributedConfig, EnvConfig, make_mesh, set_mesh
from seq_jax.logging import LoggingConfig, make_meter, init_wandb
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

    @property
    def trainable_filter(self):
        return nnx.All(nnx.Param, nnx.Not(util.PathRegex(self.model.frozen_filter)))

    @property
    def frozen_filter(self):
        return nnx.All(nnx.Param, util.PathRegex(self.model.frozen_filter))


@struct.dataclass
class TrainState:
    step: int
    params: nnx.State
    model_def: nnx.GraphDef
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
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
    trainable_params = params.filter(cfg.trainable_filter)
    frozen_params = params.filter(cfg.frozen_filter)
    print(
        "Frozen params: "
        + ", ".join(
            [
                ".".join([str(it) for it in path_tuple])
                for path_tuple in frozen_params.flat_state().paths
            ]
        ),
    )
    opt_state = tx.init(trainable_params)
    state = TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=opt_state,
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
    state: TrainState,
    data: jax.Array,
):
    """Perform a training step."""
    print("in train step", type(state), type(data))
    model = nnx.merge(state.model_def, state.params)
    model.train()

    rng = jax.random.PRNGKey(0)

    def loss_fn(model: nnx.Module, rngs: nnx.Rngs, obs: Observation, pred: Prediction):

        loss = model.compute_loss(obs, pred, deterministic=False, rngs=rngs)
        return loss

    # Compute loss and gradients
    print("train step", util.array_tree_to_info(data), sep="\n")

    obs, pred = data
    diff_state = nnx.DiffState(0, cfg.trainable_filter)
    grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state)
    loss, grads = grad_fn(model, rng, obs, pred)

    params = state.params.filter(cfg.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )
    metrics = {
        "loss": jnp.array([loss]),
        "grad_norm": jnp.array([optax.global_norm(grads)]),
        "step": jnp.array([state.step]),
        # "params": optax.global_norm(new_params),
    }
    return new_state, metrics


def train(cfg: TrainConfig):
    init_wandb(cfg)

    print(f"Jax process: {jax.process_index()} / {jax.process_count()}")
    print(f"Jax devices: {jax.device_count()}")

    rng = jax.random.PRNGKey(cfg.seed)
    rngs = nnx.Rngs(rng)

    # Setup training components
    # mesh = make_mesh(cfg.distributed)
    devices = np.array(jax.devices())
    mesh = Mesh(devices, ("data",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    dataloader = make_dataloader(
        cfg.data,
        cfg.model.context_len,
        cfg.model.pred_len,
        data_sharding,
    )
    lr_scheduler = make_lr_scheduler(cfg.optim)
    meter = make_meter(cfg.logging, cfg.optim.num_steps)

    state = make_train_state(cfg, lr_scheduler, mesh, rngs=rngs)
    ptrain_step = jax.jit(
        train_step,
        in_shardings=(replicated_sharding, data_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        # donate_argnums=(1,),
    )

    # Training loop
    for step in range(state.step, cfg.optim.num_steps):
        # Get next batch
        data = next(dataloader)

        # Perform training step
        print("ar_mask", data[0].mask.shape)
        print(util.array_tree_to_info(data))

        with set_mesh(mesh):
            state, info = ptrain_step(state, data)

        print(info)
        print(info["loss"].global_shards)
        print("done")
        # print(metrics)
        # Save a checkpoint
        if (step + 1) % cfg.logging.save_freq == 0:
            save_checkpoint(state, cfg.logdir, step)


def get_cfg() -> TrainConfig:
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(TrainConfig())
    cfg = OmegaConf.merge(default_cfg, cli_args)
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)

    # Dump config
    os.makedirs(cfg.logdir, exist_ok=True)
    with open(os.path.join(cfg.logdir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    return cfg


if __name__ == "__main__":
    cfg: TrainConfig = get_cfg()
    train(cfg)
