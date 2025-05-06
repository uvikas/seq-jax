"""Train a BC model."""

from typing import Any, Iterable

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax
import orbax.checkpoint as ocp

from absl import logging
from dataclasses import dataclass, field
from flax import nnx
from flax.training import train_state, checkpoints
from functools import partial
from jax import random
from omegaconf import OmegaConf

from seq_jax.dataloader import DataConfig, make_dataloader
from seq_jax.distributed import DistributedConfig, EnvConfig
from seq_jax.logging import LoggingConfig, make_meter
from seq_jax.actor import ModelConfig, make_model
from seq_jax.optim import OptimConfig, make_optimizer, make_lr_scheduler


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


class TrainState(train_state.TrainState):
    batch_stats: Any
    ema_params: Any


def initialized(rng: jax.Array, image_size: int, model: Any):
    """Initialize model parameters with dummy input."""
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    vars = init({"param": rng}, jnp.ones(input_shape, model.dtype))
    return vars["params"], vars["batch_stats"]


def make_train_state(
    rng: jax.Array,
    cfg: TrainConfig,
    model: Any,
    lr_scheduler: optax.Schedule,
):
    """Create initial training state."""
    params, batch_stats = initialized(rng, cfg.data.im_size, model)
    tx = make_optimizer(cfg.optim, lr_scheduler)
    state = TrainState(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx,
    )
    return state


def save_checkpoint(state: TrainState, outdir: str, step: int):
    step = int(state.step)
    logging.info(f"Saving checkpoint at step {step}")

    replicated_state = jax.tree_util.tree_map(
        ocp.utils.fully_replicated_host_array_to_global_array, state
    )
    checkpoints.save_checkpoint_multiprocess(outdir, replicated_state, step, keep=3)


def train_step(
    state: TrainState,
    batch: Any,
    lr_scheduler: optax.Schedule,
):
    """Perform a training step."""
    lr = lr_scheduler(state.step)

    def loss_fn(params: Any):
        pass

    # Compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Average gradients across devices
    grads = lax.pmean(grads, axis_name="batch")
    # Apply gradients
    loss, new_model_state = aux
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=lax.pmean(new_model_state["batch_stats"], axis_name="batch"),
    )
    metrics = {"loss": loss}

    return new_state, metrics


def train(cfg: TrainConfig):

    logging.info(f"Jax process: {jax.process_index()} / {jax.process_count()}")
    logging.info(f"Jax devices: {jax.device_count()}")

    rngs = nnx.Rngs(0)

    # Setup training components
    dataloader = make_dataloader(cfg.data)
    model = make_model(cfg.model, rngs=rngs)
    lr_scheduler = make_lr_scheduler(cfg.optim)
    meter = make_meter(cfg.logging)
    state = make_train_state(cfg, model, lr_scheduler)

    ptrain_step = jax.pmap(
        partial(
            train_step,
            lr_scheduler=lr_scheduler,
        ),
        # Replicate state, split batch
        in_axes=(None, 0),
        # Replicate new_state, stack metrics
        out_axes=(None, 0),
        axis_name="batch",
    )

    # Training loop
    for step in range(state.step, cfg.num_steps):
        data = next(dataloader)
        # Perform training step
        state, metrics = ptrain_step(state, data)
        meter.update(metrics)

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
