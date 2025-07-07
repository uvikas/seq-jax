"""Train a BC model."""

from typing import Any, Iterable, Tuple

import os
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import termcolor
import time
import torch

from clu import parameter_overview
from dataclasses import dataclass, field
from flax import nnx, struct
from functools import partial
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf

from seq_jax.dataloader import DataConfig, make_dataloader
from seq_jax.distributed import (
    DistributedConfig,
    EnvConfig,
    set_mesh,
    setup_env,
    init_dist,
    make_mesh,
    BATCH_AXIS,
)
from seq_jax.logging import (
    LoggingConfig,
    make_meter,
    init_wandb,
    init_profiler,
    start_trace,
    stop_trace,
)
from seq_jax.actor import ModelConfig, make_model, Observation, Prediction
from seq_jax.optim import OptimConfig, make_optimizer, make_lr_scheduler
from seq_jax import util


@dataclass
class TrainConfig:
    exp_name: str = "fold"
    logdir: str = "/tmp/bc/"
    seed: int = 0
    platform: str = "gpu"
    resume: bool = False

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dist: DistributedConfig = field(default_factory=DistributedConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)

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
    model_def: nnx.GraphDef = struct.field(pytree_node=False)
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

    model = make_model(cfg.model, rngs=rngs)
    params = nnx.state(model)
    graphdef = nnx.graphdef(model)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    params = jax.device_put(params, replicated_sharding)
    nnx.update(model, params)

    # Filter frozen parameters
    params = nnx.state(model)
    print(parameter_overview.get_parameter_overview(params.to_pure_dict()))
    trainable_params = params.filter(cfg.trainable_filter).to_pure_dict()
    frozen_params = params.filter(cfg.frozen_filter)
    print(
        "Frozen params: ",
        ", ".join(
            [
                ".".join(map(str, path_tuple))
                for path_tuple in frozen_params.flat_state().paths
            ]
        ),
    )

    tx = make_optimizer(cfg.optim, lr_scheduler, trainable_params)
    opt_state = tx.init(trainable_params)

    # Count number of params
    encoder_params = sum(
        np.prod(p.shape)
        for p in jax.tree_util.tree_leaves(nnx.state(model.img_encoder))
    )
    query_pool_params = sum(
        np.prod(p.shape) for p in jax.tree_util.tree_leaves(nnx.state(model.attn_pool))
    )
    policy_params = sum(
        np.prod(p.shape) for p in jax.tree_util.tree_leaves(nnx.state(model.policy))
    )
    print(f"Encoder params: {encoder_params / 1e6:.2f}M")
    print(f"Query pool params: {query_pool_params / 1e6:.2f}M")
    print(f"Policy params: {policy_params / 1e6:.2f}M")

    state = TrainState(
        step=0,
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=opt_state,
        ema_alpha=cfg.optim.ema_alpha,
        ema_params=params,
    )

    return state, replicated_sharding


def make_checkpointer(logdir: str, resume: bool = False) -> ocp.CheckpointManager:
    os.makedirs(logdir, exist_ok=True)
    mngr = ocp.CheckpointManager(
        os.path.abspath(logdir),
        item_handlers={
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            create=True,
            enable_async_checkpointing=True,
            async_options=ocp.AsyncOptions(
                timeout_secs=7200,
                create_directories_asynchronously=True,
            ),
            enable_background_delete=True,
        ),
    )

    if not resume and not (tuple(mngr.all_steps()) in [(), (0,)]):
        raise ValueError(
            f"Log directory {logdir} already exists and contains checkpoints. "
            "Set resume=True to resume training from the last checkpoint."
        )

    return mngr


def restore_state(ckptr: ocp.CheckpointManager, state: TrainState) -> TrainState:
    if tuple(ckptr.all_steps()) in [(), (0,)]:
        # No checkpoints saved, so we can't restore
        return state

    params = state.params
    state = dataclasses.replace(state, params={})
    restored = ckptr.restore(step=None, items={"train_state": state, "params": params})
    restored_state = restored["train_state"]
    restored_state = dataclasses.replace(restored_state, params=params)

    print(f"Restored training state from checkpoint at step {restored_state.step}")
    return restored_state


def save_state(
    ckptr: ocp.CheckpointManager, state: TrainState, step: int, wait: bool = False
):
    if jax.process_index() != 0:
        return

    step = int(state.step)
    print(f"Saving checkpoint at step {step} to {ckptr.directory}")
    params = state.params
    save_train_state = dataclasses.replace(state, params={})
    ckptr.save(step, items={"train_state": save_train_state, "params": params})

    if wait:
        ckptr.wait_until_finished()


def train_step(
    state: TrainState,
    data: jax.Array,
    step_rng: jax.random.PRNGKey,
    lr_scheduler: optax.Schedule,
    cfg: TrainConfig,
):
    """Perform a training step."""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    rng = nnx.Rngs(step_rng)

    def loss_fn(model: nnx.Module, rngs: nnx.Rngs, obs: Observation, pred: Prediction):
        loss, loss_dict = model.compute_loss(obs, pred, rngs=rngs)
        return loss, loss_dict

    # Compute loss and gradients
    obs, pred = data
    diff_state = nnx.DiffState(0, cfg.trainable_filter)
    grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)
    (loss, loss_dict), grads = grad_fn(model, rng, obs, pred)

    params = state.params.filter(cfg.trainable_filter)
    updates, new_opt_state = state.tx.update(
        grads.to_pure_dict(), state.opt_state, params.to_pure_dict()
    )
    new_params = optax.apply_updates(params.to_pure_dict(), updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    # TODO: grad norm + clip twice...
    clipped_grads, _ = optax.clip_by_global_norm(cfg.optim.clip_norm).update(
        grads, None
    )

    ema_params = jax.tree.map(
        lambda old, new: old + state.ema_alpha * (new - old),
        state.ema_params,
        new_params,
    )

    # Create metrics with minimal array allocations
    metrics = {
        "loss": loss_dict,
        "grad_norm": jnp.array([optax.global_norm(clipped_grads)]),
        "step": jnp.array([state.step]),
        "param_norm": optax.global_norm(new_params),
        "lr": lr_scheduler(state.step),
        "batch_size": obs.state.shape[0],
    }

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        ema_params=ema_params,
    )

    return new_state, metrics


def train(cfg: TrainConfig):
    setup_env(cfg.env)
    init_dist(cfg.dist)
    init_wandb(cfg)
    init_profiler(cfg.log.profiler)

    print(f"Jax process: {jax.process_index()} / {jax.process_count()}")
    print(f"Jax devices: {jax.device_count()}")

    root_rng = jax.random.PRNGKey(cfg.seed)
    init_rng, train_rng = jax.random.split(root_rng)
    rngs = nnx.Rngs(init_rng)

    # Setup training components
    mesh = make_mesh(cfg.dist)
    data_sharding = jax.sharding.NamedSharding(mesh, P(BATCH_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, P())
    dataloader = make_dataloader(
        cfg.data,
        cfg.model.context_len,
        cfg.model.pred_len,
        data_sharding,
    )
    lr_scheduler = make_lr_scheduler(cfg.optim)
    meter = make_meter(cfg.log, cfg.optim.num_steps)
    ckptr = make_checkpointer(cfg.logdir, resume=cfg.resume)

    # Create train state
    state, state_sharding = make_train_state(cfg, lr_scheduler, mesh, rngs)
    if cfg.resume:
        state = restore_state(ckptr, state)

    ptrain_step = jax.jit(
        partial(
            train_step,
            lr_scheduler=lr_scheduler,
            cfg=cfg,
        ),
        in_shardings=(state_sharding, data_sharding, replicated_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        # donate_argnums=(0, 1),
    )

    # Pre-generate RNG keys
    step_rngs = jax.random.split(train_rng, cfg.optim.num_steps)

    # Training loop
    for step in range(state.step, cfg.optim.num_steps):
        if cfg.log.profiler.enabled and step == cfg.log.profiler.start_step:
            start_trace(cfg.log.profiler, step)

        meter.reset()
        meter.iter_tic()

        with meter.timer("io"):
            # Get next batch
            while True:
                data = next(dataloader)
                if not jnp.isnan(data[0].img_feat).any():
                    break

        with meter.timer("fwd_bwd"):
            # Perform training step
            with set_mesh(mesh):
                state, metrics = ptrain_step(state, data, step_rngs[step])

        with meter.timer("metrics"):
            metrics = {
                key: value.item() if isinstance(value, jnp.ndarray) else value
                for key, value in metrics.items()
            }
            metrics["loss"] = {
                key: value.item() for key, value in metrics["loss"].items()
            }
            metrics["prefetch"] = dataloader.prefetched()

            # Log metrics
            meter.iter_toc()
            meter.update(metrics)
            meter.log_iter_stats(step)

        if cfg.log.profiler.enabled and step == cfg.log.profiler.stop_step:
            stop_trace(cfg.log.profiler, step)

        with meter.timer("save"):
            # Save a checkpoint
            if (step + 1) % cfg.log.save_freq == 0 or step == cfg.optim.num_steps - 1:
                # Wait for the final checkpoint to complete
                is_final_checkpoint = step == cfg.optim.num_steps - 1
                save_state(ckptr, state, step, wait=is_final_checkpoint)


def read_cfg() -> TrainConfig:
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
    cfg: TrainConfig = read_cfg()
    train(cfg)
