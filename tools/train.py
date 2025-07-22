"""Train a BC model."""

from typing import Any

import os
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import subprocess

from clu import parameter_overview
from dataclasses import dataclass, field
from flax import nnx, struct, traverse_util
from functools import partial
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf

from seq_jax.data import DataConfig, make_dataloader, array_tree_to_info
from seq_jax.dist import (
    DistributedConfig,
    EnvConfig,
    set_mesh,
    setup_env,
    init_dist,
    make_mesh,
    BATCH_AXIS,
    fsdp_sharding,
)
from seq_jax.log import (
    LoggingConfig,
    make_meter,
    init_wandb,
    init_profiler,
    start_trace,
    stop_trace,
)
from seq_jax.actor import ModelConfig, make_model, Observation, Prediction
from seq_jax.optim import OptimConfig, make_optimizer, make_lr_scheduler
from seq_jax.policy import PathRegex


@dataclass
class TrainConfig:
    exp_name: str = "fold"
    logdir: str = "/tmp/bc/"
    seed: int = 0
    resume: bool = False

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dist: DistributedConfig = field(default_factory=DistributedConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def trainable_filter(self):
        return nnx.All(nnx.Param, nnx.Not(PathRegex(self.model.frozen_filter)))

    @property
    def frozen_filter(self):
        return nnx.All(nnx.Param, PathRegex(self.model.frozen_filter))


@struct.dataclass
class TrainState:
    step: int
    params: nnx.State
    model_def: nnx.GraphDef = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    ema_alpha: float
    ema_params: Any


def load_weights(weights_path: str) -> nnx.State:
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(weights_path)
        params = ckptr.restore(
            weights_path,
            ocp.args.PyTreeRestore(
                item=metadata,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        restore_type=jax.Array,
                        sharding=jax.sharding.SingleDeviceSharding(
                            jax.local_devices()[0]
                        ),
                    ),
                    metadata,
                ),
            ),
        )
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    params = traverse_util.unflatten_dict(flat_params)
    return params


def make_train_state(
    cfg: TrainConfig,
    lr_scheduler: optax.Schedule,
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
):
    """Create initial training state."""

    def init(
        params: nnx.State,
        graphdef: nnx.GraphDef,
    ):
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
        opt_state = tx.init(trainable_params.to_pure_dict())
        state = TrainState(
            step=0,
            params=params,
            model_def=graphdef,
            tx=tx,
            opt_state=opt_state,
            ema_alpha=cfg.optim.ema_alpha,
            ema_params=params,
        )
        return state

    model = make_model(cfg.model, rngs=rngs)

    # Initialize params from checkpoint
    if cfg.model.init_weights:
        params = load_weights(cfg.model.init_weights)
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(params)
        model = nnx.merge(graphdef, state)
        print(f"Initialized model from checkpoint {cfg.model.init_weights}")

    params = nnx.state(model)
    graphdef = nnx.graphdef(model)
    tx = make_optimizer(
        cfg.optim, lr_scheduler, params.filter(cfg.trainable_filter).to_pure_dict()
    )

    train_state_shape = jax.eval_shape(init, params, graphdef)
    state_sharding = fsdp_sharding(train_state_shape, mesh, log=True)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    print(parameter_overview.get_parameter_overview(params.to_pure_dict()))

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
    print(
        f"Total params: {(encoder_params + query_pool_params + policy_params) / 1e6:.2f}M"
    )

    train_state_fn = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )
    train_state = train_state_fn(params, graphdef)

    return train_state, state_sharding


def make_checkpointer(logdir: str, resume: bool = False) -> ocp.CheckpointManager:
    os.makedirs(logdir, exist_ok=True)

    mngr = ocp.CheckpointManager(
        logdir,
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

    print("Checkpoint manager steps: ", tuple(mngr.all_steps()))

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

    params_placeholder = state.params
    state_placeholder = dataclasses.replace(state, params={})

    abstract_params = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, params_placeholder
    )
    abstract_state = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, state_placeholder
    )

    restored = ckptr.restore(
        step=None,
        args=ocp.args.Composite(
            params=ocp.args.PyTreeRestore(
                item=params_placeholder,
                restore_args=jax.tree.map(
                    lambda s: ocp.ArrayRestoreArgs(
                        restore_type=jax.Array, sharding=s.sharding
                    ),
                    abstract_params,
                ),
            ),
            train_state=ocp.args.PyTreeRestore(
                item=state_placeholder,
                restore_args=jax.tree.map(
                    lambda s: ocp.ArrayRestoreArgs(
                        restore_type=jax.Array, sharding=s.sharding
                    ),
                    abstract_state,
                ),
            ),
        ),
    )
    restored_state = restored["train_state"]
    restored_state = dataclasses.replace(restored_state, params=restored["params"])

    print(f"Restored training state from checkpoint at step {restored_state.step}")
    return restored_state


def save_state(
    ckptr: ocp.CheckpointManager,
    state: TrainState,
    step: int,
    gcs_path: str | None = None,
    wait: bool = False,
):

    step = int(state.step)
    print(f"Saving checkpoint at step {step} to {ckptr.directory}")
    params = state.params
    save_train_state = dataclasses.replace(state, params={})
    ckptr.save(step, items={"train_state": save_train_state, "params": params})

    if wait or gcs_path:
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

    if cfg.log.gcs_ckpt:
        ckpt_path = os.path.join(cfg.log.gcs_ckpt, cfg.exp_name)
    else:
        ckpt_path = os.path.abspath(os.path.join(cfg.logdir, cfg.exp_name))
    ckptr = make_checkpointer(ckpt_path, resume=cfg.resume)

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
    start_step = state.step
    for step in range(start_step, cfg.optim.num_steps):
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

            if step == start_step:
                print(array_tree_to_info(data))

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
                save_state(
                    ckptr,
                    state,
                    step,
                    gcs_path=cfg.log.gcs_ckpt,
                    wait=(step == cfg.optim.num_steps - 1),
                )


def read_cfg() -> TrainConfig:
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(TrainConfig())
    cfg = OmegaConf.merge(default_cfg, cli_args)
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)

    # Dump config
    output_dir = os.path.join(cfg.logdir, cfg.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    if cfg.log.gcs_ckpt:
        # Copy config to gcs
        subprocess.run(
            [
                "gsutil",
                "cp",
                os.path.join(output_dir, "config.yaml"),
                os.path.join(cfg.log.gcs_ckpt, cfg.exp_name, "config.yaml"),
            ],
            check=True,
        )

    return cfg


if __name__ == "__main__":
    cfg: TrainConfig = read_cfg()
    train(cfg)
