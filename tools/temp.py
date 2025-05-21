"""Train a BC model with data parallelism using Flax NNX."""

from typing import Any, Iterable, Tuple

import os
import dataclasses  # Use standard dataclasses for TrainConfig
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import optax
import orbax.checkpoint as ocp

# import termcolor # Unused

from absl import logging
from flax import nnx, struct  # nnx.Module also available directly from flax.nnx
from flax.training import (
    train_state,
)  # train_state is not directly used for NNX state management here
from functools import partial

# from jax import random # jax.random is used directly
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf

from seq_jax.dataloader import DataConfig, make_dataloader
from seq_jax.distributed import (
    DistributedConfig,
    EnvConfig,
    make_mesh,
    set_mesh,
)  # Assuming these are correctly setup
from seq_jax.logging import LoggingConfig, make_meter, init_wandb
from seq_jax.actor import (
    ModelConfig,
    make_model,
    Observation,
    Prediction,
)  # Assuming these are NNX compatible
from seq_jax.optim import OptimConfig, make_optimizer, make_lr_scheduler
from seq_jax import util


@dataclasses.dataclass  # Standard dataclass for configuration
class TrainConfig:
    exp_name: str = "fold"
    logdir: str = "/tmp/bc/"
    seed: int = 0
    platform: str = "gpu"  # Ensure this is handled by jax.config or your environment

    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    distributed: DistributedConfig = dataclasses.field(
        default_factory=DistributedConfig
    )
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)

    @property
    def trainable_filter(self):
        # Defines what parts of the model state are trainable parameters
        return nnx.All(nnx.Param, nnx.Not(util.PathRegex(self.model.frozen_filter)))

    @property
    def frozen_filter(self):
        # Defines what parts of the model state are frozen parameters
        return nnx.All(nnx.Param, util.PathRegex(self.model.frozen_filter))


@struct.dataclass
class TrainState:
    step: int
    params: (
        nnx.State
    )  # Full model state (trainable + frozen parameters, and other variables if any)
    model_def: nnx.GraphDef  # Static definition of the model
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    ema_alpha: float
    ema_params: nnx.State  # EMA copy of the full model state


def make_train_state(
    cfg: TrainConfig,
    lr_scheduler: optax.Schedule,
    # mesh: jax.sharding.Mesh, # Mesh not directly needed here, but for sharding of state later
    rngs: nnx.Rngs,  # RNGs for model initialization
) -> TrainState:
    """Create initial training state."""
    # Create model
    # Model definition and initial parameters
    model = make_model(cfg.model, rngs=rngs)  # make_model should call model.init(rngs)
    initial_params = nnx.state(
        model
    )  # Extract the initial state from the initialized model

    # Create optimizer
    tx = make_optimizer(cfg.optim, lr_scheduler)

    # Filter frozen parameters for optimizer initialization
    # The optimizer state should be initialized only for trainable parameters
    trainable_params_for_opt = initial_params.filter(cfg.trainable_filter)

    # Log frozen parameters
    frozen_params_paths = initial_params.filter(cfg.frozen_filter).flat_state().paths
    logging.info(
        "Frozen params: "
        + ", ".join(
            [
                ".".join([str(it) for it in path_tuple])
                for path_tuple in frozen_params_paths
            ]
        )
    )
    if not frozen_params_paths:
        logging.info("No parameters are frozen.")

    opt_state = tx.init(trainable_params_for_opt)

    state = TrainState(
        step=0,
        params=initial_params,
        model_def=nnx.graphdef(model),  # Store the graph definition
        tx=tx,
        opt_state=opt_state,
        ema_alpha=cfg.optim.ema_alpha,
        ema_params=initial_params,  # EMA starts as a copy of initial parameters
    )
    return state


def save_checkpoint(state: TrainState, manager: ocp.CheckpointManager, step: int):
    """Saves checkpoint using Orbax."""
    logging.info(f"Saving checkpoint at step {step}")
    # Orbax CheckpointManager can save sharded JAX arrays directly.
    # Ensure state is compatible (e.g., PyTree of JAX arrays).
    manager.save(step, args=ocp.args.StandardSave(state))
    manager.wait_until_finished()  # Wait for async save to complete if necessary


# This train_step function will be pjit'd. It should be pure and not have side effects like printing.
def train_step_fn(
    state: TrainState,
    data: Tuple[Observation, Prediction],  # Explicitly typed data
    step_rng_key: jax.random.PRNGKey,
    trainable_filter: nnx.Filter,  # Pass the filter for trainable parameters
):
    """Perform a single training step, designed for pjit."""
    # Reconstruct the model from its definition and current parameters
    model = nnx.merge(state.model_def, state.params)
    model.train()  # Set model to training mode (e.g., for dropout, batchnorm)

    # Prepare RNGs for the current step, e.g., for dropout layers
    # Assuming the model's compute_loss uses an Rngs collection with a 'dropout' key
    step_rngs = nnx.Rngs(dropout=step_rng_key)

    # Define the loss function to be differentiated
    def loss_fn(
        model_arg: nnx.Module,
        current_rngs: nnx.Rngs,
        obs: Observation,
        pred: Prediction,
    ):
        # model_arg.compute_loss should handle its own variables if any (e.g. batch norm stats)
        # and return the loss.
        loss_value = model_arg.compute_loss(
            obs, pred, deterministic=False, rngs=current_rngs
        )
        return loss_value

    obs, pred = data  # Unpack data

    # Compute gradients only for trainable parameters specified by `trainable_filter`
    # Differentiate `loss_fn` with respect to its first argument (`model_arg`).
    grad_calculator = nnx.value_and_grad(loss_fn, argnums=0, wrt=trainable_filter)
    (loss, _aux_outputs_if_any), grads_module = grad_calculator(
        model, step_rngs, obs, pred
    )

    # `grads_module` is an nnx.Module containing gradients. Extract into nnx.State.
    grads_state = nnx.state(grads_module)

    # Average gradients across all devices in the 'data' mesh axis
    averaged_grads_state = lax.pmean(grads_state, axis_name="data")

    # Average loss across all devices for metrics
    averaged_loss = lax.pmean(loss, axis_name="data")

    # Apply optimizer updates
    # Get the current trainable parameters to pass to optimizer's update function
    current_trainable_params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(
        averaged_grads_state, state.opt_state, current_trainable_params
    )
    new_trainable_params_state = optax.apply_updates(current_trainable_params, updates)

    # Update the full model state:
    # Start with a copy of the current full parameters
    new_full_params = state.params.copy()
    # Update only the trainable parts with the new values
    new_full_params.update(new_trainable_params_state)

    # Update EMA parameters
    # EMA is applied to the new full set of parameters (trainable ones updated, frozen ones unchanged)
    new_ema_params = optax.ema_update(
        new_full_params, state.ema_params, state.ema_alpha
    )

    # Create the new train state
    new_state = state.replace(  # Use .replace for flax.struct.dataclass
        step=state.step + 1,
        params=new_full_params,
        opt_state=new_opt_state,
        ema_params=new_ema_params,
    )

    # Collect metrics
    metrics = {
        "loss": averaged_loss,
        "grad_norm": optax.global_norm(
            averaged_grads_state
        ),  # Norm of the averaged gradients
        "step": state.step,  # Current step number (before increment)
    }
    # Ensure metrics are JAX arrays for proper handling by pjit and outside calls
    metrics = jax.tree_util.tree_map(jnp.asarray, metrics)

    return new_state, metrics


def train(cfg: TrainConfig):
    init_wandb(cfg)  # Initialize Weights & Biases if configured

    logging.info(f"JAX process: {jax.process_index()} / {jax.process_count()}")
    logging.info(f"JAX devices: {jax.device_count()}")

    # Initialize JAX distributed system if necessary (often handled by environment or make_mesh)
    # jax.distributed.initialize(cfg.distributed.coordinator_address, ...) if not using make_mesh to do it.

    # Setup PRNG keys
    # Root key for all operations
    root_key = jax.random.PRNGKey(cfg.seed)
    # Split keys for model initialization and the main training loop
    init_rng_key, training_loop_rng_key = jax.random.split(root_key)

    # Setup device mesh for data parallelism
    # Assumes 1D data parallelism over all available devices.
    # devices = np.array(jax.devices()) # Or use cfg.platform to select devices
    # mesh = Mesh(devices, ("data",))
    # The make_mesh utility from the original script is likely more robust:
    mesh = make_mesh(cfg.distributed, cfg.env)  # Use your existing mesh creation

    # Define sharding specifications
    # Data is sharded along the 'data' axis (batch dimension)
    data_sharding = jax.sharding.NamedSharding(mesh, P("data"))
    # Model parameters and optimizer state are replicated across all devices
    replicated_sharding = jax.sharding.NamedSharding(mesh, P())

    # Create dataloader that shards data per device
    dataloader = make_dataloader(
        cfg.data,
        cfg.model.context_len,
        cfg.model.pred_len,
        data_sharding,  # Pass sharding spec to dataloader
        mesh,  # Pass mesh if dataloader needs it for host-local sharding
    )

    # Learning rate scheduler
    lr_scheduler = make_lr_scheduler(cfg.optim)
    # Logging meter
    meter = make_meter(cfg.logging, cfg.optim.num_steps)

    # Create initial training state
    # Model initialization might need specific RNG streams (e.g., 'params', 'dropout')
    # Example: model_init_rngs = nnx.Rngs(params=init_rng_key, dropout=jax.random.fold_in(init_rng_key, 1))
    # For simplicity, assuming nnx.Rngs(params=init_rng_key) is sufficient if make_model handles it.
    model_init_rngs = nnx.Rngs(
        params=init_rng_key
    )  # Or more specific if your model needs other keys for init

    # Abstract state for checkpoint restoration (optional, but good practice)
    # abstract_state = make_train_state(cfg, lr_scheduler, model_init_rngs)
    # abstract_state_sharded = jax.jit(lambda x: x, out_shardings=replicated_sharding)(abstract_state)

    # Checkpoint manager setup
    checkpoint_dir = os.path.join(cfg.logdir, "checkpoints")
    orbax_checkpointer = (
        ocp.StandardCheckpointer()
    )  # Use StandardCheckpointer or AsyncCheckpointer
    checkpoint_manager_options = ocp.CheckpointManagerOptions(
        max_to_keep=3, create=True
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, checkpoint_manager_options
    )

    # Initialize or restore train state
    # Initial state needs to be defined on CPU first, then sharded if restored from non-sharded checkpoint
    # or if creating fresh.
    # If restoring, provide an abstract state or target Pytree.
    # target_for_restore = make_train_state(cfg, lr_scheduler, model_init_rngs) # As abstract state
    # if checkpoint_manager.latest_step() is not None:
    #    logging.info(f"Restoring from checkpoint: {checkpoint_manager.latest_step()}")
    #    state = checkpoint_manager.restore(checkpoint_manager.latest_step(), args=ocp.args.StandardRestore(target_for_restore))
    # else:
    #    state = target_for_restore
    #
    # The state needs to be sharded according to `replicated_sharding`.
    # `pjit` will handle this for arguments if they are not already global device arrays.
    # If creating fresh, make sure it's sharded before the loop or first pjit call.

    # Create initial state (on host)
    initial_host_state = make_train_state(cfg, lr_scheduler, rngs=model_init_rngs)

    # `pjit` the train_step_fn.
    # `cfg.trainable_filter` is static, so we use `partial` to bake it in.
    # This avoids passing `cfg` or the filter dynamically, which `pjit` handles more cleanly.
    p_train_step = pjit(
        partial(train_step_fn, trainable_filter=cfg.trainable_filter),
        in_shardings=(
            replicated_sharding,
            data_sharding,
            replicated_sharding,
        ),  # state, data, step_rng_key
        out_shardings=(replicated_sharding, replicated_sharding),  # new_state, metrics
        # donate_argnums can be used for optimization if memory is critical, e.g., (0,) for state
    )

    # To ensure state is correctly sharded before the loop begins:
    # If restoring, the restored state from Orbax might already be global arrays.
    # If creating new, explicitly shard it.
    # This can be done by running an identity pjit function or ensuring p_train_step's first call handles it.
    # For simplicity, let's assume the first call to p_train_step correctly handles sharding of initial_host_state.
    state = initial_host_state  # Start with host state; pjit will manage sharding.

    logging.info("Starting training loop...")
    for step_idx in range(state.step, cfg.optim.num_steps):
        # Get next batch of data (already sharded by the dataloader)
        data = next(dataloader)

        # Generate a new PRNG key for this training step (e.g., for dropout)
        step_rng = jax.random.fold_in(training_loop_rng_key, step_idx)

        # Perform the training step.
        # The `set_mesh(mesh)` context manager ensures JAX knows which mesh to use for operations
        # like `lax.pmean` called within `p_train_step`.
        # `nnx.flags(modify=True)` might be needed if NNX operations modify global graph state,
        # though with functional updates to `nnx.State`, it's often not strictly necessary for params.
        with mesh, nnx.flags(
            modify=True
        ):  # modify=True for safety with nnx.merge/update
            state, metrics = p_train_step(state, data, step_rng)

        # Log metrics (metrics are already averaged)
        # Ensure metrics are transferred to host for printing/logging if they are JAX arrays on devices
        # This usually happens automatically for small arrays or can be forced with jax.device_get()
        if (
            step_idx + 1
        ) % cfg.logging.log_freq == 0:  # Assuming log_freq in LoggingConfig
            # meter.update_metrics(metrics) # Your existing meter
            # logged_metrics = meter.log()
            log_output = {
                k: float(v.item()) if hasattr(v, "item") else float(v)
                for k, v in metrics.items()
            }
            logging.info(f"Step {step_idx+1}/{cfg.optim.num_steps}: {log_output}")
            # if cfg.logging.use_wandb: # Your wandb logging
            #    wandb.log(log_output, step=step_idx)

        # Save a checkpoint periodically
        if (step_idx + 1) % cfg.logging.save_freq == 0:
            # Ensure save_checkpoint is called in a way that it can access the sharded state
            # or handles gathering it. Orbax manager should handle this.
            save_checkpoint(state, checkpoint_manager, step_idx + 1)

    logging.info("Training finished.")
    # Final save or cleanup
    if cfg.logging.save_freq > 0:  # ensure cfg.logging.save_freq is positive
        save_checkpoint(state, checkpoint_manager, cfg.optim.num_steps)


def get_cfg() -> TrainConfig:
    cli_args = OmegaConf.from_cli()
    # Ensure TrainConfig is registered if it contains complex types not native to OmegaConf
    # OmegaConf.register_new_resolver("eval", eval) # Example if needed
    default_cfg = OmegaConf.structured(TrainConfig)
    cfg_merged = OmegaConf.merge(default_cfg, cli_args)
    print("Effective configuration:\n", OmegaConf.to_yaml(cfg_merged))

    # Convert to the actual dataclass instance
    # This might fail if OmegaConf cannot instantiate complex objects within TrainConfig
    # You might need to manually convert parts or use OmegaConf.to_container with care.
    cfg_obj: TrainConfig = OmegaConf.to_object(cfg_merged)

    # Dump config
    os.makedirs(cfg_obj.logdir, exist_ok=True)
    with open(os.path.join(cfg_obj.logdir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg_merged, f=f)

    return cfg_obj


if __name__ == "__main__":
    # Potentially initialize JAX distributed environment early if needed globally
    # distributed.initialize() # If your setup requires it here

    cfg: TrainConfig = get_cfg()
    train(cfg)
