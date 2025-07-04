from typing import List

import os

import atexit
import contextlib
import jax
import numpy as np
import shutil
import tempfile

from dataclasses import dataclass, asdict, field

BATCH_AXIS = "data"
FSDP_AXIS = "model"


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


@dataclass
class DistributedConfig:
    platform: str = "gpu"

    # (data, model)
    mesh_shape: List[int] = field(default_factory=lambda: [8, 1])

    init_dist: bool = False
    coord_addr: str | None = None
    num_processes: int | None = None
    devices_per_process: int | None = None
    process_id: int | None = None


@dataclass
class EnvConfig:
    OMP_NUM_THREADS: str = "1"
    NCCL_DEBUG: str = "WARN"
    LIBTPU_INIT_ARGS: str = (
        "--xla_tpu_megacore_fusion_allow_ags=false "
        "--xla_enable_async_collective_permute=true "
        "--xla_tpu_enable_ag_backward_pipelining=true "
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
        "--xla_tpu_data_parallel_opt_different_sized_ops=true "
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    )


def setup_env(cfg: EnvConfig):
    env_vars = asdict(cfg)

    # Triton cache in tmpfs
    triton_cache_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    env_vars["TRITON_CACHE_DIR"] = triton_cache_dir

    # Set env vars
    for name, value in env_vars.items():
        if os.environ.get(name) != str(value):
            os.environ[name] = str(value)
            print(f"WARNING: Setting {name} to {value}")

    # JAX config
    jax.config.update("jax_default_matmul_precision", "float32")

    # See https://docs.jax.dev/en/latest/xla_flags.html
    # os.environ.setdefault(
    #     "XLA_FLAGS",
    #     "--xla_gpu_enable_triton_gemm=true ",
    # )


def init_dist(cfg: DistributedConfig):
    if not cfg.init_dist:
        return

    print(f"Initializing distributed with {cfg=}")
    jax.distributed.initialize(
        coordinator_address=cfg.coord_addr,
        num_processes=cfg.num_processes,
        process_id=cfg.process_id,
    )

    print(
        f"Initialized process {cfg.process_id} of {cfg.num_processes}: {jax.devices()=} {jax.local_devices()=}"
    )


def make_mesh(cfg: DistributedConfig):
    devices = np.array(jax.devices()).reshape(cfg.mesh_shape)
    mesh = jax.sharding.Mesh(devices, (BATCH_AXIS, FSDP_AXIS))
    print(f"Created mesh {mesh=}")
    return mesh


def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree,
        jax.sharding.NamedSharding(
            _MeshState.active_mesh, jax.sharding.PartitionSpec(BATCH_AXIS)
        ),
    )


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """Plumbing the mesh deep into the module tree is extremeley cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below.
    """
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None
