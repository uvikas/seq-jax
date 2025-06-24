from typing import List

import os

import atexit
import contextlib
import jax
import numpy as np
import shutil
import tempfile

from dataclasses import dataclass, asdict, field
from jax.experimental import mesh_utils

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


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # if fsdp is not actually going to be used, replicate everything to avoid extraneous logging
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return None
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate small arrays
        if (
            arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize
        ) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    print(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(
                    mesh, jax.sharding.PartitionSpec(*spec)
                )

        # replicate if no valid sharding was found
        if log:
            print(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)


def make_mesh(cfg: DistributedConfig):
    num_devices = jax.local_device_count()
    mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((num_devices,)), (BATCH_AXIS,)
    )
    print(f"Created mesh {mesh=}")
    return mesh


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
