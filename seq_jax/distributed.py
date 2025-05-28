import contextlib
import jax
import numpy as np

from dataclasses import dataclass
from jax.sharding import Mesh


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


@dataclass
class DistributedConfig:
    platform: str = "gpu"
    num_devices: int = 8
    spawn_method: str = "fork"


@dataclass
class EnvConfig:
    OMP_NUM_THREADS: str = "1"
    NCCL_DEBUG: str = "WARN"

    # Only enable for brc
    # NCCL_P2P_DISABLE: str = "1"


def make_mesh(cfg: DistributedConfig):
    """1d data parallel sharding"""
    devices = np.array(jax.devices())
    mesh = Mesh(devices, ("data",))
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
