import jax
import numpy as np

from dataclasses import dataclass
from jax.sharding import Mesh


@dataclass
class DistributedConfig:
    platform: str = "gpu"
    num_devices: int = 1
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
