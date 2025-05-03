from dataclasses import dataclass


@dataclass
class DistributedConfig:
    platform: str = "gpu"
    num_gpus: int = 1
    spawn_method: str = "fork"


@dataclass
class EnvConfig:
    OMP_NUM_THREADS: str = "1"
    NCCL_DEBUG: str = "WARN"

    # Only enable for brc
    # NCCL_P2P_DISABLE: str = "1"
