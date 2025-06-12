import os

import atexit
import contextlib
import jax
import shutil
import tempfile

from dataclasses import dataclass, asdict


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
