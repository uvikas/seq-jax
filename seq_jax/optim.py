from dataclasses import dataclass
from typing import Tuple

import jax
import optax

from flax import nnx


@dataclass
class OptimConfig:
    lr: float = 0.0005
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    clip_norm: float = 1.0

    scheduler: str = "cosine"
    warmup_steps: int = 1000
    num_steps: int = 10_000
    final_lr_ratio: float = 0.0  # Final LR is lr * final_lr_ratio

    ema_alpha: float = 1e-5


_EXCLUDE_FROM_WEIGHT_DECAY = (
    "ln",
    "token",
    "bn",
    "bias",
    "scale",
    "pos_embed",
    "mean",
    "var",
    "img_encoder",
)


def wd_mask(params: nnx.State) -> nnx.State:
    def _keep(path, _leaf):
        path_str = ".".join(map(str, path)).lower()
        return not any(substr in path_str for substr in _EXCLUDE_FROM_WEIGHT_DECAY)

    return jax.tree_util.tree_map_with_path(_keep, params)


def make_optimizer(
    cfg: OptimConfig, lr_scheduler: optax.Schedule, params: nnx.State
) -> optax.GradientTransformation:
    """Create optimizer."""
    mask = wd_mask(params)
    #print(
    #    "Excluding from weight decay: ",
    #    ", ".join(
    #        [
    #            ".".join(map(str, [e.key if e is not None else e.index for e in path]))
    #            for path, apply_wd in jax.tree.flatten_with_path(mask)[0]
    #            if not apply_wd
    #        ]
    #    ),
    #)
    return optax.chain(
        optax.clip_by_global_norm(cfg.clip_norm),
        optax.adamw(
            learning_rate=lr_scheduler,
            weight_decay=cfg.weight_decay,
            b1=cfg.betas[0],
            b2=cfg.betas[1],
            eps=1e-8,
            mask=mask,
        ),
    )


def make_lr_scheduler(cfg: OptimConfig) -> optax.Schedule:
    """Create learning rate scheduler."""

    match cfg.scheduler:
        case "cosine":
            warmup = optax.linear_schedule(
                init_value=0.0,
                end_value=cfg.lr,
                transition_steps=cfg.warmup_steps,
            )
            cosine = optax.cosine_decay_schedule(
                init_value=cfg.lr,
                decay_steps=cfg.num_steps - cfg.warmup_steps,
                alpha=cfg.final_lr_ratio,
            )
            schedule = optax.join_schedules(
                schedules=[warmup, cosine], boundaries=[cfg.warmup_steps]
            )
            return schedule
        case _:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
