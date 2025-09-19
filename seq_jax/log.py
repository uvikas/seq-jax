"""Logging utilities."""

from typing import List, Dict, Any

import os
import wandb
import numpy as np
import jax
import decimal
import simplejson
import time
import jax.numpy as jnp

from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque


@dataclass
class WandbConfig:
    project: str = "seq-jax"
    entity: str = "my-gcp-project"
    mode: str = "disabled"
    notes: str = ""


@dataclass
class ProfilerConfig:
    enabled: bool = False
    profile_dir: str = "/tmp/jax_profile"
    port: int = 9999
    start_step: int = 5
    stop_step: int = 10


@dataclass
class LoggingConfig:
    save_freq: int = 1000
    log_freq: int = 50
    losses: List[str] = field(
        default_factory=lambda: [
            "action_loss",
            "im_head_loss",
            "im_left_loss",
            "im_right_loss",
            "state_loss",
            "total_loss",
        ]
    )
    wandb: WandbConfig = field(default_factory=WandbConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    gcs_ckpt: str | None = None


def init_wandb(cfg: "TrainConfig"):
    if jax.process_index() != 0:
        return

    if cfg.log.wandb.mode == "disabled":
        wandb.init(mode="disabled")
        return

    run_id_file = os.path.join(cfg.logdir, cfg.exp_name, "wandb_id.txt")

    if cfg.resume and os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        wandb.init(
            id=run_id,
            resume="must",
            project=cfg.log.wandb.project,
            notes=cfg.log.wandb.notes,
        )
    else:
        wandb.init(
            dir=os.path.join(cfg.logdir, cfg.exp_name),
            name=cfg.exp_name,
            project=cfg.log.wandb.project,
            entity=cfg.log.wandb.entity,
            config=asdict(cfg),
            mode=cfg.log.wandb.mode,
            notes=cfg.log.wandb.notes,
        )
        with open(run_id_file, "w") as f:
            f.write(wandb.run.id)


def init_profiler(cfg: ProfilerConfig):
    if cfg.enabled:
        os.makedirs(cfg.profile_dir, exist_ok=True)
        try:
            jax.profiler.start_server(cfg.port)
            print(f"[Profiler] JAX profiler server started on 0.0.0.0:{cfg.port}. ")
        except Exception as e:
            print(f"[Profiler] Failed to start profiler server: {e}")


def start_trace(cfg: ProfilerConfig, step: int):
    trace_dir = os.path.join(cfg.profile_dir, f"step_{step}")
    os.makedirs(trace_dir, exist_ok=True)
    print(f"[Profiler] Starting JAX trace. Output dir: {trace_dir}")
    try:
        jax.profiler.start_trace(trace_dir)
    except Exception as _e:
        print(f"[Profiler] Failed to start trace: {_e}")


def stop_trace(cfg: ProfilerConfig, step: int):
    try:
        jax.profiler.stop_trace()
        print(
            f"[Profiler] Stopped JAX trace at step {step}. Trace saved to {cfg.profile_dir}"
        )
    except Exception as e:
        print(f"[Profiler] Failed to stop trace: {e}")


def time_string(seconds):
    """Converts time in seconds to a fixed-width string format."""
    days, rem = divmod(int(seconds), 24 * 3600)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)


def float_to_decimal(data, prec=4):
    """Convert floats to decimals which allows for fixed width json."""
    if prec and isinstance(data, dict):
        return {k: float_to_decimal(v, prec) for k, v in data.items()}
    if prec and isinstance(data, float):
        return decimal.Decimal(("{:." + str(prec) + "f}").format(data))
    else:
        return data


def dump_json_stats(stats, stats_type):
    """Covert stats dict into json string for logging."""
    stats = float_to_decimal(stats)
    stats_json = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    return "{:s}: {:s}".format(stats_type, stats_json)


def cos_sim_grad(
    current_grad,
    grad_deque: deque,
    eps: float = 1e-8,
) -> jnp.ndarray:
    cur_norm = jnp.linalg.norm(current_grad)
    flats = jnp.stack(grad_deque)  # (H, D)
    dots = flats @ current_grad  # (H,)
    norms = jnp.linalg.norm(flats, axis=1) * cur_norm + eps
    return jnp.mean(dots / norms).item()  # (H,)


class Timer:
    """A simple timer."""

    def __init__(self) -> None:
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0
        self.times = deque(maxlen=500)
        self.reset()

    def tic(self) -> None:
        self.start_time = time.perf_counter()

    def toc(self) -> None:
        self.diff = time.perf_counter() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        self.times.append(self.diff)
        self.sliding_average_time = sum(self.times) / len(self.times)

    def latest(self) -> float:
        return self.diff

    def reset(self) -> None:
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0


class TimerMeter:
    """Measures timing stats for different operations."""

    def __init__(self):
        self.timers = defaultdict(Timer)

    def reset(self):
        for timer in self.timers.values():
            timer.reset()

    def __call__(self, name):
        """Returns a context manager that times a code block."""
        timer = self.timers[name]

        class TimerContext:
            def __enter__(self_):
                timer.tic()
                return timer

            def __exit__(self_, *args):
                timer.toc()

        return TimerContext()

    def get_stats(self):
        return {name: timer.total_time for name, timer in self.timers.items()}


class ScalarMeter(object):
    """Measures a scalar value."""

    def __init__(self, window_size: int):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value: float | List[float]):
        if not isinstance(value, list):
            value = [value]
        self.deque.extend(value)
        self.count += len(value)
        self.total += sum(value)

    def get_value(self, mode: str = "mean") -> float:
        if not self.deque:
            return 0.0
        if mode == "median":
            return self.get_win_median()
        elif mode == "mean":
            return self.get_win_mean()
        elif mode == "global_mean":
            return self.get_global_mean()
        elif mode == "latest":
            return self.get_latest()
        else:
            raise NotImplementedError

    def get_win_median(self) -> float:
        return np.median(list(self.deque))

    def get_win_mean(self) -> float:
        return np.mean(list(self.deque))

    def get_global_mean(self) -> float:
        return self.total / self.count

    def get_latest(self) -> float:
        return self.deque[-1]


class IterTrainMeter(object):
    """Measures training stats for iter-based training."""

    def __init__(
        self,
        loss_names: List[str],
        max_iter: int,
        log_period: int = 20,
        warmup_period: int = 20,
        phase: str = "train",
    ):
        self.max_iter = max_iter
        self.log_period = log_period
        self.warmup_period = warmup_period
        self.phase = phase
        self.iter_timer = Timer()

        self.loss_names = loss_names
        self.losses = {name: ScalarMeter(log_period) for name in loss_names}
        self.per_token_loss = None

        self.lr = None
        self.batch_size = 0
        self.prefetched_size = 0
        self.num_samples = 0
        self.grads = deque(maxlen=1)

        self.timer = TimerMeter()

    def reset(self, timer: bool = False):
        if timer:
            self.iter_timer.reset()

        for name in self.losses:
            self.losses[name].reset()
        self.timer.reset()

        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update(
        self,
        metrics: Dict[str, Any],
    ):
        for name, value in metrics["loss"].items():
            if name in self.losses:
                self.losses[name].add_value(value)
        self.per_token_loss = metrics["loss"]["per_token_loss"]
        self.lr = metrics["lr"]
        self.num_samples += metrics["batch_size"]
        self.prefetched_size = metrics["prefetch"]
        self.batch_size = metrics["batch_size"]
        self.grad_norm = metrics["grad_norm"]
        self.param_norm = metrics["param_norm"]
        self.sim = cos_sim_grad(metrics["grad"], self.grads) if self.grads else 0
        self.grads.append(metrics["grad"])

    def get_iter_stats(self, cur_iter: int):
        cur_iter_total = cur_iter + 1
        eta_sec = self.iter_timer.sliding_average_time * (
            self.max_iter - cur_iter_total
        )
        stats = {
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "eta": time_string(eta_sec),
            "lr": self.lr,
            "prefetched": self.prefetched_size,
            "batch_sz": self.batch_size,
            "grad_norm": self.grad_norm,
            "param_norm": self.param_norm,
            "grad_cos_sim": self.sim,
        }

        loss_stats = {
            name: self.losses[name].get_win_median() for name in self.loss_names
        } | {"per_token_loss": self.per_token_loss}

        latency_stats = {
            f"{name}_s": time_s for name, time_s in self.timer.get_stats().items()
        }

        return stats, loss_stats, latency_stats

    def log_iter_stats(self, cur_iter: int):
        stats, loss_stats, latency_stats = self.get_iter_stats(cur_iter)
        stats["latency"] = latency_stats
        stats["loss"] = loss_stats

        if (cur_iter + 1) <= self.warmup_period or (
            cur_iter + 1
        ) % self.log_period == 0:
            print(dump_json_stats(stats, self.phase + "_iter"))

        if jax.process_index() == 0:
            wandb.log(
                {
                    "train/iter": cur_iter + 1,
                    "train/lr": stats["lr"],
                    "train/grad_norm": stats["grad_norm"],
                    "train/param_norm": stats["param_norm"],
                    "train/grad_cos_sim": stats["grad_cos_sim"],
                    "per_token_loss": wandb.plot.line_series(
                        xs=[list(range(len(loss_stats["per_token_loss"])))],
                        ys=[loss_stats["per_token_loss"]],
                        keys=[f"step_{cur_iter}"],
                        title="Per-token loss",
                        xname="token_index",
                        yname="loss",
                    ),
                }
                | {f"train/iter_{name}": loss_stats[name] for name in self.loss_names}
                | {
                    f"latency/iter_{name}": latency_stats[name]
                    for name in latency_stats.keys()
                },
                step=cur_iter,
            )


def make_meter(cfg: LoggingConfig, total_iter: int) -> IterTrainMeter:
    return IterTrainMeter(
        cfg.losses,
        total_iter,
        log_period=cfg.log_freq,
    )
