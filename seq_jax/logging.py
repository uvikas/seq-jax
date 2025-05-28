"""Meters."""

from typing import List, Dict, Any

import os
import wandb
import numpy as np
import decimal
import simplejson
import time

from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque


@dataclass
class WandbConfig:
    project: str = "seq-jax"
    entity: str = "my-gcp-project"
    mode: str = "disabled"


@dataclass
class LoggingConfig:
    save_freq: int = 1000
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


def init_wandb(cfg: "TrainConfig"):
    name = os.path.basename(os.path.normpath(cfg.logdir))
    wandb.init(
        dir=cfg.logdir,
        name=name,
        project=cfg.log.wandb.project,
        entity=cfg.log.wandb.entity,
        config=asdict(cfg),
        mode=cfg.log.wandb.mode,
    )


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


class Timer:
    """A simple timer."""

    def __init__(self) -> None:
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0
        self.reset()

    def tic(self) -> None:
        self.start_time = time.perf_counter()

    def toc(self) -> None:
        self.diff = time.perf_counter() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

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
        log_period: int = 1,
        phase: str = "train",
    ):
        self.max_iter = max_iter
        self.log_period = log_period
        self.phase = phase
        self.iter_timer = Timer()

        self.loss_names = loss_names
        self.losses = {name: ScalarMeter(log_period) for name in loss_names}

        self.lr = None
        self.batch_size = 0
        self.prefetched_size = 0
        self.num_samples = 0

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
            self.losses[name].add_value(value)
        self.lr = metrics["lr"]
        self.num_samples += metrics["batch_size"]
        self.prefetched_size = 0
        self.batch_size = metrics["batch_size"]
        self.grad_norm = metrics["grad_norm"]
        self.param_norm = metrics["param_norm"]

    def get_iter_stats(self, cur_iter: int):
        cur_iter_total = cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        stats = {
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "eta": time_string(eta_sec),
            "lr": self.lr,
            "prefetched": self.prefetched_size,
            "batch_sz": self.batch_size,
            "grad_norm": self.grad_norm,
        }

        loss_stats = {
            name: self.losses[name].get_win_median() for name in self.loss_names
        }

        latency_stats = {
            f"{name}_s": time_s for name, time_s in self.timer.get_stats().items()
        }

        return stats, loss_stats, latency_stats

    def log_iter_stats(self, cur_iter: int):
        stats, loss_stats, latency_stats = self.get_iter_stats(cur_iter)
        stats["latency"] = latency_stats
        stats["loss"] = loss_stats

        if (cur_iter + 1) % self.log_period == 0:
            print(dump_json_stats(stats, self.phase + "_iter"))

        wandb.log(
            {
                "train/iter": cur_iter + 1,
                "train/lr": stats["lr"],
                "train/grad_norm": stats["grad_norm"],
            }
            | {f"train/iter_{name}": loss_stats[name] for name in self.loss_names}
            | {
                f"latency/iter_{name}": latency_stats[name]
                for name in latency_stats.keys()
            }
        )


def make_meter(cfg: LoggingConfig, total_iter: int):
    return IterTrainMeter(
        cfg.losses,
        total_iter,
    )
