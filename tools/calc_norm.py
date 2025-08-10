"""Calculate normalization statistics for a dataset."""

import argparse
import numpy as np

from dataclasses import dataclass

from seq_jax.data import DataConfig, make_dataset



@dataclass
class NormStats:
    mean: np.ndarray
    var: np.ndarray
    count: int

    def update(self, sample: np.ndarray):
        self.mean = (self.mean * self.count + sample) / (self.count + 1)
        self.var = (self.var * self.count + (sample - self.mean) ** 2) / (self.count + 1)
        self.count += 1

def calc_norm(cfg: DataConfig):
    """Calculate normalization statistics for a dataset."""
    dataset = make_dataset(cfg, context_len=1, pred_len=1)
    state_stats = NormStats(
        mean=np.zeros(24),
        var=np.zeros(24),
        count=0,
    )
    action_stats = NormStats(
        mean=np.zeros(24),
        var=np.zeros(24),
        count=0,
    )

    for vision, state, action, ar_mask in dataset:
        state_stats.update(state)
        action_stats.update(action)

    print(state_stats)
    print(action_stats)


if __name__ == "__main__":
    cfg = DataConfig(
        chunked=True,
        dataset_root="gs://DATA_BUCKET/chunk64-vitb",
        dataset_names=["*"],
        global_batch_size=256,
        shufflebuffer=4096,
        dl_shufflebuffer=2048,
        shardshuffle=100,
        prefetch_device_factor=5,
        num_workers=4,
    )
    calc_norm(cfg)