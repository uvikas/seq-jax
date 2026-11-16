"""Calculate normalization statistics for a dataset."""

import argparse
import os
import numpy as np
import pytest

from dataclasses import dataclass
from tqdm import tqdm

from seq_jax.data import DataConfig, make_dataset



@dataclass
class NormStats:
    mean: np.ndarray
    var: np.ndarray
    count: int

    def update_batch(self, batch: np.ndarray):
        n_batch = batch.shape[0]

        batch_mean = batch.mean(axis=0, dtype=np.float64)
        batch_var = batch.var(axis=0, dtype=np.float64)

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = n_batch
            return

        delta = batch_mean - self.mean
        total_count = self.count + n_batch

        self.mean = self.mean + delta * n_batch / total_count

        m2_self = self.var * self.count
        m2_batch = batch_var * n_batch
        m2 = m2_self + m2_batch + (delta ** 2) * self.count * n_batch / total_count
        self.var = m2 / total_count

        self.count = total_count

@pytest.mark.parametrize("num_samples, features, rng_seed", [(10000, 24, 0), (12345, 10, 42)])
def test_running_stats_match_numpy(num_samples: int, features: int, rng_seed: int):
    """Generate random data, feed it in variable-sized batches and compare statistics."""
    rng = np.random.default_rng(rng_seed)
    data = rng.standard_normal(size=(num_samples, features)).astype(np.float32)

    # Split into variable-length batches of size 1–128
    idx = 0
    stats = NormStats(mean=np.zeros(features, dtype=np.float64),
                      var=np.zeros(features, dtype=np.float64),
                      count=0)
    while idx < num_samples:
        batch_size = rng.integers(1, 129)
        batch = data[idx: idx + batch_size]
        stats.update_batch(batch)
        idx += batch_size

    # Reference statistics computed in one pass
    ref_mean = data.mean(axis=0)
    ref_var = data.var(axis=0)  # population variance (ddof=0)

    np.testing.assert_allclose(stats.mean, ref_mean, rtol=1e-6, atol=1e-7)
    # Variance accumulates a bit more numerical error – 1e-5 is still
    # far below 0.01 % relative error and adequate for a sanity check.
    np.testing.assert_allclose(stats.var, ref_var, rtol=1e-5, atol=1e-7)
    assert stats.count == num_samples 

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

    for vision, state, action, ar_mask in tqdm(dataset):
        state_stats.update_batch(state)
        action_stats.update_batch(action)

    print(state_stats)
    print(action_stats)


if __name__ == "__main__":
    cfg = DataConfig(
        chunked=True,
        dataset_root=os.environ["DATA_ROOT"],
        dataset_names=["*"],
        global_batch_size=256,
        shufflebuffer=4096,
        dl_shufflebuffer=2048,
        shardshuffle=100,
        prefetch_device_factor=5,
        num_workers=4,
    )
    calc_norm(cfg)