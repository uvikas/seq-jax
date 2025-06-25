"""Dataloader."""

from typing import Any, Iterator, List
import os

import jax
import jax.dlpack as jdl
import jax.numpy as jnp
import multiprocessing
import numpy as np
import queue
import threading
import torch
import webdataset as wds

from dataclasses import dataclass, field
from jax.sharding import Sharding
from functools import partial
from glob import glob

from seq_jax.policy import Observation, Prediction


@dataclass
class DataConfig:
    dataset_type: str = "wds"
    chunked: bool = False
    global_batch_size: int = 256
    shufflebuffer: int = 1024
    shardshuffle: int = 100
    num_workers: int = 4
    prefetch_factor: int | None = None

    dataset_root: str = "/path/to/wds/fold-h16a1fs2-vitb-5min/"
    dataset_names: List[str] = field(default_factory=lambda: ["*"])
    dtype: str = "float32"
    vision_format: str = "mp4"

    im_size: int = 224
    dof_size: int = 24
    touch_dim: int = 60
    cam_names: List[str] = field(default_factory=lambda: ["left", "head", "right"])

    action_type: str = "absolute"

    default_pos: List[float] = field(
        default_factory=lambda: [
            # Left arm
            1.89,
            -1.12,
            1.41,
            -0.03,
            1.24,
            2.82,
            # Left hand
            0.17,
            0.17,
            0.17,
            0.17,
            0.17,
            -1.22,
            # Right arm
            4.39,
            -2.02,
            -1.41,
            3.17,
            -1.24,
            3.47,
            # Right hand
            0.17,
            0.17,
            0.17,
            0.17,
            0.17,
            -1.22,
        ]
    )

    joint_noise_mean: List[float] = field(
        default_factory=lambda: [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    joint_noise_std: List[float] = field(
        default_factory=lambda: [
            0.0213,
            0.0246,
            0.0211,
            0.0378,
            0.0382,
            0.0255,
            0.0273,
            0.0373,
            0.0329,
            0.0500,
            0.0738,
            0.0741,
            0.0202,
            0.0198,
            0.0211,
            0.0417,
            0.0396,
            0.0308,
            0.0795,
            0.0652,
            0.0638,
            0.0567,
            0.0532,
            0.0575,
        ]
    )
    joint_noise_std_scale: float = 1.0
    frame_skip: int = 2


def pack_batch(
    batch,
    default_pos: np.array,
    noise_mean: np.array,
    noise_std: np.array,
    noise_scale: float,
    action_type: str,
    context_len: int,
    pred_len: int,
):
    batch = jax.tree.map(lambda x: jdl.from_dlpack(x), batch)
    vision, states, actions, masks = batch

    if action_type == "absolute":
        states -= default_pos
        actions -= default_pos
    elif action_type == "relative":
        actions = actions - states

    noise = np.random.normal(
        loc=noise_mean,
        scale=noise_std * noise_scale,
        size=states.shape,
    )

    obs = Observation(
        img_rgb=None,
        img_feat=vision,
        img_pooled=None,
        state=states,
        noisy_state=states + noise,
        prev_action=actions,
        mask=masks,
    )
    pred = Prediction(
        img_pooled=None,
        state=None,
        action=jnp.stack(
            [actions[:, i : i + pred_len, ...] for i in range(context_len)],
            axis=1,
        ),
    )

    return obs, pred


def _collate_fn(items):
    """Collate the batch elements into torch tensors for more efficient IPC."""
    val = jax.tree.map(lambda x: torch.from_numpy(x), items)
    return val


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class PrefetchLoader(Iterator):
    def __init__(
        self,
        dataloader: Iterator[Any],
        sharding: jax.sharding.Sharding,
        buffer_size: int = 5,
        num_workers: int = 4,
    ):
        self._dl_iter = iter(dataloader)
        self._sharding = sharding
        self._queue = queue.Queue(maxsize=buffer_size)
        self._sentinel = object()

        def _worker():
            for b in self._dl_iter:
                # `b` is a pytree of torch.Tensors → numpy → sharded JAX arrays
                b = jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(
                        sharding,
                        x,
                    ),
                    b,
                )
                self._queue.put(b)
            self._queue.put(self._sentinel)

        threading.Thread(target=_worker, daemon=True).start()

    def prefetched(self):
        return self._queue.qsize()

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        batch = self._queue.get()
        if batch is self._sentinel:
            raise StopIteration
        return batch


class TorchDataLoader:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        cfg: DataConfig,
        context_len: int,
        pred_len: int,
        sharding: jax.sharding.Sharding | None = None,
        seed: int = 0,
    ):
        """Create a PyTorch data loader."""
        if sharding is None:
            raise ValueError("Sharding is required")

        self._sharding = sharding
        self._batch_size = cfg.global_batch_size // jax.process_count()
        mp_context = None
        if cfg.num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=cfg.num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=generator,
            prefetch_factor=cfg.prefetch_factor,
        )

        self.pack_batch = partial(
            pack_batch,
            default_pos=np.array(cfg.default_pos),
            noise_mean=np.array(cfg.joint_noise_mean),
            noise_std=np.array(cfg.joint_noise_std),
            noise_scale=cfg.joint_noise_std_scale,
            action_type=cfg.action_type,
            context_len=context_len,
            pred_len=pred_len,
        )

    def __iter__(self):
        while True:
            data_iter = iter(self._data_loader)
            while True:
                try:
                    batch = next(data_iter)
                    batch = self.pack_batch(batch)

                    if (
                        batch[0].state.shape[0] != self._batch_size
                        or jnp.isnan(batch[0].img_feat).any()
                        or jnp.isnan(batch[0].state).any()
                        or jnp.isnan(batch[0].noisy_state).any()
                        or jnp.isnan(batch[0].prev_action).any()
                        or jnp.isnan(batch[0].mask).any()
                        or jnp.isnan(batch[1].action).any()
                    ):
                        print("Skipping batch with batch size", batch[0].state.shape[0])
                        continue
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                yield batch


def random_segmenter(ctx_len: int, action_len: int):
    """
    sample = (vision, state, action)
    vision : (T, …)  torch.Tensor / np.ndarray
    state  : (T, …)
    action : (T, …)

    Returns the same tuple but with the first dimension cropped to
    `seg_len`.  A new random start index is drawn for every sample.
    """

    def _fn(sample):
        vision, state, action, json = sample  # <- unpack
        steps = vision.shape[0]  # total time-steps

        if json["begin"]:
            start = np.random.randint(-ctx_len + 1, steps - ctx_len - action_len + 1)
            if start < 0:
                pad = -start
                start = 0
            else:
                pad = 0
        else:
            pad = 0
            start = np.random.randint(0, steps - ctx_len - action_len + 1)
        end = start + ctx_len + action_len - pad

        vision = vision[start:end]
        state = state[start:end]
        action = action[start:end]
        ar_mask = np.zeros(end - start)

        if pad > 0:
            vision = np.concatenate([np.array([vision[0]] * pad), vision], axis=0)
            state = np.concatenate([np.array([state[0]] * pad), state], axis=0)
            action = np.concatenate([np.array([action[0]] * pad), action], axis=0)
            ar_mask = np.concatenate([np.array([1] * pad), ar_mask], axis=0)

        assert vision.shape[0] == ctx_len + action_len
        assert state.shape[0] == ctx_len + action_len
        assert action.shape[0] == ctx_len + action_len
        assert ar_mask.shape[0] == ctx_len + action_len

        return (vision, state, action, ar_mask)

    return _fn


def _ctx_slice(x, context_len):
    return x[:context_len]


def _pred_slice(x, context_len, pred_len):
    return x[: context_len + pred_len - 1]


def _validate_sample(sample):
    """Validate that the sample doesn't contain any NaN/Inf values."""
    vision, state, action, mask = sample
    return (
        not np.isnan(vision).any()
        and not np.isnan(state).any()
        and not np.isnan(action).any()
        and not np.isnan(mask).any()
        and not np.isinf(vision).any()
        and not np.isinf(state).any()
        and not np.isinf(action).any()
        and not np.isinf(mask).any()
    )


def make_dataset(cfg: DataConfig, context_len: int, pred_len: int) -> wds.WebDataset:
    import subprocess
    from urllib.parse import urlparse

    def _is_gcs_path(path: str) -> bool:
        """Check if path is a Google Cloud Storage path."""
        return path.startswith("gs://")

    def _list_gcs_shards(gcs_path: str) -> List[str]:
        """List shard files in GCS using gsutil."""
        try:
            # Use gsutil to list files matching the pattern
            cmd = ["gsutil", "ls", f"{gcs_path}/*/shard-*.tar"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
        except subprocess.CalledProcessError:
            # If the exact pattern fails, try a broader search and filter
            try:
                cmd = ["gsutil", "ls", "-r", gcs_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                all_files = [
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]
                # Filter for shard files
                return [f for f in all_files if "/shard-" in f and f.endswith(".tar")]
            except subprocess.CalledProcessError:
                return []

    def _list_gcs_directories(gcs_path: str) -> List[str]:
        """List directories in GCS."""
        try:
            cmd = ["gsutil", "ls", "-d", f"{gcs_path}/*/"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            dirs = [
                line.strip().rstrip("/")
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            return [os.path.basename(d) for d in dirs]
        except subprocess.CalledProcessError:
            return []

    # Determine if we're working with GCS or local paths
    if _is_gcs_path(cfg.dataset_root):
        # Handle GCS paths
        roots = []
        names = []

        for dataset_name in cfg.dataset_names:
            if dataset_name == "*":
                # List all directories in the root
                all_dirs = _list_gcs_directories(cfg.dataset_root)
                for dir_name in all_dirs:
                    gcs_pattern = f"{cfg.dataset_root}/{dir_name}"
                    shards = _list_gcs_shards(gcs_pattern)
                    roots.extend(shards)
                    if shards:  # Only add to names if we found shards
                        names.append(dir_name)
            else:
                # Specific dataset name
                gcs_pattern = f"{cfg.dataset_root}/{dataset_name}"
                shards = _list_gcs_shards(gcs_pattern)
                roots.extend(shards)
                if shards:  # Only add to names if we found shards
                    names.append(dataset_name)
    else:
        # Handle local filesystem paths (original logic)
        roots = []
        for dataset_name in cfg.dataset_names:
            pattern = os.path.join(cfg.dataset_root, dataset_name, "*", "shard-*.tar")
            matches = glob(pattern)
            roots.extend(matches)

        names = []
        for dataset_name in cfg.dataset_names:
            pattern = os.path.join(cfg.dataset_root, dataset_name)
            matches = glob(pattern)
            names.extend([os.path.basename(item) for item in matches])

    assert (
        len(roots) > 0
    ), f"No shards found in {cfg.dataset_root} with datasets {cfg.dataset_names}"
    np.random.shuffle(roots)

    print("Webdataset using", len(roots), "shards")

    # Sort names consistently
    names = sorted(set(names), key=lambda x: x.split("_")[-1] if "_" in x else x)
    print(f"Dataset ({len(names)})", names)

    dataset = wds.WebDataset(
        roots,
        resampled=False,
        shardshuffle=cfg.shardshuffle,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.shardlists.split_by_worker,
    )

    if cfg.chunked:
        dataset = (
            dataset.shuffle(cfg.shufflebuffer)
            .decode("torch")
            .repeat()
            .to_tuple("vision.npy", "state.npy", "action.npy", "json")
            .select(
                lambda sample: sample[0].shape[0]
                >= cfg.actor.context_len + cfg.actor.action_len
            )
            .map(random_segmenter(cfg.actor.context_len, cfg.actor.action_len))
            .select(_validate_sample)
            .batched(cfg.global_batch_size // jax.process_count())
        )
    else:
        dataset = (
            dataset.shuffle(cfg.shufflebuffer)
            .decode("torch")
            .repeat()
            .to_tuple("vision.npy", "state.npy", "action.npy", "mask.npy")
            .map_tuple(
                partial(_ctx_slice, context_len=context_len),
                partial(_ctx_slice, context_len=context_len),
                partial(_pred_slice, context_len=context_len, pred_len=pred_len),
                partial(_ctx_slice, context_len=context_len),
            )
            .select(_validate_sample)
            .batched(cfg.global_batch_size // jax.process_count())
        )
    return dataset


def make_dataloader(
    cfg: DataConfig, context_len: int, pred_len: int, sharding: Sharding
) -> wds.WebLoader:
    dataset = make_dataset(cfg, context_len, pred_len)
    dataloader = TorchDataLoader(
        dataset,
        cfg,
        context_len,
        pred_len,
        sharding,
        seed=0,
    )
    dataloader = iter(dataloader)
    dataloader = PrefetchLoader(dataloader, sharding)
    return dataloader
