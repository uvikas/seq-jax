"""Dataloader."""

from typing import Any, Iterator, List, Callable
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
import fsspec

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
    dl_shufflebuffer: int = 1024
    shardshuffle: int = 100
    prefetch_device_factor: int = 5
    num_workers: int = 4
    prefetch_factor: int | None = None

    dataset_root: str = "/path/to/wds/fold-h16a1fs2-vitb-5min/"
    dataset_names: List[str] = field(default_factory=lambda: ["*"])
    dtype: str = "float32"
    vision_format: str = "feat"

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


def tree_to_info(tree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging."""
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(
        f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree
    )


def array_tree_to_info(tree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


def pack_batch(
    batch,
    default_pos: np.array,
    noise_mean: np.array,
    noise_std: np.array,
    noise_scale: float,
    action_type: str,
    context_len: int,
    pred_len: int,
    vision_format: str,
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

    if vision_format == "feat":
        obs = Observation(
            img_rgb=None,
            img_feat=vision,
            img_pooled=None,
            state=states,
            noisy_state=states + noise,
            prev_action=actions,
            mask=masks,
        )
    elif vision_format == "mp4":
        # (B, T, V, H, W, 3) -> (B, T, V, 3, H, W)
        vision = vision.transpose(0, 1, 2, 5, 3, 4)
        obs = Observation(
            img_rgb=[vision[:, :, view_idx] for view_idx in range(vision.shape[2])],
            img_feat=None,
            img_pooled=None,
            state=states,
            noisy_state=states + noise,
            prev_action=actions,
            mask=masks,
        )
    else:
        raise ValueError(f"Invalid vision format: {vision_format}")

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
        buffer_size: int = 2,
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
        self._data_loader = (
            self._data_loader.unbatched()
            .shuffle(cfg.dl_shufflebuffer)
            .batched(cfg.global_batch_size // jax.process_count())
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
            vision_format=cfg.vision_format,
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


def slide(sample, context_len: int, action_len: int):
    """Turn one contiguous sample into 16-frame samples."""
    pre_vision, pre_state, pre_action, vision, state, action, json = sample
    ar_mask = np.zeros(vision.shape[0])

    if pre_vision.shape[0] > 0:
        vision = np.concatenate([pre_vision, vision], axis=0)
        state = np.concatenate([pre_state, state], axis=0)
        action = np.concatenate([pre_action, action], axis=0)
        ar_mask = np.concatenate([np.array([0] * pre_vision.shape[0]), ar_mask], axis=0)

    if json["begin"]:
        pad_len = context_len - 1
        vision = np.concatenate([np.array([vision[0]] * pad_len), vision], axis=0)
        state = np.concatenate([np.array([state[0]] * pad_len), state], axis=0)
        action = np.concatenate([np.array([action[0]] * pad_len), action], axis=0)
        ar_mask = np.concatenate([np.array([1] * pad_len), ar_mask], axis=0)

    # torch.unfold / as_strided is fastest, but a plain loop
    # is easier to read and plenty fast for 100 frames.
    samples = []
    for i in range(vision.shape[0] - context_len - action_len + 1):
        start, end = i, i + context_len
        act_start, act_end = i, i + context_len + action_len - 1

        assert vision[start:end].shape[0] == context_len
        assert state[start:end].shape[0] == context_len
        assert action[act_start:act_end].shape[0] == context_len + action_len - 1
        assert ar_mask[start:end].shape[0] == context_len
        samples.append(
            (
                vision[start:end],
                state[start:end],
                action[act_start:act_end],
                ar_mask[start:end],
            )
        )
    return samples


def _ctx_slice(x, context_len):
    return x[:context_len]


def _pred_slice(x, context_len, pred_len):
    return x[: context_len + pred_len - 1]


def skip_broken_shard(exc):
    # exc can be OSError, tarfile.ReadError, etc.
    print(f"[WARN] problem with: {exc} - skipping shard")
    return True


def finite_sample(fields):
    """True only if every np.ndarray in the sample is finite."""
    if any(f is None for f in fields):
        return False
    return all(np.isfinite(f).all() for f in fields if isinstance(f, np.ndarray))


def glob_shards(dataset_root: str, dataset_names: List[str]) -> List[str]:
    """Return a list of shard paths."""
    if dataset_root.startswith("gs://"):
        fs = fsspec.filesystem("gcs")
        shards: List[str] = []
        for name in dataset_names:
            pattern = f"{dataset_root.rstrip('/')}/{name}/*/shard-*.tar"
            shards.extend(fs.glob(pattern))

        shards = [p if p.startswith("gs://") else f"gs://{p}" for p in shards]
        return shards

    shards: List[str] = []
    for name in dataset_names:
        shards.extend(glob(os.path.join(dataset_root, name, "*", "shard-*.tar")))
    return shards


def discover_dataset_names(dataset_root: str, dataset_names: List[str]) -> List[str]:
    """Discover dataset folders for logging."""
    if dataset_root.startswith("gs://"):
        fs = fsspec.filesystem("gcs")
        names: List[str] = []
        for name in dataset_names:
            pattern = f"{dataset_root.rstrip('/')}/{name}"
            names.extend(fs.glob(pattern))
        names = [os.path.basename(n.rstrip("/")) for n in names]
    else:
        paths = [glob(os.path.join(dataset_root, pattern)) for pattern in dataset_names]
        names = [os.path.basename(p) for sub in paths for p in sub]

    return sorted(names, key=lambda x: x.split("_")[-1])


def make_dataset(cfg: DataConfig, context_len: int, pred_len: int) -> wds.WebDataset:
    roots = glob_shards(cfg.dataset_root, cfg.dataset_names)
    if len(roots) == 0:
        raise FileNotFoundError(
            f"No shards found under root='{cfg.dataset_root}' with patterns {cfg.dataset_names}."
        )

    np.random.shuffle(roots)
    print("Webdataset using", len(roots), "shards")

    names = discover_dataset_names(cfg.dataset_root, cfg.dataset_names)
    print(f"Dataset ({len(names)})", names)

    os.environ["RANK"] = str(jax.process_index())
    os.environ["WORLD_SIZE"] = str(jax.process_count())

    dataset = wds.WebDataset(
        roots,
        handler=skip_broken_shard,
        resampled=False,
        shardshuffle=cfg.shardshuffle,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.shardlists.split_by_worker,
    )

    if cfg.chunked:
        dataset = (
            dataset.decode("torch")
            .repeat()
            .to_tuple(
                "pre_vision.npy",
                "pre_state.npy",
                "pre_action.npy",
                "vision.npy",
                "state.npy",
                "action.npy",
                "json",
                missing_is_error=False,
            )
            .select(finite_sample)
            .map(
                partial(
                    slide,
                    context_len=context_len,
                    action_len=pred_len,
                )
            )
            .unlisted()
            .shuffle(cfg.shufflebuffer)
            .batched(cfg.global_batch_size // jax.process_count(), partial=False)
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
            .batched(cfg.global_batch_size // jax.process_count(), partial=False)
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
    dataloader = PrefetchLoader(
        dataloader, sharding, buffer_size=cfg.prefetch_device_factor
    )
    return dataloader
