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
    hflip: bool = False

    # fmt: off
    state_mean: List[float] = field(
        default_factory=lambda: [ 
            1.84536481, -1.12462929,  1.41197011, -0.02487607,  1.33455319, 2.63499216,  
            0.77312651,  0.82768274,  0.79064847,  0.79948837, 0.65214008, -1.26757096,  
            4.32017923, -1.96941897, -1.4748379 , 3.31425951, -1.47706232,  3.43987484,  
            0.70836829,  0.72942468, 0.69718277,  0.698197  ,  0.5570043 , -1.25428493
        ]
    )
    state_std: List[float] = field(
        default_factory=lambda: [
            0.2347727 , 0.24163017, 0.34348805, 0.27242452, 0.17415138, 0.27901413, 
            0.4256069 , 0.44951353, 0.46495739, 0.46300634, 0.44718893, 0.08101595, 
            0.19660917, 0.26167127, 0.34266775, 0.24946231, 0.19555666, 0.24784786, 
            0.45313833, 0.47493871, 0.47512846, 0.47718881, 0.46647868, 0.0717899
        ]
    )

    action_mean: List[float] = field(
        default_factory=lambda: [ 
            1.84448056, -1.12398471,  1.41060664, -0.02533558,  1.33518984, 2.63446084,  
            0.91516366,  0.91516414,  0.87024576,  0.87024623, 0.74085207, -1.39427505,  
            4.32030675, -1.96946854, -1.47363764, 3.314219  , -1.47800851,  3.43907236,  
            0.79889305,  0.79889328, 0.7621681 ,  0.76216829,  0.62425366, -1.39427558
        ]
    )
    action_std: List[float] = field(
        default_factory=lambda: [
            0.23524328, 0.24370715, 0.34556206, 0.27444562, 0.17549883, 0.28037564, 
            0.54767225, 0.54767145, 0.55964619, 0.55964604, 0.54700286, 0.0195207 , 
            0.19729668, 0.26349596, 0.34509491, 0.25071675, 0.19587037, 0.24915961, 
            0.54916166, 0.54916251, 0.55408106, 0.55408127, 0.54903041, 0.01951983
        ]
    )

    # fmt: on
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


def mirror_arms(j: jnp.ndarray) -> jnp.ndarray:
    # return [-2*np.pi - j[0], -np.pi - j[1], -j[2], -np.pi - j[3], -j[4], -j[5]]
    bias = jnp.array([-2 * jnp.pi, -jnp.pi, 0.0, -jnp.pi, 0.0, 0.0], dtype=j.dtype)
    return -j + bias


def hflip_joints(joints: jnp.ndarray):
    larm, lhand, rarm, rhand = jnp.split(joints, 4, axis=1)
    return jnp.concatenate([mirror_arms(rarm), rhand, mirror_arms(larm), lhand], axis=1)


def hflip(batch, hflip: bool = False):

    if not hflip:
        return batch
    if np.random.random() < 0.5:
        return batch

    vision, states, actions, masks = batch
    # Reverse (B, T, V, H, W, 3)
    left, head, right = (
        vision[:, :, 0, :, :, :],
        vision[:, :, 1, :, :, :],
        vision[:, :, 2, :, :, :],
    )
    left = left[:, :, :, ::-1, :]
    right = right[:, :, :, ::-1, :]
    vision = jnp.stack([left, head, right], axis=2)

    # Hflip actions
    actions = hflip_joints(actions)
    states = hflip_joints(states)

    return vision, states, actions, masks


def pack_batch(
    batch,
    state_mean: np.array,
    state_std: np.array,
    action_mean: np.array,
    action_std: np.array,
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
        states = (states - state_mean) / state_std
        actions = (actions - action_mean) / action_std
    elif action_type == "relative":
        actions = actions - states
        states = (states - state_mean) / state_std

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
    elif vision_format == "rgb":
        # (B, T, V, H, W, 3) -> (B, T, V, 3, H, W)
        vision = vision.transpose(0, 1, 2, 5, 3, 4)
        obs = Observation(
            img_rgb=[vision[:, :, i] for i in range(vision.shape[2])],
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
        img_pooled=None,  # Need to pool in actor first
        state=jnp.stack(
            [states[:, i + 1 : i + pred_len + 1, ...] for i in range(context_len)],
            axis=1,
        ),
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
            state_mean=np.array(cfg.state_mean),
            state_std=np.array(cfg.state_std),
            action_mean=np.array(cfg.action_mean),
            action_std=np.array(cfg.action_std),
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
                        or (
                            batch[0].img_feat is not None
                            and jnp.isnan(batch[0].img_feat).any()
                        )
                        or (
                            batch[0].img_rgb is not None
                            and any(jnp.isnan(img).any() for img in batch[0].img_rgb)
                        )
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
    # is easier to read and plenty fast.
    samples = []
    for i in range(vision.shape[0] - context_len - action_len):
        start, end = i, i + context_len
        mod_start, mod_end = i, i + context_len + action_len
        act_start, act_end = i, i + context_len + action_len - 1

        assert vision[mod_start:mod_end].shape[0] == context_len + action_len
        assert state[mod_start:mod_end].shape[0] == context_len + action_len
        assert action[act_start:act_end].shape[0] == context_len + action_len - 1
        assert ar_mask[start:end].shape[0] == context_len
        samples.append(
            (
                vision[mod_start:mod_end],
                state[mod_start:mod_end],
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
            .map(partial(hflip, hflip=cfg.hflip))
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
