"""Dataloader."""

from typing import List
import os

import jax
import jax.numpy as jnp
import numpy as np
import webdataset as wds

from dataclasses import dataclass, field
from jax.sharding import NamedSharding
from functools import partial
from glob import glob

from seq_jax import util
from seq_jax.policy import Observation, Prediction


@dataclass
class DataConfig:
    dataset_type: str = "wds"
    chunked: bool = False
    mb_size: int = 256
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


def wds_collate_fn(
    batch,
    default_pos: np.array,
    noise_mean: np.array,
    noise_std: np.array,
    noise_scale: float,
    action_type: str,
    context_len: int,
    pred_len: int,
    sharding: NamedSharding,
):
    vision, states, actions, masks = batch

    if action_type == "absolute":
        states -= default_pos
        actions -= default_pos
    elif action_type == "relative":
        actions = actions - states
        # states = (states - joint_ranges[:, 0]) / (
        #     joint_ranges[:, 1] - joint_ranges[:, 0]
        # )

    noise = np.random.normal(
        loc=noise_mean,
        scale=noise_std * noise_scale,
        size=states.shape,
    )
    noisy_states = states + noise

    vision = jnp.array(vision)
    states = jnp.array(states)
    noisy_states = jnp.array(noisy_states)
    actions = jnp.array(actions)
    masks = jnp.array(masks)

    obs = Observation(
        img_rgb=None,
        img_feat=vision,
        img_pooled=None,
        state=states,
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

    obs = jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(sharding, x), obs
    )
    pred = jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(sharding, x), pred
    )

    return obs, pred


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


def make_dataset(
    cfg: DataConfig, num_devices: int, context_len: int, pred_len: int
) -> wds.WebDataset:
    roots = [
        glob(os.path.join(cfg.dataset_root, "*", "*", "shard-*.tar"))
        for dataset_name in cfg.dataset_names
    ]
    roots = [item for sublist in roots for item in sublist]
    assert len(roots) > 0, f"No shards found in {cfg.dataset_root} {cfg.dataset_names}"

    np.random.shuffle(roots)
    np.random.shuffle(roots)

    print("Webdataset using", len(roots), "shards")

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
            .batched(cfg.mb_size)
        )
    else:
        dataset = (
            dataset.shuffle(cfg.shufflebuffer)
            .decode("torch")
            .repeat()
            .to_tuple("vision.npy", "state.npy", "action.npy", "mask.npy")
            .map_tuple(
                lambda x: x[:context_len],
                lambda x: x[:context_len],
                lambda x: x[: context_len + pred_len - 1],
                lambda x: x[:context_len],
            )
            .batched(cfg.mb_size)
        )
    return dataset


def make_dataloader(
    cfg: DataConfig, context_len: int, pred_len: int, sharding: NamedSharding
) -> wds.WebLoader:
    dataset = make_dataset(cfg, sharding.num_devices, context_len, pred_len)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=cfg.num_workers,
        collate_fn=partial(
            wds_collate_fn,
            default_pos=np.array(cfg.default_pos),
            noise_mean=np.array(cfg.joint_noise_mean),
            noise_std=np.array(cfg.joint_noise_std),
            noise_scale=cfg.joint_noise_std_scale,
            # joint_ranges=np.array(cfg.joint_ranges),
            action_type=cfg.action_type,
            context_len=context_len,
            pred_len=pred_len,
            sharding=sharding,
        ),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=cfg.prefetch_factor,
    )
    # dataloader = dataloader.unbatched().shuffle(cfg.shufflebuffer).batched(cfg.mb_size)
    return iter(dataloader)
