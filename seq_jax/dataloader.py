"""Dataloader."""

from typing import List

import omegaconf

from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class DataConfig:
    dataset_type: str = "wds"
    shufflebuffer: int = 8192
    shardshuffle: int = 1000
    num_workers: int = 4

    dataset_root: str
    dataset_names: List[str] = []
    dtype: str = "float32"
    vision_format: str = "mp4"

    im_size: int = 224
    dof_size: int = 24
    touch_dim: int = 60
    cam_names: List[str] = ["left", "head", "right"]

    default_pos: List[float] = [
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

    joint_noise_mean: List[float] = [
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

    joint_noise_std: List[float] = [
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
    joint_noise_std_scale: float = 1.0
    frame_skip: int = 2


def make_dataset(cfg: omegaconf.DictConfig) -> Dataset:
    pass


def make_dataloader(cfg: omegaconf.DictConfig) -> DataLoader:
    dataset = make_dataset(cfg)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)


class Dataset:
    pass
