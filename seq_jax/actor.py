from typing import List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from flax import struct

from seq_jax.model import ViTModel


################################################################################
# Configs
################################################################################


@dataclass
class VisionEncoderConfig:
    name: str = "vitb-mae-egosoup"
    pretrain_dir: str = "/path/to/pretrained/encoders"
    freeze: bool = True
    embed_dim: int = 768
    dtype: str = "float32"
    feat_dtype: str = "float16"


@dataclass
class BasePolicyConfig:
    name: str


@dataclass
class TransformerConcatPolicyConfig(BasePolicyConfig):
    name: str = "transformer_concat"
    causal: bool = True
    in_proj_hidden_sizes: List[int] = field(default_factory=lambda: [])
    num_img_query: int = 16
    embed_dim: int = 384
    num_blocks: int = 6
    num_heads: int = 8
    mlp_ratio: float = 2.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    last_layer_norm: bool = False
    head_hidden_sizes: List[int] = field(default_factory=lambda: [])
    normalize_input: bool = True
    im_loss_weight: float = 0.1
    multihead_proj: bool = False


@dataclass
class ModelConfig:
    name: str = "transformer_concat_querypool"
    context_len: int = 16
    pred_len: int = 1

    encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    policy: BasePolicyConfig = field(default_factory=TransformerConcatPolicyConfig)


################################################################################
# Model inputs and outputs
################################################################################


@struct.dataclass
class Observation:
    # RGB images
    images: jnp.ndarray
    # Robot state
    state: jnp.ndarray
    # Autoregressive mask
    mask: jnp.ndarray


@struct.dataclass
class Action:
    # Action
    action: jnp.ndarray


################################################################################
# Transformer concat actor
################################################################################


class TransformerConcatActor(nn.Module):
    """Transformer concat actor."""

    def __init__(self, cfg: ModelConfig):
        pass


def make_model(cfg: ModelConfig):
    pass
