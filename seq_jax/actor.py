from typing import List

from dataclasses import dataclass


@dataclass
class ModelConfig:
    causal: bool = True
    in_proj_hidden_sizes: List[int] = []
    num_img_query: int = 16
    embed_dim: int = 384
    num_blocks: int = 6
    num_heads: int = 8
    mlp_ratio: float = 2.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    last_layer_norm: bool = False
    head_hidden_sizes: List[int] = []
    normalize_input: bool = True
    im_loss_weight: float = 0.1
    multihead_proj: bool = False


def make_model(cfg: ModelConfig):
    pass
