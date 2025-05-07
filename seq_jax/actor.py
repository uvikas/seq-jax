from typing import List, Tuple

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from flax import nnx

from seq_jax.policy import (
    TransformerConcatPolicy,
    TransformerConcatPolicyConfig,
    Observation,
    Prediction,
    QueryPool,
)
from seq_jax.vit import VisionEncoder, VisionEncoderConfig

################################################################################
# Configs
################################################################################


@dataclass
class ModelConfig:
    name: str = "transformer_concat_querypool"
    context_len: int = 16
    pred_len: int = 1
    num_cams: int = 3
    dofs: int = 24

    freeze_filter: str = ".*img_encoder.*"

    encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    policy: TransformerConcatPolicyConfig = field(
        default_factory=TransformerConcatPolicyConfig
    )


################################################################################
# Transformer concat actor
################################################################################


def make_attn_mask(
    ar_mask: jnp.ndarray, num_heads: int, causal: bool = True
) -> jnp.ndarray:
    """
    Make attention mask for multi-head attention.

    Args:
      ar_mask:    [batch, seq_len] 0/1 mask (1 = keep, 0 = pad)
      num_heads:  number of attention heads
      causal:     if True, add causal (upper triangle) masking

    Returns:
      attn_mask:  [batch, num_heads, seq_len, seq_len] boolean mask
    """
    batch_sz, seq_len = ar_mask.shape

    # expand to [batch, seq_len, seq_len]
    # ar_mask[:, None, :] has shape [B,1,L], broadcast to [B,L,L]
    attn = jnp.broadcast_to(ar_mask[:, None, :], (batch_sz, seq_len, seq_len))

    # zero out the diagonal (no self‐masking)
    idx = jnp.arange(seq_len)
    attn = attn.at[:, idx, idx].set(0)

    # replicate for each head and reshape to [B*H, L, L]
    attn = jnp.tile(attn[:, None, :, :], (1, num_heads, 1, 1))

    # convert to boolean mask (True = masked out)
    attn = attn == 0

    if causal:
        # build causal mask [L, L], True above the diagonal
        causal_m = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1)
        # broadcast to [B*H, L, L]
        causal_m = jnp.broadcast_to(
            causal_m, (batch_sz * num_heads, seq_len, seq_len)
        ).reshape(batch_sz, num_heads, seq_len, seq_len)
        # combine padding & causal masks
        attn = jnp.logical_or(attn, causal_m)

    return attn


class TransformerConcatActor(nnx.Module):
    """Transformer concat actor."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.context_len = cfg.context_len
        self.pred_len = cfg.pred_len

        self.img_encoder = VisionEncoder(cfg.encoder, rngs=rngs)

        # 3 images, 1 state, 1 action
        inp_dims = [cfg.policy.embed_dim] * cfg.num_cams + [cfg.dofs] + [cfg.dofs]
        # 3 images, 1 state, 1 action
        out_dims = [cfg.policy.embed_dim] * cfg.num_cams + [cfg.dofs] + [cfg.dofs]

        self.pools = [
            QueryPool(
                cfg.encoder.embed_dim,
                cfg.policy.embed_dim,
                cfg.policy.num_img_query,
                rngs=rngs,
            )
            for _ in range(cfg.num_cams)
        ]

        self.policy = TransformerConcatPolicy(
            cfg.context_len,
            cfg.pred_len,
            inp_dims[:-1],
            out_dims,
            cfg.policy,
            rngs=rngs,
        )  # exclude action from input

    def compute_loss(
        self, obs: Observation, pred: Prediction, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        attn_mask = make_attn_mask(
            obs.mask, self.cfg.policy.num_heads, causal=self.cfg.policy.causal
        )

        policy_in = jax.tree.map(
            lambda x: x[:, : self.context_len] if x is not None else None,
            obs,
            is_leaf=lambda x: x is None,
        )

        if policy_in.img_feat is None:
            # Extract features from rgb
            policy_in = policy_in.replace(
                img_feat=self.img_encoder(
                    policy_in.img_rgb, deterministic=deterministic, rngs=rngs
                )
            )

        policy_in = policy_in.replace(
            img_pooled=self.pool_feats(
                policy_in.img_feat, deterministic=deterministic, rngs=rngs
            )
        )
        pred_act = self.policy(
            policy_in, attn_mask, deterministic=deterministic, rngs=rngs
        )

        loss = jnp.mean(jnp.square(pred_act.action - pred.action))
        return loss

    def pool_feats(
        self, feat: jnp.ndarray, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jnp.ndarray:
        return [
            pool(f, deterministic=deterministic, rngs=rngs)
            for f, pool in zip(feat, self.pools)
        ]


def make_model(cfg: ModelConfig, rngs: nnx.Rngs):
    return TransformerConcatActor(cfg, rngs=rngs)
