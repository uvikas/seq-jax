from typing import List, Tuple

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from flax import nnx

from seq_jax.distributed import activation_sharding_constraint
from seq_jax.policy import (
    TransformerConcatPolicy,
    TransformerConcatPolicyConfig,
    Observation,
    Prediction,
    QueryPool,
)
from seq_jax.vit import VisionEncoder, VisionEncoderConfig, normalize

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

    frozen_filter: str = ".*img_encoder.*"

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
    attn = attn > 0

    if causal:
        # build causal mask [L, L], True above the diagonal
        causal_m = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1) > 0
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
        inp_dims = (cfg.policy.embed_dim,) * cfg.num_cams + (cfg.dofs,) + (cfg.dofs,)
        # obs_shape = sum(inp_dims[:-1])  # Exclude action dim from input
        # 3 images, 1 state, 1 action
        out_dims = (cfg.policy.embed_dim,) * cfg.num_cams + (cfg.dofs,) + (cfg.dofs,)

        self.attn_pool = [
            QueryPool(
                cfg.encoder.embed_dim,
                cfg.policy.embed_dim,
                cfg.policy.num_img_query,
                rngs=rngs,
            )
            for _ in range(cfg.num_cams)
        ]

        self.obs_normalizer = nnx.BatchNorm(
            num_features=sum(inp_dims[:-1]),
            epsilon=1e-5,
            momentum=1.0,
            use_bias=False,
            use_scale=False,
            use_fast_variance=False,
            rngs=rngs,
        )

        self.policy = TransformerConcatPolicy(
            cfg.context_len,
            cfg.pred_len,
            inp_dims[:-1],
            out_dims,
            cfg.policy,
            rngs=rngs,
        )  # exclude action from input

        self.mask_token = nnx.Param(jnp.zeros(sum(inp_dims[:-1])))
        self.action_start_token = nnx.Param(jnp.zeros((1, 1, out_dims[-1])))

    def normalize_traj(self, obs: jnp.array) -> jnp.array:
        training = not self.obs_normalizer.use_running_average

        if training:
            # Update running stats with the latest step (B, D)
            last = obs[-1:].reshape(-1, obs.shape[-1])
            _ = self.obs_normalizer(last, use_running_average=False)

        # Normalize the full trajectory *without* altering stats
        flat = obs.reshape(-1, obs.shape[-1])  # (T*B, D)
        norm_flat = self.obs_normalizer(flat, use_running_average=True)  # eval-mode
        return norm_flat.reshape(obs.shape)

    def compute_loss(
        self, obs: Observation, pred: Prediction, *, rngs: nnx.Rngs
    ) -> jax.Array:
        attn_mask = make_attn_mask(
            obs.mask, self.cfg.policy.num_heads, causal=self.cfg.policy.causal
        )

        policy_in = jax.tree.map(
            lambda x: x[:, : self.context_len] if x is not None else None,
            obs,
            is_leaf=lambda x: x is None,
        )

        # Extract features from rgb
        if policy_in.img_feat is None:
            b, t, c, h, w = policy_in.img_rgb[0].shape
            rgb_flat = [
                rgb.reshape(b * t, c, h, w).astype(jnp.float32)
                for rgb in policy_in.img_rgb
            ]
            rgb_cat = jnp.concatenate(rgb_flat, axis=0)
            rgb_cat = normalize(rgb_cat)
            feats = self.img_encoder(rgb_cat, rngs=rngs)
            feats = jnp.array_split(feats, self.cfg.num_cams, axis=0)
            feats = [feat.reshape(b, t, *feat.shape[1:]) for feat in feats]
            policy_in = policy_in.replace(img_feat=feats)
        else:
            policy_in = policy_in.replace(
                img_feat=[policy_in.img_feat[:, :, i] for i in range(self.cfg.num_cams)]
            )

        # Pool image features
        img_pooled = self.pool_feats(
            policy_in.img_feat,
            rngs=rngs,
        )

        policy_in = policy_in.replace(img_pooled=img_pooled)

        # Concat + normalize
        obs_cat = jnp.concatenate(
            policy_in.img_pooled
            + [
                policy_in.state,
                # obs.prev_action,
            ],
            axis=-1,
        )
        obs_cat = activation_sharding_constraint(obs_cat)
        # (B, T, D) -> (T, B, D)
        obs_cat = obs_cat.transpose(1, 0, 2)
        obs_cat = self.normalize_traj(obs_cat)

        # Predict actions + compute loss
        pred_act = self.policy(obs_cat, attn_mask, rngs=rngs)
        pred_act = activation_sharding_constraint(pred_act)

        loss = jnp.mean(jnp.square(pred_act.action - pred.action))
        loss_dict = {
            "action_loss": loss,
            "im_head_loss": jnp.array([0.0]),
            "im_left_loss": jnp.array([0.0]),
            "im_right_loss": jnp.array([0.0]),
            "state_loss": jnp.array([0.0]),
            "total_loss": loss,
        }
        return loss, loss_dict

    def pool_feats(self, feat: jnp.ndarray, *, rngs: nnx.Rngs) -> jnp.ndarray:
        return [pool(f, rngs=rngs) for f, pool in zip(feat, self.attn_pool)]

    def sample_actions(self, obs: Observation, *, rngs: nnx.Rngs) -> Prediction:
        attn_mask = make_attn_mask(
            obs.mask, self.cfg.policy.num_heads, causal=self.cfg.policy.causal
        )

        policy_in = jax.tree.map(
            lambda x: x[:, : self.context_len] if x is not None else None,
            obs,
            is_leaf=lambda x: x is None,
        )

        # Extract features from rgb
        if policy_in.img_feat is None:
            b, t, c, h, w = policy_in.img_rgb[0].shape
            rgb_flat = [
                rgb.reshape(b * t, c, h, w).astype(jnp.float32)
                for rgb in policy_in.img_rgb
            ]
            rgb_cat = jnp.concatenate(rgb_flat, axis=0)
            rgb_cat = normalize(rgb_cat)
            feats = self.img_encoder(rgb_cat, rngs=rngs)
            feats = jnp.array_split(feats, self.cfg.num_cams, axis=0)
            feats = [feat.reshape(b, t, *feat.shape[1:]) for feat in feats]
            policy_in = policy_in.replace(img_feat=feats)

        # Pool image features
        img_pooled = self.pool_feats(
            policy_in.img_feat,
            rngs=rngs,
        )
        policy_in = policy_in.replace(img_pooled=img_pooled)

        # Concat + normalize
        obs_cat = jnp.concatenate(
            policy_in.img_pooled
            + [
                policy_in.state,
                # obs.prev_action,
            ],
            axis=-1,
        )
        # (B, T, D) -> (T, B, D)
        obs_cat = obs_cat.transpose(1, 0, 2)
        obs_cat = self.normalize_traj(obs_cat)

        # Predict actions
        pred_act = self.policy(obs_cat, attn_mask, rngs=rngs)
        return pred_act.action


def make_model(cfg: ModelConfig, rngs: nnx.Rngs):
    return TransformerConcatActor(cfg, rngs=rngs)
