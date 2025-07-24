from typing import List

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from enum import Enum
from flax import nnx

from seq_jax.dist import activation_sharding_constraint
from seq_jax.policy import (
    TransformerConcatPolicy,
    TransformerMultimodalPolicy,
    TransformerPolicyConfig,
    Observation,
    Prediction,
    QueryPool,
    SeqPool,
    PoolConfig,
    PoolType,
    MLP,
)
from seq_jax.vision import VisionEncoder, VisionEncoderConfig, normalize

################################################################################
# Configs
################################################################################


class ModelType(str, Enum):
    concat = "concat"
    multimodal = "multimodal"


@dataclass
class ModelConfig:
    name: str = ModelType.concat
    context_len: int = 16
    pred_len: int = 1
    num_cams: int = 3
    dofs: int = 24

    frozen_filter: str = ".*img_encoder.*"

    encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    policy: TransformerPolicyConfig = field(default_factory=TransformerPolicyConfig)

    init_weights: str | None = None


################################################################################
# Transformer concat actor
################################################################################


def make_attn_mask(
    ar_mask: jnp.ndarray, num_heads: int, causal: bool = True
) -> jnp.ndarray:
    """
    Make attention mask for multi-head attention.
    ar_mask:    [batch, seq_len] (0 = keep, 1 = pad)
    """
    batch_sz, seq_len = ar_mask.shape

    # (batch, seq_len) -> (batch, seq_len, seq_len)
    attn = jnp.broadcast_to(ar_mask[:, None, :], (batch_sz, seq_len, seq_len))
    attn = attn.at[:, jnp.arange(seq_len), jnp.arange(seq_len)].set(0)
    # (batch, seq_len, seq_len) -> (batch, num_heads, seq_len, seq_len)
    attn = jnp.tile(attn[:, None, :, :], (1, num_heads, 1, 1))
    attn = attn > 0

    if causal:
        # upper trianglar causal mask (seq_len, seq_len)
        causal_m = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1) > 0
        # (seq_len, seq_len) -> (batch, num_heads, seq_len, seq_len)
        causal_m = jnp.broadcast_to(
            causal_m, (batch_sz * num_heads, seq_len, seq_len)
        ).reshape(batch_sz, num_heads, seq_len, seq_len)

        attn = jnp.logical_or(attn, causal_m)

    return attn


def make_block_attn_mask(
    ar_mask: jnp.ndarray, num_heads: int, block_size: int, causal: bool = True
) -> jnp.ndarray:
    """
    Make block attention mask for multi-head attention.
    Bidirectional attention in each block. Blocks are causal.
    ar_mask:    [batch, seq_len] (0 = keep, 1 = pad)
    """
    batch_sz, seq_len = ar_mask.shape

    # (batch, seq_len) -> (batch, seq_len, seq_len)
    attn = jnp.broadcast_to(ar_mask[:, None, :], (batch_sz, seq_len, seq_len))
    attn = attn.at[:, jnp.arange(seq_len), jnp.arange(seq_len)].set(0)
    # (batch, seq_len, seq_len) -> (batch, num_heads, seq_len, seq_len)
    attn = jnp.tile(attn[:, None, :, :], (1, num_heads, 1, 1))
    attn = attn > 0

    if causal:
        # upper trianglar causal mask (seq_len, seq_len)
        causal_m = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1) > 0
        # (seq_len, seq_len) -> (batch, num_heads, seq_len, seq_len)
        causal_m = jnp.broadcast_to(
            causal_m, (batch_sz * num_heads, seq_len, seq_len)
        ).reshape(batch_sz, num_heads, seq_len, seq_len)
        attn = jnp.logical_or(attn, causal_m)

    # (batch, num_heads, seq_len, seq_len) -> (batch, num_heads, seq_len * block_size, seq_len * block_size)
    attn = attn.repeat(block_size, axis=-1).repeat(block_size, axis=-2)
    return attn


def make_pool(cfg: ModelConfig, rngs: nnx.Rngs):
    match cfg.pool.type:
        case PoolType.query:
            return QueryPool(
                cfg.encoder.embed_dim,
                cfg.policy.embed_dim,
                cfg.pool.num_query,
                rngs=rngs,
            )
        case PoolType.seq:
            return SeqPool(
                cfg.pool.num_query,
                cfg.pool.num_blocks,
                cfg.pool.embed_dim,
                cfg.pool.num_heads,
                cfg.pool.mlp_ratio,
                cfg.pool.attn_dropout,
                cfg.pool.mlp_dropout,
                flatten_proj=(cfg.name == ModelType.concat),
                rngs=rngs,
            )
        case _:
            raise ValueError(f"Invalid pool type: {cfg.pool.type}")


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

        self.attn_pool = [make_pool(cfg, rngs) for _ in range(cfg.num_cams)]

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

    def pool_feats(
        self, feat: List[jnp.ndarray], *, rngs: nnx.Rngs
    ) -> List[jnp.ndarray]:
        return [pool(f, rngs=rngs) for f, pool in zip(feat, self.attn_pool)]

    def sample_actions(self, obs: Observation, *, rngs: nnx.Rngs) -> Prediction:
        attn_mask = make_attn_mask(
            obs.mask, self.cfg.policy.num_heads, causal=self.cfg.policy.causal
        )

        # Concat + normalize
        obs_cat = jnp.concatenate(
            obs.img_pooled
            + [
                obs.state,
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

    def encode_im(self, ims: jnp.ndarray, *, rngs: nnx.Rngs) -> jnp.ndarray:
        # ims: (b, c, h, w)
        feats = self.img_encoder(ims, rngs=rngs)
        return feats


class TransformerMultimodalActor(nnx.Module):
    """Transformer actor with interleaved tokens of different modalities."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.context_len = cfg.context_len
        self.pred_len = cfg.pred_len

        # 3 images, 1 state, 1 action
        self.tokens_per_mod = (cfg.pool.num_query,) * cfg.num_cams + (1,) + (1,)
        self.block_size = sum(self.tokens_per_mod)
        # 3 images, 1 state, 1 action
        self.out_dims = (
            (cfg.policy.embed_dim,) * cfg.num_cams + (cfg.dofs,) + (cfg.dofs,)
        )

        self.img_encoder = VisionEncoder(cfg.encoder, rngs=rngs)
        self.state_encoder = MLP(
            cfg.dofs,
            cfg.policy.embed_dim,
            rngs=rngs,
        )
        self.action_encoder = MLP(
            cfg.dofs,
            cfg.policy.embed_dim,
            rngs=rngs,
        )
        self.attn_pool = [make_pool(cfg, rngs) for _ in range(cfg.num_cams)]
        self.policy = TransformerMultimodalPolicy(
            cfg.context_len,
            cfg.pred_len,
            self.tokens_per_mod,
            self.out_dims,
            cfg.pool.embed_dim,
            cfg.policy,
            rngs=rngs,
        )

        self.action_token = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (1, 1, 1, cfg.policy.embed_dim),
            )
            * 0.02
        )

    def compute_loss(
        self, obs: Observation, pred: Prediction, *, rngs: nnx.Rngs
    ) -> jax.Array:
        # (B, H, T*block_size, T*block_size)
        attn_mask = make_block_attn_mask(
            obs.mask,
            self.cfg.policy.num_heads,
            self.block_size,
            causal=self.cfg.policy.causal,
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
        # (B, T, dofs) -> (B, T, 1, D)
        state_embed = self.state_encoder(
            policy_in.noisy_state[:, :, None, :], rngs=rngs
        )
        # (1, 1, D) -> (B, T, 1, D)
        action_token = jnp.broadcast_to(self.action_token, state_embed.shape)

        # Tokens per mod List[B, T, N, D] * num_mod
        obs_each_mod = img_pooled + [state_embed, action_token]

        # Predict actions + compute loss
        pred_act = self.policy(obs_each_mod, attn_mask, rngs=rngs)
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

    def pool_feats(
        self, feat: List[jnp.ndarray], *, rngs: nnx.Rngs
    ) -> List[jnp.ndarray]:
        # (B, T, N, D) * num_cams
        return [pool(f, rngs=rngs) for f, pool in zip(feat, self.attn_pool)]

    def sample_actions(self, obs: Observation, *, rngs: nnx.Rngs) -> Prediction:
        attn_mask = make_attn_mask(
            obs.mask, self.cfg.policy.num_heads, causal=self.cfg.policy.causal
        )

        # Concat + normalize
        obs_cat = jnp.concatenate(
            obs.img_pooled
            + [
                obs.state,
                # obs.prev_action,
            ],
            axis=-1,
        )
        # (B, T, D) -> (T, B, D)
        obs_cat = obs_cat.transpose(1, 0, 2)

        # Predict actions
        pred_act = self.policy(obs_cat, attn_mask, rngs=rngs)
        return pred_act.action

    def encode_im(self, ims: jnp.ndarray, *, rngs: nnx.Rngs) -> jnp.ndarray:
        # ims: (b, c, h, w)
        feats = self.img_encoder(ims, rngs=rngs)
        return feats


def make_model(cfg: ModelConfig, rngs: nnx.Rngs):
    match cfg.name:
        case ModelType.concat:
            return TransformerConcatActor(cfg, rngs=rngs)
        case ModelType.multimodal:
            return TransformerMultimodalActor(cfg, rngs=rngs)
        case _:
            raise ValueError(f"Invalid model type: {cfg.name}")
