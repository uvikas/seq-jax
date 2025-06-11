"""Model definitions."""

from typing import List, Optional, Sequence, Union, Callable

import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass, field
from flax import nnx, struct
from functools import partial

################################################################################
#                                                                              #
#                                 Inputs and Outputs                           #
#                                                                              #
################################################################################


@struct.dataclass
class Observation:
    # Image rgb (B, T, H, W, 3) * V
    img_rgb: List[jnp.ndarray] | None
    # Image features (B, T, P, C) * V
    img_feat: List[jnp.ndarray] | None
    # Pooled img features (B, T, C) * V
    img_pooled: List[jnp.ndarray] | None
    # Robot state (B, T, D)
    state: jnp.ndarray
    # Noisy robot state (B, T, D)
    noisy_state: jnp.ndarray
    # Robot prev action (B, T, D)
    prev_action: jnp.ndarray
    # Autoregressive mask (B, T)
    mask: jnp.ndarray


@struct.dataclass
class Prediction:
    # Pooled img features (B, T, A, C) * V
    img_pooled: List[jnp.ndarray] | None
    # Robot state (B, T, A, D)
    state: jnp.ndarray | None
    # Action (B, T, A, D)
    action: jnp.ndarray


################################################################################
#                                                                              #
#                                 Layers                                       #
#                                                                              #
################################################################################


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> jnp.ndarray:
    """
    Args:
      embed_dim: must be even. Final embedding dim = embed_dim
      length:    number of positions (M)

    Returns:
      pos_embed: [length, embed_dim] array of sin/cos embeddings
    """
    assert embed_dim % 2 == 0, "embed_dim must be divisible by 2"
    half = embed_dim // 2

    # compute the inverse frequencies (D/2,)
    omega = jnp.arange(half) / (embed_dim / 2.0)
    omega = 1.0 / (10000**omega)

    # positions (M,)
    pos = jnp.arange(length)

    # outer product → (M, D/2)
    angles = jnp.einsum("m,d->md", pos, omega)

    # sin on first half, cos on second half → (M, D)
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=1)
    return emb


class MLP(nnx.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int] = [],
        stable_init: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        layers = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                layers.append(nnx.Linear(input_dim, size, rngs=rngs))
            else:
                layers.append(nnx.Linear(hidden_sizes[i - 1], size, rngs=rngs))
            layers.append(nnx.elu)
        if len(hidden_sizes) > 0:
            layers.append(nnx.Linear(hidden_sizes[-1], output_dim, rngs=rngs))
        else:
            layers.append(nnx.Linear(input_dim, output_dim, rngs=rngs))
        self.net = nnx.Sequential(*layers)

        # So-called stable init
        if stable_init:
            gain = 1.0 if output_dim == 1 else 0.01
            # Create an orthogonal initializer with the specified gain
            ortho_init = jax.nn.initializers.orthogonal(scale=gain)
            # Get the last layer (which is the final Linear layer)
            last_layer = self.net.layers[-1]
            # Initialize new kernel with orthogonal initialization
            kernel_shape = last_layer.kernel.shape
            kernel_dtype = last_layer.kernel.dtype
            new_kernel = ortho_init(rngs.params(), kernel_shape, kernel_dtype)
            # Create new Linear layer with updated kernel
            new_last_layer = nnx.Linear(
                in_features=kernel_shape[0],
                out_features=kernel_shape[1],
                kernel_init=lambda *_: new_kernel,
                bias_init=lambda *_: last_layer.bias,
                rngs=rngs,
            )
            # Update the Sequential layers
            self.net.layers[-1] = new_last_layer

    def __call__(self, x, *, rngs: nnx.Rngs):
        return self.net(x)


################################################################################
#                                                                              #
#                                 Pooling                                      #
#                                                                              #
################################################################################


class QueryPool(nnx.Module):
    """Attention pooling with a learnable query"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_query: int,
        head_dim: int = 64,
        qkv_bias: bool = True,
        drop: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        assert (
            input_dim % head_dim == 0
        ), f"input_dim {input_dim} must be divisible by head_dim {head_dim}"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_query = num_query
        self.head_dim = head_dim
        self.num_heads = input_dim // head_dim
        self.scale = head_dim ** (-0.5)
        self.drop = drop

        # learnable query: (1, num_query, num_heads, head_dim)
        self.q = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_query, self.num_heads, head_dim))
        )

        # combined key/value projection
        self.kv = nnx.Linear(
            in_features=input_dim,
            out_features=2 * input_dim,
            use_bias=qkv_bias,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=(
                nnx.initializers.normal(stddev=1e-6)
                if qkv_bias
                else nnx.initializers.zeros()
            ),
        )

        self.proj_1 = nnx.Linear(
            in_features=input_dim * num_query,
            out_features=output_dim * 4,
            rngs=rngs,
        )
        self.proj_2 = nnx.Linear(
            in_features=output_dim * 4,
            out_features=output_dim,
            rngs=rngs,
        )
        self.act = partial(nnx.gelu, approximate=False)
        self.proj_drop = nnx.Dropout(rate=drop)

        # MLP head
        self.mlp = MLP(
            input_dim * num_query,
            output_dim,
            [output_dim * 4],
            rngs=rngs,
        )

    def __call__(self, im_emb: jnp.ndarray, *, rngs: nnx.Rngs) -> jnp.ndarray:
        """
        im_emb: [batch, T, N, C]  (C == input_dim)
        returns: [batch, T, output_dim]
        """
        b, T, N, C = im_emb.shape
        assert C == self.input_dim, f"Expected C={self.input_dim}, got {C}"

        B = b * T
        x = im_emb.reshape((B, N, C))

        # project to keys and values, then split
        kv = self.kv(x)  # (B, N, 2*input_dim)
        kv = kv.reshape((B, N, 2, self.num_heads, self.head_dim))
        kv = jnp.transpose(kv, (2, 0, 3, 1, 4))  # (2, B, heads, N, head_dim)
        k, v = kv[0], kv[1]  # each (B, heads, N, head_dim)

        # expand and reorder query to (B, heads, num_query, head_dim)
        q = jnp.tile(self.q, (B, 1, 1, 1))
        q = jnp.transpose(q, (0, 2, 1, 3))

        # scaled dot‐product attention
        q = q * self.scale
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        attn = jax.nn.softmax(attn_logits, axis=-1)
        weighted = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        # reshape back to (b, T, num_query * C)
        out = jnp.transpose(weighted, (0, 2, 1, 3))  # (B, num_query, heads, head_dim)
        out = out.reshape((b, T, self.num_query * C))

        # MLP head + dropout
        out = self.proj_1(out)
        out = self.act(out)
        out = self.proj_2(out)
        out = self.proj_drop(out)

        return out


################################################################################
#                                                                              #
#                                 Transformer                                  #
#                                                                              #
################################################################################


class MLPBlock(nnx.Module):
    """Transformer MLP block: Dense → Dropout → GELU → Dense → Dropout."""

    w_in: int
    mlp_d: int
    dropout: float = 0.0

    def __init__(
        self,
        w_in: int,
        mlp_d: int,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.w_in = w_in
        self.mlp_d = mlp_d
        self.dropout = dropout

        self.linear_1 = nnx.Linear(
            in_features=w_in,
            out_features=mlp_d,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.normal(stddev=1e-6),
        )
        self.linear_2 = nnx.Linear(
            in_features=mlp_d,
            out_features=w_in,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.normal(stddev=1e-6),
        )
        self.af = partial(nnx.gelu, approximate=True)

        self.dropout_1 = nnx.Dropout(rate=dropout)
        self.dropout_2 = nnx.Dropout(rate=dropout)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        rngs: nnx.Rngs,
    ) -> jnp.ndarray:
        x = self.linear_1(x)
        x = self.dropout_1(x, rngs=rngs)
        x = self.af(x)
        x = self.linear_2(x)
        x = self.dropout_2(x, rngs=rngs)
        return x


class TransformerEncoderBlock(nnx.Module):
    """Transformer encoder block, following https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        hidden_d: int,
        n_heads: int,
        mlp_d: int,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        layernorm: Callable = partial(nnx.LayerNorm, epsilon=1e-6),
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_d = hidden_d
        self.ln_1 = layernorm(hidden_d, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=hidden_d,
            qkv_features=hidden_d,
            out_features=hidden_d,
            dropout_rate=attn_dropout,
            decode=False,
            rngs=rngs,
        )
        self.ln_2 = layernorm(hidden_d, rngs=rngs)
        self.mlp_block = MLPBlock(
            hidden_d,
            mlp_d,
            dropout=mlp_dropout,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        attn_mask: jax.Array,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        x_p = self.ln_1(x)
        x_p = self.attention(
            x_p,
            x_p,
            x_p,
            mask=~attn_mask,
            rngs=rngs,
        )

        x = x + x_p
        x_p = self.mlp_block(self.ln_2(x), rngs=rngs)
        return x + x_p


class TransformerEncoder(nnx.Module):
    """Transformer encoder (sequence of TransformerEncoderBlock)."""

    def __init__(
        self,
        n_layers: int,
        hidden_d: int,
        n_heads: int,
        mlp_d: int,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        layernorm: Callable = partial(nnx.LayerNorm, epsilon=1e-6),
        last_layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        # Initialize a list of TransformerEncoderBlock instances
        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append(
                TransformerEncoderBlock(
                    hidden_d=hidden_d,
                    n_heads=n_heads,
                    mlp_d=mlp_d,
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout,
                    rngs=rngs,
                )
            )
        # Optional final layer normalization
        if last_layer_norm:
            self.ln = layernorm(hidden_d)

    def __call__(
        self,
        x: jax.Array,
        attn_mask: jax.Array,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        # Apply each encoder block in sequence
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, rngs=rngs)

        # Apply final layer norm if defined
        if hasattr(self, "ln"):
            x = self.ln(x)
        return x


################################################################################
#                                                                              #
#                                 Policies                                     #
#                                                                              #
################################################################################


@dataclass
class TransformerConcatPolicyConfig:
    name: str = "transformer_concat"
    causal: bool = True
    in_proj_hidden_sizes: List[int] = field(default_factory=lambda: [])
    num_img_query: int = 16
    embed_dim: int = 768
    num_blocks: int = 16
    num_heads: int = 8
    mlp_ratio: float = 2.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    last_layer_norm: bool = False
    head_hidden_sizes: List[int] = field(default_factory=lambda: [])
    normalize_input: bool = True
    im_loss_weight: float = 0.1
    multihead_proj: bool = False


class TransformerConcatPolicy(nnx.Module):
    """Transformer-based policy."""

    def __init__(
        self,
        context_len: int,
        pred_len: int,
        inp_dims: List[int],
        out_dims: List[int],
        cfg: TransformerConcatPolicyConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.context_len = context_len
        self.pred_len = pred_len
        self.multihead_proj = cfg.multihead_proj

        self.input_dim = sum(inp_dims)
        self.output_dim = sum(out_dims)
        # Must be static tuple
        self.split_output = tuple(np.array(out_dims).cumsum())

        self.in_proj1 = MLP(
            self.input_dim,
            self.input_dim // 2,
            cfg.in_proj_hidden_sizes,
            stable_init=False,
            rngs=rngs,
        )
        self.in_proj2 = MLP(
            self.input_dim // 2,
            cfg.embed_dim,
            cfg.in_proj_hidden_sizes,
            stable_init=False,
            rngs=rngs,
        )
        self.act = partial(nnx.gelu, approximate=False)

        self.pos_embed = nnx.Variable(
            jnp.expand_dims(get_1d_sincos_pos_embed(cfg.embed_dim, context_len), axis=0)
        )

        self.encoder = TransformerEncoder(
            cfg.num_blocks,
            cfg.embed_dim,
            cfg.num_heads,
            int(cfg.embed_dim * cfg.mlp_ratio),
            cfg.attn_dropout,
            cfg.mlp_dropout,
            cfg.last_layer_norm,
            rngs=rngs,
        )

        if cfg.multihead_proj:
            self.out_proj = [
                MLP(
                    cfg.embed_dim,
                    self.output_dim,
                    cfg.head_hidden_sizes,
                    stable_init=True,
                    rngs=rngs,
                )
                for _ in range(pred_len)
            ]
        else:
            self.out_proj = MLP(
                cfg.embed_dim,
                self.output_dim,
                cfg.head_hidden_sizes,
                stable_init=True,
                rngs=rngs,
            )

    def __call__(
        self,
        obs: jnp.array,
        attn_mask: jnp.ndarray,
        *,
        rngs: nnx.Rngs,
    ) -> Prediction:
        # (B, T, 3*C + D + D)

        x = self.in_proj1(obs, rngs=rngs)
        x = self.act(x)
        x = self.in_proj2(x, rngs=rngs)
        x = x.transpose(1, 0, 2)
        x = x + self.pos_embed
        x = self.encoder(x, attn_mask, rngs=rngs)

        if self.multihead_proj:
            img_pooled, state, action = [], [], []
            for proj in self.out_proj:
                out = proj(x)
                splits = jnp.split(out, self.output_dims, axis=-1)
                img_pooled.append(splits[:3])
                state.append(splits[3])
                action.append(splits[4])
            img_pooled = jnp.stack(img_pooled, axis=1)
            state = jnp.stack(state, axis=1)
            action = jnp.stack(action, axis=1)
            return Prediction(img_pooled=img_pooled, state=state, action=action)
        else:
            x = self.out_proj(x, rngs=rngs)
            splits = jnp.split(x, self.split_output, axis=-1)
            img_pooled, state, action = splits[:3], splits[3], splits[4]
            return Prediction(
                img_pooled=[jnp.expand_dims(img, axis=2) for img in img_pooled],
                state=jnp.expand_dims(state, axis=2),
                action=jnp.expand_dims(action, axis=2),
            )
