"""Model definitions."""

from typing import List, Callable, Any

import jax
import jax.numpy as jnp
import numpy as np
import re

from dataclasses import dataclass, field
from enum import Enum
from flax import nnx, struct
from functools import partial

from seq_jax.dist import activation_sharding_constraint

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
    img_pooled: jnp.ndarray | None
    # Robot state (B, T, A, D)
    state: jnp.ndarray | None
    # Action (B, T, A, D)
    action: jnp.ndarray


################################################################################
#                                                                              #
#                                 Parameter Filters                            #
#                                                                              #
################################################################################


@dataclass(frozen=True)
class PathRegex:
    """NNX Filter that matches paths using a regex.

    By default, paths are joined with a `/` separator. This can be overridden by setting the `sep` argument.
    """

    pattern: str | re.Pattern
    sep: str = "/"

    def __post_init__(self):
        if not isinstance(self.pattern, re.Pattern):
            object.__setattr__(self, "pattern", re.compile(self.pattern))

    def __call__(self, path: nnx.filterlib.PathParts, x: Any) -> bool:
        joined_path = self.sep.join(str(x) for x in path)
        assert isinstance(self.pattern, re.Pattern)
        return self.pattern.fullmatch(joined_path) is not None


################################################################################
#                                                                              #
#                                 Layers                                       #
#                                                                              #
################################################################################


def get_act_fn(act_fn: str) -> Callable:
    match act_fn:
        case "gelu":
            return partial(nnx.gelu, approximate=False)
        case "relu":
            return jax.nn.relu
        case _:
            raise ValueError(f"Unknown activation function: {act_fn}")


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> jnp.ndarray:
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


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def _rope_sin_cos(
    seq_len: int, dim: int, base: int = 10000, dtype: jnp.dtype = jnp.float32
) -> tuple[jnp.ndarray, jnp.ndarray]:
    assert dim % 2 == 0, "RoPE requires the head dimension to be even"
    half = dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(0, half, dtype=dtype) / half))
    positions = jnp.arange(seq_len, dtype=dtype)
    freqs = jnp.einsum("t,d->td", positions, inv_freq)  # [T, half]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    sin = jnp.sin(emb)[None, :, None, :]  # [1, T, 1, D]
    cos = jnp.cos(emb)[None, :, None, :]  # [1, T, 1, D]
    return sin, cos


def _apply_rope(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray) -> jnp.ndarray:
    return (x * cos) + (_rotate_half(x) * sin)


def rope_dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: jnp.ndarray | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: jnp.dtype | None = None,
    precision=None,
    module=None,
):
    # Shapes: [..., T_q, H, D], [..., T_kv, H, D]
    d = query.shape[-1]
    assert d % 2 == 0, "RoPE requires the head dimension to be even"
    q_len = query.shape[-3]
    k_len = key.shape[-3]

    sin_q, cos_q = _rope_sin_cos(
        q_len, d, dtype=query.dtype if dtype is None else dtype
    )
    sin_k, cos_k = _rope_sin_cos(k_len, d, dtype=key.dtype if dtype is None else dtype)

    query = _apply_rope(query, sin_q, cos_q)
    key = _apply_rope(key, sin_k, cos_k)

    return nnx.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
        module=module,
    )


class MLP(nnx.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int] = [],
        stable_init: bool = False,
        act_fn: str = "gelu",
        *,
        rngs: nnx.Rngs,
    ):
        layers = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                layers.append(nnx.Linear(input_dim, size, rngs=rngs))
            else:
                layers.append(nnx.Linear(hidden_sizes[i - 1], size, rngs=rngs))
            layers.append(get_act_fn(act_fn))
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


class TokenwiseMLP(nnx.Module):
    """MLP that operates on each token independently."""

    def __init__(
        self,
        input_dims: List[int],
        num_tokens: int,
        output_dim: int,
        hidden_sizes: List[int] = [],
        stable_init: bool = False,
        act_fn: str = "gelu",
        *,
        rngs: nnx.Rngs,
    ):
        self.input_dims = input_dims
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.stable_init = stable_init

        self.mlps = [
            MLP(
                input_dim=input_dims[i],
                output_dim=output_dim,
                hidden_sizes=hidden_sizes,
                stable_init=stable_init,
                act_fn=act_fn,
                rngs=rngs,
            )
            for i in range(num_tokens)
        ]

    def __call__(self, x: jnp.ndarray, *, rngs: nnx.Rngs) -> jnp.ndarray:
        # x: [B, T, C] * N
        assert len(x) == self.num_tokens

        out = []
        for i in range(self.num_tokens):
            # [B, T, C]
            token_x = x[i]
            # (B, T, output_dim)
            token_out = self.mlps[i](token_x, rngs=rngs)
            out.append(token_out)
        # (B, T, N, output_dim)
        return jnp.stack(out, axis=2)


################################################################################
#                                                                              #
#                                 Pooling                                      #
#                                                                              #
################################################################################


class PoolType(str, Enum):
    query = "query"
    seq = "seq"
    mean = "mean"


@dataclass
class PoolConfig:
    type: PoolType = PoolType.query
    flatten_proj: bool = False
    num_query: int = 16
    num_blocks: int = 4
    embed_dim: int = 768
    num_heads: int = 12
    mlp_ratio: float = 2.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0


class MeanPool(nnx.Module):
    """Mean pooling"""

    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.proj = nnx.Linear(
            in_features=input_dim,
            out_features=output_dim,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, *, rngs: nnx.Rngs) -> jnp.ndarray:
        return self.proj(x.mean(axis=-2))


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
        flatten_proj: bool = False,
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
        self.flatten_proj = flatten_proj

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

        if flatten_proj:
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

    def __call__(self, im_emb: jnp.ndarray, *, rngs: nnx.Rngs) -> jnp.ndarray:
        """
        im_emb: [batch, T, N, C]
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

        # MLP head + dropout
        if self.flatten_proj:
            out = out.reshape((b, T, self.num_query * C))
            out = self.proj_1(out)
            out = self.act(out)
            out = self.proj_2(out)
            out = self.proj_drop(out, rngs=rngs)
        else:
            out = out.reshape((b, T, self.num_query, C))

        return out


class SeqPool(nnx.Module):
    """
    Pool latent features using `n_tokens` extra summary tokens in-sequence
    """

    def __init__(
        self,
        num_tokens: int,
        num_blocks: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        attn_dropout: float,
        mlp_dropout: float,
        last_layer_norm: bool = False,
        flatten_proj: bool = False,
        layernorm: Callable = partial(nnx.LayerNorm, epsilon=1e-6),
        n_patches: int = 197,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.summary_tokens = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, num_tokens, embed_dim))
        )
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_tokens + n_patches, embed_dim))
            * 0.02
        )
        self.encoder = TransformerEncoder(
            n_layers=num_blocks,
            hidden_d=embed_dim,
            n_heads=num_heads,
            mlp_d=int(embed_dim * mlp_ratio),
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            layernorm=layernorm,
            last_layer_norm=last_layer_norm,
            rngs=rngs,
        )

        self.flatten_proj = flatten_proj
        if flatten_proj:
            self.out_proj = nnx.Linear(
                in_features=embed_dim * num_tokens,
                out_features=embed_dim,
                rngs=rngs,
            )

    def __call__(self, x: jnp.ndarray, *, rngs: nnx.Rngs) -> jnp.ndarray:
        batch_sz, seq_len, n_input_tokens, d = x.shape
        summary_tokens = jnp.tile(self.summary_tokens, (batch_sz, seq_len, 1, 1))
        # (B, T, N+K, C)
        x = jnp.concatenate([summary_tokens, x], axis=2)

        # summary tokens attend to all
        # input tokens may not attend to summary tokens
        total_tokens = self.num_tokens + n_input_tokens
        # fully bidirectional attn mask (N+K, N+K)
        mask = jnp.zeros((total_tokens, total_tokens), dtype=bool)
        # (N+K, N+K) -> (B, T, num_heads, N+K, N+K)
        mask = jnp.tile(
            mask[None, None, None, :, :],
            (batch_sz, seq_len, self.num_heads, 1, 1),
        )

        # (B, T, N+K, C)
        x = x + self.pos_embed
        x = self.encoder(x, attn_mask=mask, rngs=rngs)
        # (B, T, K, C)
        x = x[:, :, : self.num_tokens, :]

        if self.flatten_proj:
            # (B, T, K, C) -> (B, T, K*C)
            x = x.reshape(batch_sz, seq_len, -1)
            # (B, T, K*C) -> (B, T, C)
            x = self.out_proj(x)

        return x


################################################################################
#                                                                              #
#                                 Transformer                                  #
#                                                                              #
################################################################################


class MLPBlock(nnx.Module):
    """Transformer MLP block: Dense -> Dropout -> GELU -> Dense -> Dropout."""

    def __init__(
        self,
        w_in: int,
        mlp_d: int,
        dropout: float = 0.0,
        act_fn: str = "gelu",
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
        self.af = get_act_fn(act_fn)

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
        act_fn: str = "gelu",
        rope: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_d = hidden_d
        self.ln_1 = layernorm(hidden_d, rngs=rngs)
        attn_fn = rope_dot_product_attention if rope else nnx.dot_product_attention
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=hidden_d,
            qkv_features=hidden_d,
            out_features=hidden_d,
            dropout_rate=attn_dropout,
            attention_fn=attn_fn,
            decode=False,
            rngs=rngs,
        )
        self.ln_2 = layernorm(hidden_d, rngs=rngs)
        self.mlp_block = MLPBlock(
            hidden_d,
            mlp_d,
            dropout=mlp_dropout,
            act_fn=act_fn,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        attn_mask: jax.Array,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        x = activation_sharding_constraint(x)
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
        act_fn: str = "gelu",
        rope: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append(
                TransformerEncoderBlock(
                    hidden_d=hidden_d,
                    n_heads=n_heads,
                    mlp_d=mlp_d,
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout,
                    act_fn=act_fn,
                    rope=rope,
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
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, rngs=rngs)

        # Apply final layer norm if defined
        if hasattr(self, "ln"):
            x = self.ln(x)
        return x


################################################################################
#                                                                              #
#                                 Policy                                       #
#                                                                              #
################################################################################


@dataclass
class TransformerPolicyConfig:
    causal: bool = True
    in_proj_hidden_sizes: List[int] = field(default_factory=lambda: [1024])

    embed_dim: int = 768
    num_blocks: int = 12
    num_heads: int = 12
    mlp_ratio: float = 2.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    last_layer_norm: bool = False
    qk_norm: bool = False

    head_hidden_sizes: List[int] = field(default_factory=lambda: [])
    im_loss_begin: int = 0
    im_loss_weight: float = 0.0
    multihead_proj: bool = False

    rope: bool = False
    activation_fn: str = "gelu"
    mask_ratio: float = 0.0


class TransformerConcatPolicy(nnx.Module):
    """Transformer-based policy that concatenates all modalities from a single timestep into a single token."""

    def __init__(
        self,
        context_len: int,
        pred_len: int,
        inp_dims: List[int],
        out_dims: List[int],
        cfg: TransformerPolicyConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.context_len = context_len
        self.pred_len = pred_len
        self.multihead_proj = cfg.multihead_proj

        self.input_dim = tuple(inp_dims)
        self.output_dim = tuple(out_dims)
        self.rope = cfg.rope

        # Must be static tuple
        self.split_output = tuple(np.array(out_dims).cumsum())

        self.in_proj = MLP(
            sum(self.input_dim),
            cfg.embed_dim,
            cfg.in_proj_hidden_sizes,
            stable_init=False,
            act_fn=cfg.activation_fn,
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
            cfg.activation_fn,
            cfg.rope,
            rngs=rngs,
        )

        if cfg.multihead_proj:
            self.out_proj = [
                MLP(
                    cfg.embed_dim,
                    sum(self.output_dim),
                    cfg.head_hidden_sizes,
                    act_fn=cfg.activation_fn,
                    rngs=rngs,
                )
                for _ in range(pred_len)
            ]
        else:
            self.out_proj = MLP(
                cfg.embed_dim,
                sum(self.output_dim),
                cfg.head_hidden_sizes,
                act_fn=cfg.activation_fn,
                rngs=rngs,
            )

    def __call__(
        self,
        obs: jnp.array,
        attn_mask: jnp.ndarray,
        *,
        rngs: nnx.Rngs,
    ) -> Prediction:
        # (T, B, 3*C + D + D)
        x = self.in_proj(obs, rngs=rngs)
        # (T, B, d) -> (B, T, d)
        x = x.transpose(1, 0, 2)
        if not self.rope:
            x = x + self.pos_embed
        x = self.encoder(x, attn_mask, rngs=rngs)

        if self.multihead_proj:
            img_pooled, state, action = [], [], []
            for proj in self.out_proj:
                out = proj(x)
                splits = jnp.split(out, self.split_output, axis=-1)
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


class TransformerMultimodalPolicy(nnx.Module):
    """Transformer-based policy that uses a separate token for each modality."""

    def __init__(
        self,
        context_len: int,
        pred_len: int,
        tokens_per_mod: List[int],
        inp_dims: List[int],
        out_dims: List[int],
        cfg: TransformerPolicyConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.context_len = context_len
        self.pred_len = pred_len
        self.multihead_proj = cfg.multihead_proj
        self.rope = cfg.rope
        self.mask_ratio = cfg.mask_ratio

        # (1, T, D)
        self.pos_embed = nnx.Variable(
            jnp.expand_dims(get_1d_sincos_pos_embed(cfg.embed_dim, context_len), axis=0)
        )

        # List[D] * num_mod
        inp_embed_dims = []
        for tokens, dim in zip(tokens_per_mod, inp_dims):
            inp_embed_dims.extend([dim] * tokens)

        self.in_projs = TokenwiseMLP(
            inp_embed_dims,
            sum(tokens_per_mod),
            cfg.embed_dim,
            act_fn=cfg.activation_fn,
            rngs=rngs,
        )

        self.encoder = TransformerEncoder(
            cfg.num_blocks,
            cfg.embed_dim,
            cfg.num_heads,
            int(cfg.embed_dim * cfg.mlp_ratio),
            attn_dropout=cfg.attn_dropout,
            mlp_dropout=cfg.mlp_dropout,
            last_layer_norm=cfg.last_layer_norm,
            act_fn=cfg.activation_fn,
            rope=cfg.rope,
            rngs=rngs,
        )

        self.out_projs = [
            [
                MLP(
                    cfg.embed_dim,
                    out_dims[i],
                    act_fn=cfg.activation_fn,
                    rngs=rngs,
                )
                for _ in range(pred_len)
            ]
            for i in range(len(out_dims))
        ]

        self.mask_token = nnx.Param(jax.random.normal(rngs.params(), (cfg.embed_dim,)))

    def __call__(
        self,
        obs_each_mod: List[jnp.array],
        attn_mask: jnp.ndarray,
        *,
        rngs: nnx.Rngs,
    ) -> Prediction:
        # obs_each_mod: list of (B, T, N, D)
        # assumes all modalities are hidden_d embedded

        tokens_per_mod = [obs.shape[-2] for obs in obs_each_mod]

        # List[B, T, N_total, D_i] * mods -> (B, T, D_i) * N_total
        token_list = [
            mod_tokens[:, :, i, :]
            for mod_tokens in obs_each_mod
            for i in range(mod_tokens.shape[-2])
        ]
        # List[B, T, D_i] * N_total -> (B, T, N_total, D)
        x = self.in_projs(token_list, rngs=rngs)

        # Apply pos embed across time. Tokens in the same timestep get the same pos embed.
        # pos_embed: (1, T, D) -> (1, T, 1, D)
        if not self.rope:
            x = x + jnp.expand_dims(self.pos_embed, axis=2)

        batch_sz, ctx_len, n_token_per_step, hidden_d = x.shape

        # Dropout wrist tokens
        if self.mask_ratio > 0.0:
            # List[B, T, N_i, D] * num_mod
            emb_each_mod = jnp.split(x, np.cumsum(tokens_per_mod), axis=-2)

            b, t, n, d = emb_each_mod[0].shape

            left_mask = jax.random.bernoulli(
                rngs.dropout(),
                1 - self.mask_ratio,
                (b, t, n, 1),
            )
            right_mask = jax.random.bernoulli(
                rngs.dropout(),
                1 - self.mask_ratio,
                (b, t, n, 1),
            )
            emb_each_mod[0] = emb_each_mod[0] * left_mask + self.mask_token[
                None, None, None, :
            ] * (1 - left_mask)
            emb_each_mod[2] = emb_each_mod[2] * right_mask + self.mask_token[
                None, None, None, :
            ] * (1 - right_mask)

            # List[B, T, N_i, D] * num_mod -> (B, T, N_total, D)
            x = jnp.concatenate(emb_each_mod, axis=-2)

        # (B, T, N_total, D) -> (B, T * N_total, D)
        x = jnp.reshape(x, (x.shape[0], -1, *x.shape[3:]))

        # (B, T * N_total, D) -> (B, T * N_total, D)
        x = self.encoder(x, attn_mask, rngs=rngs)

        # (B, T * N_total, D) -> (B, T, N_total, D)
        x = jnp.reshape(x, (x.shape[0], ctx_len, n_token_per_step, x.shape[-1]))

        # (B, T, N_total, D) -> List[B, T, N, D] * num_mod
        pred_each_mod = jnp.split(x, np.cumsum(tokens_per_mod), axis=-2)

        # Output projection on the last step
        out_each_mod = [
            # (B, T, N, D)
            # -> List[List[B, T, N, D_out] * pred_len] * num_mod
            # -> List[B, T, pred_len, N, D_out] * num_mod
            jnp.stack(
                [out_proj[i](out, rngs=rngs) for i in range(self.pred_len)], axis=-3
            )
            for out, out_proj in zip(pred_each_mod, self.out_projs)
        ]
        return Prediction(
            img_pooled=out_each_mod[:3],
            state=None,
            action=out_each_mod[3].squeeze(axis=-2),
        )
