"""
Model definitions.
"""

from typing import List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


################################################################################
#                                                                              #
#                                 Vision Transformer                           #
#                                                                              #
################################################################################


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    elif typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    else:
        raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        """Applies Transformer MlpBlock module."""
        inits = dict(
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )

        d = x.shape[-1]
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        # In some extreme batch-size cases, this is needed as of Sept 2024:
        x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = nn.Dense(d, dtype=self.dtype_mm, **inits)(x)
        return x


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}
        x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
        y = nn.LayerNorm()(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm()(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(
                x, deterministic
            )
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, l=lyr: o[l], scan_out)
        else:
            # Input Encoder
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm")(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        # TODO
        n, l, d = x.shape  # pylint: disable=unused-variable
        probe = self.param(
            "probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype
        )
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform()
        )(probe, x)

        # TODO: dropout on head?
        y = nn.LayerNorm()(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
        return x[:, 0]


class _Model(nn.Module):
    """ViT model."""

    num_classes: Optional[int] = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: Union[int, bool] = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        image = jnp.asarray(image, self.dtype_mm)

        # Patch extraction
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=self.dtype_mm,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(
            self, self.posemb, (h, w), c, "pos_embedding", x.dtype
        )

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, l, c = x.shape  # pylint: disable=unused-variable
        x = nn.Dropout(rate=self.dropout)(x, not train)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads, mlp_dim=self.mlp_dim
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


class ViTModel(nn.Module):
    pass


################################################################################
#                                                                              #
#                                 Transformer Decoder                          #
#                                                                              #
################################################################################
