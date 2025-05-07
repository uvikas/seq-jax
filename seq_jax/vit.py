from typing import Optional, Tuple, Type, Callable, Union, List
import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from dataclasses import dataclass
from flax import nnx
from functools import partial


# Helper to create tuples (similar to _ntuple(n))
def to_tuple(n: int, x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return tuple(x for _ in range(n))
    return x


class Identity(nnx.Module):
    """A simple NNX Module that returns the input unchanged."""

    def __init__(self, *, rngs: Optional[nnx.Rngs] = None):
        # No parameters or state needed for Identity
        pass

    def __call__(
        self, x: jax.Array, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        return x


class DropPath(nnx.Module):
    """Drop paths (Stochastic Depth) per sample. FLAX NNX version."""

    def __init__(self, drop_prob: float, *, rngs: nnx.Rngs):
        self.drop_prob = drop_prob
        # No parameters needed for initialization

    def __call__(
        self, x: jax.Array, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        if self.drop_prob == 0.0 or deterministic:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with different dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + jax.random.bernoulli(
            rngs.dropout(), keep_prob, shape
        )
        # Unlike PyTorch impl, JAX bernoulli gives 0/1, so we don't need floor
        # random_tensor = jnp.floor(random_tensor) # no need in JAX

        # Scale by keep_prob is often done by default (scale_by_keep=True)
        output = (x / keep_prob) * random_tensor
        return output


class PatchEmbed(nnx.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Type[nnx.Module]] = None,
        flatten: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.img_size = to_tuple(2, img_size)
        self.patch_size = to_tuple(2, patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim  # Store embed_dim for potential later use

        # NNX Conv expects kernel_size, stride as sequences
        self.proj = nnx.Conv(
            in_features=in_chans,
            out_features=embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            rngs=rngs,
        )
        self.norm = norm_layer(embed_dim, rngs=rngs) if norm_layer else Identity()

    def __call__(
        self, x: jax.Array, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        B, C, H, W = x.shape
        # JAX assertions typically happen outside JIT-compiled functions if possible,
        # or raise errors during runtime if inside.
        assert (
            H == self.img_size[0]
        ), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."

        # (B, C, H, W) -> (B, H, W, C)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.proj(x)
        # (B, H, W, C) -> (B, C, H, W)
        x = jnp.transpose(x, (0, 3, 1, 2))
        # Output shape: (B, H', W', Cout) if channels_last, or (B, Cout, H', W') if channels_first
        # nnx.Conv default is NCHW format similar to pytorch Conv2d
        # Output shape: (B, embed_dim, grid_size[0], grid_size[1])

        if self.flatten:
            # Reshape to (B, embed_dim, num_patches) then transpose to (B, num_patches, embed_dim)
            x = x.reshape(B, self.embed_dim, -1).transpose(
                (0, 2, 1)
            )  # NCHW -> B C (H*W) -> B (H*W) C

        x = self.norm(x, deterministic=deterministic, rngs=rngs)
        return x


class Attention(nnx.Module):
    """Multi-head Self-Attention using JAX native scaled_dot_product_attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (
            self.head_dim**-0.5
        )  # Often included in scaled_dot_product_attention

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.attn_drop_rate = attn_drop  # Store rate for scaled_dot_product_attention
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop)

    def __call__(
        self, x: jax.Array, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3 * C)
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, self.head_dim
        )  # (B, N, 3, num_heads, head_dim)
        # Jax expects (3, B, N, num_heads, head_dim)
        qkv = qkv.transpose((2, 0, 1, 3, 4))  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, N, head_dim)

        # Use JAX's efficient attention implementation
        # Note: jax.nn.scaled_dot_product_attention includes the scaling factor
        attn_output = jax.nn.dot_product_attention(
            q, k, v, is_causal=False
        )  # (B, N, num_heads, head_dim)

        # Combine heads back
        # Transpose to (B, N, num_heads, head_dim) then reshape to (B, N, C)
        x = attn_output.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic, rngs=rngs)
        return x


class LayerScale(nnx.Module):
    """Layer Scaling implementation."""

    def __init__(self, dim: int, init_values: float = 1e-5, *, rngs: nnx.Rngs):
        # Initialize gamma parameter
        # Note: NNX handles parameter creation slightly differently.
        # We define the shape and initializer, NNX manages the state.
        self.gamma = nnx.Param(
            jnp.full((dim,), init_values)
        )  # No explicit rng needed for jnp.full

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.gamma.value  # Access parameter value using .value


class Mlp(nnx.Module):
    """MLP as used in Vision Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = partial(jax.nn.gelu, approximate=False),
        bias: bool = True,
        drop: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        use_bias = to_tuple(2, bias)

        self.fc1 = nnx.Linear(
            in_features, hidden_features, use_bias=use_bias[0], rngs=rngs
        )
        self.act = act_layer
        self.drop1 = nnx.Dropout(drop, rngs=rngs)
        self.fc2 = nnx.Linear(
            hidden_features, out_features, use_bias=use_bias[1], rngs=rngs
        )
        self.drop2 = nnx.Dropout(drop, rngs=rngs)

    def __call__(
        self, x: jax.Array, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, deterministic=deterministic, rngs=rngs)
        x = self.fc2(x)
        x = self.drop2(x, deterministic=deterministic, rngs=rngs)
        return x


class Block(nnx.Module):
    """Transformer Block"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable = partial(jax.nn.gelu, approximate=False),
        norm_layer: Type[nnx.LayerNorm] = nnx.LayerNorm,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = norm_layer(dim, rngs=rngs)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            rngs=rngs,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values, rngs=rngs)
            if init_values
            else Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        )

        self.norm2 = norm_layer(dim, rngs=rngs)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            rngs=rngs,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values, rngs=rngs)
            if init_values
            else Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        )

    def __call__(
        self, x: jax.Array, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        # Need to split RNG for parallel paths if both use dropout, but here it's sequential
        shortcut1 = x
        x_norm1 = self.norm1(x)
        attn_out = self.attn(x_norm1, deterministic=deterministic, rngs=rngs)
        ls_out = self.ls1(attn_out, deterministic=deterministic, rngs=rngs)
        drop_path_out = self.drop_path1(
            ls_out,
            deterministic=deterministic,
            rngs=rngs,
        )
        x = shortcut1 + drop_path_out

        shortcut2 = x
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2, deterministic=deterministic, rngs=rngs)
        x = shortcut2 + self.drop_path2(
            self.ls2(mlp_out, deterministic=deterministic, rngs=rngs),
            deterministic=deterministic,
            rngs=rngs,
        )

        return x


class VisionTransformer(nnx.Module):
    """Vision Transformer using FLAX NNX."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",  # 'token', 'avg' or ''
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: Type[PatchEmbed] = PatchEmbed,
        norm_layer: Optional[Type[nnx.LayerNorm]] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Type[Block] = Block,
        *,
        rngs: nnx.Rngs,  # Top-level RNGs needed
    ):
        assert global_pool in ("", "avg", "token")
        assert (
            class_token or global_pool != "token"
        ), "Cannot use global_pool='token' without class_token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nnx.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or partial(jax.nn.gelu, approximate=False)

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.num_features = embed_dim  # for consistency
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.class_token_enabled = class_token

        # Patch Embedding
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=None,  # Norm is applied after flattening usually
            flatten=True,
            rngs=rngs,
        )
        num_patches = self.patch_embed.num_patches

        # CLS Token
        if self.class_token_enabled:
            self.cls_token = nnx.Param(
                jnp.zeros((1, 1, embed_dim))
            )  # Default Flax initializer (lecun_normal usually)
        else:
            self.cls_token = None  # Or handle differently if needed

        # Positional Embedding
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        # Use jax.random for initialization, nnx.Param stores it
        pos_embed_init = (
            jax.random.normal(rngs.params(), (1, embed_len, embed_dim)) * 0.02
        )
        self.pos_embed = nnx.Param(pos_embed_init)
        self.pos_drop = nnx.Dropout(drop_rate, rngs=rngs)

        # Stochastic Depth Decay Rule
        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]

        # Transformer Blocks
        # Use a standard list; NNX will discover modules within it
        self.blocks = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                rngs=rngs,  # Pass rngs for each block's init
            )
            for i in range(depth)
        ]

        # Final Norm Layer
        self.norm = norm_layer(embed_dim, rngs=rngs) if not use_fc_norm else Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim, rngs=rngs) if use_fc_norm else Identity()
        self.head = (
            nnx.Linear(embed_dim, num_classes, rngs=rngs)
            if num_classes > 0
            else Identity()
        )

        # Note: Complex weight initialization like timm's 'jax' or 'moco' modes
        # would require custom initialization logic after the module is constructed
        # or by passing specific initializers to nnx.Linear/nnx.Conv.

    def _pos_embed(self, x: jax.Array, B: int) -> jax.Array:
        if self.no_embed_class:
            # Position embedding does not overlap with class token
            x = x + self.pos_embed.value  # Add pos embed to patches
            if self.class_token_enabled:
                # Expand cls_token to batch size and concatenate
                cls_tokens = jnp.broadcast_to(
                    self.cls_token.value, (B, 1, self.embed_dim)
                )
                x = jnp.concatenate([cls_tokens, x], axis=1)
        else:
            # Position embedding has entry for class token
            if self.class_token_enabled:
                # Expand cls_token and concatenate first
                cls_tokens = jnp.broadcast_to(
                    self.cls_token.value, (B, 1, self.embed_dim)
                )
                x = jnp.concatenate([cls_tokens, x], axis=1)
            # Then add positional embedding
            x = x + self.pos_embed.value

        return x  # Dropout is applied after this in __call__

    def forward_features(
        self, x: jax.Array, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        B = x.shape[0]
        x = self.patch_embed(
            x, deterministic=deterministic, rngs=rngs
        )  # (B, num_patches, embed_dim)
        x = self._pos_embed(x, B)  # (B, num_patches + num_prefix_tokens, embed_dim)
        x = self.pos_drop(x, deterministic=deterministic, rngs=rngs)

        # Iterate through blocks
        for block in self.blocks:
            # We might need to split the dropout RNG key for each block if they all use it
            # For simplicity here, assuming rngs object handles potential splitting needs
            # or that sub-modules internally request fresh keys if designed that way.
            # A safer pattern is explicitly splitting:
            # block_rngs = nnx.Rngs(dropout=jax.random.split(rngs.dropout(), 1)[0]) # Example
            # x = block(x, deterministic=deterministic, rngs=block_rngs)
            # Let's pass the main rngs down for now.
            x = block(x, deterministic=deterministic, rngs=rngs)

        x = self.norm(x)
        return x

    def forward_head(self, x: jax.Array, *, pre_logits: bool = False) -> jax.Array:
        if self.global_pool == "avg":
            # Average pool features except prefix tokens (e.g., class token)
            x = x[:, self.num_prefix_tokens :].mean(axis=1)
        elif self.global_pool == "token":
            # Take the first token (e.g., class token)
            x = x[:, 0]
        # If global_pool == '', x remains (B, N, C) - maybe useful for FCNs

        x = self.fc_norm(x)
        # Don't apply final head if pre_logits is True
        return x if pre_logits else self.head(x)

    def __call__(
        self, x: jax.Array, *, train: bool, rngs: Optional[nnx.Rngs] = None
    ) -> jax.Array:
        # Determine deterministic flag based on train
        deterministic = not train

        # Ensure rngs are provided if needed (dropout, etc.)
        if not deterministic and rngs is None:
            # This is just an example; ideally, the caller manages RNGs properly.
            # In a real scenario, raise an error or handle default RNG creation.
            print(
                "Warning: RNGs not provided during training, creating default dropout RNG."
            )
            rngs = nnx.Rngs(dropout=jax.random.key(0))  # Example default

        # If deterministic, we don't strictly need RNGs for dropout, but submodules might expect the object.
        # Create dummy RNGs if None and deterministic.
        if deterministic and rngs is None:
            rngs = nnx.Rngs(dropout=jax.random.key(0))  # Dummy key, won't be used

        x = self.forward_features(x, deterministic=deterministic, rngs=rngs)
        x = self.forward_head(x)  # pre_logits=False by default
        return x


class VisionTransformerExtractor(VisionTransformer):
    """
    NNX Vision Transformer for feature extraction.
    Removes the classification head and provides different feature extraction modes.
    """

    CLS_MODE = 0
    MEAN_MODE = 1
    ALL_MODE = 2

    def __init__(self, *, rngs: nnx.Rngs, **kwargs):
        # Ensure head is removed by setting num_classes=0
        # Also disable fc_norm as it's part of the head logic
        kwargs_modified = kwargs.copy()
        kwargs_modified["num_classes"] = 0
        kwargs_modified["fc_norm"] = False  # Usually associated with avg pooling head
        super().__init__(**kwargs_modified, rngs=rngs)

        # This class's methods often assume class token exists
        if not self.class_token_enabled:
            raise ValueError(
                "VisionTransformerFeaturesNNX requires class_token=True during initialization."
            )

    def extract_feat(
        self, x: jax.Array, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        """Extracts features before the final LayerNorm. Assumes class_token=True."""
        B = x.shape[0]
        x = self.patch_embed(
            x, deterministic=deterministic, rngs=rngs
        )  # (B, num_patches, embed_dim)

        # --- Replicating specific pos_embed logic from PyTorch version ---
        # Assumes class_token=True and no_embed_class=False
        if self.cls_token is None:
            # Should have been caught by __init__ check, but defensive coding
            raise ValueError("cls_token is None, but required by extract_feat")

        # Expand cls_token and concatenate
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, 1 + num_patches, embed_dim)

        # Add positional embedding (assumes pos_embed has entries for cls token + patches)
        # Ensure shape compatibility if base class init used different logic
        if self.pos_embed.value.shape[1] != x.shape[1]:
            raise ValueError(
                f"Shape mismatch: pos_embed ({self.pos_embed.value.shape}) vs input ({x.shape})"
            )
        x = x + self.pos_embed.value
        x = self.pos_drop(x, deterministic=deterministic, rngs=rngs)
        # --- End specific pos_embed logic ---

        # Apply transformer blocks
        for block in self.blocks:
            # Pass RNGs and deterministic flag for dropout/drop_path inside blocks
            x = block(x, deterministic=deterministic, rngs=rngs)

        return x  # Return features *before* final norm

    def forward_norm(self, x: jax.Array) -> jax.Array:
        """Applies the final LayerNorm."""
        return self.norm(x)

    def forward_feature(
        self, x: jax.Array, *, deterministic: bool, rngs: nnx.Rngs
    ) -> jax.Array:
        """Extracts features and applies the final LayerNorm."""
        x = self.extract_feat(x, deterministic=deterministic, rngs=rngs)
        x = self.forward_norm(x)
        return x

    def __call__(
        self, x: jax.Array, *, mode: int, train: bool, rngs: Optional[nnx.Rngs] = None
    ) -> jax.Array:
        """
        Main forward pass for feature extraction with different modes.
        Applies stop_gradient to the output.
        """
        deterministic = not train

        # Handle RNGs for stochastic operations (dropout/droppath)
        if not deterministic and rngs is None:
            # In real usage, caller should provide RNGs for training
            print("Warning: RNGs not provided for training mode, using dummy key.")
            rngs = nnx.Rngs(dropout=jax.random.key(0))  # Example default
        elif deterministic and rngs is None:
            # Create dummy RNGs if None and deterministic, as submodules might expect the object
            rngs = nnx.Rngs(dropout=jax.random.key(0))

        # Get features after blocks and final norm
        features = self.forward_feature(x, deterministic=deterministic, rngs=rngs)

        # Apply mode-based selection/pooling
        if mode == self.CLS_MODE:
            # Select the CLS token (index 0)
            output = features[:, 0]
        elif mode == self.MEAN_MODE:
            # Select patch tokens (starting from index 1) and average
            output = features[:, self.num_prefix_tokens :].mean(axis=1)
        elif mode == self.ALL_MODE:
            # Return all tokens
            output = features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

        # Replicate .detach() behavior - stop gradients here
        output = jax.lax.stop_gradient(output)

        # Ensure float32, although JAX typically uses it by default
        output = output.astype(jnp.float32)

        return output


_MODELS = {
    "vits-mae-hoi": "mae_pretrain_hoi_vit_small.jax",
    "vits-mae-egosoup": "mae_pretrain_egosoup_vit_small.jax",
    "vitb-mae-egosoup": "mae_pretrain_egosoup_vit_base.jax",
    "vitl-mae-egosoup": "mae_pretrain_egosoup_vit_large_256.jax",
    "vith-mae-egosoup": "mae_pretrain_egosoup_vit_huge_256.jax",
}


def vit_b16(pretrained: str, rngs: nnx.Rngs, **kwargs) -> VisionTransformerExtractor:
    scratch = pretrained.endswith("none")
    assert scratch or os.path.exists(pretrained)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),
        class_token=True,
        **kwargs,
    )
    # model = VisionTransformerExtractor(**model_kwargs, rngs=rngs)

    checkpointer = ocp.StandardCheckpointer()
    abstract_model = nnx.eval_shape(
        lambda: VisionTransformerExtractor(**model_kwargs, rngs=nnx.Rngs(0))
    )
    graphdef, rng_state, other_state = nnx.split(abstract_model, nnx.RngState, ...)

    state_restored = checkpointer.restore(pretrained, other_state)
    state_restored = nnx.merge_state(state_restored, rng_state)
    model = nnx.merge(graphdef, state_restored)

    print("Loaded encoder from: {}".format(pretrained))
    return model


@dataclass
class VisionEncoderConfig:
    name: str = "vitb-mae-egosoup"
    pretrain_dir: str = "/path/to/pretrained/encoders_jax"
    im_dim: int = 224
    embed_dim: int = 768
    dtype: str = "float32"
    feat_dtype: str = "float16"


class VisionEncoder(nnx.Module):
    """
    NNX Vision Transformer encoder wrapper.
    Instantiates a ViT backbone and handles configuration.
    Freezing is handled externally via optimizer configuration.
    """

    def __init__(
        self,
        cfg: VisionEncoderConfig,
        rngs: nnx.Rngs,  # Required for backbone initialization
        **kwargs,  # Pass additional args (like img_size) to backbone
    ):
        self.name = cfg.name

        # Determine pretrain path
        pretrain_path: Optional[str] = "none"  # Default to scratch
        if cfg.pretrain_dir and cfg.name in _MODELS:
            path = os.path.join(cfg.pretrain_dir, _MODELS[cfg.name])
            if os.path.exists(path):
                pretrain_path = path
            else:
                print(
                    f"Warning: Pretrain path not found: {path}. Training from scratch."
                )
        elif cfg.name not in _MODELS and cfg.name not in [
            "regnet_y_3_2gf",
            "regnet_y_16gf",
            "regnet_y_800mf",
        ]:  # Add other known non-ViT models here
            print(
                f"Warning: Model name '{cfg.name}' not in known models or RegNets. Assuming scratch training."
            )

        # Instantiate the appropriate backbone
        # Pass rngs down for initialization
        match cfg.name:
            # case "vits-mae-hoi" | "vits-mae-egosoup":
            #     self.backbone = vit_s16_nnx(pretrain_path, rngs=rngs, **kwargs)
            #     self.emb_dim = 384
            case "vitb-mae-egosoup":
                self.backbone = vit_b16(pretrain_path, rngs=rngs, **kwargs)
                self.emb_dim = 768
            # case "vitl-mae-egosoup":
            #     # Pass img_size specifically if needed, from kwargs
            #     merged_kwargs = {"img_size": 256, **kwargs}
            #     self.backbone = vit_l16_nnx(pretrain_path, rngs=rngs, **merged_kwargs)
            #     self.emb_dim = 1024
            # case "vith-mae-egosoup":
            #     merged_kwargs = {"img_size": 256, **kwargs}
            #     self.backbone = vit_h16_nnx(pretrain_path, rngs=rngs, **merged_kwargs)
            #     self.emb_dim = 1280
            case _:
                raise NotImplementedError(f"Unknown model name: {cfg.name}")

    def __call__(
        self,
        x: jax.Array,
        *,
        mode: Optional[int] = VisionTransformerExtractor.ALL_MODE,
        train: bool,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jax.Array:
        # Pass arguments to the ViT backbone's __call__
        out = self.backbone(x, mode=mode, train=train, rngs=rngs)

        # Dtype handling: JAX usually propagates dtypes. Explicit cast removed.
        # If mixed precision policies are used, JAX handles it.
        # out = out.astype(x.dtype) # Removed - generally not needed

        return out
