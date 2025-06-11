from typing import Optional, Tuple, Type, Callable, Union, List
import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from dataclasses import dataclass
from flax import nnx
from functools import partial


def normalize(ims: jax.Array) -> jax.Array:
    """Normalize images using ImageNet mean and std."""
    ims = ims.astype(jnp.float32) / 255.0
    mean = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32).reshape(1, 3, 1, 1)
    std = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32).reshape(1, 3, 1, 1)
    ims = (ims - mean) / std
    return ims


def to_tuple(n: int, x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return tuple(x for _ in range(n))
    return x


class Identity(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        return x


class DropPath(nnx.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float):
        self.drop_prob = drop_prob

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        if self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + jax.random.bernoulli(
            rngs.dropout(), keep_prob, shape
        )
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
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        self.proj = nnx.Conv(
            in_features=in_chans,
            out_features=embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            rngs=rngs,
        )
        self.norm = norm_layer(embed_dim, rngs=rngs) if norm_layer else Identity()

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, H, W, C)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.proj(x)
        # (B, H, W, C) -> (B, C, H, W)
        x = jnp.transpose(x, (0, 3, 1, 2))

        if self.flatten:
            # Reshape to (B, embed_dim, num_patches)
            # Then transpose to (B, num_patches, embed_dim)
            x = x.reshape(B, self.embed_dim, -1).transpose(
                (0, 2, 1)
            )  # NCHW -> B C (H*W) -> B (H*W) C

        x = self.norm(x, rngs=rngs)
        return x


class Attention(nnx.Module):

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
        self.scale = self.head_dim**-0.5

        self.qkv = nnx.Linear(
            dim,
            dim * 3,
            use_bias=qkv_bias,
            rngs=rngs,
        )
        self.attn_drop = nnx.Dropout(attn_drop)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:

        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3 * C)
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, self.head_dim
        )  # (B, N, 3, num_heads, head_dim)

        # Attn fn expects (3, B, N, num_heads, head_dim)
        qkv = qkv.transpose((2, 0, 1, 3, 4))  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, N, head_dim)

        attn_output = jax.nn.dot_product_attention(
            q, k, v, is_causal=False
        )  # (B, N, num_heads, head_dim)

        # Transpose to (B, N, num_heads, head_dim) then reshape to (B, N, C)
        x = attn_output.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x, rngs=rngs)
        return x


class LayerScale(nnx.Module):

    def __init__(self, dim: int, init_values: float = 1e-5):
        self.gamma = nnx.Param(jnp.full((dim,), init_values))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.gamma.value


class Mlp(nnx.Module):

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
        self.drop1 = nnx.Dropout(drop)
        self.fc2 = nnx.Linear(
            hidden_features, out_features, use_bias=use_bias[1], rngs=rngs
        )
        self.drop2 = nnx.Dropout(drop)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, rngs=rngs)
        x = self.fc2(x)
        x = self.drop2(x, rngs=rngs)
        return x


class Block(nnx.Module):

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
            LayerScale(dim, init_values=init_values) if init_values else Identity()
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
            LayerScale(dim, init_values=init_values) if init_values else Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        )

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), rngs=rngs), rngs=rngs), rngs=rngs
        )
        x = x + self.drop_path2(
            self.ls2(self.mlp(self.norm2(x), rngs=rngs), rngs=rngs), rngs=rngs
        )
        return x


class VisionTransformer(nnx.Module):

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
        rngs: nnx.Rngs,
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
        self.num_features = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.class_token_enabled = class_token

        # Patch Embedding
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            rngs=rngs,
        )
        num_patches = self.patch_embed.num_patches

        # CLS Token
        if self.class_token_enabled:
            self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        else:
            self.cls_token = None

        # Positional Embedding
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )

        pos_embed_init = (
            jax.random.normal(rngs.params(), (1, embed_len, embed_dim)) * 0.02
        )
        self.pos_embed = nnx.Param(pos_embed_init)
        self.pos_drop = nnx.Dropout(drop_rate)

        # Stochastic Depth Decay Rule
        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]

        # Transformer Blocks
        self.blocks = nnx.Sequential(
            *[
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
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )

        # Final Norm Layer
        self.norm = norm_layer(embed_dim, rngs=rngs) if not use_fc_norm else Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim, rngs=rngs) if use_fc_norm else Identity()
        self.head = (
            nnx.Linear(embed_dim, num_classes, rngs=rngs)
            if num_classes > 0
            else Identity()
        )

    def _pos_embed(self, x: jax.Array, B: int) -> jax.Array:
        if self.no_embed_class:
            x = x + self.pos_embed.value
            if self.class_token_enabled:
                cls_tokens = jnp.broadcast_to(
                    self.cls_token.value, (B, 1, self.embed_dim)
                )
                x = jnp.concatenate([cls_tokens, x], axis=1)
        else:
            if self.class_token_enabled:
                cls_tokens = jnp.broadcast_to(
                    self.cls_token.value, (B, 1, self.embed_dim)
                )
                x = jnp.concatenate([cls_tokens, x], axis=1)
            x = x + self.pos_embed.value

        return x

    def forward_features(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        B = x.shape[0]
        x = self.patch_embed(x, rngs=rngs)  # (B, num_patches, embed_dim)
        x = self._pos_embed(x, B)  # (B, num_patches + num_prefix_tokens, embed_dim)
        x = self.pos_drop(x, rngs=rngs)

        for block in self.blocks:
            x = block(x, rngs=rngs)

        x = self.norm(x)
        return x

    def forward_head(self, x: jax.Array, *, pre_logits: bool = False) -> jax.Array:
        if self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(axis=1)
        elif self.global_pool == "token":
            x = x[:, 0]

        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        x = self.forward_features(x, rngs=rngs)
        x = self.forward_head(x)
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
        kwargs_modified = kwargs.copy()
        kwargs_modified["num_classes"] = 0
        kwargs_modified["fc_norm"] = False
        super().__init__(**kwargs_modified, rngs=rngs)

        if not self.class_token_enabled:
            raise ValueError(
                "VisionTransformerFeaturesNNX requires class_token=True during initialization."
            )

    def extract_feat(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Extracts features before the final LayerNorm. Assumes class_token=True."""
        B = x.shape[0]
        x = self.patch_embed(x, rngs=rngs)

        # Expand cls_token and concatenate
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, 1 + num_patches, embed_dim)

        if self.pos_embed.value.shape[1] != x.shape[1]:
            raise ValueError(
                f"Shape mismatch: pos_embed ({self.pos_embed.value.shape}) vs input ({x.shape})"
            )
        x = x + self.pos_embed.value
        x = self.blocks(x, rngs=rngs)
        return x

    def forward_norm(self, x: jax.Array) -> jax.Array:
        """Applies the final LayerNorm."""
        return self.norm(x)

    def forward_feature(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Extracts features and applies the final LayerNorm."""
        x = self.extract_feat(x, rngs=rngs)
        x = self.forward_norm(x)
        return x

    def __call__(self, x: jax.Array, *, mode: int, rngs: nnx.Rngs) -> jax.Array:
        """
        Main forward pass for feature extraction with different modes.
        Applies stop_gradient to the output.
        """

        # Get features after blocks and final norm
        features = self.forward_feature(x, rngs=rngs)

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

        output = jax.lax.stop_gradient(output)
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
    model = VisionTransformerExtractor(**model_kwargs, rngs=rngs)

    # checkpointer = ocp.StandardCheckpointer()
    # abstract_model = nnx.eval_shape(
    #     lambda: VisionTransformerExtractor(**model_kwargs, rngs=nnx.Rngs(0))
    # )
    # graphdef, rng_state, other_state = nnx.split(abstract_model, nnx.RngState, ...)

    # state_restored = checkpointer.restore(pretrained, other_state)
    # state_restored = nnx.merge_state(state_restored, rng_state)
    # model = nnx.merge(graphdef, state_restored)

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
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.name = cfg.name
        pretrain_path: Optional[str] = "none"
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
        ]:
            print(
                f"Warning: Model name '{cfg.name}' not in known models or RegNets. Assuming scratch training."
            )

        match cfg.name:
            case "vitb-mae-egosoup":
                self.backbone = vit_b16(pretrain_path, rngs=rngs, **kwargs)
                self.emb_dim = 768
            case _:
                raise NotImplementedError(f"Unknown model name: {cfg.name}")

    def __call__(
        self,
        x: jax.Array,
        *,
        mode: int = VisionTransformerExtractor.ALL_MODE,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        out = self.backbone(x, mode=mode, rngs=rngs)
        return out
