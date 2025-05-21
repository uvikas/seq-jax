"""Load full pretrained and compare outputs between torch and jax"""

import jax
import torch
from flax import nnx, serialization
import numpy as np
from flax import traverse_util


import seq_jax.vit as vit_jax
import torch_ref.vision.vit as vit_torch

import jax.numpy as jnp

import orbax.checkpoint as ocp
import time

ckpt_dir = ocp.test_utils.erase_and_create_empty("/tmp/jax_vitb/")


def rename(k: str):
    k = ".".join([str(j) for j in k])
    k = k.replace("fc1.kernel", "fc1.weight")
    k = k.replace("fc2.kernel", "fc2.weight")
    k = k.replace("proj.kernel", "proj.weight")
    k = k.replace("qkv.kernel", "qkv.weight")
    k = k.replace("norm.scale", "norm.weight")
    k = k.replace("norm1.scale", "norm1.weight")
    k = k.replace("norm2.scale", "norm2.weight")
    return k


torch_enc = vit_torch.Encoder(
    "vitb-mae-egosoup", "/path/to/pretrained/encoders"
)


key = jax.random.key(0)
init_rngs = nnx.Rngs(params=key, dropout=key)
jax_enc = vit_jax.VisionEncoder(
    cfg=vit_jax.VisionEncoderConfig(),
    rngs=init_rngs,
)
backbone = jax_enc.backbone

dummy_input = jnp.ones((1, 3, 224, 224))
out = jax_enc(dummy_input, train=False)
print(out.shape, out.dtype)
jax_to_torch = torch.from_numpy(np.asarray(out))

dummy_torch_inp = torch.ones((1, 3, 224, 224))
out_torch = torch_enc(dummy_torch_inp)
print(out_torch.shape, out_torch.dtype)

print("max abs diff", (jax_to_torch - out_torch).abs().max())

print("Done!!")
