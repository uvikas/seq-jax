"""Load regnet pretrained and compare outputs between torch and jax"""

import jax
import torch
from flax import nnx
import numpy as np

import seq_jax.regnet as regnet_jax
import torch_ref.vision.vit as regnet_torch

torch_enc = regnet_torch.Encoder("regnet_y_800mf")

key = jax.random.key(2)
init_rngs = nnx.Rngs(params=key, dropout=key)
jax_enc = regnet_jax.VisionEncoder(
    cfg=regnet_jax.VisionEncoderConfig(),
    rngs=init_rngs,
)
backbone = jax_enc.backbone

dummy_input = jax.random.uniform(key, shape=(1, 3, 224, 224), minval=0, maxval=255)
out = jax_enc(dummy_input, rngs=init_rngs)
print(out.shape, out.dtype)
jax_to_torch = torch.from_numpy(np.asarray(out))

dummy_torch_inp = torch.from_numpy(np.array(dummy_input))
out_torch = torch_enc(dummy_torch_inp)
print(out_torch.shape, out_torch.dtype)

print("max abs diff", (jax_to_torch - out_torch).abs().max())

print("Done!!")
