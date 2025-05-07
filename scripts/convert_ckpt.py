"""Convert checkpoint from torch to jax. Very hacky"""

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
    vit_jax.VisionEncoderConfig(
        name="vitb-mae-egosoup",
        pretrain_dir="/path/to/pretrained/encoders_jax",
    ),
    rngs=init_rngs,
)

backbone = jax_enc.backbone

param_state = nnx.state(backbone, nnx.Param)
wrote = []
for path, var_state in nnx.to_flat_state(param_state):
    path = rename(path)
    true_weights = torch_enc.backbone.state_dict()[path].numpy()
    jax_weights = jnp.array(true_weights)

    if tuple(reversed(jax_weights.shape)) == tuple(var_state.value.shape):
        jax_weights = jnp.transpose(jax_weights)

    assert (
        jax_weights.shape == var_state.value.shape
    ), f"{jax_weights.shape} != {var_state.value.shape} for {path}"
    var_state.value = jax_weights
    wrote.append(path)

nnx.update(backbone, param_state)

# print(torch_enc.backbone.state_dict())
# print(jax_enc.backbone.state_dict())

# state = nnx.state(jax_enc, nnx.Param)
# print(state)

# all_state_keys = list(state.keys())
# print("--- All State Keys ---")
# for key in all_state_keys:
#     # Keys are tuples representing the path
#     print(f"{'/'.join(map(str, key))}: {type(state[key]).__name__}")


dummy_input = jnp.ones((1, 3, 224, 224))
out = jax_enc(dummy_input, train=False)
print(out.shape, out.dtype)
jax_to_torch = torch.from_numpy(np.asarray(out))

dummy_torch_inp = torch.ones((1, 3, 224, 224))
out_torch = torch_enc(dummy_torch_inp)
print(out_torch.shape, out_torch.dtype)

print("max abs diff", (jax_to_torch - out_torch).abs().max())


print("jax_keys")
params = nnx.state(backbone, nnx.Param)
d = nnx.to_pure_dict(params)
jax_dict = {rename(k): v.shape for k, v in traverse_util.flatten_dict(d).items()}
print(jax_dict)

print("torch_keys")
torch_dict = {k: tuple(v.shape) for k, v in torch_enc.backbone.state_dict().items()}
print(torch_dict)

jax_keys = set(jax_dict.keys())
torch_keys = set(torch_enc.backbone.state_dict().keys())
wrote = set(wrote)

print("missing", jax_keys - torch_keys)
print("extra", torch_keys - jax_keys)

print("missing", jax_keys - wrote)
print("extra", wrote - jax_keys)

p = nnx.to_pure_dict(nnx.state(backbone, nnx.Param))
dd = traverse_util.flatten_dict(p)
for k in dd:
    key = rename(k)
    jax_weights = dd[k]
    if jax_weights.shape == tuple(reversed(torch_enc.backbone.state_dict()[key].shape)):
        jax_weights = jnp.transpose(jax_weights)

    torch_weights = torch_enc.backbone.state_dict()[key].numpy()
    jax_weights = np.asarray(jax_weights)

    assert np.allclose(torch_weights, jax_weights)


_, state = nnx.split(jax_enc.backbone)
state = nnx.filter_state(state, nnx.Param)
nnx.display(state)
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(ckpt_dir / "mae_pretrain_egosoup_vit_base.jax", state)
checkpointer.wait_until_finished()

print("Done!!")
