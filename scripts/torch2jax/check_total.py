"""Load full model (vitb + action head) and compare outputs between torch and jax"""

import os
import jax
import torch
from flax import nnx
import numpy as np
import omegaconf
import orbax.checkpoint as ocp
import jax.numpy as jnp
import joblib
import torch.nn.functional as F

import seq_jax
import seq_jax.actor as actor_jax
from torch_ref.actors.model import make_model as make_torch_model


def preprocess_rgb(rgb):
    rgb = torch.from_numpy(np.asarray(rgb))[:, 80:-80, :]
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)
    rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    # (h, w, c) -> (c, h, w) -> (1, 1, c, h, w)
    rgb = rgb.unsqueeze(0).repeat(4, 16, 1, 1, 1)
    rgb = jnp.asarray(rgb)
    return rgb


data = joblib.load("scratch/data/0000.pkl")

batch_size = 4
num_cams = 3
ctx_len = 16
dofs = 24
im_size = 224

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
jax.config.update("jax_default_matmul_precision", "float32")

CHECKPOINT = "/path/to/torch_checkpoints/fold-h16a1-concat-19hr-lr5e4-vitb-mb256-it100k"
omg_cfg = omegaconf.OmegaConf.load(os.path.join(CHECKPOINT, "config.yaml"))
omg_cfg.encoder.pretrain_dir = "/path/to/pretrained/encoders"
torch_model = (
    make_torch_model(
        omg_cfg,
        obs_shapes=[omg_cfg.transformer_concat.embed_dim] * 3 + [omg_cfg.data.dofs] * 2,
        output_dims=[omg_cfg.transformer_concat.embed_dim] * 3
        + [omg_cfg.data.dofs] * 2,
    )
    .eval()
    .cuda()
)
torch_model = torch.compile(torch_model, mode="reduce-overhead")

state_dict = torch.load(
    os.path.join(CHECKPOINT, "model_iter100000.pt"),
    map_location="cpu",
    weights_only=False,
)["model_state"]
mks, uks = torch_model.load_state_dict(state_dict, strict=True)
print(mks, uks)


# Initialize model directly without eval_shape
key = jax.random.key(42)
init_rngs = nnx.Rngs(params=key, dropout=key)
jax_model = actor_jax.TransformerConcatActor(
    cfg=actor_jax.ModelConfig(),
    rngs=init_rngs,
)
graphdef, state = nnx.split(jax_model)

ckptr = ocp.PyTreeCheckpointer()
restored_state = ckptr.restore("/tmp/jax_model/pure/", item=nnx.state(jax_model))
jax_model = nnx.merge(graphdef, restored_state)
jax_model.eval()

obs = seq_jax.policy.Observation(
    img_rgb=[
        preprocess_rgb(data[rgb_key])
        for rgb_key in ["rgb_left", "rgb_head", "rgb_right"]
    ],
    img_feat=None,
    img_pooled=None,
    state=jnp.asarray(
        torch.from_numpy(np.asarray(data["joint_positions"]))
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, ctx_len, 1)
    ),
    noisy_state=jnp.asarray(
        torch.from_numpy(np.asarray(data["joint_positions"]))
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, ctx_len, 1)
    ),
    prev_action=jnp.asarray(
        torch.from_numpy(np.asarray(data["joint_positions"]))
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, ctx_len, 1)
    ),
    mask=jnp.zeros((batch_size, ctx_len)),
)

out_jax = jax_model.sample_actions(obs, rngs=init_rngs)
print("sample actions", out_jax.shape, out_jax.device)
# out_jax = torch.from_numpy(np.asarray(out_jax))
dlpack_capsule = jax.dlpack.to_dlpack(out_jax)
out_jax = torch.utils.dlpack.from_dlpack(dlpack_capsule)
print("sample actions torch", out_jax.shape, out_jax.device)

# (b, t, c, h, w) -> (b, t, h, w, c)
torch_rgb = [
    torch.from_numpy(np.asarray(rgb)).permute(0, 1, 3, 4, 2).cuda()
    for rgb in obs.img_rgb
]
torch_state = torch.from_numpy(np.asarray(obs.state)).cuda()
torch_mask = torch.from_numpy(np.asarray(obs.mask)).cuda()

out_torch = torch_model(
    torch_rgb,
    torch_state,
    torch_state,
    torch_state,
    torch_mask,
    torch_mask,
    extract_feat=True,
)

print(out_torch.shape, out_jax.shape)
print(out_torch.dtype, out_jax.dtype)
print(out_torch.device, out_jax.device)

print("max abs diff", (out_jax - out_torch).abs().max())
print("close", torch.allclose(out_jax, out_torch, atol=1e-6))

print("Done!!")
