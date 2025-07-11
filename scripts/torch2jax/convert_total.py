"""
Convert checkpoint from torch to jax

See the following refs:
https://flax-linen.readthedocs.io/en/latest/guides/converting_and_upgrading/convert_pytorch_to_flax.html
https://docs.jaxstack.ai/en/latest/JAX_porting_PyTorch_model.html
"""

import os
import argparse
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
import torch.nn as nn
import flax
from flax import nnx, serialization, traverse_util
import omegaconf
from functools import partial
import orbax
import orbax.checkpoint as ocp


import torch_ref.vision.vit as vit_torch
import seq_jax.vision as vit_jax
import seq_jax.actor as actor_jax
from torch_ref.actors.model import make_model as make_torch_model

NAME_MAP = {"im_enc": "img_encoder"}


class Torch2Flax:
    @staticmethod
    def conv_params_permute(name, torch_param):
        if name == "weight":
            # (C_out, C_in, kH, kW) -> (kH, kW, C_in, C_out)
            return torch_param.permute(2, 3, 1, 0)
        return torch_param

    @staticmethod
    def linear_params_permute(name, torch_param):
        if name == "weight":
            return torch_param.permute(1, 0)
        return torch_param

    @staticmethod
    def default_params_transform(name, param):
        return param

    modules_mapping_info = {
        nn.Conv2d: {
            "type": nnx.Conv,
            "params_mapping": {
                "weight": "kernel",
                "bias": "bias",
            },
            "params_transform": conv_params_permute,
        },
        nn.BatchNorm2d: {
            "type": nnx.BatchNorm,
            "params_mapping": {
                "weight": "scale",
                "bias": "bias",
                "running_mean": "mean",
                "running_var": "var",
            },
        },
        nn.BatchNorm1d: {
            "type": nnx.BatchNorm,
            "params_mapping": {
                # "weight": "scale",
                # "bias": "bias",
                "running_mean": "mean",
                "running_var": "var",
            },
        },
        nn.Linear: {
            "type": nnx.Linear,
            "params_mapping": {
                "weight": "kernel",
                "bias": "bias",
            },
            "params_transform": linear_params_permute,
        },
        nn.MultiheadAttention: {
            "type": nnx.MultiHeadAttention,
            "params_mapping": {
                "weight": "kernel",
                "bias": "bias",
            },
            # "params_transform": mha_params_permute,
        },
        nn.LayerNorm: {
            "type": nnx.LayerNorm,
            "params_mapping": {
                "weight": "scale",
                "bias": "bias",
            },
        },
    } | {
        torch_mod: {
            "type": nnx_fn_type,
            "params_mapping": {},
        }
        for torch_mod, nnx_fn_type in [
            (nn.Identity, vit_jax.Identity),
            (nn.Flatten, type(lambda x: x)),
            (nn.ReLU, type(nnx.relu)),
            (nn.GELU, type(partial(nnx.gelu, approximate=False))),
            (nn.SELU, type(nnx.selu)),
            (nn.SiLU, type(nnx.silu)),
            (nn.Tanh, type(nnx.tanh)),
            (nn.Dropout, nnx.Dropout),
            (nn.Sigmoid, type(nnx.sigmoid)),
            (nn.AvgPool2d, type(lambda x: nnx.avg_pool(x, (2, 2)))),
            (
                nn.AdaptiveAvgPool2d,
                type(lambda x: nnx.avg_pool(x, (x.shape[1], x.shape[2]))),
            ),
        ]
    }

    def _copy_params_buffers(self, torch_nn_module, nnx_module):
        torch_module_type = type(torch_nn_module)
        assert torch_module_type in self.modules_mapping_info, torch_module_type
        module_mapping_info = self.modules_mapping_info[torch_module_type]
        assert isinstance(nnx_module, module_mapping_info["type"]), (
            torch_nn_module.__class__.__name__,
            nnx_module,
            type(nnx_module),
            module_mapping_info["type"],
        )

        for torch_key, nnx_key in module_mapping_info["params_mapping"].items():

            torch_value = getattr(torch_nn_module, torch_key)
            nnx_param = getattr(nnx_module, nnx_key)
            assert nnx_param is not None, (torch_key, nnx_key, nnx_module)

            if torch_value is None:
                if torch_key == "weight" and nnx_key == "scale":
                    print(list(torch_nn_module.state_dict().keys()))
                assert nnx_param.value is None, (torch_key, nnx_key, nnx_param)
                continue

            params_transform = module_mapping_info.get(
                "params_transform", Torch2Flax.default_params_transform
            )
            torch_value = params_transform(torch_key, torch_value)

            assert nnx_param.value.shape == torch_value.data.shape, (
                torch_nn_module.__class__.__name__,
                nnx_module.__class__.__name__,
                nnx_key,
                nnx_param.value.shape,
                torch_key,
                torch_value.data.shape,
            )
            nnx_param.value = jnp.asarray(torch_value.data)

    def _copy_mha(self, torch_nn_module, nnx_module):
        torch_module_type = type(torch_nn_module)
        assert torch_module_type in self.modules_mapping_info, torch_module_type
        module_mapping_info = self.modules_mapping_info[torch_module_type]
        assert isinstance(nnx_module, module_mapping_info["type"]), (
            torch_nn_module.__class__.__name__,
            nnx_module,
            type(nnx_module),
            module_mapping_info["type"],
        )

        # In proj
        torch_in_proj_weight = torch_nn_module.in_proj_weight
        torch_in_proj_bias = torch_nn_module.in_proj_bias
        torch_out_proj = torch_nn_module.out_proj

        nnx_query = nnx_module.query
        nnx_key = nnx_module.key
        nnx_value = nnx_module.value
        nnx_out = nnx_module.out

        torch_in_proj_weight = Torch2Flax.linear_params_permute(
            "weight", torch_in_proj_weight
        )
        torch_query_weight, torch_key_weight, torch_value_weight = torch.chunk(
            torch_in_proj_weight, 3, dim=1
        )
        torch_query_bias, torch_key_bias, torch_value_bias = torch.chunk(
            torch_in_proj_bias, 3, dim=0
        )
        torch_out_proj_weight = Torch2Flax.linear_params_permute(
            "weight", torch_out_proj.weight
        )
        torch_out_proj_bias = torch_out_proj.bias

        torch_query_weight = torch_query_weight.reshape(768, 8, 96)
        torch_key_weight = torch_key_weight.reshape(768, 8, 96)
        torch_value_weight = torch_value_weight.reshape(768, 8, 96)
        torch_query_bias = torch_query_bias.reshape(8, 96)
        torch_key_bias = torch_key_bias.reshape(8, 96)
        torch_value_bias = torch_value_bias.reshape(8, 96)
        torch_out_proj_weight = torch_out_proj_weight.reshape(8, 96, 768)

        assert nnx_query.kernel.shape == torch_query_weight.shape
        assert nnx_key.kernel.shape == torch_key_weight.shape
        assert nnx_value.kernel.shape == torch_value_weight.shape
        assert nnx_query.bias.shape == torch_query_bias.shape
        assert nnx_key.bias.shape == torch_key_bias.shape
        assert nnx_value.bias.shape == torch_value_bias.shape
        assert nnx_out.kernel.shape == torch_out_proj_weight.shape
        assert nnx_out.bias.shape == torch_out_proj_bias.shape

        nnx_query.kernel.value = jnp.asarray(torch_query_weight.data)
        nnx_query.bias.value = jnp.asarray(torch_query_bias.data)
        nnx_key.kernel.value = jnp.asarray(torch_key_weight.data)
        nnx_key.bias.value = jnp.asarray(torch_key_bias.data)
        nnx_value.kernel.value = jnp.asarray(torch_value_weight.data)
        nnx_value.bias.value = jnp.asarray(torch_value_bias.data)
        nnx_out.kernel.value = jnp.asarray(torch_out_proj_weight.data)
        nnx_out.bias.value = jnp.asarray(torch_out_proj_bias.data)

    def _copy_sequential(self, torch_nn_seq, nnx_seq, skip_modules=None):
        assert isinstance(torch_nn_seq, (nn.Sequential, nn.ModuleList)), type(
            torch_nn_seq
        )
        assert isinstance(nnx_seq, nnx.Sequential), (type(nnx_seq), nnx_seq)
        print("Copying sequential", len(torch_nn_seq), len(nnx_seq.layers))
        for i, index in enumerate(torch_nn_seq):
            torch_module = torch_nn_seq[i]
            nnx_module = nnx_seq.layers[i]
            self.copy_module(torch_module, nnx_module, skip_modules=skip_modules)

    def _copy_module_list(self, torch_nn_seq, nnx_seq, skip_modules=None):
        assert isinstance(torch_nn_seq, nn.ModuleList), type(torch_nn_seq)
        assert isinstance(nnx_seq, list), (
            type(nnx_seq),
            nnx_seq.__class__.__name__,
            type(torch_nn_seq),
            torch_nn_seq.__class__.__name__,
            nnx_seq,
        )
        print("Copying module list", len(torch_nn_seq), len(nnx_seq))
        for i, index in enumerate(torch_nn_seq):
            torch_module = torch_nn_seq[i]
            nnx_module = nnx_seq[i]
            self.copy_module(torch_module, nnx_module, skip_modules=skip_modules)

    def copy_module(self, torch_module, nnx_module, skip_modules=None):
        print("Copying", torch_module.__class__.__name__, type(torch_module))
        if skip_modules is None:
            skip_modules = []

        if isinstance(torch_module, nn.Sequential):
            self._copy_sequential(torch_module, nnx_module, skip_modules=skip_modules)
        elif isinstance(torch_module, nn.ModuleList):
            self._copy_module_list(torch_module, nnx_module, skip_modules=skip_modules)
        elif isinstance(torch_module, nn.MultiheadAttention):
            self._copy_mha(torch_module, nnx_module)
        elif type(torch_module) in self.modules_mapping_info:
            self._copy_params_buffers(torch_module, nnx_module)
        else:
            if skip_modules is not None:
                if torch_module.__class__.__name__ in skip_modules:
                    return

            named_children = list(torch_module.named_children())
            assert len(named_children) > 0, type(torch_module)

            for name, torch_child in named_children:
                if name in NAME_MAP:
                    name = NAME_MAP[name]
                if name == "actor":
                    nnx_child = nnx_module.policy
                else:
                    nnx_child = getattr(nnx_module, name, None)
                assert nnx_child is not None, f"{name} not found in {nnx_module}"
                self.copy_module(torch_child, nnx_child, skip_modules=skip_modules)
            # Copy buffers and params of the module itself (not its children)
            for name, torch_buffer in torch_module.named_buffers():
                if "." in name:
                    # This is child's buffer
                    continue
                if "cache" in name or "att_mask" in name:
                    # This is a cache buffer
                    continue
                nnx_buffer = getattr(nnx_module, name)
                assert isinstance(nnx_buffer, nnx.Variable), (
                    name,
                    nnx_buffer,
                    nnx_module,
                )

                assert nnx_buffer.value.shape == torch_buffer.shape, (
                    name,
                    nnx_buffer.value.shape,
                    torch_buffer.shape,
                )
                nnx_buffer.value = jnp.asarray(torch_buffer)

            for name, torch_param in torch_module.named_parameters():
                if "." in name:
                    # This is child's parameter
                    continue
                nnx_param = getattr(nnx_module, name)
                assert isinstance(nnx_param, nnx.Param), (name, nnx_param, nnx_module)

                assert nnx_param.value.shape == torch_param.data.shape, (
                    name,
                    nnx_param.value.shape,
                    torch_param.data.shape,
                )
                nnx_param.value = jnp.asarray(torch_param.data)


def convert_total(ckpt_dir: str):
    t2f = Torch2Flax()

    CHECKPOINT = "/path/to/torch_checkpoints/fold-h16a1-concat-19hr-lr5e4-vitb-mb256-it100k"
    omg_cfg = omegaconf.OmegaConf.load(os.path.join(CHECKPOINT, "config.yaml"))
    omg_cfg.encoder.pretrain_dir = "/path/to/pretrained/encoders"
    torch_model = make_torch_model(
        omg_cfg,
        obs_shapes=[omg_cfg.transformer_concat.embed_dim] * 3 + [omg_cfg.data.dofs] * 2,
        output_dims=[omg_cfg.transformer_concat.embed_dim] * 3
        + [omg_cfg.data.dofs] * 2,
    ).eval()
    torch_model = torch.compile(torch_model, mode="reduce-overhead")

    state_dict = torch.load(
        os.path.join(CHECKPOINT, "model_iter100000.pt"),
        map_location="cpu",
        weights_only=False,
    )["model_state"]
    torch_model.load_state_dict(state_dict, strict=True)

    key = jax.random.key(0)
    init_rngs = nnx.Rngs(params=key, dropout=key)
    jax_model = actor_jax.TransformerConcatActor(
        cfg=actor_jax.ModelConfig(),
        rngs=init_rngs,
    )

    print("copying torch -> jax")
    t2f.copy_module(torch_model._orig_mod, jax_model)

    CKPT_DIR = "/tmp/jax_model/pure/"
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(CKPT_DIR, nnx.state(jax_model), force=True)

    print("Done!!")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ckpt_dir", type=str, default="/tmp/jax_model/")
    args = args.parse_args()
    convert_total(args.ckpt_dir)
