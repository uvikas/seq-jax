import math
import numpy as np
import torch
import jax
import jax.numpy as jnp

# PyTorch function (PyTorch ≥2.0) :contentReference[oaicite:0]{index=0}
from torch.nn.functional import scaled_dot_product_attention

# JAX function :contentReference[oaicite:1]{index=1}
from jax.nn import dot_product_attention


def compare_attention_outputs(
    batch_size=2,
    seq_len_q=4,
    seq_len_kv=4,
    num_heads=3,
    head_dim=5,
    seed=0,
    use_causal=False,
):
    # 1) Seed both frameworks
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    # 2) Generate random inputs for PyTorch: shape (B, H, L, D)
    q_t = torch.randn(batch_size, num_heads, seq_len_q, head_dim)
    k_t = torch.randn(batch_size, num_heads, seq_len_kv, head_dim)
    v_t = torch.randn(batch_size, num_heads, seq_len_kv, head_dim)

    # 3) Compute PyTorch attention
    #    Default scale = 1/sqrt(D)
    out_t = scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=use_causal,
        scale=None,
    )

    # 4) Convert to JAX format: (B, L, H, D)
    q_j = jnp.array(q_t.permute(0, 2, 1, 3).cpu().numpy())
    k_j = jnp.array(k_t.permute(0, 2, 1, 3).cpu().numpy())
    v_j = jnp.array(v_t.permute(0, 2, 1, 3).cpu().numpy())

    # 5) Compute JAX attention
    #    Default scale = 1/sqrt(D) when scale=None
    out_j = dot_product_attention(
        q_j,
        k_j,
        v_j,
        bias=None,
        mask=None,
        scale=None,
        is_causal=use_causal,
    )

    # 6) Convert JAX output back to PyTorch shape: (B, H, L, D)
    out_j_t = torch.from_numpy(np.array(out_j)).permute(0, 2, 1, 3)

    # 7) Compare
    diff = out_t - out_j_t
    print(f"Max absolute difference: {diff.abs().max().item():.3e}")
    print(f"Mean absolute difference: {diff.abs().mean().item():.3e}")


if __name__ == "__main__":
    print("=== Unmasked (non-causal) Attention ===")
    compare_attention_outputs(use_causal=False)
    print("\n=== Causal Attention ===")
    compare_attention_outputs(use_causal=True)
