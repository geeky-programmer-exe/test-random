import math
import os
import sys

# hw3lib lives in the directory that contains tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax
import jax.numpy as jnp
from flax import nnx
import torch
import torch.nn as nn


def test_attention(attention):
    '''
    Test the MultiHeadAttention module in JAX.
    Args:
        attention (nnx.Module): The multi-head attention module class.
    '''
    # Structural Test
    test_initialization(attention)

    # Functional Tests
    test_shapes(attention)

    # Behavioural Tests
    test_equivalence_to_torch(attention)
    test_masking(attention)
    test_rope_applied_to_qk_only(attention)
    test_qk_norm(attention)


def test_initialization(attention):
    '''
    Test the projections and the head split.
    '''
    print("Testing initialization ...")
    embed_dim, num_heads = 32, 4
    model = attention(embed_dim, num_heads, rngs=nnx.Rngs(0))

    for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
        assert hasattr(model, name), f"MultiHeadAttention is missing {name}"
        proj = getattr(model, name)
        assert isinstance(proj, nnx.Linear), f"{name} should be an nnx.Linear"
        assert proj.in_features == embed_dim,  f"{name} in_features: expected {embed_dim} but got {proj.in_features}"
        assert proj.out_features == embed_dim, f"{name} out_features: expected {embed_dim} but got {proj.out_features}"

    assert model.embed_dim == embed_dim, f"embed_dim: expected {embed_dim} but got {model.embed_dim}"
    assert model.num_heads == num_heads, f"num_heads: expected {num_heads} but got {model.num_heads}"
    assert model.head_dim == embed_dim // num_heads, \
        f"head_dim: expected {embed_dim // num_heads} but got {model.head_dim}"

    # embed_dim must divide evenly into heads
    try:
        attention(embed_dim, 5, rngs=nnx.Rngs(0))
        raise AssertionError("An embed_dim not divisible by num_heads should raise a ValueError")
    except ValueError:
        pass

    print("Test Passed: The projections are set up correctly")


def test_shapes(attention):
    '''
    Test the output and attention weight shapes, including when the query and key
    lengths differ as they do in cross-attention.
    '''
    print("Testing shapes ...")
    N, L, S, E, H = 3, 7, 11, 32, 4
    model = attention(E, H, rngs=nnx.Rngs(0))

    key = jax.random.PRNGKey(10)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (N, L, E))
    k = jax.random.normal(k2, (N, S, E))
    v = jax.random.normal(k3, (N, S, E))
    out, weights = model(q, k, v)

    assert out.shape == (N, L, E), f"Output shape: expected {(N, L, E)} but got {tuple(out.shape)}"
    assert weights.shape == (N, L, S), \
        f"Attention weight shape: expected {(N, L, S)}, averaged over heads, but got {tuple(weights.shape)}"

    # Attention weights are a distribution over the keys
    assert jnp.allclose(weights.sum(axis=-1), jnp.ones((N, L)), rtol=1e-4, atol=1e-4), \
        "Attention weights should sum to 1 over the key dimension"

    print("Test Passed: Shapes are correct")


def test_equivalence_to_torch(attention):
    '''
    Test against nn.MultiheadAttention.

    With rotary embeddings and QK-Norm disabled, this module computes exactly what
    PyTorch's does, so we can copy the weights across and compare directly. Note
    that nn packs the three input projections into one in_proj_weight, so the first
    third is the query projection, the second the key, and the third the value.
    '''
    print("Testing equivalence to nn.MultiheadAttention ...")
    N, L, S, E, H = 3, 7, 11, 32, 4
    model = attention(E, H, dropout=0.0, rngs=nnx.Rngs(0))
    ref = nn.MultiheadAttention(E, H, dropout=0.0, batch_first=True)

    with torch.no_grad():
        W, b = ref.in_proj_weight, ref.in_proj_bias
        # PyTorch Linear weight is (out_features, in_features)
        # Flax NNX Linear kernel is (in_features, out_features)
        model.q_proj.kernel.value = jnp.array(W[:E].detach().numpy().T)
        model.q_proj.bias.value = jnp.array(b[:E].detach().numpy())
        model.k_proj.kernel.value = jnp.array(W[E:2*E].detach().numpy().T)
        model.k_proj.bias.value = jnp.array(b[E:2*E].detach().numpy())
        model.v_proj.kernel.value = jnp.array(W[2*E:].detach().numpy().T)
        model.v_proj.bias.value = jnp.array(b[2*E:].detach().numpy())
        model.out_proj.kernel.value = jnp.array(ref.out_proj.weight.detach().numpy().T)
        model.out_proj.bias.value = jnp.array(ref.out_proj.bias.detach().numpy())

    ref.eval()

    q_np = torch.randn(N, L, E).numpy()
    k_np = torch.randn(N, S, E).numpy()
    v_np = torch.randn(N, S, E).numpy()

    q_jax, k_jax, v_jax = jnp.array(q_np), jnp.array(k_np), jnp.array(v_np)
    q_th, k_th, v_th = torch.tensor(q_np), torch.tensor(k_np), torch.tensor(v_np)

    key_padding_mask_np = (torch.arange(S) >= S - 3).expand(N, S).numpy()
    attn_mask_np = (torch.arange(S).unsqueeze(0).expand(L, S) >= S - 1).numpy()

    cases = [
        ("no mask",          {}, {}),
        ("key padding mask", 
         {"key_padding_mask": jnp.array(key_padding_mask_np)}, 
         {"key_padding_mask": torch.tensor(key_padding_mask_np)}),
        ("attention mask",   
         {"attn_mask": jnp.array(attn_mask_np)}, 
         {"attn_mask": torch.tensor(attn_mask_np)}),
        ("both masks",       
         {"key_padding_mask": jnp.array(key_padding_mask_np), "attn_mask": jnp.array(attn_mask_np)}, 
         {"key_padding_mask": torch.tensor(key_padding_mask_np), "attn_mask": torch.tensor(attn_mask_np)}),
    ]

    for name, jax_kwargs, th_kwargs in cases:
        out, weights = model(q_jax, k_jax, v_jax, **jax_kwargs)
        with torch.no_grad():
            ref_out, ref_weights = ref(q_th, k_th, v_th, need_weights=True, average_attn_weights=True, **th_kwargs)
        
        assert jnp.allclose(out, jnp.array(ref_out.numpy()), rtol=1e-4, atol=1e-5), \
            f"Output does not match nn.MultiheadAttention with {name}. Check the scaling by sqrt(head_dim)."
        assert jnp.allclose(weights, jnp.array(ref_weights.numpy()), rtol=1e-4, atol=1e-5), \
            f"Attention weights do not match nn.MultiheadAttention with {name}"

    print("Test Passed: Matches nn.MultiheadAttention exactly")


def test_masking(attention):
    '''
    Test that masked positions receive no attention.
    '''
    print("Testing masking ...")
    N, L, S, E, H = 2, 6, 8, 16, 2
    model = attention(E, H, dropout=0.0, rngs=nnx.Rngs(1))

    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (N, L, E))
    k = jax.random.normal(k2, (N, S, E))
    v = jax.random.normal(k3, (N, S, E))

    to_pad = 3
    key_padding_mask = jnp.zeros((N, S), dtype=bool).at[:, -to_pad:].set(True)
    _, weights = model(q, k, v, key_padding_mask=key_padding_mask)
    assert jnp.all(weights[:, :, -to_pad:] == 0.0), "Padded key positions should receive zero attention"

    causal = jnp.triu(jnp.ones((L, L), dtype=bool), k=1)
    _, weights = model(q, q, q, attn_mask=causal)
    # Masked upper triangular positions (diagonal=1) should be zero
    upper_tri_indices = jnp.triu_indices(L, k=1)
    for b in range(N):
        assert jnp.all(weights[b][upper_tri_indices] == 0.0), "Future positions should receive zero attention"

    print("Test Passed: Masking is applied correctly")


def test_rope_applied_to_qk_only(attention):
    '''
    Test that rotary embeddings reach the queries and keys but not the values.
    '''
    print("Testing that rotary embeddings reach only the queries and keys ...")
    from hw3lib.model import RotaryPositionalEncoding
    N, L, E, H = 2, 9, 32, 4
    rope = RotaryPositionalEncoding(E // H, 64)

    model = attention(E, H, dropout=0.0, rope=rope, rngs=nnx.Rngs(3))
    assert model.rope is not None, "The rope argument was not stored"

    x = jax.random.normal(jax.random.PRNGKey(4), (N, L, E))
    with_rope, _ = model(x, x, x)
    model.rope = None
    without_rope, _ = model(x, x, x)
    assert not jnp.allclose(with_rope, without_rope, atol=1e-4), \
        "Enabling rotary embeddings did not change the output, so they are not being applied"

    # Rotation is norm preserving, so a value path left unrotated keeps the output
    # magnitude in the same range. A rotation wrongly applied to v would not.
    ratio = float(jnp.linalg.norm(with_rope) / jnp.linalg.norm(without_rope))
    assert 0.5 < ratio < 2.0, \
        f"Output magnitude changed by {ratio:.2f}x, which suggests the rotation was applied to the values as well"

    print("Test Passed: Rotary embeddings reach only the queries and keys")


def test_qk_norm(attention):
    '''
    Test the optional QK-Norm.
    '''
    print("Testing QK-Norm ...")
    from hw3lib.model import RMSNorm
    E, H = 32, 4

    off = attention(E, H, qk_norm=False, rngs=nnx.Rngs(5))
    assert getattr(off, 'q_norm', None) is None, "q_norm should be None when qk_norm is False"

    on = attention(E, H, qk_norm=True, rngs=nnx.Rngs(6))
    assert isinstance(on.q_norm, RMSNorm), "q_norm should be an RMSNorm when qk_norm is True"
    assert isinstance(on.k_norm, RMSNorm), "k_norm should be an RMSNorm when qk_norm is True"
    assert on.q_norm.weight.shape == (E // H,), \
        f"QK-Norm normalizes over head_dim, so its weight should have shape {(E // H,)} but got {tuple(on.q_norm.weight.shape)}"

    x = jax.random.normal(jax.random.PRNGKey(7), (2, 6, E))
    out, weights = on(x, x, x)
    assert out.shape == x.shape, "QK-Norm changed the output shape"
    assert not jnp.isnan(out).any(), "QK-Norm produced NaN"

    print("Test Passed: QK-Norm is applied correctly")


def main():
    """
    Main function to run the multi-head attention tests using the testing framework.
    """
    from hw3lib.model import MultiHeadAttention
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'MultiHeadAttention': [
                {
                    'func': lambda: test_attention(MultiHeadAttention),
                    'description': 'Test the multi-head attention module in JAX'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()


if __name__ == '__main__':
    main()
