import os
import sys

# hw3lib lives in the directory that contains tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax
import jax.numpy as jnp
from flax import nnx
from hw3lib.model import MultiHeadAttention, RMSNorm


def test_sublayer_selfattention(self_attn):
    '''
    Test the self-attention sublayer in JAX.
    Args:
        self_attn (nnx.Module): The self-attention sublayer class.
    '''
    # Structural Test
    test_initialization(self_attn)  

    # Functional Tests
    test_forward_shapes(self_attn)

    # Behavioural Test
    test_padding_mask_behaviour(self_attn)
    test_self_attention_mask_behaviour(self_attn)
    test_self_attention_residual(self_attn)


def test_initialization(self_attn):
    '''
    Test if the layers exist in the sublayer.
    '''
    print("Testing initialization ...")
    d_model   = 10    
    num_heads = 2
    dropout   = 0.0
    model     = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout, rngs=nnx.Rngs(0))

    # Check if the layers exist in the model 
    expected_attributes = {"mha", "norm", "dropout"}
    assert expected_attributes.issubset(dir(model)), "Required attributes are missing"
   
    # Check if the layers are not None
    assert model.mha is not None, "Multi-Head Attention layer is None"
    assert model.norm is not None, "Normalization layer is None"
    assert model.dropout is not None, "Dropout layer is None"   
   
    # Check if the layers are of the correct type
    assert isinstance(model.mha, MultiHeadAttention), "Multi-Head Attention layer is not of the correct type"
    assert model.mha.embed_dim == d_model, f"Multi-Head Attention embed_dim: expected {d_model} but got {model.mha.embed_dim}"
    assert model.mha.num_heads == num_heads, f"Multi-Head Attention num_heads: expected {num_heads} but got {model.mha.num_heads}"
    assert isinstance(model.dropout, nnx.Dropout), "Dropout layer is not of the correct type"
    drop_p = getattr(model.dropout, 'rate', getattr(model.dropout, 'p', None))
    assert drop_p == dropout, f"Dropout layer p: expected {dropout} but got {drop_p}"

    # Check the normalization layer for both norm types
    layernorm_model = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout, norm_type='layernorm', rngs=nnx.Rngs(0))
    rmsnorm_model   = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout, norm_type='rmsnorm', rngs=nnx.Rngs(0))
    assert isinstance(layernorm_model.norm, nnx.LayerNorm), "Normalization layer is not of the correct type for norm_type='layernorm'"
    ln_shape = getattr(layernorm_model.norm, 'normalized_shape', (layernorm_model.norm.num_features,))
    assert ln_shape == (d_model,), f"Normalization layer normalized_shape: expected {d_model} but got {ln_shape}"
    assert isinstance(rmsnorm_model.norm, RMSNorm), "Normalization layer is not of the correct type for norm_type='rmsnorm'"

    print("Test Passed: All layers exist and are instantiated correctly")


def test_forward_shapes(self_attn):
    '''
    Test if the forward pass returns the correct shapes.
    '''
    print("Testing forward shapes ...")
    d_model   = 10
    num_heads = 2
    dropout   = 0.1
    model     = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout, rngs=nnx.Rngs(0))

    batch_size = 4
    seq_length = 8
    key = jax.random.PRNGKey(1)
    input_tensor = jax.random.normal(key, (batch_size, seq_length, d_model))

    pad_mask = jnp.zeros((batch_size, seq_length), dtype=bool)
    attn_mask = jnp.zeros((seq_length, seq_length), dtype=bool)

    output, attn_weights = model.forward(input_tensor, pad_mask, attn_mask)

    assert output.shape == (batch_size, seq_length, d_model), \
        f"Output shape: expected {(batch_size, seq_length, d_model)} but got {output.shape}"  
    assert attn_weights.shape == (batch_size, seq_length, seq_length), \
        f"Attention weights shape: expected {(batch_size, seq_length, seq_length)} but got {attn_weights.shape}"
    print("Test Passed: Forward pass returns the correct shapes")


def test_padding_mask_behaviour(self_attn):
    '''
    Test if the padding mask is applied correctly.
    '''
    print("Testing padding mask behaviour ...")
    d_model   = 10
    num_heads = 2
    dropout   = 0.1
    model     = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout, rngs=nnx.Rngs(2))
    
    batch_size = 4
    seq_length = 8
    to_pad = 2
    input_tensor = jax.random.normal(jax.random.PRNGKey(3), (batch_size, seq_length, d_model))

    pad_mask = jnp.zeros((batch_size, seq_length), dtype=bool).at[:, to_pad:].set(True)
    attn_mask = jnp.zeros((seq_length, seq_length), dtype=bool)

    _, attn_weights = model.forward(input_tensor, pad_mask, attn_mask)

    assert jnp.all(attn_weights[:, :, to_pad:] == 0.0), "Attention weights for padded positions should be zero"
    print("Test Passed: Padding mask is applied correctly") 


def test_self_attention_mask_behaviour(self_attn):
    '''
    Test if the self-attention mask is applied correctly.
    '''
    print("Testing self-attention mask behaviour ...")
    d_model   = 10
    num_heads = 2
    dropout   = 0.1
    model     = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout, rngs=nnx.Rngs(4))
    
    batch_size = 4
    seq_length = 8
    input_tensor = jax.random.normal(jax.random.PRNGKey(5), (batch_size, seq_length, d_model))

    pad_mask = jnp.zeros((batch_size, seq_length), dtype=bool)
    attn_mask = jnp.triu(jnp.ones((seq_length, seq_length), dtype=bool), k=1)

    _, attn_weights = model.forward(input_tensor, pad_mask, attn_mask)

    triu_indices = jnp.triu_indices(seq_length, k=1)
    for b in range(batch_size):
        assert jnp.all(attn_weights[b][triu_indices] == 0.0), "Future positions should not be attended"
    print("Test Passed: Self-attention mask is applied correctly")  


def test_self_attention_residual(self_attn):
    '''
    Test if the self-attention residual is applied correctly.
    '''
    print("Testing self-attention residual ...")
    d_model   = 4
    num_heads = 2
    dropout   = 0.0
    model     = self_attn(d_model=d_model, num_heads=num_heads, dropout=dropout, rngs=nnx.Rngs(6))
    
    # Zero out key and value projections so attention context is zero, leaving only the residual
    model.mha.q_proj.kernel.value = jnp.eye(d_model)
    model.mha.k_proj.kernel.value = jnp.zeros((d_model, d_model))
    model.mha.v_proj.kernel.value = jnp.zeros((d_model, d_model))
    model.mha.out_proj.kernel.value = jnp.eye(d_model)
    for proj in (model.mha.q_proj, model.mha.k_proj, model.mha.v_proj, model.mha.out_proj):
        proj.bias.value = jnp.zeros_like(proj.bias.value)
    
    batch_size   = 4
    seq_length   = 10
    input_tensor = jax.random.normal(jax.random.PRNGKey(7), (batch_size, seq_length, d_model))
    
    pad_mask  = jnp.zeros((batch_size, seq_length), dtype=bool)
    attn_mask = jnp.zeros((seq_length, seq_length), dtype=bool)

    output, _ = model.forward(input_tensor, pad_mask, attn_mask)

    assert jnp.allclose(output, input_tensor, rtol=1e-5, atol=1e-5), "Residual connection is not applied correctly."
    print("Test Passed: Residual connection is applied correctly")


def main():
    from hw3lib.model import SelfAttentionLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'SelfAttentionLayer': [
                {
                    'func': lambda: test_sublayer_selfattention(SelfAttentionLayer),
                    'description': 'Test the self-attention sublayer in JAX'
                }
            ]
        }
    )   

    framework.run_tests()
    framework.summarize_results()


if __name__ == '__main__':
    main()
