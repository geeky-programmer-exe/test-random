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


def test_sublayer_crossattention(cross_attention):
    '''
    Test the cross-attention sublayer in JAX.
    Args:
        cross_attention (nnx.Module): The cross-attention sublayer class.
    '''
    # Structural Test
    test_initialization(cross_attention)

    # Functional Tests
    test_forward_shapes(cross_attention)

    # Behavioural Tests
    test_padding_mask_behaviour(cross_attention)
    test_cross_attention_behaviour(cross_attention)
    test_cross_attention_residual(cross_attention)


def test_initialization(cross_attention):
    '''
    Test if the layers exist in the decoder cross-attention sublayer.
    '''
    print("Testing initialization ...")
    d_model   = 10
    num_heads = 2
    dropout   = 0.1
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout, rngs=nnx.Rngs(0))

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

    # Cross-attention never carries rotary embeddings, the queries and keys come
    # from two different sequences
    assert model.mha.rope is None, "Cross-attention should not use rotary embeddings"

    # Check the normalization layer for both norm types
    layernorm_model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout, norm_type='layernorm', rngs=nnx.Rngs(0))
    rmsnorm_model   = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout, norm_type='rmsnorm', rngs=nnx.Rngs(0))
    assert isinstance(layernorm_model.norm, nnx.LayerNorm), "Normalization layer is not of the correct type for norm_type='layernorm'"
    ln_shape = getattr(layernorm_model.norm, 'normalized_shape', (layernorm_model.norm.num_features,))
    assert ln_shape == (d_model,), f"Normalization layer normalized_shape: expected {d_model} but got {ln_shape}"
    assert isinstance(rmsnorm_model.norm, RMSNorm), "Normalization layer is not of the correct type for norm_type='rmsnorm'"

    print("Test Passed: All layers exist and are instantiated correctly")


def test_forward_shapes(cross_attention):
    '''
    Test if the forward pass returns the correct shapes.
    '''
    print("Testing forward shapes ...")
    d_model   = 10
    num_heads = 2
    dropout   = 0.1
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout, rngs=nnx.Rngs(0))

    batch_size = 4
    dec_seq_length = 8
    enc_seq_length = 6
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    decoder_input = jax.random.normal(k1, (batch_size, dec_seq_length, d_model))
    encoder_output = jax.random.normal(k2, (batch_size, enc_seq_length, d_model))

    pad_mask_enc = jnp.zeros((batch_size, enc_seq_length), dtype=bool)

    output, attn_weights = model.forward(decoder_input, encoder_output, pad_mask_enc, None)

    assert output.shape == (batch_size, dec_seq_length, d_model), \
        f"Output shape: expected {(batch_size, dec_seq_length, d_model)} but got {output.shape}"
    assert attn_weights.shape == (batch_size, dec_seq_length, enc_seq_length), \
        f"Attention weights shape: expected {(batch_size, dec_seq_length, enc_seq_length)} but got {attn_weights.shape}"
    print("Test Passed: Forward pass returns the correct shapes")


def test_padding_mask_behaviour(cross_attention):
    '''
    Test if the padding mask is applied correctly.
    '''
    print("Testing padding mask behaviour ...")
    d_model   = 10
    num_heads = 2
    dropout   = 0.1
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout, rngs=nnx.Rngs(2))

    batch_size = 4
    dec_seq_length = 8
    enc_seq_length = 6
    to_pad = 2
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    decoder_input  = jax.random.normal(k1, (batch_size, dec_seq_length, d_model))
    encoder_output = jax.random.normal(k2, (batch_size, enc_seq_length, d_model))

    pad_mask_enc = jnp.zeros((batch_size, enc_seq_length), dtype=bool).at[:, to_pad:].set(True)

    _, attn_weights = model.forward(decoder_input, encoder_output, pad_mask_enc, None)

    assert jnp.all(attn_weights[:, :, to_pad:] == 0.0), "Attention weights for padded positions should be zero"
    print("Test Passed: Padding mask is applied correctly")


def test_cross_attention_behaviour(cross_attention):
    '''
    Test if the cross-attention mechanism correctly attends to encoder outputs.
    '''
    print("Testing cross-attention behaviour ...")
    d_model   = 4
    num_heads = 1
    dropout   = 0.0
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout, norm_type='rmsnorm', rngs=nnx.Rngs(4))

    # Initialize MHA weights to identity projections
    model.mha.q_proj.kernel.value = jnp.eye(d_model)
    model.mha.k_proj.kernel.value = jnp.eye(d_model)
    model.mha.v_proj.kernel.value = jnp.eye(d_model)
    for proj in (model.mha.q_proj, model.mha.k_proj, model.mha.v_proj):
        proj.bias.value = jnp.zeros_like(proj.bias.value)
    
    model.mha.out_proj.kernel.value = jnp.eye(d_model)
    model.mha.out_proj.bias.value = jnp.zeros_like(model.mha.out_proj.bias.value)

    batch_size     = 1
    dec_seq_length = 3
    enc_seq_length = 4
    
    encoder_output = jnp.zeros((batch_size, enc_seq_length, d_model))
    encoder_output = encoder_output.at[0, 0].set(jnp.array([1.0, -1.0, -1.0, -1.0]))
    encoder_output = encoder_output.at[0, 1].set(jnp.array([-1.0, 1.0, -1.0, -1.0]))
    encoder_output = encoder_output.at[0, 2].set(jnp.array([-1.0, -1.0, 1.0, -1.0]))
    encoder_output = encoder_output.at[0, 3].set(jnp.array([-1.0, -1.0, -1.0, 1.0]))

    decoder_input = jnp.zeros((batch_size, dec_seq_length, d_model))
    decoder_input = decoder_input.at[0, 0].set(jnp.array([1.0, -1.0, -1.0, -1.0]))
    decoder_input = decoder_input.at[0, 1].set(jnp.array([-1.0, 1.0, -1.0, -1.0]))
    decoder_input = decoder_input.at[0, 2].set(jnp.array([-1.0, -1.0, 1.0, -1.0]))

    pad_mask_enc = jnp.zeros((batch_size, enc_seq_length), dtype=bool)

    _, attn_weights = model.forward(decoder_input, encoder_output, pad_mask_enc, None)

    for i in range(dec_seq_length):
        max_attention_pos = int(jnp.argmax(attn_weights[0, i]))
        assert max_attention_pos == i, f"Position {i} attended most to position {max_attention_pos} instead of position {i}"
        
        max_attention = attn_weights[0, i, i]
        other_indices = jnp.array([j for j in range(enc_seq_length) if j != i])
        other_positions_attention = attn_weights[0, i, other_indices]
        assert jnp.all(max_attention > other_positions_attention), \
            f"Position {i} does not have significantly higher attention to its corresponding position"

    print("Test Passed: Cross-attention behavior is correct")


def test_cross_attention_residual(cross_attention):
    '''
    Test if the residual connection is applied correctly.
    '''
    print("Testing residual connection ...")
    d_model   = 4
    num_heads = 2
    dropout   = 0.0
    model = cross_attention(d_model=d_model, num_heads=num_heads, dropout=dropout, rngs=nnx.Rngs(5))

    # Force MHA context to be zero
    model.mha.q_proj.kernel.value = jnp.eye(d_model)
    model.mha.k_proj.kernel.value = jnp.zeros((d_model, d_model))
    model.mha.v_proj.kernel.value = jnp.zeros((d_model, d_model))
    model.mha.out_proj.kernel.value = jnp.eye(d_model)
    for proj in (model.mha.q_proj, model.mha.k_proj, model.mha.v_proj, model.mha.out_proj):
        proj.bias.value = jnp.zeros_like(proj.bias.value)

    batch_size     = 4
    dec_seq_length = 10
    enc_seq_length = 10
    decoder_input  = jax.random.normal(jax.random.PRNGKey(6), (batch_size, dec_seq_length, d_model))
    encoder_output = decoder_input

    pad_mask_enc = jnp.zeros((batch_size, enc_seq_length), dtype=bool)

    output, _ = model.forward(decoder_input, encoder_output, pad_mask_enc, None)

    assert jnp.allclose(output, decoder_input, rtol=1e-5, atol=1e-5), "Residual connection is not applied correctly."
    print("Test Passed: Residual connection is applied correctly")


def main():
    from hw3lib.model import CrossAttentionLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'CrossAttentionLayer': [
                {   
                    'func': lambda: test_sublayer_crossattention(CrossAttentionLayer),
                    'description': 'Test the cross-attention sublayer in JAX'
                }
            ]
        }   
    )   

    framework.run_tests()
    framework.summarize_results()


if __name__ == '__main__':
    main()
