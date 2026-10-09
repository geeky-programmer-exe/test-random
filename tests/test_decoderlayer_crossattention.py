import os
import sys

# hw3lib lives in the directory that contains tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax
import jax.numpy as jnp
from flax import nnx


def test_decoderlayer_crossattention(decoder_layer):
    '''
    Test the decoder layer with both self-attention and cross-attention mechanisms integrated in JAX.   
    '''
    # Structural Test
    test_initialization(decoder_layer)

    # Integration Test
    test_forward_shapes(decoder_layer)  
    test_sublayer_integration(decoder_layer)
    test_cross_attention_integration(decoder_layer)


def test_initialization(decoder_layer):
    '''
    Test if the sublayers exist and are properly initialized.
    '''
    print("Testing initialization ...")
    d_model   = 16
    num_heads = 4
    d_ff      = 32
    dropout   = 0.1
    model = decoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(0))

    # Check if sublayers exist
    expected_attributes = {"self_attn", "cross_attn", "ffn"}
    assert expected_attributes.issubset(dir(model)), "Required sublayers are missing"

    # Check if sublayers are not None
    assert model.self_attn is not None, "self_attn sublayer is None"
    assert model.cross_attn is not None, "cross_attn sublayer is None"
    assert model.ffn is not None, "ffn sublayer is None"

    print("Test Passed: All sublayers exist and are initialized correctly")


def test_forward_shapes(decoder_layer):
    '''
    Test the shapes of the output of the decoder layer.
    '''
    print("Testing forward shapes ...")

    d_model   = 16
    num_heads = 4
    d_ff      = 32
    dropout   = 0.0
    model = decoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(1))

    # Create test inputs
    batch_size = 2
    dec_seq_length = 8
    enc_seq_length = 10
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (batch_size, dec_seq_length, d_model))
    enc_output = jax.random.normal(k2, (batch_size, enc_seq_length, d_model))
    pad_mask_dec  = jnp.zeros((batch_size, dec_seq_length), dtype=bool)
    pad_mask_enc  = jnp.zeros((batch_size, enc_seq_length), dtype=bool)
    slf_attn_mask = jnp.zeros((dec_seq_length, dec_seq_length), dtype=bool)

    # Forward pass
    output, self_attn_weights, cross_attn_weights = model(
        x, enc_output, pad_mask_dec, pad_mask_enc, slf_attn_mask
    )

    # Check shapes
    assert output.shape == (batch_size, dec_seq_length, d_model), \
        f"Output shape mismatch: expected {(batch_size, dec_seq_length, d_model)} but got {output.shape}"
    assert self_attn_weights.shape == (batch_size, dec_seq_length, dec_seq_length), \
        f"Self attention weights shape mismatch: expected {(batch_size, dec_seq_length, dec_seq_length)} but got {self_attn_weights.shape}"
    assert cross_attn_weights.shape == (batch_size, dec_seq_length, enc_seq_length), \
        f"Cross attention weights shape mismatch: expected {(batch_size, dec_seq_length, enc_seq_length)} but got {cross_attn_weights.shape}"

    print("Test Passed: Forward shapes are as expected")


def test_sublayer_integration(decoder_layer):
    '''
    Test the integration of the sublayers.
    '''
    print("Testing sublayer integration ...")

    d_model   = 16
    num_heads = 4
    d_ff      = 32
    dropout   = 0.0
    model = decoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(3))

    batch_size = 2
    dec_seq_length = 8
    enc_seq_length = 10
    key = jax.random.PRNGKey(4)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (batch_size, dec_seq_length, d_model))
    enc_output = jax.random.normal(k2, (batch_size, enc_seq_length, d_model))
    pad_mask_dec  = jnp.zeros((batch_size, dec_seq_length), dtype=bool)
    pad_mask_enc  = jnp.zeros((batch_size, enc_seq_length), dtype=bool)
    slf_attn_mask = jnp.zeros((dec_seq_length, dec_seq_length), dtype=bool)

    # Trace step by step
    self_attn_out, _ = model.self_attn(x, pad_mask=pad_mask_dec, attn_mask=slf_attn_mask)
    cross_attn_out, _ = model.cross_attn(self_attn_out, memory=enc_output, memory_pad_mask=pad_mask_enc, attn_mask=None)
    ffn_out = model.ffn(cross_attn_out)

    output, _, _ = model(x, enc_output, pad_mask_dec, pad_mask_enc, slf_attn_mask)

    assert not jnp.array_equal(self_attn_out, x), "self_attn did not transform the input"
    assert not jnp.array_equal(cross_attn_out, self_attn_out), "cross_attn did not transform self_attn's output"
    assert not jnp.array_equal(ffn_out, cross_attn_out), "ffn did not transform cross_attn's output"
    assert jnp.allclose(output, ffn_out, atol=1e-6), "Final output should match ffn's output"

    print("Test Passed: Sublayers interact correctly")


def test_cross_attention_integration(decoder_layer):
    '''
    Test the integration of the cross-attention mechanism.
    '''
    print("Testing cross-attention behavior ...")
    
    d_model   = 16
    num_heads = 4
    d_ff      = 32
    dropout   = 0.0
    model = decoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(5))

    batch_size = 2
    dec_seq_length = 8
    enc_seq_length = 10
    x = jax.random.normal(jax.random.PRNGKey(6), (batch_size, dec_seq_length, d_model))
    pad_mask_dec  = jnp.zeros((batch_size, dec_seq_length), dtype=bool)
    pad_mask_enc  = jnp.zeros((batch_size, enc_seq_length), dtype=bool)
    slf_attn_mask = jnp.zeros((dec_seq_length, dec_seq_length), dtype=bool)

    # Test 1: Different encoder outputs should produce different results
    enc_output1 = jax.random.normal(jax.random.PRNGKey(7), (batch_size, enc_seq_length, d_model))
    enc_output2 = jax.random.normal(jax.random.PRNGKey(8), (batch_size, enc_seq_length, d_model))

    output1, _, cross_attn1 = model(x, enc_output1, pad_mask_dec, pad_mask_enc, slf_attn_mask)
    output2, _, cross_attn2 = model(x, enc_output2, pad_mask_dec, pad_mask_enc, slf_attn_mask)

    assert not jnp.allclose(output1, output2), "Different encoder outputs should produce different results"
    assert not jnp.allclose(cross_attn1, cross_attn2), "Different encoder outputs should produce different attention patterns"

    # Test 2: Masked encoder positions should be ignored
    pad_mask_enc_masked = jnp.zeros((batch_size, enc_seq_length), dtype=bool).at[:, -5:].set(True)

    _, _, cross_attn_masked = model(x, enc_output1, pad_mask_dec, pad_mask_enc_masked, slf_attn_mask)
    
    assert jnp.all(cross_attn_masked[:, :, -5:] == 0.0), "Masked encoder positions should be ignored in cross-attention"

    print("Test Passed: Cross-attention behaves correctly")


def main():
    '''
    Main function to run the tests.
    '''
    from hw3lib.model import CrossAttentionDecoderLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'CrossAttentionDecoderLayer': [
                {
                    'func': lambda: test_decoderlayer_crossattention(CrossAttentionDecoderLayer),
                    'description': 'Test the cross-attention decoder layer in JAX'
                }
            ]
        }
    )      
    framework.run_tests()
    framework.summarize_results()


if __name__ == "__main__":
    main()
