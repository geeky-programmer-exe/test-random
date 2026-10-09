import os
import sys

# hw3lib lives in the directory that contains tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax
import jax.numpy as jnp
from flax import nnx


def test_encoderlayer_selfattention(encoder_layer):
    '''
    Test the encoder layer self-attention mechanism in JAX.
    '''
    # Structural Test
    test_initialization(encoder_layer)

    # Integration Test
    test_forward_shapes(encoder_layer)
    test_sublayer_integration(encoder_layer)
    test_bidirectional_attention(encoder_layer)


def test_initialization(encoder_layer):
    '''
    Test if the sublayers exist and are properly initialized.
    '''
    print("Testing initialization ...")
    d_model   = 16
    num_heads = 4
    d_ff      = 32
    dropout   = 0.1
    model = encoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(0))

    # Check if sublayers exist
    expected_attributes = {"self_attn", "ffn"}
    assert expected_attributes.issubset(dir(model)), "Required sublayers are missing"

    # Check if sublayers are not None
    assert model.self_attn is not None, "self_attn sublayer is None"
    assert model.ffn is not None, "ffn sublayer is None"

    print("Test Passed: All sublayers exist and are initialized correctly")


def test_forward_shapes(encoder_layer):
    '''
    Test the shapes of the output of the encoder layer.
    '''
    print("Testing forward shapes ...")
    
    d_model   = 16
    num_heads = 4
    d_ff      = 32
    dropout   = 0.0
    model = encoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(1))

    # Create test inputs
    batch_size = 2
    seq_length = 8
    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (batch_size, seq_length, d_model))
    pad_mask = jnp.zeros((batch_size, seq_length), dtype=bool)

    # Forward pass
    output, attn_weights = model(x, pad_mask)

    # Check shapes
    assert output.shape == (batch_size, seq_length, d_model), \
        f"Output shape mismatch: expected {(batch_size, seq_length, d_model)} but got {output.shape}"
    assert attn_weights.shape == (batch_size, seq_length, seq_length), \
        f"Attention weights shape mismatch: expected {(batch_size, seq_length, seq_length)} but got {attn_weights.shape}"

    print("Test Passed: Forward shapes are as expected")


def test_sublayer_integration(encoder_layer):
    '''
    Test the integration of the sublayers.
    '''
    print("Testing sublayer interaction ...")
    
    d_model   = 16
    num_heads = 4
    d_ff      = 32
    dropout   = 0.0
    model = encoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(3))

    batch_size = 2
    seq_length = 8
    x = jax.random.normal(jax.random.PRNGKey(4), (batch_size, seq_length, d_model))
    pad_mask = jnp.zeros((batch_size, seq_length), dtype=bool)

    # Step by step execution
    self_attn_output, attn_weights = model.self_attn(x, pad_mask=pad_mask)
    ffn_output = model.ffn(self_attn_output)
    output, _ = model(x, pad_mask)

    # Verify data flow
    assert not jnp.array_equal(self_attn_output, x), "self_attn did not transform the input"
    assert not jnp.array_equal(ffn_output, self_attn_output), "ffn did not transform self_attn's output"
    assert jnp.allclose(output, ffn_output, atol=1e-6), "Final output should match ffn's output"

    print("Test Passed: Sublayers interact correctly")


def test_bidirectional_attention(encoder_layer):
    '''
    Test that the encoder can attend to all positions (bidirectional attention).
    '''
    print("Testing bidirectional attention ...")
    
    d_model   = 4
    num_heads = 1
    d_ff      = 8
    dropout   = 0.0
    model = encoder_layer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(5))

    # Initialize MHA weights to zeros for key/value so attention distribution is uniform
    model.self_attn.mha.q_proj.kernel.value = jnp.eye(d_model)
    model.self_attn.mha.k_proj.kernel.value = jnp.zeros((d_model, d_model))
    model.self_attn.mha.v_proj.kernel.value = jnp.zeros((d_model, d_model))
    for proj in (model.self_attn.mha.q_proj, model.self_attn.mha.k_proj, model.self_attn.mha.v_proj):
        proj.bias.value = jnp.zeros_like(proj.bias.value)
    model.self_attn.mha.out_proj.kernel.value = jnp.eye(d_model)
    model.self_attn.mha.out_proj.bias.value = jnp.zeros_like(model.self_attn.mha.out_proj.bias.value)

    # Create input with distinct patterns
    batch_size = 1
    seq_length = 4
    x = jnp.zeros((batch_size, seq_length, d_model))
    x = x.at[0, 0].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    x = x.at[0, 1].set(jnp.array([0.0, 1.0, 0.0, 0.0]))
    x = x.at[0, 2].set(jnp.array([0.0, 0.0, 1.0, 0.0]))
    x = x.at[0, 3].set(jnp.array([0.0, 0.0, 0.0, 1.0]))

    pad_mask = jnp.zeros((batch_size, seq_length), dtype=bool)

    # Forward pass
    _, attn_weights = model(x, pad_mask)

    # Test that each position can attend to all other positions
    for i in range(seq_length):
        for j in range(seq_length):
            assert attn_weights[0, i, j] > 0, \
                f"Position {i} cannot attend to position {j}"

    # Test that attention weights for each query sum to 1
    assert jnp.allclose(attn_weights.sum(axis=-1), jnp.ones_like(attn_weights.sum(axis=-1)), atol=1e-5), \
        "Attention weights do not sum to 1"

    print("Test Passed: Bidirectional attention is working correctly")


def main():
    '''
    Main function to run the tests.
    '''
    from hw3lib.model import SelfAttentionEncoderLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'SelfAttentionEncoderLayer': [
                {
                    'func': lambda: test_encoderlayer_selfattention(SelfAttentionEncoderLayer),
                    'description': 'Test the self-attention encoder layer in JAX'
                }
            ]
        }
    )      

    framework.run_tests()
    framework.summarize_results()


if __name__ == "__main__":
    main()
