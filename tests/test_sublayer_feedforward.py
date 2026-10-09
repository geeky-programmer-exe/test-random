import os
import sys

# hw3lib lives in the directory that contains tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax
import jax.numpy as jnp
from flax import nnx
from hw3lib.model import RMSNorm, SwiGLU


def test_sublayer_feedforward(feedforward):
    '''
    Test the feedforward sublayer in JAX.
    Args:
        feedforward (nnx.Module): The feedforward sublayer class.
    '''
    # Structural Test
    test_initialization(feedforward)
    test_swiglu_structure(feedforward)

    # Functional Tests
    test_forward_shapes(feedforward)
    
    # Behavioral Tests
    test_ffn_behavior(feedforward)
    test_residual_connection(feedforward)
    test_layer_norm(feedforward)
    test_forward_order(feedforward)


def test_initialization(feedforward):
    '''
    Test if the layers exist and match the reference implementation.
    '''
    print("Testing initialization ...")
    d_model = 10
    d_ff    = 40
    dropout = 0.1
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout, ffn_type='gelu', rngs=nnx.Rngs(0))

    # Check if the layers exist in the model
    expected_attributes = {"ffn", "norm", "dropout"}
    assert expected_attributes.issubset(dir(model)), "Required attributes are missing"

    # Check if the layers are not None
    assert model.ffn is not None, "Feed-forward network is None"
    assert model.norm is not None, "Normalization layer is None"
    assert model.dropout is not None, "Dropout layer is None"

    # Check if the layers are of the correct type
    assert isinstance(model.dropout, nnx.Dropout), "Dropout layer is not of the correct type"

    # Check FFN structure
    assert len(model.ffn) == 4, "FFN should have exactly 4 layers"
    assert isinstance(model.ffn[0], nnx.Linear), "First FFN layer should be Linear"
    assert callable(model.ffn[1]), "Second FFN layer should be activation"
    assert isinstance(model.ffn[2], nnx.Dropout), "Third FFN layer should be Dropout"
    assert isinstance(model.ffn[3], nnx.Linear), "Fourth FFN layer should be Linear"

    # Check dimensions
    assert model.ffn[0].in_features == d_model, f"FFN input dimension: expected {d_model} but got {model.ffn[0].in_features}"
    assert model.ffn[0].out_features == d_ff, f"FFN hidden dimension: expected {d_ff} but got {model.ffn[0].out_features}"
    assert model.ffn[3].in_features == d_ff, f"FFN input dimension: expected {d_ff} but got {model.ffn[3].in_features}"
    assert model.ffn[3].out_features == d_model, f"FFN output dimension: expected {d_model} but got {model.ffn[3].out_features}"

    # Check the normalization layer for both norm types
    layernorm_model = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout, norm_type='layernorm', rngs=nnx.Rngs(0))
    rmsnorm_model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout, norm_type='rmsnorm', rngs=nnx.Rngs(0))
    assert isinstance(layernorm_model.norm, nnx.LayerNorm), "Normalization layer is not of the correct type for norm_type='layernorm'"
    ln_shape = getattr(layernorm_model.norm, 'normalized_shape', (layernorm_model.norm.num_features,))
    assert ln_shape == (d_model,), f"Normalization layer normalized_shape: expected {d_model} but got {ln_shape}"
    assert isinstance(rmsnorm_model.norm, RMSNorm), "Normalization layer is not of the correct type for norm_type='rmsnorm'"

    print("Test Passed: All layers exist and match reference implementation")


def test_swiglu_structure(feedforward):
    '''
    Test the gated feed-forward network.
    '''
    print("Testing SwiGLU structure ...")
    d_model = 10
    d_ff    = 40
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=0.0, ffn_type='swiglu', rngs=nnx.Rngs(0))

    assert isinstance(model.ffn, SwiGLU), "Feed-forward network should be a SwiGLU when ffn_type='swiglu'"

    # d_hidden should be 2/3 of d_ff, so that swiglu and gelu have comparable parameter counts
    d_hidden = int(2 * d_ff / 3)
    assert model.ffn.gate_proj.in_features  == d_model,  f"gate_proj input dimension: expected {d_model} but got {model.ffn.gate_proj.in_features}"
    assert model.ffn.gate_proj.out_features == d_hidden, f"gate_proj output dimension: expected {d_hidden} but got {model.ffn.gate_proj.out_features}"
    assert model.ffn.up_proj.out_features   == d_hidden, f"up_proj output dimension: expected {d_hidden} but got {model.ffn.up_proj.out_features}"
    assert model.ffn.down_proj.in_features  == d_hidden, f"down_proj input dimension: expected {d_hidden} but got {model.ffn.down_proj.in_features}"
    assert model.ffn.down_proj.out_features == d_model,  f"down_proj output dimension: expected {d_model} but got {model.ffn.down_proj.out_features}"

    # The three projections should have no bias
    assert getattr(model.ffn.gate_proj, 'bias', None) is None, "gate_proj should not have a bias"
    assert getattr(model.ffn.up_proj, 'bias', None) is None,   "up_proj should not have a bias"
    assert getattr(model.ffn.down_proj, 'bias', None) is None, "down_proj should not have a bias"

    # Forward pass should preserve the input shape
    x = jax.random.normal(jax.random.PRNGKey(10), (4, 8, d_model))
    assert model.ffn(x).shape == x.shape, "SwiGLU should preserve the input shape"

    print("Test Passed: SwiGLU is structured correctly")


def test_forward_shapes(feedforward):
    '''
    Test the forward shapes of the feedforward sublayer.
    '''
    print("Testing forward shapes ...")
    d_model = 10
    d_ff    = 40
    dropout = 0.1
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(1))

    batch_sizes = [1, 4, 8]
    seq_lengths = [10, 20, 15]
    
    for batch_size, seq_length in zip(batch_sizes, seq_lengths):
        input_tensor = jax.random.normal(jax.random.PRNGKey(batch_size + seq_length), (batch_size, seq_length, d_model))
        output = model.forward(input_tensor)
        
        assert output.shape == input_tensor.shape, \
            f"Output shape mismatch: expected {input_tensor.shape} but got {output.shape}"
    
    print("Test Passed: Forward pass returns correct shapes for various input dimensions")


def test_ffn_behavior(feedforward):
    '''
    Test the behavior of the feedforward sublayer.
    '''
    print("Testing feed-forward network behavior ...")
    d_model = 4
    d_ff    = 8
    dropout = 0.0
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(2))
    
    input_tensor = jnp.ones((2, 3, d_model))
    output = model.forward(input_tensor)
    
    assert not jnp.allclose(output, input_tensor), \
        "FFN output is identical to input, suggesting no transformation"
    assert jnp.all(jnp.isfinite(output)), "Output contains NaN or infinite values"
    
    print("Test Passed: Feed-forward network transforms input appropriately")


def test_residual_connection(feedforward):
    '''
    Test the residual connection of the feedforward sublayer.
    '''
    print("Testing residual connection ...")
    d_model = 4
    d_ff    = 8
    dropout = 0.0
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout, ffn_type='gelu', rngs=nnx.Rngs(3))
    
    # Force FFN to be zero
    model.ffn.linear1.kernel.value = jnp.zeros((d_model, d_ff))
    model.ffn.linear1.bias.value   = jnp.zeros((d_ff,))
    model.ffn.linear2.kernel.value = jnp.zeros((d_ff, d_model))
    model.ffn.linear2.bias.value   = jnp.zeros((d_model,))
    
    batch_size = 2
    seq_length = 3
    input_tensor = jax.random.normal(jax.random.PRNGKey(4), (batch_size, seq_length, d_model))
    
    # Test 1: With zero FFN, output should equal input (residual only)
    output = model(input_tensor)
    assert jnp.allclose(output, input_tensor, rtol=1e-5, atol=1e-5), \
        "Residual connection is not applied correctly with identity FFN"
    
    # Test 2: With non-zero FFN, verify both paths contribute
    key = jax.random.PRNGKey(5)
    k1, k2 = jax.random.split(key)
    model.ffn.linear1.kernel.value = jax.random.normal(k1, (d_model, d_ff)) * 0.1
    model.ffn.linear2.kernel.value = jax.random.normal(k2, (d_ff, d_model)) * 0.1
    
    output_with_ffn = model(input_tensor)
    ffn_only = model.ffn(model.norm(input_tensor))
    assert not jnp.allclose(output_with_ffn, input_tensor, rtol=1e-3, atol=1e-3), \
        "Output is identical to input, suggesting FFN path is not active"
    assert not jnp.allclose(output_with_ffn, ffn_only, rtol=1e-3, atol=1e-3), \
        "Output is identical to FFN output, suggesting residual path is not active"
    
    # Test 3: Output equals input + dropout(ffn(norm(input)))
    assert jnp.allclose(output_with_ffn, input_tensor + model.dropout(ffn_only), rtol=1e-5, atol=1e-5), \
        "Output is not the sum of input and FFN output as expected"
    
    print("Test Passed: Residual connection is working correctly")  


def test_layer_norm(feedforward):
    '''
    Test the layer normalization of the feedforward sublayer.
    '''
    print("Testing layer normalization ...")
    d_model = 4
    d_ff    = 8
    dropout = 0.0
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(6))
    
    input_tensor = jax.random.normal(jax.random.PRNGKey(7), (2, 3, d_model))
    norm_out = model.norm(input_tensor)
    
    # Verify norm modifies input
    assert not jnp.allclose(norm_out, input_tensor), "Normalization did not change input"
    print("Test Passed: Layer normalization is being applied correctly")


def test_forward_order(feedforward):
    '''
    Test if the forward pass follows the correct order of operations:
    1. Apply normalization to input
    2. Apply FFN with dropout to normalized input
    3. Add residual connection: x + ffn_out
    '''
    print("Testing forward pass order ...")
    d_model = 4
    d_ff    = 8
    dropout = 0.0
    model   = feedforward(d_model=d_model, d_ff=d_ff, dropout=dropout, rngs=nnx.Rngs(8))
    
    input_tensor = jax.random.normal(jax.random.PRNGKey(9), (2, 3, d_model))
    
    expected_output = input_tensor + model.dropout(model.ffn(model.norm(input_tensor)))
    actual_output = model(input_tensor)
    
    assert jnp.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5), \
        "Forward pass does not follow Pre-LN Transformer order"
    
    print("Test Passed: Forward pass operations are in correct order")


def main():
    from hw3lib.model import FeedForwardLayer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'FeedForwardLayer': [
                {
                    'func': lambda: test_sublayer_feedforward(FeedForwardLayer),
                    'description': 'Test the feedforward sublayer in JAX'
                }
            ]
        }
    )   

    framework.run_tests()  
    framework.summarize_results()


if __name__ == '__main__':
    main()
