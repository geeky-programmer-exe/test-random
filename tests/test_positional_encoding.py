import sys, os
from typing import Callable
import jax
import jax.numpy as jnp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


try:
    from tests.jax.testing_framework import TestingFramework
except ImportError:
    from testing_framework import TestingFramework

def test_positional_encoding(positional_encoding_fn: Callable):
    """Test the PositionalEncoding class in JAX."""
    test_pe_shape(positional_encoding_fn)
    test_pe_values(positional_encoding_fn)
    test_pe_forward(positional_encoding_fn)

def test_pe_shape(positional_encoding_fn: Callable):
    print("Testing Positional Encoding Shape ...")
    d_model = 16
    max_len = 50
    pe_layer = positional_encoding_fn(d_model, max_len)
    input_tensor = jnp.zeros((4, 10, d_model))
    output = pe_layer(input_tensor)
    assert output.shape == input_tensor.shape, f"Output shape {output.shape} does not match input shape {input_tensor.shape}"
    print("Test Passed: Positional Encoding Shape is Correct")

def test_pe_values(positional_encoding_fn: Callable):
    print("Testing Positional Encoding Values ...")
    d_model = 4
    max_len = 10
    pe_layer = positional_encoding_fn(d_model, max_len)
    
    expected_pe = jnp.array([
        [ 0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.8415,  0.5403,  0.0100,  0.9999],
        [ 0.9093, -0.4161,  0.0200,  0.9998],
        [ 0.1411, -0.9900,  0.0300,  0.9996],
        [-0.7568, -0.6536,  0.0400,  0.9992],
        [-0.9589,  0.2837,  0.0500,  0.9988],
        [-0.2794,  0.9602,  0.0600,  0.9982],
        [ 0.6570,  0.7539,  0.0699,  0.9976],
        [ 0.9894, -0.1455,  0.0799,  0.9968],
        [ 0.4121, -0.9111,  0.0899,  0.9960]
    ], dtype=jnp.float32)

    pe_buffer = jnp.squeeze(pe_layer.pe.value, axis=0)[:max_len]
    assert jnp.allclose(pe_buffer, expected_pe, rtol=1e-4, atol=1e-4), \
        "Positional Encoding Values do not match expected values"
    print("Test Passed: Positional Encoding Values are Correct")

def test_pe_forward(positional_encoding_fn: Callable):
    print("Testing Positional Encoding Forward ...")
    d_model = 8
    max_len = 20
    pe_layer = positional_encoding_fn(d_model, max_len)
    input_tensor = jnp.ones((2, 15, d_model))
    output = pe_layer(input_tensor)
    expected_output = input_tensor + pe_layer.pe.value[:, :input_tensor.shape[1], :]
    assert jnp.allclose(output, expected_output, rtol=1e-5, atol=1e-5), \
        "Positional Encoding Forward does not match expected values"
    print("Test Passed: Positional Encoding Forward is Correct")

def main():
    from hw3lib.model.positional_encoding import PositionalEncoding

    framework = TestingFramework(
        test_categories={
            'PositionalEncoding': [
                {
                    'func': lambda: test_positional_encoding(PositionalEncoding),
                    'description': 'Test the positional encoding in JAX'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()
