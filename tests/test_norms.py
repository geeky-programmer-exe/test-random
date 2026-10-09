import sys, os
import jax
import jax.numpy as jnp
from flax import nnx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


try:
    from tests.testing_framework import TestingFramework
except ImportError:
    from testing_framework import TestingFramework


def test_norms(rms_norm):
    """
    Test the RMSNorm module in JAX / Flax NNX.
    """
    test_initialization(rms_norm)
    test_shapes(rms_norm)
    test_values(rms_norm)
    test_no_mean_subtraction(rms_norm)
    test_learnable_scale(rms_norm)
    test_mixed_precision(rms_norm)

def test_initialization(rms_norm):
    print("Testing initialization ...")
    d_model = 16
    model = rms_norm(d_model)

    assert hasattr(model, 'weight'), "RMSNorm is missing its weight attribute"
    assert isinstance(model.weight, nnx.Param), "weight should be an nnx.Param so that it is learnable"
    assert model.weight.shape == (d_model,), f"weight shape: expected {(d_model,)} but got {model.weight.shape}"
    assert jnp.allclose(model.weight.value, jnp.ones(d_model)), "weight should be initialized to ones"
    assert not hasattr(model, 'bias') or model.bias is None, "RMSNorm should not have a bias"

    print("Test Passed: RMSNorm is initialized correctly")

def test_shapes(rms_norm):
    print("Testing shapes ...")
    d_model = 16
    model = rms_norm(d_model)

    for shape in [(4, d_model), (4, 8, d_model), (2, 3, 8, d_model)]:
        x = jax.random.normal(jax.random.PRNGKey(0), shape)
        assert model(x).shape == shape, f"Output shape: expected {shape} but got {model(x).shape}"

    print("Test Passed: Output shape matches input shape")

def test_values(rms_norm):
    print("Testing values ...")
    d_model = 8
    model = rms_norm(d_model)
    x = jax.random.normal(jax.random.PRNGKey(42), (4, 6, d_model)) * 3.0

    expected = (x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + model.eps)) * model.weight.value
    assert jnp.allclose(model(x), expected, rtol=1e-4, atol=1e-5), \
        "RMSNorm values do not match reference formula"

    # Normalized root mean square should be 1
    rms = jnp.sqrt(jnp.mean(model(x)**2, axis=-1))
    assert jnp.allclose(rms, jnp.ones_like(rms), rtol=1e-3, atol=1e-3), \
        "Normalized vectors should have root mean square of 1"

    print("Test Passed: RMSNorm values are correct")

def test_no_mean_subtraction(rms_norm):
    print("Testing that the mean is not subtracted ...")
    d_model = 8
    model = rms_norm(d_model)

    x = jnp.ones((1, 1, d_model)) * 5.0
    out = model(x)

    assert not jnp.allclose(jnp.mean(out, axis=-1), jnp.zeros((1, 1)), atol=1e-3), \
        "Output is zero-mean, meaning mean was subtracted. RMSNorm should not center the input"
    assert jnp.allclose(out, jnp.ones_like(out), rtol=1e-3, atol=1e-3), \
        "A constant input should normalize to a constant output of ones"

    print("Test Passed: The mean is not subtracted")

def test_learnable_scale(rms_norm):
    print("Testing that the scale is learnable ...")
    d_model = 8
    model = rms_norm(d_model)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, d_model))

    def loss_fn(m):
        out = m(x)
        return jnp.sum(out ** 2)

    grad = nnx.grad(loss_fn)(model)
    scale_grad = grad.weight.value
    assert not jnp.allclose(scale_grad, jnp.zeros_like(scale_grad)), "Gradients with respect to weight should be non-zero"

    print("Test Passed: The scale is learnable")

def test_mixed_precision(rms_norm):
    print("Testing float16 numerical stability ...")
    d_model = 8
    model = rms_norm(d_model)

    # In float16, values above 256 square to > 65536 which overflows float16.
    # Computing the statistic in float32 prevents this overflow.
    x_fp16 = (jnp.ones((2, d_model), dtype=jnp.float16) * 300.0)
    out = model(x_fp16)
    
    assert not jnp.any(jnp.isnan(out)), "Output contains NaNs under float16 input"
    assert not jnp.any(jnp.isinf(out)), "Output contains Infs under float16 input"
    assert jnp.allclose(out, jnp.ones_like(out), rtol=1e-2, atol=1e-2), \
        "Float16 constant input should normalize to ones without underflow/overflow"

    print("Test Passed: float16 numerical stability verified")

def main():
    from hw3lib.model.norms import RMSNorm

    framework = TestingFramework(
        test_categories={
            'RMSNorm': [
                {
                    'func': lambda: test_norms(RMSNorm),
                    'description': 'Test RMSNorm implementation in JAX'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()
