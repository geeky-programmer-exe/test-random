import sys, os
from typing import Callable
import jax
import jax.numpy as jnp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


try:
    from tests.testing_framework import TestingFramework
except ImportError:
    from testing_framework import TestingFramework

def test_mask_causal(mask_gen_fn: Callable[[jax.Array], jax.Array]):
    """
    Test the causal mask generation function in JAX.
    
    Args:
        mask_gen_fn (Callable): The function to generate the causal mask.
    """
    print("Testing JAX Causal Mask ...")

    batch_sizes = [1, 2, 4]
    seq_lengths = [2, 4, 8]
    feat_lens   = [0, 5, 10]

    expected_mask1 = jnp.array([
        [False, True],
        [False, False]
    ])

    expected_mask2 = jnp.array([
        [False, True,  True,  True],
        [False, False, True,  True],
        [False, False, False, True],
        [False, False, False, False]
    ])

    expected_mask3 = jnp.array([
        [False, True,  True,  True,  True,  True,  True,  True],
        [False, False, True,  True,  True,  True,  True,  True],
        [False, False, False, True,  True,  True,  True,  True],
        [False, False, False, False, True,  True,  True,  True],
        [False, False, False, False, False, True,  True,  True],
        [False, False, False, False, False, False, True,  True],
        [False, False, False, False, False, False, False, True],
        [False, False, False, False, False, False, False, False]
    ])

    masks = [expected_mask1, expected_mask2, expected_mask3]

    for (batch_size, seq_length, feat_len, expected_mask) in zip(batch_sizes, seq_lengths, feat_lens, masks):
        if feat_len == 0:
            input_tensor = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length))
        else:
            input_tensor = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, feat_len))
        mask = mask_gen_fn(input_tensor)
        assert jnp.array_equal(mask, expected_mask), (
            f"Causal mask generation failed for batch size {batch_size}, "
            f"sequence length {seq_length}, and feature length {feat_len}."
        )

    print("Test Passed: JAX causal mask generation is correct.")

def main():
    from hw3lib.model.masks import CausalMask

    framework = TestingFramework(
        test_categories={
            'CausalMask': [
                {
                    'func': lambda: test_mask_causal(CausalMask),
                    'description': 'Test the causal mask generation in JAX'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()
