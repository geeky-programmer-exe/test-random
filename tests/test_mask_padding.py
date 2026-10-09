import sys, os
from typing import Callable
import jax
import jax.numpy as jnp

# hw3lib lives in the directory that contains tests/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


try:
    from tests.jax.testing_framework import TestingFramework
except ImportError:
    from testing_framework import TestingFramework

def test_mask_padding(mask_gen_fn: Callable[[jax.Array, jax.Array], jax.Array]):
    """
    Test the padding mask generation function in JAX.
    
    Args:
        mask_gen_fn (Callable): The function to generate the padding mask.
    """
    print("Testing JAX Padding Mask ...")

    batch_sizes = [1, 2, 4]
    seq_lengths = [2, 4, 8]
    feat_lens   = [0, 5, 10]
    input_lengths = [
        jnp.array([1]),
        jnp.array([2, 3]),
        jnp.array([1, 3, 4, 7])
    ]

    expected_mask_input_length1 = jnp.array([
        [False, True]
    ])
    expected_mask_input_length2 = jnp.array([
        [False, False, True, True],
        [False, False, False, True]
    ])
    expected_mask_input_length3 = jnp.array([
        [False, True, True, True, True, True, True, True],
        [False, False, False, True, True, True, True, True],
        [False, False, False, False, True, True, True, True],
        [False, False, False, False, False, False, False, True]
    ])

    masks_input_length = [expected_mask_input_length1, expected_mask_input_length2, expected_mask_input_length3]

    for (batch_size, seq_length, feat_len, input_len, expected_mask) in zip(
        batch_sizes, seq_lengths, feat_lens, input_lengths, masks_input_length
    ):
        if feat_len == 0:
            input_tensor = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length))
        else:
            input_tensor = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, feat_len))
            
        mask = mask_gen_fn(input_tensor, input_lengths=input_len)
        assert jnp.array_equal(mask, expected_mask), (
            f"Padding mask generation failed for batch size {batch_size}, "
            f"sequence length {seq_length}, feature length {feat_len}, and input length {input_len}."
        )

    print("Test Passed: JAX padding mask generation is correct.")

def main():
    from hw3lib.model.masks import PadMask

    framework = TestingFramework(
        test_categories={
            'PaddingMask': [
                {
                    'func': lambda: test_mask_padding(PadMask),
                    'description': 'Test the padding mask generation in JAX'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()
