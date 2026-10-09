import jax
import jax.numpy as jnp

'''
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean array of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions (time_idx >= length) that should be masked
- False values indicate valid positions that should not be masked
- Supports inputs of shape (N, T) or (N, T, ...)
'''
def PadMask(padded_input: jax.Array, input_lengths: jax.Array) -> jax.Array:
    """
    Create a mask for padding positions.
    Args:
        padded_input (jax.Array): The input array, shape (N, T, ...).
        input_lengths (jax.Array): Actual lengths before padding, shape (N,).
    Returns:
        jax.Array: Boolean mask array of shape (N, T).
    """
    # TODO: Implement PadMask
    raise NotImplementedError  # Remove once implemented
    T = NotImplementedError
    lengths = NotImplementedError
    time_idx = NotImplementedError
    mask = NotImplementedError
    return mask


'''
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean array of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other (future positions)
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be upper triangular (excluding the main diagonal)
- Supports inputs of shape (N, T) or (N, T, ...)
'''
def CausalMask(padded_input: jax.Array) -> jax.Array:
    """
    Create a causal mask for self-attention.
    Args:
        padded_input (jax.Array): Input array, shape (N, T, ...).
    Returns:
        jax.Array: Boolean mask array of shape (T, T).
    """
    # TODO: Implement CausalMask
    raise NotImplementedError  # Remove once implemented
    T = NotImplementedError
    mask = NotImplementedError
    return mask
