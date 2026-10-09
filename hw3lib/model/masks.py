from typing import Optional
import jax
import jax.numpy as jnp

def PadMask(padded_input: jax.Array, input_lengths: jax.Array) -> jax.Array:
    """
    Create a mask for padding positions in JAX.
    
    Specification:
    - Mask is a boolean array of shape (N, T) where:
      * N = batch size from padded_input
      * T = sequence length from padded_input
    - True values indicate padding positions (time_idx >= length) that should be masked.
    - False values indicate valid positions that should NOT be masked.
    - Supports inputs of shape (N, T) or (N, T, ...).
    
    Args:
        padded_input (jax.Array): The input array, shape (N, T, ...).
        input_lengths (jax.Array): Actual lengths before padding, shape (N,).
        
    Returns:
        jax.Array: Boolean mask array of shape (N, T).
    """
    T = padded_input.shape[1]
    lengths = jnp.asarray(input_lengths).reshape(-1, 1)
    time_idx = jnp.arange(T)[None, :]
    mask = time_idx >= lengths
    return mask

def CausalMask(padded_input: jax.Array) -> jax.Array:
    """
    Create a causal mask for self-attention in JAX.
    
    Specification:
    - Mask is a boolean array of shape (T, T) where T is sequence length from padded_input.
    - True values indicate positions that should NOT attend to each other (future positions).
    - False values indicate positions that can attend to each other.
    - Mask is upper triangular (excluding the main diagonal).
    - Supports inputs of shape (N, T) or (N, T, ...).
    
    Args:
        padded_input (jax.Array): Input array, shape (N, T, ...).
        
    Returns:
        jax.Array: Boolean mask array of shape (T, T).
    """
    T = padded_input.shape[1]
    mask = jnp.triu(jnp.ones((T, T), dtype=bool), k=1)
    return mask
