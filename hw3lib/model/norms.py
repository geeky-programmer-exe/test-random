from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx

'''
TODO: Implement this Module.

RMSNorm: Root Mean Square Layer Normalization.

Specification:
- Normalizes over the last (feature) dimension only, so each vector is
  normalized independently of every other vector
- y = x / sqrt(mean(x^2) + eps) * weight
- Has a learnable per-feature scale, initialized to ones as self.weight
- Has no bias, and does not subtract the mean
- Computes the statistic in float32 and casts the result back to the input dtype
- Reused as QK-Norm inside the attention layer, where it is applied to the
  queries and keys over head_dim
'''

class RMSNorm(nnx.Module):
    '''
    Root Mean Square Layer Normalization.
    Normalizes over the last dimension and applies a learnable per-feature scale.
    '''
    def __init__(self, d_model: int, eps: float = 1e-6, *, rngs: Optional[nnx.Rngs] = None):
        '''
        Initialize the RMSNorm.
        Args:
            d_model (int): The dimension being normalized over.
            eps   (float): Added inside the square root for numerical stability.
            rngs (nnx.Rngs, optional): RNGs container for NNX module initialization.
        '''
        self.d_model = d_model
        self.eps = eps
        # TODO: Initialize the learnable scale
        raise NotImplementedError  # Remove once implemented
        self.weight = NotImplementedError
        self.bias = None  # RMSNorm has no bias

    def __call__(self, x: jax.Array) -> jax.Array:
        '''
        Forward pass for the RMSNorm.
        Args:
            x (jax.Array): Input array, shape (..., d_model)
        Returns:
            jax.Array: Normalized array of the same shape and dtype
        '''
        # TODO: Implement forward
        raise NotImplementedError  # Remove once implemented
        # Step 1: Compute the root mean square over the last dimension in float32, keeping the dimension
        x_f32 = NotImplementedError
        rms = NotImplementedError
        # Step 2: Divide the input by it and multiply by self.weight
        normalized = NotImplementedError
        # Step 3: Cast the result back to the input dtype
        out = NotImplementedError
        return out

    def forward(self, x: jax.Array) -> jax.Array:
        """PyTorch-compatible forward alias."""
        return self(x)


def create_norm(
    norm_type: str,
    d_model: int,
    eps: float = 1e-6,
    *,
    rngs: Optional[nnx.Rngs] = None
) -> nnx.Module:
    """
    Factory function for layer normalization. (DO NOT MODIFY)
    Args:
        norm_type (str): Either 'rmsnorm' or 'layernorm'.
        d_model (int): Feature dimension.
        eps (float): Epsilon for numerical stability.
        rngs (nnx.Rngs, optional): NNX RNGs container.
    Returns:
        nnx.Module: Normalization layer instance.
    """
    norm_type = norm_type.lower()
    if norm_type == 'rmsnorm':
        return RMSNorm(d_model=d_model, eps=eps, rngs=rngs)
    elif norm_type == 'layernorm':
        return nnx.LayerNorm(num_features=d_model, epsilon=eps, rngs=rngs)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Must be 'rmsnorm' or 'layernorm'.")
