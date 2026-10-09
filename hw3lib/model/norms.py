from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx

class RMSNorm(nnx.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) in Flax NNX.
    
    Specification:
    - Normalizes over the last (feature) dimension only, so each vector is
      normalized independently.
    - y = x / sqrt(mean(x^2) + eps) * weight
    - Has a learnable per-feature scale (weight), initialized to ones.
    - Has no bias, and does not subtract the mean.
    - Computes the statistic in float32 and casts back to input dtype.
    - Reused as QK-Norm inside attention layers.
    """
    def __init__(self, d_model: int, eps: float = 1e-6, *, rngs: Optional[nnx.Rngs] = None):
        """
        Initialize RMSNorm.
        
        Args:
            d_model (int): The dimension being normalized over.
            eps (float): Added inside the square root for numerical stability.
            rngs (nnx.Rngs, optional): RNGs container for NNX module initialization.
        """
        self.d_model = d_model
        self.eps = eps
        # TODO: Initialize the learnable scale
        # self.weight = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
        self.weight = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
        self.bias = None  # RMSNorm has no bias

    @property
    def scale(self):
        """Alias for weight to match Flax conventions."""
        return self.weight

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass for RMSNorm.
        
        Args:
            x (jax.Array): Input array, shape (..., d_model).
            
        Returns:
            jax.Array: Normalized array of same shape and dtype.
        """
        # TODO: Implement forward
        # Step 1: Compute the root mean square over the last dimension in float32, keeping the dimension
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        
        # Step 2: Divide the input by it and multiply by self.weight
        normalized = (x_f32 / rms) * self.weight.value
        
        # Step 3: Cast the result back to the input dtype
        return normalized.astype(x.dtype)

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
    Factory function for layer normalization.
    
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
