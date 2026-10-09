import math
from typing import Optional, Any, Tuple
import jax
import jax.numpy as jnp
from flax import nnx

class PositionalEncoding(nnx.Module):
    """
    Fixed Sinusoidal Positional Encoding in Flax NNX.
    
    Specification:
    - P_{t, 2i}   = sin(t / 10000^(2i/d))
    - P_{t, 2i+1} = cos(t / 10000^(2i/d))
    - Table shape: (1, max_len, d_model), broadcasting across batch dimension.
    - Adds positional information directly to embeddings.
    """
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 5000, 
        div_factor: float = 10000.0,
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        """
        Initialize the PositionalEncoding.
        
        Args:
            d_model (int): Embedding dimensionality. Must be even.
            max_len (int): Maximum sequence length.
            div_factor (float): Geometric progression base (default 10000.0).
        """
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be divisible by 2, got {d_model}")
            
        self.d_model = d_model
        self.max_len = max_len
        self.div_factor = div_factor
        
        # Build base table
        pe = self.create_pe_table(d_model, max_len, div_factor)
        # TODO: Store the table as self.pe = nnx.Variable(...), shape (1, max_len, d_model)
        self.pe = nnx.Variable(pe)

    @staticmethod
    def create_pe_table(d_model: int, max_len: int, div_factor: float = 10000.0) -> jax.Array:
        # TODO: Implement create_pe_table
        # Step 1: Compute positions 0..max_len-1 and div_term = exp(-log(10000) * 2i / d_model)
        pos = jnp.arange(max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(div_factor) / d_model))

        # Step 2: sin on even dimensions, cos on odd dimensions
        pe_table = jnp.zeros((max_len, d_model), dtype=jnp.float32)
        pe_table = pe_table.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe_table = pe_table.at[:, 1::2].set(jnp.cos(pos * div_term))
        return pe_table[None, :, :]  # (1, max_len, d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (jax.Array): Input embeddings of shape (N, T, d_model).
            
        Returns:
            jax.Array: Embeddings with positional encoding added, shape (N, T, d_model).
        """
        # TODO: Implement __call__
        # Step 1: Get sequence length from the input
        seq_len = x.shape[1]
        # Step 3: Add positional encodings to the input: x + self.pe.value[:, :seq_len, :]
        return x + self.pe.value[:, :seq_len, :].astype(x.dtype)

    def forward(self, x: jax.Array) -> jax.Array:
        return self(x)


class RotaryPositionalEncoding(nnx.Module):
    """
    Rotary Positional Embedding (RoPE) in Flax NNX.
    
    Specification:
    - Rotates queries and keys per-head after projections.
    - Uses split-half pairing convention: dimension i pairs with i + head_dim // 2.
    - cos and sin tables are registered with shape (1, 1, max_len, head_dim)
      so they broadcast across batch and num_heads dimensions.
    - Position 0 has angle 0 everywhere (cos is all 1s, sin is all 0s).
    - Preserves vector norm: |RoPE(x)| = |x|.
    - Relative position property: <RoPE(q, m), RoPE(k, n)> depends only on (m - n).
    """
    def __init__(
        self, 
        head_dim: int, 
        max_len: int = 5000, 
        theta: float = 10000.0,
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        """
        Initialize the RotaryPositionalEncoding.
        
        Args:
            head_dim (int): Dimensionality of each attention head. Must be even.
            max_len (int): Maximum sequence length supported.
            theta (float): Base angle parameter (default 10000.0).
        """
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be divisible by 2, got {head_dim}")
            
        self.head_dim = head_dim
        self.max_len = max_len
        self.theta = theta
        
        cos_table, sin_table = self.create_rope_table(head_dim, max_len, theta)
        # TODO: Store cos and sin as nnx.Variable, each of shape (1, 1, max_len, head_dim)
        self.cos = nnx.Variable(cos_table)
        self.sin = nnx.Variable(sin_table)

    @staticmethod
    def create_rope_table(head_dim: int, max_len: int, theta: float = 10000.0) -> tuple[jax.Array, jax.Array]:
        # TODO: Implement create_rope_table
        # Step 1: Compute the inverse frequencies for the head_dim // 2 pairs
        half = head_dim // 2
        freqs = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
        # Step 2: Take the outer product with the positions to get the angles
        pos = jnp.arange(max_len, dtype=jnp.float32)[:, None]
        angles = pos * freqs[None, :]  # (max_len, half)
        
        cos_half = jnp.cos(angles)
        sin_half = jnp.sin(angles)
        
        # Step 3: Widen the angles to head_dim so each pair shares an angle, then take cos and sin
        cos = jnp.concatenate([cos_half, cos_half], axis=-1)  # (max_len, head_dim)
        sin = jnp.concatenate([sin_half, sin_half], axis=-1)  # (max_len, head_dim)
        
        # Shape: (1, 1, max_len, head_dim) for broadcasting over (N, H, T, D)
        return cos[None, None, :, :], sin[None, None, :, :]

    def rotate_half(self, x: jax.Array) -> jax.Array:
        """
        Quarter-turn rotation helper for split-half pairing:
        rotate_half([x1, x2]) = [-x2, x1]
        """
        # TODO: Implement rotate_half
        # Split the last dimension in half and return concat(-x[..., half:], x[..., :half])
        half = self.head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return jnp.concatenate([-x2, x1], axis=-1)

    def named_buffers(self):
        """Duck-typing helper for PyTorch-style buffer introspection."""
        return [('cos', self.cos.value), ('sin', self.sin.value)]

    def __call__(
        self, 
        q: jax.Array, 
        k: Optional[jax.Array] = None, 
        offset: int = 0
    ) -> Any:
        """
        Apply RoPE rotation to queries, and optionally keys.
        
        Args:
            x (jax.Array): Input array of shape (N, H, L, head_dim) or (N, L, head_dim).
            q (jax.Array): Queries of shape (..., L, head_dim).
            k (jax.Array, optional): Keys of shape (..., S, head_dim).
            offset (int): Starting position offset for autoregressive generation.
            
        Returns:
            jax.Array or Tuple[jax.Array, jax.Array]: Rotated queries, or (rotated_q, rotated_k) if k is provided.
        """
        def _apply_rope(x: jax.Array, pos_offset: int) -> jax.Array:
            # Step 2: Verify offset + seq_len does not exceed max_len, and raise ValueError if it does
            seq_len = x.shape[-2]
            if pos_offset + seq_len > self.max_len:
                raise ValueError(
                    f"Requested sequence range [{pos_offset}, {pos_offset + seq_len}) exceeds max_len {self.max_len}"
                )
                
            # Step 3: Slice the cos and sin tables to the positions in use
            cos = self.cos.value[:, :, pos_offset:pos_offset + seq_len, :].astype(x.dtype)
            sin = self.sin.value[:, :, pos_offset:pos_offset + seq_len, :].astype(x.dtype)
            
            # If input has 3 dimensions (N, L, D), broadcast cos and sin appropriately
            if x.ndim == 3:
                cos = cos.squeeze(1)  # (1, L, D)
                sin = sin.squeeze(1)  # (1, L, D)
                
            # Step 4: Rotate each of q and k: x * cos + self.rotate_half(x) * sin
            return (x * cos) + (self.rotate_half(x) * sin)

        # TODO: Implement __call__
        # Step 1: Accept q alone, or q and k. A lone q returns one array.
        q_rot = _apply_rope(q, offset)
        if k is not None:
            # If key sequence length matches query or is cross-attention, apply offset
            k_rot = _apply_rope(k, 0 if q.shape[-2] != k.shape[-2] else offset)
            return q_rot, k_rot
        return q_rot

    def forward(
        self, 
        q: jax.Array, 
        k: Optional[jax.Array] = None, 
        offset: int = 0
    ) -> Any:
        return self(q, k=k, offset=offset)

