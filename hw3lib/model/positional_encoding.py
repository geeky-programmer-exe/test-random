import math
from typing import Optional, Any
import jax
import jax.numpy as jnp
from flax import nnx

'''
TODO: Implement these Modules.

This file contains two ways of giving a transformer positional information:

1. PositionalEncoding: Fixed sinusoidal encoding added to the token embeddings
   - Added once, before the first layer
   - Encodes absolute position

2. RotaryPositionalEncoding: Rotates queries and keys inside each attention operation
   - Applied per head, after the q/k projections and the reshape into heads
   - Encodes relative position

Specification for RotaryPositionalEncoding:
- Applied to queries and keys only, never to values
- Pairs dimension i with dimension i + head_dim//2 (the "split-half" convention
  used by Llama and HuggingFace), not with dimension i+1
- The angle for position t and pair i is t / (theta ** (2i / head_dim))
- cos and sin are stored as nnx.Variable with shape (1, 1, max_len, head_dim),
  so that they broadcast over batch and head
'''

class PositionalEncoding(nnx.Module):
    """
    Fixed Sinusoidal Positional Encoding.

    Specification:
    - P_{t, 2i}   = sin(t / 10000^(2i/d))
    - P_{t, 2i+1} = cos(t / 10000^(2i/d))
    - Table shape: (1, max_len, d_model), broadcasting across the batch dimension.
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

        # create_pe_table returns shape (1, max_len, d_model)
        pe = self.create_pe_table(d_model, max_len, div_factor)
        self.pe = nnx.Variable(pe)

    @staticmethod
    def create_pe_table(d_model: int, max_len: int, div_factor: float = 10000.0) -> jax.Array:
        """
        Create the sinusoidal positional encoding table.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
            div_factor (float): Geometric progression base.
        Returns:
            jax.Array: Table of shape (1, max_len, d_model).
        """
        # TODO: Implement create_pe_table
        raise NotImplementedError  # Remove once implemented
        # Step 1: Compute positions 0..max_len-1 and div_term = exp(-log(div_factor) * 2i / d_model)
        pos = NotImplementedError
        div_term = NotImplementedError
        # Step 2: sin on even dimensions, cos on odd dimensions
        pe_table = NotImplementedError
        return pe_table

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Add positional encoding to input embeddings.
        Args:
            x (jax.Array): Input embeddings of shape (N, T, d_model).
        Returns:
            jax.Array: Embeddings with positional encoding added, shape (N, T, d_model).
        """
        # TODO: Implement __call__
        raise NotImplementedError  # Remove once implemented
        # Step 1: Get sequence length from the input
        seq_len = NotImplementedError
        # Step 2: Add positional encodings to the input: x + self.pe.value[:, :seq_len, :]
        out = NotImplementedError
        return out

    def forward(self, x: jax.Array) -> jax.Array:
        return self(x)


class RotaryPositionalEncoding(nnx.Module):
    """
    Rotary Positional Embedding (RoPE).

    Specification:
    - Rotates queries and keys per-head after projections.
    - Uses split-half pairing: dimension i pairs with i + head_dim // 2.
    - cos and sin tables have shape (1, 1, max_len, head_dim)
      so they broadcast across batch and num_heads.
    - Position 0 has angle 0 everywhere (cos is all 1s, sin is all 0s).
    - Preserves vector norm: |RoPE(x)| = |x|.
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

        # create_rope_table returns cos and sin, each of shape (1, 1, max_len, head_dim)
        cos_table, sin_table = self.create_rope_table(head_dim, max_len, theta)
        self.cos = nnx.Variable(cos_table)
        self.sin = nnx.Variable(sin_table)

    @staticmethod
    def create_rope_table(head_dim: int, max_len: int, theta: float = 10000.0) -> tuple[jax.Array, jax.Array]:
        """
        Create the rotation tables.
        Args:
            head_dim (int): The dimension of each attention head.
            max_len  (int): The maximum length of the input sequence.
            theta  (float): The base of the geometric progression of wavelengths.
        Returns:
            tuple[jax.Array, jax.Array]: cos and sin, each of shape (1, 1, max_len, head_dim).
        """
        # TODO: Implement create_rope_table
        raise NotImplementedError  # Remove once implemented
        # Step 1: Compute the inverse frequencies for the head_dim // 2 pairs
        half = NotImplementedError
        freqs = NotImplementedError
        # Step 2: Take the outer product with the positions to get the angles
        pos = NotImplementedError
        angles = NotImplementedError
        cos_half = NotImplementedError
        sin_half = NotImplementedError
        # Step 3: Widen the angles to head_dim so each pair shares an angle, then take cos and sin
        cos = NotImplementedError
        sin = NotImplementedError
        return cos, sin

    def rotate_half(self, x: jax.Array) -> jax.Array:
        """
        Quarter-turn rotation helper for split-half pairing:
        rotate_half([x1, x2]) = [-x2, x1]
        Args:
            x (jax.Array): Input array, shape (..., head_dim)
        Returns:
            jax.Array: Rotated array of the same shape
        """
        # TODO: Implement rotate_half
        raise NotImplementedError  # Remove once implemented
        # Split the last dimension in half and return concat(-x[..., half:], x[..., :half])
        half = NotImplementedError
        x1 = NotImplementedError
        x2 = NotImplementedError
        rotated = NotImplementedError
        return rotated

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
            q (jax.Array): Queries of shape (..., L, head_dim).
            k (jax.Array, optional): Keys of shape (..., S, head_dim).
            offset (int): Starting position offset for autoregressive generation.
        Returns:
            jax.Array or Tuple[jax.Array, jax.Array]: Rotated queries, or (rotated_q, rotated_k) if k is provided.
        """
        # TODO: Implement __call__
        raise NotImplementedError  # Remove once implemented

        def _apply_rope(x: jax.Array, pos_offset: int) -> jax.Array:
            # Step 2: Verify offset + seq_len does not exceed max_len, and raise ValueError if it does
            seq_len = NotImplementedError
            if pos_offset + seq_len > self.max_len:
                raise ValueError(
                    f"Requested sequence range [{pos_offset}, {pos_offset + seq_len}) exceeds max_len {self.max_len}"
                )
            # Step 3: Slice the cos and sin tables to the positions in use
            cos = NotImplementedError
            sin = NotImplementedError
            # If input has 3 dimensions (N, L, D), broadcast cos and sin appropriately
            if x.ndim == 3:
                cos = NotImplementedError
                sin = NotImplementedError
            # Step 4: Rotate: x * cos + self.rotate_half(x) * sin
            rotated = NotImplementedError
            return rotated

        # Step 1: Accept q alone, or q and k. A lone q returns one array.
        # A key whose length differs from the query is rotated from position 0.
        q_rot = NotImplementedError
        if k is not None:
            k_rot = NotImplementedError
            return q_rot, k_rot
        return q_rot

    def forward(
        self,
        q: jax.Array,
        k: Optional[jax.Array] = None,
        offset: int = 0
    ) -> Any:
        return self(q, k=k, offset=offset)
