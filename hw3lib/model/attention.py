import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from .norms import RMSNorm
from .positional_encoding import RotaryPositionalEncoding

'''
TODO: Implement this Module.

This file contains the multi-head attention used by the transformer sublayers:

MultiHeadAttention: Scaled dot-product attention over multiple heads
- Optionally applies rotary embeddings to the queries and keys
- Optionally applies QK-Norm to the queries and keys
- Uses nnx.Linear for q, k, v, and out projections
- Returns (output, attn_weights) where attn_weights is averaged across heads

Specification:
- key_padding_mask has shape (N, S), attn_mask has shape (L, S), and in both
  True marks a position to ignore
- Rotary embeddings run before QK-Norm. Both are applied after _split_heads,
  and to the queries and keys only
- Attention weights are returned averaged over the heads, shape (N, L, S)
'''

class MultiHeadAttention(nnx.Module):
    '''
    Multi-Head Attention.
    '''
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        rope: Optional[RotaryPositionalEncoding] = None,
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        '''
        Initialize the MultiHeadAttention.
        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate applied to the attention weights.
            qk_norm (bool): Whether to normalize the queries and keys before scoring.
            rope (Optional[RotaryPositionalEncoding]): Rotary embeddings, or None to disable.
            rngs (nnx.Rngs, optional): NNX random number generators container.
        '''
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # DO NOT MODIFY THESE ATTRIBUTES
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qk_norm_enabled = qk_norm
        self.rope = rope

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, rngs=rngs)
            self.k_norm = RMSNorm(self.head_dim, rngs=rngs)
        else:
            self.q_norm = None
            self.k_norm = None

        # TODO: Implement __init__
        raise NotImplementedError  # Remove once implemented
        # TODO: Initialize the dropout applied to the attention weights
        self.dropout = NotImplementedError
        # TODO: Initialize the projections, each embed_dim -> embed_dim
        self.q_proj = NotImplementedError
        self.k_proj = NotImplementedError
        self.v_proj = NotImplementedError
        self.out_proj = NotImplementedError

    def _split_heads(self, x: jax.Array) -> jax.Array:
        '''
        Split the last dimension into (num_heads, head_dim).
        Args:
            x (jax.Array): Input array. shape: (N, L, embed_dim)
        Returns:
            jax.Array: Reshaped array. shape: (N, num_heads, L, head_dim)
        '''
        # TODO: Implement _split_heads
        raise NotImplementedError  # Remove once implemented
        # (N, L, embed_dim) -> (N, num_heads, L, head_dim)
        N, T, _ = x.shape
        x = NotImplementedError
        out = NotImplementedError
        return out

    def _concat_heads(self, x: jax.Array) -> jax.Array:
        '''
        Concatenate the heads back into the embedding dimension.
        Args:
            x (jax.Array): Input array. shape: (N, num_heads, L, head_dim)
        Returns:
            jax.Array: Reshaped array. shape: (N, L, embed_dim)
        '''
        # TODO: Implement _concat_heads
        raise NotImplementedError  # Remove once implemented
        # (N, num_heads, L, head_dim) -> (N, L, embed_dim)
        N, H, T, D = x.shape
        x = NotImplementedError
        out = NotImplementedError
        return out

    def _merge_masks(
        self,
        key_padding_mask: Optional[jax.Array],
        attn_mask: Optional[jax.Array],
        N: int,
        L: int,
        S: int
    ) -> Optional[jax.Array]:
        '''
        Merge the two mask types into a single mask.
        Args:
            key_padding_mask (Optional[jax.Array]): Positions to ignore. shape: (N, S)
            attn_mask (Optional[jax.Array]): Positions to ignore. shape: (L, S)
            N (int): The batch size.
            L (int): The query sequence length.
            S (int): The key sequence length.
        Returns:
            Optional[jax.Array]: The combined mask, broadcastable to (N, num_heads, L, S),
                                 or None if both inputs were None.
        '''
        # TODO: Implement _merge_masks
        raise NotImplementedError  # Remove once implemented
        mask = NotImplementedError
        return mask

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        key_padding_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None,
        offset: int = 0
    ) -> Tuple[jax.Array, jax.Array]:
        '''
        Forward pass for the MultiHeadAttention.
        Args:
            query (jax.Array): The queries. shape: (N, L, embed_dim)
            key (jax.Array): The keys. shape: (N, S, embed_dim)
            value (jax.Array): The values. shape: (N, S, embed_dim)
            key_padding_mask (Optional[jax.Array]): Positions to ignore. shape: (N, S)
            attn_mask (Optional[jax.Array]): Positions to ignore. shape: (L, S)
            offset (int): Position offset for RoPE.
        Returns:
            output (jax.Array): The output array. shape: (N, L, embed_dim)
            attn_weights (jax.Array): The attention weights, averaged over heads. shape: (N, L, S)
        '''
        # TODO: Implement __call__
        raise NotImplementedError  # Remove once implemented
        N, L, E = query.shape
        _, S, _ = key.shape

        # TODO: Project the inputs
        q = NotImplementedError
        k = NotImplementedError
        v = NotImplementedError

        # TODO: Reshape for multiple heads
        q = NotImplementedError
        k = NotImplementedError
        v = NotImplementedError

        # TODO: Apply rotary embeddings to the queries and keys, if enabled
        # RoPE runs before QK-Norm, and only on queries and keys.
        if self.rope is not None:
            q = NotImplementedError
            k = NotImplementedError

        # TODO: Apply QK-Norm to the queries and keys, if enabled
        if self.qk_norm_enabled:
            q = NotImplementedError
            k = NotImplementedError

        # TODO: Combine the padding and attention masks
        # _merge_masks returns a mask broadcastable to (N, num_heads, L, S), or None
        mask = NotImplementedError

        # TODO: Score, mask, normalize and apply dropout to get the attention weights
        # Scale the scores by self.scale before masking.
        # True marks a position to ignore: set those scores to a large negative value,
        # softmax over the last axis, then apply self.dropout.
        attn_weights = NotImplementedError

        # TODO: Apply the attention weights to the values
        attn_output = NotImplementedError

        # TODO: Merge the heads
        attn_output = NotImplementedError

        # TODO: Final projection, and average the attention weights over the heads
        output = NotImplementedError
        attn_weights = NotImplementedError
        return output, attn_weights

    def forward(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        key_padding_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None,
        offset: int = 0
    ) -> Tuple[jax.Array, jax.Array]:
        return self(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            offset=offset,
        )
