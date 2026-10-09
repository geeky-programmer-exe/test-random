import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from .norms import RMSNorm
from .positional_encoding import RotaryPositionalEncoding

class MultiHeadAttention(nnx.Module):
    """
    Multi-Head Attention module in Flax NNX.
    
    Specification:
    - Splits queries, keys, and values into `num_heads` heads of dimension `head_dim`.
    - Projections: `q_proj`, `k_proj`, `v_proj`, `out_proj`.
    - Supports optional RoPE (`RotaryPositionalEncoding`) applied to Q and K per-head.
    - Supports optional QK-Norm (`RMSNorm`) applied to Q and K per-head.
    - Supports `key_padding_mask` (N, S) and `attn_mask` (L, S), where True marks positions to ignore.
    - Returns `(output, attn_weights)` where `attn_weights` is shape (N, L, S), averaged over heads.
    """
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
        """
        Initialize MultiHeadAttention.
        
        Args:
            embed_dim (int): Model feature dimension.
            num_heads (int): Number of attention heads. Must divide embed_dim evenly.
            dropout (float): Dropout probability for attention weights.
            qk_norm (bool): Whether to apply RMSNorm to Q and K before dot product.
            rope (RotaryPositionalEncoding, optional): RoPE instance for query/key rotation.
            rngs (nnx.Rngs, optional): NNX random number generators container.
        """
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qk_norm_enabled = qk_norm
        self.rope = rope

        # TODO: Initialize the projections, each embed_dim -> embed_dim
        # Input and output projections
        self.q_proj = nnx.Linear(in_features=embed_dim, out_features=embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(in_features=embed_dim, out_features=embed_dim, rngs=rngs)
        self.v_proj = nnx.Linear(in_features=embed_dim, out_features=embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(in_features=embed_dim, out_features=embed_dim, rngs=rngs)

        # Optional QK normalization
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, rngs=rngs)
            self.k_norm = RMSNorm(self.head_dim, rngs=rngs)
        else:
            self.q_norm = None
            self.k_norm = None

        # TODO: Initialize the dropout applied to the attention weights
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def _split_heads(self, x: jax.Array) -> jax.Array:
        """
        Reshape (N, T, embed_dim) -> (N, num_heads, T, head_dim).
        """
        # TODO: Reshape for multiple heads
        # _split_heads: (N, L, embed_dim) -> (N, num_heads, L, head_dim)
        N, T, _ = x.shape
        x = x.reshape((N, T, self.num_heads, self.head_dim))
        return jnp.swapaxes(x, 1, 2)

    def _concat_heads(self, x: jax.Array) -> jax.Array:
        """
        Reshape (N, num_heads, T, head_dim) -> (N, T, embed_dim).
        """
        # TODO: Merge the heads
        # _concat_heads: (N, num_heads, L, head_dim) -> (N, L, embed_dim)
        N, H, T, D = x.shape
        x = jnp.swapaxes(x, 1, 2)
        return x.reshape((N, T, H * D))

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        key_padding_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None,
        offset: int = 0
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            query (jax.Array): Shape (N, L, E).
            key (jax.Array): Shape (N, S, E).
            value (jax.Array): Shape (N, S, E).
            key_padding_mask (jax.Array, optional): Shape (N, S), True marks padding to ignore.
            attn_mask (jax.Array, optional): Shape (L, S), True marks positions to ignore.
            offset (int): Position offset for RoPE.
            
        Returns:
            Tuple[jax.Array, jax.Array]: (context_output of shape (N, L, E), attention_weights of shape (N, L, S))
        """
        N, L, E = query.shape
        _, S, _ = key.shape

        # TODO: Project the inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # TODO: Reshape for multiple heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # TODO: Apply rotary embeddings to the queries and keys, if enabled
        # RoPE runs before QK-Norm, and only on queries and keys.
        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=0 if query.shape[1] != key.shape[1] else offset)

        # TODO: Apply QK-Norm to the queries and keys, if enabled
        if self.qk_norm_enabled:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # TODO: Score, mask, normalize and apply dropout to get the attention weights
        # Scale the scores by the square root of head_dim before masking.
        # True marks a position to ignore: set those scores to a large negative value,
        # softmax over the last axis, then apply self.dropout.
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * self.scale

        # TODO: Combine the padding and attention masks
        if key_padding_mask is not None:
            # key_padding_mask shape: (N, S) -> broadcast to (N, 1, 1, S)
            k_mask = key_padding_mask[:, None, None, :]
            scores = jnp.where(k_mask, -1e9, scores)

        if attn_mask is not None:
            # attn_mask shape: (L, S) or (1, L, S) -> broadcast to (1, 1, L, S)
            if attn_mask.ndim == 2:
                a_mask = attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                a_mask = attn_mask[:, None, :, :]
            else:
                a_mask = attn_mask
            scores = jnp.where(a_mask, -1e9, scores)

        # 7. Softmax and dropout
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # TODO: Apply the attention weights to the values
        context = jnp.matmul(attn_weights_dropped, v)
        # TODO: Merge the heads
        out = self._concat_heads(context)
        # TODO: Final projection, and average the attention weights over the heads
        out = self.out_proj(out)

        # 9. Return (output, attention_weights averaged over heads)
        avg_weights = jnp.mean(attn_weights, axis=1)  # (N, L, S)
        return out, avg_weights

    def forward(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        key_padding_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None,
        offset: int = 0
    ) -> Tuple[jax.Array, jax.Array]:
        return self(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask, offset=offset)
