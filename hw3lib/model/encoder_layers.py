from typing import Optional, Tuple
import jax
from flax import nnx
from .sublayers import SelfAttentionLayer, FeedForwardLayer
from .positional_encoding import RotaryPositionalEncoding

'''
TODO: Implement this Module.

SelfAttentionEncoderLayer: Used in the encoder part of transformers
- Contains self-attention and feed-forward sublayers
- Does NOT use causal masking (bidirectional attention over speech features)
- Pre-LN residual structure
'''

class SelfAttentionEncoderLayer(nnx.Module):
    '''
    Pre-LN Encoder Layer with self-attention.
    Input -> SelfAttentionLayer -> FeedForwardLayer -> Output
    '''
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = 'rmsnorm',
        ffn_type: str = 'gelu',
        qk_norm: bool = False,
        rope: Optional[RotaryPositionalEncoding] = None,
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        '''
        Initialize the SelfAttentionEncoderLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            ffn_type  (str): The feed-forward network to use, 'swiglu' or 'gelu'.
            qk_norm  (bool): Whether to normalize the queries and keys before scoring.
            rope (Optional[RotaryPositionalEncoding]): Rotary embeddings, or None to disable.
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        # TODO: Implement __init__
        raise NotImplementedError  # Remove once implemented
        # Step 1: Initialize self.self_attn as SelfAttentionLayer (no causal mask is stored here)
        self.self_attn = NotImplementedError
        # Step 2: Initialize self.ffn as FeedForwardLayer
        self.ffn = NotImplementedError

    def __call__(
        self,
        x: jax.Array,
        pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        '''
        Forward pass for the encoder layer.
        Args:
            x (jax.Array): Input array, shape (N, T, d_model).
            pad_mask (Optional[jax.Array]): Padding mask, shape (N, T).
            attn_mask (Optional[jax.Array]): Attention mask, shape (T, T).
        Returns:
            Tuple[jax.Array, jax.Array]: (output of shape (N, T, d_model), attn_weights of shape (N, T, T))
        '''
        # TODO: Implement __call__. What differs from decoder self-attention is the missing causal mask.
        raise NotImplementedError  # Remove once implemented
        # Step 1: Run self_attn with the padding mask and no causal mask
        x, attn_weights = NotImplementedError, NotImplementedError
        # Step 2: Run self.ffn
        x = NotImplementedError
        # Step 3: Return (output, attn_weights)
        return x, attn_weights

    def forward(
        self,
        x: jax.Array,
        pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        return self(x, pad_mask=pad_mask, attn_mask=attn_mask)
