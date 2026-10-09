from typing import Optional, Tuple
import jax
from flax import nnx
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer
from .positional_encoding import RotaryPositionalEncoding

'''
TODO: Implement this Module.

CrossAttentionDecoderLayer: Pre-LN decoder layer with masked self-attention,
cross-attention, and a feed-forward sublayer.
- self_attn receives rope
- cross_attn never receives rope
- The encoder states are passed as memory
'''

class CrossAttentionDecoderLayer(nnx.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention, cross-attention, and feed-forward sublayers.
    Target -> SelfAttentionLayer -> CrossAttentionLayer -> FeedForwardLayer -> Output
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
        Initialize the CrossAttentionDecoderLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            ffn_type  (str): The feed-forward network to use, 'swiglu' or 'gelu'.
            qk_norm  (bool): Whether to normalize the queries and keys before scoring.
            rope (Optional[RotaryPositionalEncoding]): Rotary embeddings for self-attention.
                       Cross-attention never uses them.
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        # TODO: Implement __init__
        raise NotImplementedError  # Remove once implemented
        # Step 1: Initialize self.self_attn as SelfAttentionLayer (this sublayer receives rope)
        self.self_attn = NotImplementedError
        # Step 2: Initialize self.cross_attn as CrossAttentionLayer (never receives rope)
        self.cross_attn = NotImplementedError
        # Step 3: Initialize self.ffn as FeedForwardLayer
        self.ffn = NotImplementedError

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        pad_mask_dec: Optional[jax.Array] = None,
        pad_mask_enc: Optional[jax.Array] = None,
        slf_attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        '''
        Forward pass for the decoder layer.
        Args:
            x (jax.Array): Target representations, shape (N, T_dec, d_model).
            memory (jax.Array): Encoder representations, shape (N, T_enc, d_model).
            pad_mask_dec (Optional[jax.Array]): Decoder padding mask, shape (N, T_dec).
            pad_mask_enc (Optional[jax.Array]): Encoder memory padding mask, shape (N, T_enc).
            slf_attn_mask (Optional[jax.Array]): Causal mask, shape (T_dec, T_dec).
        Returns:
            Tuple[jax.Array, jax.Array, jax.Array]:
                (output, self_attn_weights, cross_attn_weights)
        '''
        # TODO: Implement __call__. Follow the figure in the writeup.
        raise NotImplementedError  # Remove once implemented
        # Step 1: Run self_attn with the causal mask and the decoder padding mask
        x, slf_attn_weights = NotImplementedError, NotImplementedError
        # Step 2: Run cross_attn. The encoder output is the memory argument.
        x, crs_attn_weights = NotImplementedError, NotImplementedError
        # Step 3: Run self.ffn
        x = NotImplementedError
        # Step 4: Return (output, self_attn_weights, cross_attn_weights)
        return x, slf_attn_weights, crs_attn_weights

    def forward(
        self,
        x: jax.Array,
        memory: jax.Array,
        pad_mask_dec: Optional[jax.Array] = None,
        pad_mask_enc: Optional[jax.Array] = None,
        slf_attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        return self(
            x,
            memory=memory,
            pad_mask_dec=pad_mask_dec,
            pad_mask_enc=pad_mask_enc,
            slf_attn_mask=slf_attn_mask,
        )
