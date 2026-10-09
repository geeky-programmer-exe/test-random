from typing import Optional, Tuple
import jax
from flax import nnx
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer
from .positional_encoding import RotaryPositionalEncoding

class CrossAttentionDecoderLayer(nnx.Module):
    """
    Cross-Attention Decoder Layer for Automatic Speech Recognition in Flax NNX.
    
    Architecture:
    Target -> SelfAttentionLayer (Causal + Padding) -> CrossAttentionLayer (Target attends to Encoder Memory) -> FeedForwardLayer -> Output
    """
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
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # TODO: Implement __init__
        # Step 1: Initialize self.self_attn as SelfAttentionLayer (this sublayer receives rope)
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            norm_type=norm_type,
            qk_norm=qk_norm,
            rope=rope,
            rngs=rngs
        )
        # Step 2: Initialize self.cross_attn as CrossAttentionLayer (never receives rope)
        self.cross_attn = CrossAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            norm_type=norm_type,
            qk_norm=qk_norm,
            rngs=rngs
        )
        # Step 3: Initialize self.ffn as FeedForwardLayer
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            norm_type=norm_type,
            ffn_type=ffn_type,
            rngs=rngs
        )

    @property
    def feed_forward(self):
        """Attribute alias for compatibility with test assertions."""
        return self.ffn

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        pad_mask_dec: Optional[jax.Array] = None,
        pad_mask_enc: Optional[jax.Array] = None,
        slf_attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Forward pass for decoder layer.
        
        Args:
            x (jax.Array): Target representations, shape (N, T_dec, d_model).
            memory (jax.Array): Encoder representations, shape (N, T_enc, d_model).
            pad_mask_dec (jax.Array, optional): Decoder padding mask, shape (N, T_dec).
            pad_mask_enc (jax.Array, optional): Encoder memory padding mask, shape (N, T_enc).
            slf_attn_mask (jax.Array, optional): Causal mask, shape (T_dec, T_dec).
            
        Returns:
            Tuple[jax.Array, jax.Array, jax.Array]:
                (output of shape (N, T_dec, d_model), self_attn_weights, cross_attn_weights)
        """
        # TODO: Implement __call__. Follow the figure in the writeup.
        # Step 1: Run self_attn with the causal mask and the decoder padding mask
        x, slf_attn_weights = self.self_attn(x, pad_mask=pad_mask_dec, attn_mask=slf_attn_mask)
        
        # Step 2: Run cross_attn with the encoder output and the encoder padding mask
        x, crs_attn_weights = self.cross_attn(x, memory=memory, memory_pad_mask=pad_mask_enc, attn_mask=None)
        
        # Step 3: Run self.ffn
        x = self.ffn(x)

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
        return self(x, memory=memory, pad_mask_dec=pad_mask_dec, pad_mask_enc=pad_mask_enc, slf_attn_mask=slf_attn_mask)
