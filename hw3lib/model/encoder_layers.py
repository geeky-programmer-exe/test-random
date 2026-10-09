from typing import Optional, Tuple
import jax
from flax import nnx
from .sublayers import SelfAttentionLayer, FeedForwardLayer
from .positional_encoding import RotaryPositionalEncoding

class SelfAttentionEncoderLayer(nnx.Module):
    """
    Self-Attention Encoder Layer for Automatic Speech Recognition in Flax NNX.
    
    Architecture:
    Input -> SelfAttentionLayer (Pre-LN + MHA + Residual) -> FeedForwardLayer (Pre-LN + FFN + Residual) -> Output
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
        # Step 1: Initialize self.self_attn as SelfAttentionLayer (no causal mask is stored here)
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            norm_type=norm_type,
            qk_norm=qk_norm,
            rope=rope,
            rngs=rngs
        )
        # Step 2: Initialize self.ffn as FeedForwardLayer
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
        pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Forward pass for encoder layer.
        
        Args:
            x (jax.Array): Input tensor, shape (N, T, d_model).
            pad_mask (jax.Array, optional): Padding mask, shape (N, T).
            attn_mask (jax.Array, optional): Attention mask, shape (T, T).
            
        Returns:
            Tuple[jax.Array, jax.Array]: (encoded_output of shape (N, T, d_model), attn_weights of shape (N, T, T))
        """
        # TODO: Implement __call__. What differs from decoder self-attention is the missing causal mask.
        # Step 1: Run self_attn with the padding mask and no causal mask
        x, attn_weights = self.self_attn(x, pad_mask=pad_mask, attn_mask=attn_mask)
        # Step 2: Run self.ffn
        x = self.ffn(x)
        # Step 3: Return (output, attn_weights)
        return x, attn_weights

    def forward(
        self,
        x: jax.Array,
        pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        return self(x, pad_mask=pad_mask, attn_mask=attn_mask)
