from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from .norms import create_norm
from .attention import MultiHeadAttention
from .positional_encoding import RotaryPositionalEncoding

class SwiGLU(nnx.Module):
    """
    SwiGLU Gated Feed-Forward Network in Flax NNX.
    
    Specification:
    - 2/3 rule: hidden_dim = int(2 * d_ff / 3) to keep parameter count comparable to GELU FFN.
    - w_gate: Linear(d_model -> hidden_dim)
    - w_up:   Linear(d_model -> hidden_dim)
    - w_down: Linear(hidden_dim -> d_model)
    - y = w_down(silu(w_gate(x)) * w_up(x))
    """
    def __init__(self, d_model: int, d_ff: int, *, rngs: Optional[nnx.Rngs] = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # TODO: Implement __init__
        # Step 1: Compute d_hidden = int(2 * d_ff / 3)
        self.hidden_dim = int(2 * d_ff / 3)

        # Step 2: Initialize gate_proj, up_proj, down_proj using nnx.Linear with use_bias=False
        self.gate_proj = nnx.Linear(in_features=d_model, out_features=self.hidden_dim, use_bias=False, rngs=rngs)
        self.up_proj   = nnx.Linear(in_features=d_model, out_features=self.hidden_dim, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(in_features=self.hidden_dim, out_features=d_model, use_bias=False, rngs=rngs)
        self.w_gate = self.gate_proj
        self.w_up   = self.up_proj
        self.w_down = self.down_proj

    def __call__(self, x: jax.Array) -> jax.Array:
        # TODO: Implement forward pass: down_proj(silu(gate_proj(x)) * up_proj(x))
        gate = jax.nn.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)

    def forward(self, x: jax.Array) -> jax.Array:
        return self(x)


class StandardFFN(nnx.Module):
    """
    Standard two-layer Feed-Forward Network with GELU/ReLU/SiLU activation.
    Structure: Linear(d_model -> d_ff) -> Act -> Dropout -> Linear(d_ff -> d_model).
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        act: str = 'gelu',
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nnx.Linear(in_features=d_model, out_features=d_ff, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.dropout.p = dropout
        self.linear2 = nnx.Linear(in_features=d_ff, out_features=d_model, rngs=rngs)
        self.act_type = act.lower()
        if self.act_type == 'gelu':
            self.act = nnx.gelu
        elif self.act_type == 'relu':
            self.act = nnx.relu
        elif self.act_type == 'silu':
            self.act = nnx.silu
        else:
            self.act = nnx.gelu

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.linear1(x)
        if self.act_type == 'gelu':
            h = nnx.gelu(h)
        elif self.act_type == 'relu':
            h = nnx.relu(h)
        elif self.act_type == 'silu':
            h = nnx.silu(h)
        else:
            h = nnx.gelu(h)
        h = self.dropout(h)
        return self.linear2(h)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        return [self.linear1, self.act, self.dropout, self.linear2][idx]


class FeedForwardLayer(nnx.Module):
    """
    Transformer Feed-Forward Sublayer with Pre-Normalization and residual connection.
    y = x + dropout(ffn(norm(x)))
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = 'rmsnorm',
        ffn_type: str = 'gelu',
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.norm_type = norm_type
        self.ffn_type = ffn_type.lower()
        
        # TODO: Implement __init__
        # Step 1: Initialize self.norm using create_norm
        self.norm = create_norm(norm_type, d_model, rngs=rngs)
        # Step 3: Initialize self.dropout using nnx.Dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.dropout.p = dropout

        # Step 2: Initialize self.ffn as SwiGLU or StandardFFN
        if self.ffn_type == 'swiglu':
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, rngs=rngs)
        else:
            self.ffn = StandardFFN(d_model=d_model, d_ff=d_ff, dropout=dropout, act=self.ffn_type, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # TODO: Implement forward pass
        # Step 1: Store residual = x
        # Step 2: Normalize x
        norm_x = self.norm(x)
        # Step 3: Compute ffn output and apply dropout
        ffn_out = self.ffn(norm_x)
        # Step 4: Add residual and return
        return x + self.dropout(ffn_out)

    def forward(self, x: jax.Array) -> jax.Array:
        return self(x)


class SelfAttentionLayer(nnx.Module):
    """
    Transformer Self-Attention Sublayer with Pre-Normalization and residual connection.
    out = x + dropout(mha(norm(x), norm(x), norm(x)))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        norm_type: str = 'rmsnorm',
        qk_norm: bool = False,
        rope: Optional[RotaryPositionalEncoding] = None,
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # TODO: Implement __init__
        # Step 1: Initialize self.norm using create_norm
        self.norm = create_norm(norm_type, d_model, rngs=rngs)
        # Step 2: Initialize self.mha using MultiheadAttention
        self.mha = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            rope=rope,
            rngs=rngs
        )
        # Step 3: Initialize self.dropout using nnx.Dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.dropout.p = dropout

    def __call__(
        self,
        x: jax.Array,
        pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None,
        offset: int = 0
    ) -> Tuple[jax.Array, jax.Array]:
        # TODO: Implement forward pass
        # Step 1: Store residual = x
        # Step 2: Apply pre-normalization norm_x = self.norm(x)
        norm_x = self.norm(x)
        # Step 3: Compute self-attention out, attn_w = self.mha(norm_x, norm_x, norm_x, key_padding_mask, attn_mask)
        attn_out, attn_weights = self.mha(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=pad_mask,
            attn_mask=attn_mask,
            offset=offset
        )
        # Step 4: Add residual with dropout: residual + self.dropout(out)
        out = x + self.dropout(attn_out)
        # Step 5: Return (output, attn_w)
        return out, attn_weights

    def forward(
        self,
        x: jax.Array,
        pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None,
        offset: int = 0
    ) -> Tuple[jax.Array, jax.Array]:
        return self(x, pad_mask=pad_mask, attn_mask=attn_mask, offset=offset)


class CrossAttentionLayer(nnx.Module):
    """
    Transformer Cross-Attention Sublayer with Pre-Normalization and residual connection.
    out = x + dropout(mha(norm(x), memory, memory))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        norm_type: str = 'rmsnorm',
        qk_norm: bool = False,
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # TODO: Implement __init__
        # Step 1: Initialize one self.norm with create_norm. It is applied to the decoder queries only.
        self.norm = create_norm(norm_type, d_model, rngs=rngs)
        # Step 2: Initialize self.mha with MultiheadAttention. Do not pass RoPE.
        # Cross-attention never uses RoPE because query and key originate from separate sequences
        self.mha = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            rope=None,
            rngs=rngs
        )
        # Step 3: Initialize self.dropout with nnx.Dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.dropout.p = dropout

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        memory_pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        # TODO: Implement forward pass
        # Step 1: Store residual = x
        # Step 2: Pre-normalize the decoder queries only: norm_q = self.norm(x). Leave the encoder states unnormalized.
        norm_q = self.norm(x)
        # Step 3: Cross-attention uses the normalized queries and the encoder states as key and value
        attn_out, attn_weights = self.mha(
            query=norm_q,
            key=memory,
            value=memory,
            key_padding_mask=memory_pad_mask,
            attn_mask=attn_mask,
            offset=0
        )
        # Step 4: Add the residual with dropout: residual + self.dropout(out)
        out = x + self.dropout(attn_out)
        # Step 5: Return (output, attn_w)
        return out, attn_weights

    def forward(
        self,
        x: jax.Array,
        memory: jax.Array,
        memory_pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        return self(x, memory=memory, memory_pad_mask=memory_pad_mask, attn_mask=attn_mask)
