from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from .norms import create_norm
from .attention import MultiHeadAttention
from .positional_encoding import RotaryPositionalEncoding

'''
TODO: Implement these Modules.

The file contains four key sublayers used in transformer decoders:
1. SelfAttentionLayer: For masked self-attention
2. CrossAttentionLayer: For cross-attention between encoder and decoder
3. SwiGLU: The gated feed-forward network
4. FeedForwardLayer: For position-wise feed-forward processing

Each layer follows a Pre-LN architecture where:
- Normalization is applied before the main operation
- A residual connection wraps around the operation

The attention and feed-forward layers are configurable:
- norm_type selects the normalization, 'rmsnorm' or 'layernorm'
- ffn_type selects the feed-forward network, 'swiglu' or 'gelu'
- rope, when given, supplies rotary embeddings to the self-attention layers
'''

class SwiGLU(nnx.Module):
    '''
    Gated feed-forward network.
    Where the GELU network applies one projection and one activation, this one
    applies two projections and gates the first by the second:

        down_proj(silu(gate_proj(x)) * up_proj(x))

    Specification:
    - Three linear layers, none of them with a bias
    - gate_proj and up_proj both map d_model -> hidden_dim, down_proj maps hidden_dim -> d_model
    - hidden_dim is 2/3 of d_ff, rounded down
    '''
    def __init__(self, d_model: int, d_ff: int, *, rngs: Optional[nnx.Rngs] = None):
        '''
        Initialize the SwiGLU.
        Args:
            d_model (int): The dimension of the model.
            d_ff    (int): The dimension of the equivalent feedforward network.
        '''
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # TODO: Implement __init__
        raise NotImplementedError  # Remove once implemented
        # Step 1: Compute hidden_dim = int(2 * d_ff / 3)
        self.hidden_dim = NotImplementedError
        # Step 2: Initialize gate_proj, up_proj, down_proj using nnx.Linear with use_bias=False
        self.gate_proj = NotImplementedError
        self.up_proj = NotImplementedError
        self.down_proj = NotImplementedError

    def __call__(self, x: jax.Array) -> jax.Array:
        '''
        Forward pass for the SwiGLU.
        Args:
            x (jax.Array): Input array, shape (batch_size, seq_len, d_model)
        Returns:
            jax.Array: Output array, shape (batch_size, seq_len, d_model)
        '''
        # TODO: Implement forward pass: down_proj(silu(gate_proj(x)) * up_proj(x))
        raise NotImplementedError  # Remove once implemented
        gate = NotImplementedError
        up = NotImplementedError
        out = NotImplementedError
        return out

    def forward(self, x: jax.Array) -> jax.Array:
        return self(x)


class StandardFFN(nnx.Module):
    """
    Standard two-layer Feed-Forward Network with GELU/ReLU/SiLU activation. (DO NOT MODIFY)
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

    def forward(self, x: jax.Array) -> jax.Array:
        return self(x)


class FeedForwardLayer(nnx.Module):
    '''
    Pre-LN feed-forward sublayer.
    y = x + dropout(ffn(norm(x)))
    '''
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
        '''
        Initialize the FeedForwardLayer.
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            ffn_type  (str): The feed-forward network to use, 'swiglu' or 'gelu'.
        '''
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.norm_type = norm_type
        self.ffn_type = ffn_type.lower()
        # TODO: Implement __init__
        raise NotImplementedError  # Remove once implemented
        # Step 1: Initialize self.norm using create_norm
        self.norm = NotImplementedError
        # Step 2: Initialize self.dropout using nnx.Dropout
        self.dropout = NotImplementedError
        # Step 3: Initialize self.ffn as SwiGLU or StandardFFN
        if self.ffn_type == 'swiglu':
            self.ffn = NotImplementedError
        else:
            self.ffn = NotImplementedError

    def __call__(self, x: jax.Array) -> jax.Array:
        '''
        Forward pass for the FeedForwardLayer.
        Args:
            x (jax.Array): Input array, shape (batch_size, seq_len, d_model)
        Returns:
            jax.Array: Output array, shape (batch_size, seq_len, d_model)
        '''
        # TODO: Implement forward pass
        raise NotImplementedError  # Remove once implemented
        # Step 1: Store residual = x
        residual = NotImplementedError
        # Step 2: Normalize x
        norm_x = NotImplementedError
        # Step 3: Compute the feed-forward output
        ffn_out = NotImplementedError
        # Step 4: Add the residual with dropout
        out = NotImplementedError
        return out

    def forward(self, x: jax.Array) -> jax.Array:
        return self(x)


class SelfAttentionLayer(nnx.Module):
    '''
    Pre-LN self-attention sublayer.
    out = x + dropout(mha(norm(x), norm(x), norm(x)))
    '''
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
        '''
        Initialize the SelfAttentionLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            qk_norm  (bool): Whether to normalize the queries and keys before scoring.
            rope (Optional[RotaryPositionalEncoding]): Rotary embeddings, or None to disable.
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # TODO: Implement __init__
        raise NotImplementedError  # Remove once implemented
        # Step 1: Initialize self.norm using create_norm
        self.norm = NotImplementedError
        # Step 2: Initialize self.mha using MultiHeadAttention
        self.mha = NotImplementedError
        # Step 3: Initialize self.dropout using nnx.Dropout
        self.dropout = NotImplementedError

    def __call__(
        self,
        x: jax.Array,
        pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None,
        offset: int = 0
    ) -> Tuple[jax.Array, jax.Array]:
        '''
        Forward pass for the SelfAttentionLayer.
        Args:
            x (jax.Array): Input array, shape (batch_size, seq_len, d_model)
            pad_mask (Optional[jax.Array]): Padding mask, shape (batch_size, seq_len)
            attn_mask (Optional[jax.Array]): Attention mask, shape (seq_len, seq_len)
            offset (int): Position offset for RoPE.
        Returns:
            Tuple[jax.Array, jax.Array]: (output, attn_weights)
        '''
        # TODO: Implement forward pass
        raise NotImplementedError  # Remove once implemented
        # Step 1: Store residual = x
        residual = NotImplementedError
        # Step 2: Apply pre-normalization norm_x = self.norm(x)
        norm_x = NotImplementedError
        # Step 3: Compute self-attention out, attn_w = self.mha(norm_x, norm_x, norm_x, ...)
        attn_out, attn_weights = NotImplementedError, NotImplementedError
        # Step 4: Add residual with dropout
        out = NotImplementedError
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
    '''
    Pre-LN cross-attention sublayer.
    out = x + dropout(mha(norm(x), memory, memory))

    Rotary embeddings are not applied here. The queries and the keys come
    from two different sequences, so the difference between their positions
    does not mean anything. One self.norm is applied to the decoder queries only.
    '''
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
        '''
        Initialize the CrossAttentionLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
            qk_norm  (bool): Whether to normalize the queries and keys before scoring.
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # TODO: Implement __init__
        raise NotImplementedError  # Remove once implemented
        # Step 1: Initialize one self.norm with create_norm. It is applied to the decoder queries only.
        self.norm = NotImplementedError
        # Step 2: Initialize self.mha with MultiHeadAttention. Do not pass RoPE.
        self.mha = NotImplementedError
        # Step 3: Initialize self.dropout with nnx.Dropout
        self.dropout = NotImplementedError

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        memory_pad_mask: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        '''
        Forward pass for the CrossAttentionLayer.
        Args:
            x (jax.Array): Decoder queries, shape (batch_size, dec_len, d_model)
            memory (jax.Array): Encoder states, shape (batch_size, enc_len, d_model)
            memory_pad_mask (Optional[jax.Array]): Padding mask for memory, shape (batch_size, enc_len)
            attn_mask (Optional[jax.Array]): Attention mask, shape (dec_len, enc_len)
        Returns:
            Tuple[jax.Array, jax.Array]: (output, attn_weights)
        '''
        # TODO: Implement forward pass
        raise NotImplementedError  # Remove once implemented
        # Step 1: Store residual = x
        residual = NotImplementedError
        # Step 2: Pre-normalize the decoder queries only: norm_q = self.norm(x). Leave memory unnormalized.
        norm_q = NotImplementedError
        # Step 3: Cross-attention uses the normalized queries and the encoder states as key and value
        attn_out, attn_weights = NotImplementedError, NotImplementedError
        # Step 4: Add the residual with dropout
        out = NotImplementedError
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
