from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import nnx

from .masks import PadMask, CausalMask
from .norms import create_norm
from .positional_encoding import PositionalEncoding, RotaryPositionalEncoding
from .speech_embedding import SpeechEmbedding
from .encoder_layers import SelfAttentionEncoderLayer
from .decoder_layers import CrossAttentionDecoderLayer

'''
TODO: Implement this Module.

EncoderDecoderTransformer: Used for ASR (Automatic Speech Recognition)
- Contains an encoder stack for processing speech features
- Contains a decoder stack for generating text tokens
- Uses both self-attention and cross-attention
- Includes a CTC head. log_softmax is applied inside encode, not inside the head

There is no weight tying, no layer drop, and no device placement.
'''

class EncoderDecoderTransformer(nnx.Module):
    """
    Encoder-Decoder Transformer for Automatic Speech Recognition.
    """
    def __init__(
        self,
        input_dim: int = 80,
        time_reduction: int = 2,
        reduction_method: str = 'both',
        num_encoder_layers: int = 2,
        num_encoder_heads: int = 4,
        d_ff_encoder: int = 32,
        num_decoder_layers: int = 2,
        num_decoder_heads: int = 4,
        d_ff_decoder: int = 32,
        d_model: int = 16,
        dropout: float = 0.1,
        max_len: int = 500,
        num_classes: int = 10,
        norm_type: str = 'rmsnorm',
        encoder_pos_encoding: str = 'sinusoidal',
        decoder_pos_encoding: str = 'sinusoidal',
        ffn_type: str = 'gelu',
        qk_norm: bool = False,
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_len = max_len
        self.time_reduction = time_reduction
        self.norm_type = norm_type
        self.ffn_type = ffn_type
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_pos_type = encoder_pos_encoding.lower()
        self.decoder_pos_type = decoder_pos_encoding.lower()

        # Each stack rotates over its own head dimension.
        head_dim_enc = d_model // num_encoder_heads
        head_dim_dec = d_model // num_decoder_heads

        # TODO: Implement __init__
        raise NotImplementedError  # Remove once implemented

        # TODO: Create the rotary embeddings, one per stack, or None where the stack
        # does not use them. Each stack rotates over its own head dimension, which is
        # d_model // num_heads for that stack, and every layer in a stack must share
        # the same object rather than building its own.
        self.enc_rope = NotImplementedError
        self.dec_rope = NotImplementedError

        # TODO: Initialize self.source_embedding as SpeechEmbedding
        self.source_embedding = NotImplementedError
        # TODO: Initialize self.positional_encoding as PositionalEncoding
        self.positional_encoding = NotImplementedError
        # TODO: Initialize self.dropout as nnx.Dropout
        self.dropout = NotImplementedError

        # TODO: Initialize self.enc_layers as an nnx.List of SelfAttentionEncoderLayer.
        # Pass norm_type, ffn_type, qk_norm, and the shared encoder RoPE into each layer.
        self.enc_layers = NotImplementedError
        # TODO: Initialize self.encoder_norm using create_norm
        self.encoder_norm = NotImplementedError

        # TODO: Initialize self.target_embedding as nnx.Embed(num_embeddings=num_classes, features=d_model, rngs=rngs)
        # nnx.Embed stores the width as `features`. Tests read embedding_dim.
        self.target_embedding = NotImplementedError
        if isinstance(self.target_embedding, nnx.Module):
            self.target_embedding.embedding_dim = d_model

        # TODO: Initialize self.dec_layers as an nnx.List of CrossAttentionDecoderLayer.
        # Pass norm_type, ffn_type, qk_norm, and the shared decoder RoPE into each layer.
        self.dec_layers = NotImplementedError
        # TODO: Initialize self.decoder_norm using create_norm
        self.decoder_norm = NotImplementedError

        # TODO: Initialize self.final_linear as nnx.Linear(d_model, num_classes, rngs=rngs)
        self.final_linear = NotImplementedError
        # TODO: Create the CTC head as nnx.Linear(d_model, num_classes, rngs=rngs).
        # Apply log_softmax inside encode, not inside the head.
        self.ctc_head = NotImplementedError

    def encode(
        self,
        features: jax.Array,
        input_lengths: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, Optional[jax.Array], dict, dict]:
        """
        Encode acoustic speech features into hidden representations.
        Args:
            features (jax.Array): Filterbank features, shape (N, T_in, input_dim).
            input_lengths (jax.Array, optional): Unpadded lengths, shape (N,).
        Returns:
            Tuple: (encoder_output, pad_mask, encoder_attention, ctc_input)
                   where ctc_input is {'log_probs': ..., 'lengths': ...}
                   and log_probs has shape (T, N, num_classes).
        """
        # TODO: Implement encode
        raise NotImplementedError  # Remove once implemented

        # TODO: Apply source embedding, positional encoding, and dropout.
        # Add the sinusoidal encoding here only when encoder_pos_encoding is 'sinusoidal'.
        # Rotary embeddings are applied inside the attention layers, not here.
        h, enc_lens = NotImplementedError, NotImplementedError
        if self.encoder_pos_type == 'sinusoidal':
            h = NotImplementedError
        h = NotImplementedError  # dropout

        # TODO: Create the source padding mask with PadMask from the embedded lengths
        if enc_lens is not None:
            pad_mask = NotImplementedError
        else:
            pad_mask = None

        # TODO: Pass through encoder layers (self.enc_layers) and save attention
        encoder_attention = {}
        for i, layer in enumerate(self.enc_layers):
            h, attn_w = NotImplementedError, NotImplementedError
            encoder_attention[f'layer{i+1}_enc_self'] = attn_w

        # TODO: Final normalization and CTC projection.
        # encoder_norm, then ctc_head, then jax.nn.log_softmax over the class axis,
        # swapped to (T, N, num_classes).
        h = NotImplementedError
        ctc_logits = NotImplementedError
        ctc_log_probs = NotImplementedError
        ctc_log_probs_t = NotImplementedError
        ctc_input = NotImplementedError

        # TODO: Return the encoded representation, padding mask, running attention weights,
        # and CTC inputs {'log_probs': ..., 'lengths': ...}
        return h, pad_mask, encoder_attention, ctc_input

    def decode(
        self,
        targets: jax.Array,
        memory: jax.Array,
        target_lengths: Optional[jax.Array] = None,
        pad_mask_src: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, dict]:
        """
        Decode target token sequences with cross-attention to encoder representations.
        Args:
            targets (jax.Array): Target token IDs, shape (N, T_dec).
            memory (jax.Array): Encoded acoustic features, shape (N, T_enc, d_model).
            target_lengths (jax.Array, optional): Target lengths, shape (N,).
            pad_mask_src (jax.Array, optional): Encoder memory padding mask, shape (N, T_enc).
        Returns:
            Tuple[jax.Array, dict]: (output of shape (N, T_dec, num_classes), decoder_attention)
        """
        # TODO: Implement decode
        raise NotImplementedError  # Remove once implemented

        # TODO: Apply target embedding, positional encoding, and dropout.
        # Add the sinusoidal encoding here only when decoder_pos_encoding is 'sinusoidal'.
        # Rotary embeddings are applied inside the attention layers.
        h = NotImplementedError  # target embedding
        if self.decoder_pos_type == 'sinusoidal':
            h = NotImplementedError
        h = NotImplementedError  # dropout

        # TODO: Create the target padding mask with PadMask when target_lengths is given,
        # and the causal mask with CausalMask. There is no device placement.
        causal_mask = NotImplementedError
        if target_lengths is not None:
            pad_mask_dec = NotImplementedError
        else:
            pad_mask_dec = None

        # TODO: Pass through decoder layers and save attention
        decoder_attention = {}
        for i, layer in enumerate(self.dec_layers):
            h, self_w, cross_w = NotImplementedError, NotImplementedError, NotImplementedError
            decoder_attention[f'layer{i+1}_dec_self'] = self_w
            decoder_attention[f'layer{i+1}_dec_cross'] = cross_w

        # TODO: Final normalization and projection
        # decoder_norm, then final_linear
        h = NotImplementedError
        output = NotImplementedError
        # TODO: Return the output sequence and running attention weights
        return output, decoder_attention

    def __call__(
        self,
        features: jax.Array,
        targets: jax.Array,
        input_lengths: Optional[jax.Array] = None,
        target_lengths: Optional[jax.Array] = None,
        return_ctc: bool = True
    ) -> Tuple[jax.Array, dict, dict]:
        """
        Full forward pass for training and inference.
        Returns:
            Tuple[jax.Array, dict, dict]: (output, attention_weights, ctc_input)
        """
        # TODO: Implement __call__
        raise NotImplementedError  # Remove once implemented
        # TODO: Run encode, then decode
        enc_out, pad_mask_src, enc_attn, ctc_input = (
            NotImplementedError, NotImplementedError, NotImplementedError, NotImplementedError
        )
        output, dec_attn = NotImplementedError, NotImplementedError
        # TODO: Combine the encoder and decoder attention dictionaries
        attention_weights = NotImplementedError
        # TODO: Return (logits, attention dictionary, ctc inputs)
        return output, attention_weights, ctc_input

    def forward(
        self,
        features: jax.Array,
        targets: jax.Array,
        input_lengths: Optional[jax.Array] = None,
        target_lengths: Optional[jax.Array] = None,
        return_ctc: bool = True
    ):
        return self(
            features,
            targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            return_ctc=return_ctc,
        )

    def score(
        self,
        batch_prompts: jax.Array,
        encoder_output: jax.Array,
        pad_mask_src: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Score next token for given encoder representations and prompt prefix. (DO NOT MODIFY)
        Returns logits for the last token position: shape (batch_size, num_classes).
        """
        logits, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        return logits[:, -1, :]


# Backward compatibility alias
SpeechTransformer = EncoderDecoderTransformer
