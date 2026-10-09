from typing import Optional, Tuple, List, Union
import jax
import jax.numpy as jnp
from flax import nnx

from .masks import PadMask, CausalMask
from .norms import create_norm
from .positional_encoding import PositionalEncoding, RotaryPositionalEncoding
from .speech_embedding import SpeechEmbedding
from .encoder_layers import SelfAttentionEncoderLayer
from .decoder_layers import CrossAttentionDecoderLayer

class EncoderDecoderTransformer(nnx.Module):
    """
    Encoder-Decoder Transformer for Automatic Speech Recognition (ASR) in Flax NNX.
    
    Architecture:
    1. SpeechEmbedding: Convolutional downsampling of acoustic features.
    2. PositionalEncoding / RoPE: Temporal position representations.
    3. Encoder: Stack of SelfAttentionEncoderLayer blocks followed by encoder_norm.
    4. CTC Head: Auxiliary linear projection on encoder representations to vocabulary.
    5. Decoder: Stack of CrossAttentionDecoderLayer blocks followed by decoder_norm.
    6. Final Linear: Projection from decoder representations to target vocabulary logits.
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
        weight_tying: bool = False,
        layer_drop_rate: float = 0.0,
        *,
        rngs: Optional[nnx.Rngs] = None,
        **kwargs
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
        self.weight_tying = weight_tying
        self.layer_drop_rate = layer_drop_rate

        # TODO: Initialize self.source_embedding as SpeechEmbedding
        self.source_embedding = SpeechEmbedding(
            input_dim=input_dim,
            output_dim=d_model,
            time_reduction=time_reduction,
            reduction_method=reduction_method,
            dropout=dropout,
            rngs=rngs
        )

        # 2. Positional encodings & dropout
        self.encoder_pos_type = encoder_pos_encoding.lower()
        self.decoder_pos_type = decoder_pos_encoding.lower()

        head_dim_enc = d_model // num_encoder_heads
        head_dim_dec = d_model // num_decoder_heads

        # TODO: Create the rotary embeddings, one per stack, or None where the stack
        # does not use them. Each stack rotates over its own head dimension, which is
        # d_model // num_heads for that stack, and every layer in a stack must share
        # the same object rather than building its own.
        self.enc_rope = (
            RotaryPositionalEncoding(head_dim=head_dim_enc, max_len=max_len, rngs=rngs)
            if self.encoder_pos_type == 'rope' else None
        )
        self.dec_rope = (
            RotaryPositionalEncoding(head_dim=head_dim_dec, max_len=max_len, rngs=rngs)
            if self.decoder_pos_type == 'rope' else None
        )

        # TODO: Initialize self.positional_encoding as PositionalEncoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, rngs=rngs)
        # TODO: Initialize self.dropout as nnx.Dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

        # TODO: Initialize self.enc_layers as an nnx.List of SelfAttentionEncoderLayer.
        # Pass norm_type, ffn_type, qk_norm, and the shared encoder RoPE into each layer.
        self.enc_layers = nnx.List([
            SelfAttentionEncoderLayer(
                d_model=d_model,
                num_heads=num_encoder_heads,
                d_ff=d_ff_encoder,
                dropout=dropout,
                norm_type=norm_type,
                ffn_type=ffn_type,
                qk_norm=qk_norm,
                rope=self.enc_rope,
                rngs=rngs
            ) for _ in range(num_encoder_layers)
        ])
        # TODO: Initialize self.encoder_norm using create_norm
        self.encoder_norm = create_norm(norm_type, d_model, rngs=rngs)

        # TODO: Initialize self.target_embedding as nnx.Embed(num_embeddings=num_classes, features=d_model, rngs=rngs)
        self.target_embedding = nnx.Embed(num_embeddings=num_classes, features=d_model, rngs=rngs)
        self.target_embedding.embedding_dim = d_model
        # TODO: Initialize self.dec_layers as an nnx.List of CrossAttentionDecoderLayer.
        # Pass norm_type, ffn_type, qk_norm, and the shared decoder RoPE into each layer.
        self.dec_layers = nnx.List([
            CrossAttentionDecoderLayer(
                d_model=d_model,
                num_heads=num_decoder_heads,
                d_ff=d_ff_decoder,
                dropout=dropout,
                norm_type=norm_type,
                ffn_type=ffn_type,
                qk_norm=qk_norm,
                rope=self.dec_rope,
                rngs=rngs
            ) for _ in range(num_decoder_layers)
        ])
        # TODO: Initialize self.decoder_norm using create_norm
        self.decoder_norm = create_norm(norm_type, d_model, rngs=rngs)

        # TODO: Initialize self.final_linear as nnx.Linear(d_model, num_classes, rngs=rngs)
        self.final_linear = nnx.Linear(in_features=d_model, out_features=num_classes, rngs=rngs)
        # TODO: Create the CTC head as nnx.Linear(d_model, num_classes, rngs=rngs).
        # Apply log_softmax inside encode, not inside the head.
        self.ctc_head = nnx.Linear(in_features=d_model, out_features=num_classes, rngs=rngs)

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
            Tuple[jax.Array, Optional[jax.Array], dict, dict]:
                (encoder_output, pad_mask_src, encoder_attention, ctc_input)
        """
        # TODO: Apply source embedding, positional encoding, and dropout.
        # Add the sinusoidal encoding here only when encoder_pos_encoding is 'sinusoidal'.
        # Rotary embeddings are applied inside the attention layers, not here.
        h, enc_lens = self.source_embedding(features, input_lengths)

        if self.encoder_pos_type == 'sinusoidal':
            h = self.positional_encoding(h)
        h = self.dropout(h)

        # TODO: Create the source padding mask with PadMask from the embedded lengths
        if enc_lens is not None:
            pad_mask = PadMask(h, enc_lens)
        else:
            pad_mask = None

        # TODO: Pass through encoder layers (self.enc_layers) and save attention
        # running_att[f'layer{i+1}_enc_self'] = attention
        encoder_attention = {}
        for i, layer in enumerate(self.enc_layers):
            h, attn_w = layer(h, pad_mask=pad_mask)
            encoder_attention[f'layer{i+1}_enc_self'] = attn_w

        # TODO: Final normalization and CTC projection.
        # encoder_norm, then ctc_head, then jax.nn.log_softmax over the class axis,
        # swapped to (T, N, num_classes).
        h = self.encoder_norm(h)

        ctc_logits = self.ctc_head(h)
        ctc_log_probs = jax.nn.log_softmax(ctc_logits, axis=-1)
        ctc_log_probs_t = jnp.swapaxes(ctc_log_probs, 0, 1)  # (T, N, num_classes)
        ctc_input = {'log_probs': ctc_log_probs_t, 'lengths': enc_lens}

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
        # TODO: Apply target embedding, positional encoding, and dropout.
        # Add the sinusoidal encoding here only when decoder_pos_encoding is 'sinusoidal'.
        # Rotary embeddings are applied inside the attention layers.
        h = self.target_embedding(targets)

        if self.decoder_pos_type == 'sinusoidal':
            h = self.positional_encoding(h)
        h = self.dropout(h)

        # TODO: Create the target padding mask with PadMask when target_lengths is given,
        # and the causal mask with CausalMask. There is no device placement.
        causal_mask = CausalMask(h)
        pad_mask_dec = PadMask(h, target_lengths) if target_lengths is not None else None

        # TODO: Pass through decoder layers and save attention
        # running_att[f'layer{i+1}_dec_self'] and running_att[f'layer{i+1}_dec_cross']
        decoder_attention = {}
        for i, layer in enumerate(self.dec_layers):
            h, self_w, cross_w = layer(
                h,
                memory=memory,
                pad_mask_dec=pad_mask_dec,
                pad_mask_enc=pad_mask_src,
                slf_attn_mask=causal_mask
            )
            decoder_attention[f'layer{i+1}_dec_self'] = self_w
            decoder_attention[f'layer{i+1}_dec_cross'] = cross_w

        # TODO: Final normalization and projection
        # decoder_norm, then final_linear
        h = self.decoder_norm(h)
        output = self.final_linear(h)
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
        # TODO: Run encode, then decode
        enc_out, pad_mask_src, enc_attn, ctc_input = self.encode(features, input_lengths)
        output, dec_attn = self.decode(targets, enc_out, target_lengths=target_lengths, pad_mask_src=pad_mask_src)

        # TODO: Combine the encoder and decoder attention dictionaries
        attention_weights = {**enc_attn, **dec_attn}
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
        return self(features, targets, input_lengths=input_lengths, target_lengths=target_lengths, return_ctc=return_ctc)


# Backward compatibility alias
SpeechTransformer = EncoderDecoderTransformer
