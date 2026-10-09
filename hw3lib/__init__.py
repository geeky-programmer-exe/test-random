"""
hw3lib for JAX: Automatic Speech Recognition with Encoder-Decoder Transformer
Built with JAX, Flax NNX, and Optax.
"""

from .model import (
    PadMask,
    CausalMask,
    PositionalEncoding,
    RotaryPositionalEncoding,
    RMSNorm,
    create_norm,
    MultiHeadAttention,
    SelfAttentionLayer,
    CrossAttentionLayer,
    SwiGLU,
    FeedForwardLayer,
    CrossAttentionDecoderLayer,
    SelfAttentionEncoderLayer,
    SpeechEmbedding,
    Conv2DSubsampling,
    ConvolutionalEmbedding,
    EncoderDecoderTransformer,
)

from .data import (
    H3Tokenizer,
    TextTokenizer,
    ASRDataset,
    ASRDataSource,
    verify_dataloader
)

from .decoding import SequenceGenerator
from .trainers import BaseTrainer, ASRTrainer, AverageMeter
from .utils import create_scheduler, create_warmup_scheduler, create_optimizer, plot_lr_schedule

__all__ = [
    # Model components
    'PadMask',
    'CausalMask',
    'PositionalEncoding',
    'RotaryPositionalEncoding',
    'RMSNorm',
    'create_norm',
    'MultiHeadAttention',
    'SelfAttentionLayer',
    'CrossAttentionLayer',
    'SwiGLU',
    'FeedForwardLayer',
    'CrossAttentionDecoderLayer',
    'SelfAttentionEncoderLayer',
    'SpeechEmbedding',
    'Conv2DSubsampling',
    'ConvolutionalEmbedding',
    'EncoderDecoderTransformer',
    # Data components
    'H3Tokenizer',
    'TextTokenizer',
    'ASRDataset',
    'ASRDataSource',
    # Decoding
    'SequenceGenerator',
    # Trainers
    'BaseTrainer',
    'ASRTrainer',
    'AverageMeter',
    # Utils
    'create_scheduler',
    'create_warmup_scheduler',
    'create_optimizer'
]