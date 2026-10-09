from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import nnx

class Conv2DSubsampling(nnx.Module):
    """
    Conv2D Subsampling module with time-only downsampling in Flax NNX.
    
    Specification:
    - Input: (batch_size, seq_len, input_dim)
    - Adds channel dimension -> (batch_size, seq_len, input_dim, 1)
    - Two Conv2D layers with GELU:
      * Conv1: strides (time_stride1, 1)
      * Conv2: strides (time_stride2, 1)
    - Flattens frequency and channels, then projects with Linear layer to output_dim.
    - Output: (batch_size, new_seq_len, output_dim)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        time_reduction: int = 2,
        kernel_size: int = 3,
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.time_stride1, self.time_stride2 = self.closest_factors(time_reduction)

        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=output_dim,
            kernel_size=(kernel_size, kernel_size),
            strides=(self.time_stride1, 1),
            padding='VALID',
            rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=output_dim,
            out_features=output_dim,
            kernel_size=(kernel_size, kernel_size),
            strides=(self.time_stride2, 1),
            padding='VALID',
            rngs=rngs
        )

        downsampled_freq = self.calculate_downsampled_length(input_dim, 1, 1)
        linear_in_dim = int(downsampled_freq * output_dim)
        self.linear_out = nnx.Linear(in_features=linear_in_dim, out_features=output_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    @staticmethod
    def closest_factors(n: int) -> Tuple[int, int]:
        factor = int(n**0.5)
        while n % factor != 0:
            factor -= 1
        return max(factor, n // factor), min(factor, n // factor)

    def calculate_downsampled_length(self, lengths: jax.Array, stride1: int, stride2: int) -> jax.Array:
        k = self.kernel_size
        l1 = (lengths - (k - 1) - 1) // stride1 + 1
        l2 = (l1 - (k - 1) - 1) // stride2 + 1
        return jnp.maximum(l2, 0)

    def __call__(self, x: jax.Array, x_len: Optional[jax.Array] = None) -> Tuple[jax.Array, Optional[jax.Array]]:
        # x shape: (N, T, input_dim) -> add channel dimension: (N, T, input_dim, 1)
        x_in = jnp.expand_dims(x, axis=-1)
        
        # Conv 1 + GELU
        h = nnx.gelu(self.conv1(x_in))
        # Conv 2 + GELU
        h = nnx.gelu(self.conv2(h))
        
        # h shape: (N, new_T, new_freq, output_dim)
        N, new_T, new_F, C = h.shape
        h_flat = h.reshape((N, new_T, new_F * C))
        
        out = self.dropout(self.linear_out(h_flat))
        
        if x_len is not None:
            out_len = self.calculate_downsampled_length(x_len, self.time_stride1, self.time_stride2)
        else:
            out_len = None
            
        return out, out_len

    def forward(self, x: jax.Array, x_len: Optional[jax.Array] = None) -> Tuple[jax.Array, Optional[jax.Array]]:
        return self(x, x_len)


class LinearEmbedding(nnx.Module):
    """Simple linear projection when no time downsampling is required."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, *, rngs: Optional[nnx.Rngs] = None):
        super().__init__()
        self.linear = nnx.Linear(in_features=input_dim, out_features=output_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array, x_len: Optional[jax.Array] = None) -> Tuple[jax.Array, Optional[jax.Array]]:
        out = self.dropout(self.linear(x))
        return out, x_len

    def forward(self, x: jax.Array, x_len: Optional[jax.Array] = None) -> Tuple[jax.Array, Optional[jax.Array]]:
        return self(x, x_len)


class SpeechEmbedding(nnx.Module):
    """
    Flexible speech feature processor supporting time downsampling.
    
    Args:
        input_dim (int): Frequency dimension of speech features (e.g. 80 fbank).
        output_dim (int): Output embedding dimension (d_model).
        time_reduction (int): Downsampling factor along time.
        reduction_method (str): 'conv', 'linear', 'both', or 'lstm'.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        time_reduction: int = 2,
        reduction_method: str = 'conv',
        dropout: float = 0.0,
        *,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_reduction = time_reduction
        self.reduction_method = reduction_method.lower()

        if time_reduction > 1 and self.reduction_method != 'linear':
            self.cnn = Conv2DSubsampling(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
                time_reduction=time_reduction,
                rngs=rngs
            )
        else:
            self.cnn = LinearEmbedding(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
                rngs=rngs
            )

    def calculate_downsampled_length(self, lengths: jax.Array) -> jax.Array:
        if hasattr(self.cnn, 'calculate_downsampled_length'):
            return self.cnn.calculate_downsampled_length(lengths, self.cnn.time_stride1, self.cnn.time_stride2)
        return lengths

    def __call__(self, x: jax.Array, x_len: Optional[jax.Array] = None) -> Tuple[jax.Array, Optional[jax.Array]]:
        return self.cnn(x, x_len)

    def forward(self, x: jax.Array, x_len: Optional[jax.Array] = None) -> Tuple[jax.Array, Optional[jax.Array]]:
        return self(x, x_len)


# Alias for compatibility with previous JAX code
ConvolutionalEmbedding = SpeechEmbedding
