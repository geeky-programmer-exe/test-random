from typing import Callable, Optional, Tuple, Union, List
import jax
import jax.numpy as jnp
import numpy as np
from ..data.tokenizer import H3Tokenizer

class SequenceGenerator:
    """
    Autoregressive Sequence Generator for ASR in JAX.
    
    Supports:
    - Greedy search decoding (`generate_greedy`)
    - Beam search decoding (`generate_beam`)
    - Post-processing sequence cleanup (`post_process_sequence`)
    - Temperature scaling and repetition penalties
    """
    def __init__(
        self,
        score_fn: Callable[[jax.Array], jax.Array],
        tokenizer: H3Tokenizer,
        max_length: int = 100,
        device: Optional[str] = None,
        num_classes: Optional[int] = None
    ):
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes or tokenizer.vocab_size

    def apply_repeat_penalty(
        self,
        logits: jax.Array,
        prompts: jax.Array,
        penalty: float = 1.0
    ) -> jax.Array:
        if penalty == 1.0:
            return logits

        # Convert to numpy for flexible indexing if inside eager mode
        if logits.ndim == 2:
            B, vocab_size = logits.shape
            batch_idx = jnp.arange(B)[:, None]
            mask = jnp.zeros_like(logits, dtype=bool).at[batch_idx, prompts].set(True)
            penalties = jnp.where(logits > 0, penalty, 1.0 / penalty)
            penalized_logits = logits / penalties
            return jnp.where(mask, penalized_logits, logits)
        else:
            B, beam_width, vocab_size = logits.shape
            batch_idx = jnp.arange(B)[:, None, None]
            beam_idx = jnp.arange(beam_width)[None, :, None]
            mask = jnp.zeros_like(logits, dtype=bool).at[batch_idx, beam_idx, prompts].set(True)
            penalties = jnp.where(logits > 0, penalty, 1.0 / penalty)
            penalized_logits = logits / penalties
            return jnp.where(mask, penalized_logits, logits)

    def generate_greedy(
        self,
        x: Union[jax.Array, np.ndarray],
        temperature: float = 1.0,
        repeat_penalty: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedy search decoding.
        
        Args:
            x: Starting prompt token sequence, shape (batch_size, seq_len)
            temperature: Softmax temperature scale
            repeat_penalty: Penalty factor for previously generated tokens
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (generated_sequences of shape (batch_size, max_length), scores)
        """
        prompt = np.array(x, dtype=np.int64)
        batch_size = prompt.shape[0]
        
        sequences = np.pad(
            prompt,
            ((0, 0), (0, max(0, self.max_length - prompt.shape[1]))),
            constant_values=self.tokenizer.pad_id
        )
        scores = np.zeros(batch_size, dtype=np.float32)
        finished = np.zeros(batch_size, dtype=bool)
        
        cur_len = prompt.shape[1]
        eos_id = self.tokenizer.eos_id

        for t in range(cur_len, self.max_length):
            if np.all(finished):
                break

            # Forward pass through model score function
            cur_prompt = jnp.asarray(sequences[:, :t])
            logits = self.score_fn(cur_prompt)
            if hasattr(logits, 'shape') and logits.ndim == 3:
                logits = logits[:, -1, :]  # Take last time step
            logits = jnp.asarray(logits)
            
            # Repetition penalty and temperature
            logits = self.apply_repeat_penalty(logits, cur_prompt, repeat_penalty)
            logits = logits / max(temperature, 1e-5)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            log_probs_np = np.asarray(log_probs)

            next_tokens = np.argmax(log_probs_np, axis=-1)
            next_scores = np.max(log_probs_np, axis=-1)

            # Update finished status
            newly_finished = (next_tokens == eos_id)
            finished = finished | newly_finished

            scores += np.where(finished, 0.0, next_scores)
            sequences[:, t] = np.where(finished & ~newly_finished, self.tokenizer.pad_id, next_tokens)

        return sequences, scores

    # Alias
    decode_greedy = generate_greedy

    def generate_beam(
        self,
        x: Union[jax.Array, np.ndarray],
        beam_width: int = 8,
        temperature: float = 1.0,
        repeat_penalty: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Beam Search Decoding.
        
        Args:
            x: Initial prompt token IDs, shape (batch_size, seq_len)
            beam_width: Number of beam hypotheses to maintain
            temperature: Logit scaling temperature
            repeat_penalty: Repetition penalty factor
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                (hypotheses of shape (batch_size, beam_width, sequence_length), scores of shape (batch_size, beam_width))
        """
        x_np = np.array(x, dtype=np.int64)
        if x_np.ndim != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x_np.shape[1]:
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        batch_size, seq_len = x_np.shape
        eos_id = self.tokenizer.eos_id

        # Initialize beams: (batch_size, beam_width, seq_len)
        x_beams = np.broadcast_to(x_np[:, None, :], (batch_size, beam_width, seq_len)).copy()
        scores = np.full((batch_size, beam_width), -1e9, dtype=np.float32)
        scores[:, 0] = 0.0
        finished = np.zeros((batch_size, beam_width), dtype=bool)

        for _ in range(self.max_length - seq_len):
            if np.all(finished):
                break

            beam_logits = []
            for beam_idx in range(beam_width):
                cur_prompt = jnp.asarray(x_beams[:, beam_idx, :])
                out = self.score_fn(cur_prompt)
                if hasattr(out, 'shape') and out.ndim == 3:
                    out = out[:, -1, :]
                beam_logits.append(np.asarray(out))
            logits = np.stack(beam_logits, axis=1)  # (batch_size, beam_width, vocab_size)

            if repeat_penalty != 1.0:
                logits = np.asarray(self.apply_repeat_penalty(jnp.asarray(logits), jnp.asarray(x_beams), repeat_penalty))
            logits = logits / max(temperature, 1e-5)
            log_probs = np.asarray(jax.nn.log_softmax(jnp.asarray(logits), axis=-1))

            if np.any(finished):
                log_probs = np.where(finished[:, :, None], -1e9, log_probs)
                log_probs[:, :, eos_id] = np.where(finished, 0.0, log_probs[:, :, eos_id])

            total_scores = scores[:, :, None] + log_probs  # (batch_size, beam_width, vocab_size)
            vocab_size = log_probs.shape[-1]
            total_scores_flat = total_scores.reshape(batch_size, -1)

            topk_indices = np.argsort(total_scores_flat, axis=-1)[:, ::-1][:, :beam_width]
            topk_scores = np.take_along_axis(total_scores_flat, topk_indices, axis=-1)

            beam_indices = topk_indices // vocab_size
            next_tokens = topk_indices % vocab_size

            new_x_beams = np.take_along_axis(x_beams, beam_indices[:, :, None], axis=1)
            x_beams = np.concatenate([new_x_beams, next_tokens[:, :, None]], axis=-1)

            finished = np.take_along_axis(finished, beam_indices, axis=1) | (next_tokens == eos_id)
            scores = topk_scores

        sort_idx = np.argsort(scores, axis=-1)[:, ::-1]
        scores = np.take_along_axis(scores, sort_idx, axis=-1)
        x_beams = np.take_along_axis(x_beams, sort_idx[:, :, None], axis=1)
        return x_beams, scores

    # Alias
    decode_beam = generate_beam

    def post_process_sequence(
        self,
        seq: Union[jax.Array, np.ndarray],
        tokenizer: Optional[H3Tokenizer] = None
    ) -> np.ndarray:
        """
        Trims special tokens after EOS.
        
        Args:
            seq: Array of shape (L,) or (beam_width, L)
            tokenizer: Tokenizer instance
            
        Returns:
            np.ndarray: Trimmed token sequence(s)
        """
        tok = tokenizer or self.tokenizer
        eos_id = tok.eos_id
        sos_id = tok.sos_id
        pad_id = tok.pad_id

        seq_np = np.asarray(seq)

        if seq_np.ndim == 2:
            processed = []
            for s in seq_np:
                processed.append(self.post_process_sequence(s, tok))
            return np.array(processed, dtype=object)

        tokens = list(seq_np)
        # Strip SOS if at beginning
        if len(tokens) > 0 and tokens[0] == sos_id:
            tokens = tokens[1:]

        # Truncate at first EOS
        if eos_id in tokens:
            eos_idx = tokens.index(eos_id)
            tokens = tokens[:eos_idx]

        # Strip any trailing pads
        tokens = [t for t in tokens if t != pad_id]
        return np.array(tokens, dtype=np.int64)
