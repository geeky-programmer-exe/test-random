import os
from functools import partial
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from ..data.tokenizer import H3Tokenizer
from ..decoding.sequence_generator import SequenceGenerator
from .base_trainer import BaseTrainer, AverageMeter

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


class ASRTrainer(BaseTrainer):
    """
    ASR Trainer managing training, validation, and speech transcription in JAX.
    Supports both PyTorch-style and JAX functional training loop configurations.
    """
    def __init__(
        self,
        arg1: Optional[Any] = None,
        arg2: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        config_file: Optional[str] = None,
        device: Optional[str] = None,
        model: Optional[nnx.Module] = None,
        tokenizer: Optional[H3Tokenizer] = None,
        optimizer: Optional[nnx.Optimizer] = None,
        **kwargs
    ):
        if isinstance(arg1, dict):
            cfg = arg1
            m = arg2 or model
            tok = tokenizer
        else:
            m = arg1 or model
            tok = arg2 or tokenizer
            cfg = config if config is not None else (kwargs.get('config') or {})

        super().__init__(arg1=m, arg2=tok, config=cfg, run_name=run_name, config_file=config_file, device=device, **kwargs)
        self.model = m
        self.optimizer = optimizer
        
        token_type = cfg.get('token_type', 'char')
        self.tokenizer = tok or H3Tokenizer(token_type=token_type)
        
        loss_cfg = cfg.get('loss', {}) if isinstance(cfg.get('loss'), dict) else {}
        self.ctc_weight = cfg.get('ctc_weight', loss_cfg.get('ctc_weight', 0.2))
        self.max_length = cfg.get('max_length', 200)

    def create_loss_pad_mask(self, sequences: jax.Array, lengths: jax.Array) -> jax.Array:
        """Create padding mask where 0.0 represents valid positions and 1.0 represents padding."""
        max_len = sequences.shape[1]
        indices = jnp.arange(max_len)[None, :]
        lens = lengths[:, None]
        return jnp.where(indices < lens, 0.0, 1.0)

    def ctc_and_ce_loss(
        self,
        model: nnx.Module,
        batch,
        training: bool = True
    ) -> jax.Array:
        """
        Compute joint CTC and Cross-Entropy loss.

        Accepts batch as either:
          - A tuple: (feats, shifted, golden, feat_lens, text_lens)  <- DataLoader default
          - A dict with keys 'spectrogram'/'features', 't_shifted'/'shifted', etc.
        """
        # TODO: Unpack features, shifted targets, golden targets, and lengths from the batch
        if isinstance(batch, (list, tuple)):
            # Standard collate_fn output: (padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths)
            feats, shifted, golden, feat_lens, text_lens = batch
            feats     = jnp.asarray(feats)
            shifted   = jnp.asarray(shifted)
            golden    = jnp.asarray(golden)
            feat_lens = jnp.asarray(feat_lens)
            text_lens = jnp.asarray(text_lens)
        else:
            feats     = batch.get('spectrogram', batch.get('features'))
            shifted   = batch.get('t_shifted',   batch.get('shifted'))
            feat_lens = batch.get('spectrogram_lengths', batch.get('feature_lengths'))
            text_lens = batch.get('transcript_lengths',  batch.get('text_lengths'))
            golden    = batch.get('t_golden', batch.get('golden'))

        # TODO: Run the model to get decoder logits and CTC inputs/logits
        model_out = model(
            feats, shifted,
            input_lengths=feat_lens,
            target_lengths=text_lens,
            return_ctc=True
        )
        if len(model_out) == 3:
            model_logits, _, ctc_data = model_out
        else:
            model_logits, ctc_data = model_out[0], model_out[1]

        # Extract log_probs and lengths if ctc_data is a dict (standard in transformer.encode)
        if isinstance(ctc_data, dict):
            ctc_log_probs_t = ctc_data.get('log_probs')
            # ctc_log_probs_t is shape (T, N, V) -> swap back to (N, T, V) for optax.losses.ctc_loss
            ctc_log_probs = jnp.swapaxes(ctc_log_probs_t, 0, 1) if ctc_log_probs_t is not None else None
            ctc_lens = ctc_data.get('lengths', feat_lens)
        else:
            ctc_log_probs = jax.nn.log_softmax(ctc_data, axis=-1) if ctc_data is not None else None
            ctc_lens = feat_lens

        # TODO: Build padding masks (0 on valid positions, 1 on pad) for the transcript and, for CTC, the frames
        transcript_paddings = self.create_loss_pad_mask(golden, text_lens)
        
        # TODO: Cross-entropy with optax.losses.softmax_cross_entropy_with_integer_labels, averaged over non-pad tokens
        ce_loss = optax.losses.softmax_cross_entropy_with_integer_labels(model_logits, golden)
        ce_loss_masked = jnp.where(transcript_paddings == 0.0, ce_loss, 0.0)
        total_valid = jnp.maximum(jnp.sum(transcript_paddings == 0.0), 1.0)
        batch_ce_loss = jnp.sum(ce_loss_masked) / total_valid

        # TODO: If self.ctc_weight > 0, CTC loss with optax.losses.ctc_loss (blank_id is the tokenizer blank)
        # TODO: Joint loss = (1 - ctc_weight) * ce_loss + ctc_weight * ctc_loss. Otherwise return the CE loss.
        if self.ctc_weight > 0.0 and ctc_log_probs is not None:
            logit_paddings = self.create_loss_pad_mask(ctc_log_probs, ctc_lens)
            ctc_loss = optax.losses.ctc_loss(
                logits=ctc_log_probs,
                logit_paddings=logit_paddings,
                labels=golden,
                label_paddings=transcript_paddings,
                blank_id=self.tokenizer.blank_id
            )
            batch_ctc_loss = jnp.mean(ctc_loss)
            loss = (1.0 - self.ctc_weight) * batch_ce_loss + self.ctc_weight * batch_ctc_loss
        else:
            loss = batch_ce_loss

        return loss

    @partial(nnx.jit, static_argnums=(0,))
    def _train_step_jit(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: Any
    ) -> jax.Array:
        """JIT-compiled forward, backward via nnx.value_and_grad, and optimizer parameter update."""
        # TODO: Define loss_fn(m) that returns self.ctc_and_ce_loss(m, batch, training=True)
        def loss_fn(m):
            return self.ctc_and_ce_loss(m, batch, training=True)

        # TODO: loss, grads = nnx.value_and_grad(loss_fn)(model)
        vg_fn = nnx.value_and_grad(loss_fn)
        loss, grads = vg_fn(model)
        # TODO: Update parameters with optimizer.update(model, grads) or optimizer.update(grads)
        try:
            optimizer.update(model, grads)
        except TypeError:
            optimizer.update(grads)
        # TODO: Return float(loss)
        return loss

    def train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: Any
    ) -> float:
        """Performs a single forward, backward via nnx.value_and_grad, and optimizer parameter update."""
        loss = self._train_step_jit(model, optimizer, batch)
        return float(loss)

    def train_epoch(
        self,
        model: Optional[nnx.Module] = None,
        optimizer: Optional[nnx.Optimizer] = None,
        dataloader = None,
        epoch: int = 1
    ) -> Dict[str, float]:
        m = model or self.model
        opt = optimizer or self.optimizer
        m.train()
        loss_meter = AverageMeter()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
        for batch in pbar:
            loss = self.train_step(m, opt, batch)
            loss_meter.update(loss)
            pbar.set_postfix({'loss': f"{loss:.4f}", 'avg': f"{loss_meter.avg:.4f}"})

        return {'train_loss': loss_meter.avg}

    # Alias for Torch parity
    _train_epoch = train_epoch

    @partial(nnx.jit, static_argnums=(0,))
    def _eval_step_jit(
        self,
        model: nnx.Module,
        batch: Any
    ) -> jax.Array:
        """JIT-compiled evaluation loss step."""
        return self.ctc_and_ce_loss(model, batch, training=False)

    def validate_epoch(
        self,
        model: Optional[nnx.Module] = None,
        dataloader = None,
        epoch: int = 1
    ) -> Dict[str, float]:
        m = model or self.model
        m.eval()
        loss_meter = AverageMeter()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [Val]", leave=False)
        for batch in pbar:
            loss = float(self._eval_step_jit(m, batch))
            loss_meter.update(loss)
            pbar.set_postfix({'val_loss': f"{loss:.4f}", 'avg': f"{loss_meter.avg:.4f}"})

        return {'val_loss': loss_meter.avg}

    # Alias for Torch parity
    _validate_epoch = validate_epoch

    def train(
        self,
        train_loader = None,
        val_loader = None,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        num_epochs = epochs or self.config.get('epochs', 10)
        history = {'train_loss': [], 'val_loss': []}

        for ep in range(1, num_epochs + 1):
            if train_loader is not None:
                tr_metrics = self.train_epoch(dataloader=train_loader, epoch=ep)
                history['train_loss'].append(tr_metrics['train_loss'])
            if val_loader is not None:
                val_metrics = self.validate_epoch(dataloader=val_loader, epoch=ep)
                history['val_loss'].append(val_metrics['val_loss'])
                self.log_metrics(ep, {**tr_metrics, **val_metrics})
            elif train_loader is not None:
                self.log_metrics(ep, tr_metrics)

        return history

    def recognize(
        self,
        arg1: Any = None,
        arg2: Any = None,
        config_name: Optional[str] = None,
        max_length: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """
        Transcribe speech features into text transcripts.
        Supports both:
          1. Student Notebook / Evaluation signature:
             recognize(dataloader, recognition_config, config_name='test', max_length=...)
             -> List[Dict[str, Any]] with keys 'generated', 'score'
          2. Direct tensor signature:
             recognize(model=..., features=..., feature_lengths=..., decode_type='greedy', beam_width=4)
             -> List[str]
        """
        # Case 1: Called with a dataloader
        if hasattr(arg1, '__iter__') and not isinstance(arg1, (jnp.ndarray, np.ndarray, nnx.Module)):
            dataloader = arg1
            recognition_config = arg2 or {}
            decode_type = 'beam' if recognition_config.get('beam_width', 1) > 1 else 'greedy'
            beam_width = recognition_config.get('beam_width', 4)
            temperature = recognition_config.get('temperature', 1.0)
            repeat_penalty = recognition_config.get('repeat_penalty', 1.0)
            max_len = max_length or self.max_length

            m = self.model
            m.eval()
            results = []

            pbar = tqdm(dataloader, desc=f"[Recognizing ASR] : {config_name or 'eval'}", leave=False)
            for i, batch in enumerate(pbar):
                if isinstance(batch, (list, tuple)):
                    feats, _, golden, feat_lens, text_lens = batch
                else:
                    feats = batch.get('spectrogram', batch.get('features'))
                    feat_lens = batch.get('spectrogram_lengths', batch.get('feature_lengths'))
                    golden = batch.get('t_golden', batch.get('golden'))

                feats = jnp.asarray(feats)
                feat_lens = jnp.asarray(feat_lens) if feat_lens is not None else None
                enc_res = m.encode(feats, feat_lens)
                if len(enc_res) == 4:
                    enc_out, pad_mask_src, _, ctc_data = enc_res
                    enc_lens = ctc_data.get('lengths', feat_lens) if isinstance(ctc_data, dict) else feat_lens
                else:
                    enc_out = enc_res[0]
                    pad_mask_src = enc_res[1] if len(enc_res) > 1 else None
                    enc_lens = feat_lens

                batch_size = feats.shape[0]

                def score_fn(seq):
                    dec_res = m.decode(seq, enc_out, pad_mask_src=pad_mask_src)
                    logits = dec_res[0] if isinstance(dec_res, tuple) else dec_res
                    return logits[:, -1, :]

                generator = SequenceGenerator(
                    score_fn=score_fn,
                    tokenizer=self.tokenizer,
                    max_length=max_len
                )

                init_prompts = jnp.full((batch_size, 1), self.tokenizer.sos_id, dtype=jnp.int32)
                if decode_type == 'beam':
                    beams, scores = generator.generate_beam(init_prompts, beam_width=beam_width, temperature=temperature, repeat_penalty=repeat_penalty)
                    best_seqs = beams[:, 0, :]
                    best_scores = scores[:, 0]
                else:
                    best_seqs, best_scores = generator.generate_greedy(init_prompts, temperature=temperature, repeat_penalty=repeat_penalty)

                for j, seq in enumerate(best_seqs):
                    clean = generator.post_process_sequence(seq, self.tokenizer)
                    text = self.tokenizer.decode(clean.tolist(), skip_special_tokens=True)
                    item = {
                        'generated': text,
                        'score': float(best_scores[j]) if best_scores is not None else 0.0
                    }
                    if golden is not None:
                        target_clean = generator.post_process_sequence(golden[j], self.tokenizer)
                        item['target'] = self.tokenizer.decode(target_clean.tolist(), skip_special_tokens=True)
                    results.append(item)

                if recognition_config.get('num_batches') is not None and i >= recognition_config['num_batches'] - 1:
                    break

            return results

        # Case 2: Called with model/tensor arguments directly
        m = (arg1 if isinstance(arg1, nnx.Module) else kwargs.get('model')) or self.model
        features = kwargs.get('features', arg1 if not isinstance(arg1, nnx.Module) else arg2)
        feature_lengths = kwargs.get('feature_lengths', arg2 if not isinstance(arg1, nnx.Module) else None)
        decode_type = kwargs.get('decode_type', 'greedy')
        beam_width = kwargs.get('beam_width', 4)

        m.eval()
        # TODO: Encode speech features with model.encode
        enc_res = m.encode(features, feature_lengths)
        if len(enc_res) == 4:
            enc_out, pad_mask_src, _, ctc_data = enc_res
            enc_lens = ctc_data.get('lengths', feature_lengths) if isinstance(ctc_data, dict) else feature_lengths
        else:
            enc_out, enc_lens = enc_res[0], enc_res[1]
            pad_mask_src = None

        batch_size = features.shape[0]

        # TODO: Define score_fn(seq) from decode, then take the last-step logits
        def score_fn(seq):
            dec_res = m.decode(seq, enc_out, pad_mask_src=pad_mask_src)
            logits = dec_res[0] if isinstance(dec_res, tuple) else dec_res
            return logits[:, -1, :]

        generator = SequenceGenerator(
            score_fn=score_fn,
            tokenizer=self.tokenizer,
            max_length=max_length or self.max_length
        )

        # TODO: Initialize prompts as a batch of SOS tokens
        init_prompts = jnp.full((batch_size, 1), self.tokenizer.sos_id, dtype=jnp.int64)
        # TODO: Generate sequences with beam search when decode_type == 'beam', otherwise greedy search
        if decode_type == 'beam':
            beams, _ = generator.generate_beam(init_prompts, beam_width=beam_width)
            best_seqs = beams[:, 0, :]
        else:
            best_seqs, _ = generator.generate_greedy(init_prompts)

        # TODO: Post-process each sequence and decode it to text with the tokenizer
        transcripts = []
        for seq in best_seqs:
            clean = generator.post_process_sequence(seq, self.tokenizer)
            transcripts.append(self.tokenizer.decode(clean.tolist(), skip_special_tokens=True))
        return transcripts

