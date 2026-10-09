import os
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
from flax import nnx

class AverageMeter:
    """Tracks and computes the running average and current value of a scalar metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

class BaseTrainer:
    """
    Base trainer providing checkpointing and metrics logging for JAX models.
    Supports both Torch-style (model, tokenizer, config, ...) and JAX-style (config, ...) initialization.
    """
    def __init__(
        self,
        arg1: Optional[Any] = None,
        arg2: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        config_file: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        if isinstance(arg1, dict):
            self.config = arg1
            self.model = arg2
            self.tokenizer = None
        else:
            self.model = arg1
            self.tokenizer = arg2
            self.config = config if config is not None else (kwargs.get('config') or {})

        self.run_name = run_name or self.config.get('run_name', 'default_run')
        self.config_file = config_file
        self.device = device or 'cpu'
        self.save_dir = self.config.get('save_directory', os.path.join('checkpoints', self.run_name))
        os.makedirs(self.save_dir, exist_ok=True)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Format metrics cleanly for stdout."""
        parts = [f"Epoch {epoch:03d}"]
        for k, v in metrics.items():
            parts.append(f"{k}: {v:.4f}")
        print(" | ".join(parts))
