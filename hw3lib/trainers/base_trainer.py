import os
import gc
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
from flax import nnx

# Platform-safe memory allocation defaults (on-demand allocation prevents GPU OOM)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

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
        # Automatically detect device accelerator: 'gpu' (CUDA), 'tpu', or fallback to 'cpu'
        self.device = device or jax.default_backend()
        self.save_dir = self.config.get('save_directory', os.path.join('checkpoints', self.run_name))
        os.makedirs(self.save_dir, exist_ok=True)

        # Weights & Biases initialization
        training_cfg = self.config.get('training', {})
        self.use_wandb = training_cfg.get('use_wandb', False)
        if isinstance(self.use_wandb, str):
            self.use_wandb = self.use_wandb.lower() in ('true', '1', 'yes')

        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb
                project = training_cfg.get('wandb_project', 'hw3p2-jax-asr')
                run_id = training_cfg.get('wandb_run_id', None)
                entity = training_cfg.get('wandb_entity', training_cfg.get('entity', None))
                if run_id and str(run_id).lower() != 'none':
                    self.wandb_run = wandb.init(
                        project=project,
                        id=str(run_id),
                        resume='allow',
                        entity=entity,
                        config=self.config
                    )
                else:
                    self.wandb_run = wandb.init(
                        project=project,
                        entity=entity,
                        config=self.config,
                        name=self.run_name
                    )
                print(f"[W&B] Initialized run: {self.wandb_run.name} ({self.wandb_run.url if hasattr(self.wandb_run, 'url') else 'active'})")
            except Exception as e:
                print(f"[W&B] Warning: wandb.init failed: {e}")

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Format metrics cleanly for stdout and log to Weights & Biases."""
        parts = [f"Epoch {epoch:03d}"]
        for k, v in metrics.items():
            parts.append(f"{k}: {v:.4f}")
        print(" | ".join(parts))

        # Log to wandb if enabled
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({**metrics, 'epoch': epoch}, step=epoch)
            except Exception as e:
                pass

    def cleanup(self):
        """Cleanup trainer resources, release device memory, and finish logging."""
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass
