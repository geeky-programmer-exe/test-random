from typing import Dict, Any, Optional, Union
import optax
from flax import nnx

def create_optimizer(
    arg1: Union[Dict[str, Any], nnx.Module],
    arg2: Union[Dict[str, Any], nnx.Module],
    schedule: Optional[optax.Schedule] = None
) -> nnx.Optimizer:
    """
    Creates an Optax optimizer and wraps it with Flax NNX Optimizer.
    Accepts either (config, model) or (model, config) for seamless compatibility.
    
    Args:
        arg1: Either configuration dictionary or model instance.
        arg2: Either model instance or configuration dictionary.
        schedule (optax.Schedule, optional): Learning rate schedule.
        
    Returns:
        nnx.Optimizer: NNX optimizer instance.
    """
    if isinstance(arg1, nnx.Module):
        model = arg1
        config = arg2 if isinstance(arg2, dict) else {}
    else:
        config = arg1 if isinstance(arg1, dict) else {}
        model = arg2

    lr = schedule if schedule is not None else config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    grad_clip = config.get('grad_clip_amount', 1.0)
    grad_accum_steps = config.get('gradient_accumulation_steps', 1)

    opt_chain = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    )

    if grad_accum_steps > 1:
        opt_chain = optax.MultiSteps(opt_chain, every_k_schedule=grad_accum_steps)

    return nnx.Optimizer(model, opt_chain, wrt=nnx.Param)
