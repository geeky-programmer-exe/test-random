from typing import Dict, Any, Optional, Union
import optax
from flax import nnx

def create_optimizer(
    arg1: Union[Dict[str, Any], nnx.Module, None] = None,
    arg2: Union[Dict[str, Any], nnx.Module, None] = None,
    schedule: Optional[optax.Schedule] = None,
    *,
    # Explicit keyword aliases — match the notebook calling style:
    #   create_optimizer(model=model, arg2=config['optimizer'])
    # or the fully-explicit style:
    #   create_optimizer(model=model, config=config['optimizer'])
    model: Optional[nnx.Module] = None,
    config: Optional[Dict[str, Any]] = None,
) -> nnx.Optimizer:
    """
    Creates an Optax optimizer wrapped with Flax NNX Optimizer.

    Supports multiple calling conventions (all equivalent):

    Positional (model-first):
        create_optimizer(model, config_dict)

    Positional (config-first):
        create_optimizer(config_dict, model)

    Keyword (notebook style):
        create_optimizer(model=model, arg2=config['optimizer'])
        create_optimizer(model=model, config=config['optimizer'])

    Args:
        arg1: Model or config dict (positional slot 1).
        arg2: Config dict or model (positional slot 2).
        schedule: Optional pre-built optax schedule to use as lr.
        model: Explicit model keyword argument.
        config: Explicit config keyword argument.

    Returns:
        nnx.Optimizer wrapping the model parameters.
    """
    # Resolve model from keyword args first, then positional slots
    _model = model
    _config = config

    if _model is None and _config is None:
        # Pure positional call
        if isinstance(arg1, nnx.Module):
            _model = arg1
            _config = arg2 if isinstance(arg2, dict) else {}
        elif isinstance(arg2, nnx.Module):
            _model = arg2
            _config = arg1 if isinstance(arg1, dict) else {}
        else:
            raise TypeError(
                "create_optimizer: could not resolve model. "
                "Pass model as first positional arg or as model=<model>."
            )
    elif _model is None:
        # config= was set explicitly; model must be in arg1/arg2
        if isinstance(arg1, nnx.Module):
            _model = arg1
        elif isinstance(arg2, nnx.Module):
            _model = arg2
        else:
            raise TypeError("create_optimizer: model not found in positional args.")
    elif _config is None:
        # model= was set explicitly; config must be in arg1/arg2 or arg2 (notebook style)
        if isinstance(arg2, dict):
            _config = arg2
        elif isinstance(arg1, dict):
            _config = arg1
        else:
            _config = {}

    lr = schedule if schedule is not None else _config.get('lr', 1e-4)
    weight_decay = _config.get('weight_decay', 0.01)
    grad_clip = _config.get('grad_clip_amount', 1.0)
    grad_accum_steps = _config.get('gradient_accumulation_steps', 1)

    opt_chain = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    )

    if grad_accum_steps > 1:
        opt_chain = optax.MultiSteps(opt_chain, every_k_schedule=grad_accum_steps)

    return nnx.Optimizer(_model, opt_chain, wrt=nnx.Param)
