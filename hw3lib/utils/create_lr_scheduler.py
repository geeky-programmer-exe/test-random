from typing import Dict, Any, Callable
import optax

def create_scheduler(config: Dict[str, Any]) -> optax.Schedule:
    """
    Creates an Optax learning rate schedule based on configuration parameters.
    
    Args:
        config (dict): Configuration containing epochs, lr, pct_start, etc.
        
    Returns:
        optax.Schedule: Learning rate schedule function mapping step -> lr.
    """
    total_steps = config.get('total_steps', 1000)
    lr = config.get('lr', 1e-4)
    pct_start = config.get('pct_start', 0.2)
    div_factor = config.get('div_factor', 25.0)
    final_div_factor = config.get('final_div_factor', 1000.0)

    schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )
    return schedule

def create_warmup_scheduler(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0
) -> optax.Schedule:
    """Creates a warmup cosine decay schedule."""
    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_value
    )

def plot_lr_schedule(
    scheduler: Any,
    total_steps: int = 1000,
    num_epochs: Any = None
):
    """
    Plots the Optax learning rate schedule over steps or simulated epochs.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.arange(total_steps)
    if callable(scheduler):
        lrs = [float(scheduler(s)) for s in steps]
    else:
        lrs = [1e-4] * total_steps

    plt.figure(figsize=(10, 4))
    if num_epochs is not None and num_epochs > 0:
        x = np.linspace(0, num_epochs, total_steps)
        plt.plot(x, lrs, label='Learning Rate', color='#1f77b4', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
    else:
        plt.plot(steps, lrs, label='Learning Rate', color='#1f77b4', linewidth=2)
        plt.xlabel('Step', fontsize=12)

    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule (Optax)', fontsize=14, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
