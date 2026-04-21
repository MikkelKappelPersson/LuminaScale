"""Weight scheduler callback for decaying loss weights during training.

Allows exponential, linear, step-based, or cosine decay of individual loss weights
(e.g., charbonnier, gradient matching, TV) over the course of training.

Initial values are taken from the loss config; only minimums are specified in weight_schedule.

Usage:
    callback = WeightSchedulerCallback(
        loss_config={
            "charbonnier_weight": 3.0,
            "grad_match_weight": 2.0,
        },
        weight_schedule_config={
            "charbonnier_weight_min": 0.01,
            "grad_match_weight_min": 0.01,
            "decay_schedule": "exponential",
        },
        num_epochs=50,
    )
    trainer.callbacks.append(callback)
"""

from __future__ import annotations

from typing import Literal, Any

import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback
import numpy as np


class WeightSchedulerCallback(Callback):
    """Decay loss weights during training using various schedules.
    
    Automatically extracts initial values from loss config and minimum values from weight_schedule config.
    Decay curves are automatically calculated based on initial/minimum values.
    
    Supports:
    - exponential: weight follows exponential decay curve from initial to minimum
    - linear: weight interpolates linearly from initial to minimum
    - step: weight drops by fixed factor every step_size epochs
    - cosine: weight follows smooth cosine decay curve from initial to minimum
    """
    
    def __init__(
        self,
        loss_config: dict[str, Any] | None = None,
        weight_schedule_config: dict[str, Any] | None = None,
        num_epochs: int = 50,
    ) -> None:
        """Initialize weight scheduler.
        
        Args:
            loss_config: Loss weights dict from config (e.g., {"charbonnier_weight": 3.0}).
                        Initial weights are extracted from here.
            weight_schedule_config: Weight schedule dict from config containing:
                        - Minimum values: "{weight_name}_min" (e.g., "charbonnier_weight_min": 0.01)
                        - decay_schedule: "exponential", "linear", "step", or "cosine"
                        - step_size: (optional) for "step" schedule
            num_epochs: Total number of training epochs.
        """
        super().__init__()
        self.loss_config = loss_config or {}
        self.weight_schedule_config = weight_schedule_config or {}
        self.num_epochs = num_epochs
        
        # Extract decay schedule and step_size
        self.decay_schedule = self.weight_schedule_config.get("decay_schedule", "exponential")
        self.step_size = self.weight_schedule_config.get("step_size", 10)
        
        # Validate schedule
        if self.decay_schedule not in ["exponential", "linear", "step", "cosine"]:
            raise ValueError(f"Unknown decay schedule: {self.decay_schedule}")
        
        # Build initial_weights and minimum_weights from config
        self.initial_weights: dict[str, float] = {}
        self.minimum_weights: dict[str, float] = {}
        
        # Look for {weight_name}_min in weight_schedule_config
        for key, min_value in self.weight_schedule_config.items():
            if key.endswith("_min"):
                weight_name = key[:-4]  # Remove "_min" suffix
                
                # Get initial value from loss config
                if weight_name not in self.loss_config:
                    raise ValueError(
                        f"Weight '{weight_name}' has minimum '{key}' in weight_schedule, "
                        f"but no initial value in loss config. "
                        f"Please add '{weight_name}' to loss weights."
                    )
                
                initial_value = self.loss_config[weight_name]
                
                # Skip weights that are already zero (nothing to decay)
                if initial_value <= 0:
                    continue
                
                self.initial_weights[weight_name] = initial_value
                self.minimum_weights[weight_name] = min_value
    
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Update weights at the start of each epoch."""
        if not self.initial_weights:
            return
        
        current_epoch = trainer.current_epoch
        
        # Update each weight in the module
        for weight_name, initial_value in self.initial_weights.items():
            if hasattr(pl_module, weight_name):
                min_value = self.minimum_weights.get(weight_name, 0.0)
                new_value = self._compute_weight(weight_name, current_epoch, initial_value, min_value)
                
                setattr(pl_module, weight_name, new_value)
                
                # Log to TensorBoard
                if hasattr(trainer, "logger") and trainer.logger is not None:
                    try:
                        trainer.logger.experiment.add_scalar(
                            f"weight_schedule/{weight_name}",
                            new_value,
                            global_step=current_epoch
                        )
                    except Exception:
                        pass  # Silently ignore logging errors
    
    def _compute_weight(
        self, weight_name: str, epoch: int, initial: float, minimum: float
    ) -> float:
        """Compute weight value for the given epoch.
        
        Automatically calculates the appropriate decay curve based on the schedule type,
        ensuring the weight transitions from initial to minimum over num_epochs.
        
        Args:
            weight_name: Name of the weight (for debugging)
            epoch: Current epoch number (0-indexed)
            initial: Initial (starting) weight value from loss config
            minimum: Minimum (floor) weight value from weight_schedule config
        
        Returns:
            Weight value for this epoch, clamped to [minimum, initial]
        """
        if self.decay_schedule == "exponential":
            # Calculate decay_rate to reach minimum at last epoch
            # Formula: minimum = initial * (decay_rate ^ (num_epochs - 1))
            # Solving: decay_rate = (minimum / initial) ^ (1 / (num_epochs - 1))
            if self.num_epochs <= 1 or initial <= 0 or initial == minimum:
                return minimum
            
            decay_rate = (minimum / initial) ** (1.0 / (self.num_epochs - 1))
            value = initial * (decay_rate ** epoch)
        
        elif self.decay_schedule == "linear":
            # Linear interpolation from initial to minimum
            # At epoch 0: value = initial
            # At epoch (num_epochs-1): value = minimum
            progress = epoch / max(1, self.num_epochs - 1)
            value = initial + (minimum - initial) * progress
        
        elif self.decay_schedule == "step":
            # Calculate step decay rate to reach minimum at the last step
            # Drops by fixed factor every step_size epochs
            num_steps = epoch // self.step_size
            num_total_steps = max(1, (self.num_epochs - 1) // self.step_size + 1)
            
            if num_total_steps <= 1 or initial <= 0 or initial == minimum:
                step_decay_rate = 1.0
            else:
                step_decay_rate = (minimum / initial) ** (1.0 / num_total_steps)
            
            value = initial * (step_decay_rate ** num_steps)
        
        elif self.decay_schedule == "cosine":
            # Cosine annealing from initial to minimum
            # Uses smooth cosine curve for gentle, continuous decay
            # At epoch 0: value = initial
            # At epoch (num_epochs-1): value = minimum
            progress = epoch / max(1, self.num_epochs - 1)
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            value = minimum + (initial - minimum) * cosine_factor
        
        else:
            value = initial
        
        # Clamp to [minimum, initial] to handle any numerical issues
        return max(minimum, min(initial, value))
