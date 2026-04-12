"""Learning rate scheduling utilities for training."""

from __future__ import annotations

import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class CosineAnnealingWarmupScheduler(LambdaLR):
    """Cosine annealing with linear warmup and minimum learning rate floor.
    
    Warms up linearly for first `warmup_epochs`, then follows cosine decay to eta_min.
    Good for escaping early plateaus and achieving better convergence.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 2,
        total_epochs: int = 50,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for linear warmup
            total_epochs: Total training epochs
            eta_min: Minimum learning rate (as fraction of base_lr)
                     e.g., eta_min=1e-6 means LR decays down to 1e-6 * base_lr
            last_epoch: Index of last epoch (for resuming)
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup from 0 to 1
                return float(epoch) / float(max(1, warmup_epochs))
            else:
                # Cosine decay from 1 to eta_min
                progress = float(epoch - warmup_epochs) / float(
                    max(1, total_epochs - warmup_epochs)
                )
                # Cosine decays from 1 to -1 as progress goes from 0 to 1
                # Scale to decay from 1 to eta_min instead of 1 to 0
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Map [1, 0] → [1, eta_min]
                return eta_min + (1.0 - eta_min) * cosine_decay
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class ExponentialWarmupScheduler(LambdaLR):
    """Exponential decay with linear warmup.
    
    More aggressive than cosine; useful for quickly escaping local minima.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 2,
        decay_rate: float = 0.95,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for linear warmup
            decay_rate: Multiplicative decay per epoch after warmup
            last_epoch: Index of last epoch (for resuming)
        """
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup from 0 to 1
                return float(epoch) / float(max(1, warmup_epochs))
            else:
                # Exponential decay
                steps_after_warmup = epoch - warmup_epochs
                return self.decay_rate ** steps_after_warmup
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class CosineAnnealingWarmupStepScheduler(LambdaLR):
    """Cosine annealing with linear warmup, step-based (for batch updates).
    
    Works with step-level updates (every batch) instead of epoch-level.
    Provides smooth continuous curve in TensorBoard.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int = 100,
        total_steps: int = 5000,
        eta_min: float = 1e-6,
        last_step: int = -1,
        debug: bool = True,
    ) -> None:
        """Initialize step-based scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of steps for linear warmup
            total_steps: Total training steps
            eta_min: Minimum learning rate (as fraction of base_lr)
            last_step: Index of last step (for resuming)
            debug: Enable debug logging to console
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.debug = debug
        self.base_lr = optimizer.defaults['lr']
        
        if self.debug:
            print(f"\n[CosineAnnealingWarmupStepScheduler] Initialization:")
            print(f"  base_lr: {self.base_lr}")
            print(f"  warmup_steps: {warmup_steps}")
            print(f"  total_steps: {total_steps}")
            print(f"  eta_min: {eta_min}")
            print(f"  decay_steps: {total_steps - warmup_steps}")
            print(f"  target_lr (at end): {self.base_lr * eta_min:.2e}\n")
        
        def lr_lambda(step: int) -> float:
            lr_multiplier = 0.0
            
            if step < warmup_steps:
                # Linear warmup from 0 to 1
                lr_multiplier = float(step) / float(max(1, warmup_steps))
                phase = "WARMUP"
            else:
                # Cosine decay from 1 to eta_min
                progress = float(step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                # Clamp progress to [0, 1] to avoid going past the endpoint
                progress = min(1.0, max(0.0, progress))
                # Cosine decays from 1 to -1 as progress goes from 0 to 1
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Map [1, 0] → [1, eta_min]
                lr_multiplier = eta_min + (1.0 - eta_min) * cosine_decay
                phase = "DECAY"
            
            # Debug output every N steps
            if self.debug and step % 100 == 0:
                actual_lr = self.base_lr * lr_multiplier
                print(f"[Step {step:5d}] {phase:6s} | LR_mult={lr_multiplier:.6f} | LR={actual_lr:.2e}")
            
            return lr_multiplier
        
        super().__init__(optimizer, lr_lambda, last_step)
