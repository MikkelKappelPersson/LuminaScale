"""Learning rate scheduling utilities for training."""

from __future__ import annotations

import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class CosineAnnealingWarmupScheduler(LambdaLR):
    """Cosine annealing with linear warmup.
    
    Warms up linearly for first `warmup_epochs`, then follows cosine decay.
    Good for escaping early plateaus and achieving better convergence.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 2,
        total_epochs: int = 50,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for linear warmup
            total_epochs: Total training epochs
            last_epoch: Index of last epoch (for resuming)
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup from 0 to 1
                return float(epoch) / float(max(1, warmup_epochs))
            else:
                # Cosine decay from 1 to 0
                progress = float(epoch - warmup_epochs) / float(
                    max(1, total_epochs - warmup_epochs)
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
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
