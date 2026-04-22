"""Base LightningModule for all LuminaScale trainers."""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as L
import torch
import torch.nn as nn
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

class BaseLuminaScaleTrainer(L.LightningModule):
    """Shared logic for all LuminaScale training tasks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        # Shared initialization logic (e.g., logging setup, throughput tracking)
        pass

    def on_train_start(self) -> None:
        """Shared hook for training start."""
        if self.logger and hasattr(self.logger, "log_hyperparams"):
            # Ensure we don't double-log if already handled by training script
            pass
