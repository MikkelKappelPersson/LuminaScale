"""Custom TensorBoard logger for proper hparams logging.

Prevents PyTorch Lightning's automatic early log_hyperparams call (which deletes hparams)
and provides log_hyperparams_metrics for explicit control during training lifecycle.

Based on: https://github.com/PyTorchLightning/pytorch-lightning/issues/5584
"""

from __future__ import annotations

from typing import Any

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class CustomTensorBoardLogger(TensorBoardLogger):
    """Custom TensorBoardLogger that disables automatic hparams logging.
    
    Issue: PyTorch Lightning calls log_hyperparams early (before training starts),
    which causes hparams to be deleted because there are no metrics yet.
    
    Solution: Override log_hyperparams to do nothing, use log_hyperparams_metrics
    explicitly when we're ready (on_train_start and on_train_epoch_end).
    """

    def log_hyperparams(self, params: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
        """Override to disable automatic hparams logging.
        
        PyTorch Lightning calls this too early (before metrics exist).
        We'll use log_hyperparams_metrics instead.
        """
        pass

    @rank_zero_only
    def log_hyperparams_metrics(
        self, params: dict[str, Any], metrics: dict[str, Any] | None = None
    ) -> None:
        """Explicitly log hparams + metrics when we're ready.
        
        Calls the parent TensorBoardLogger's log_hyperparams method directly
        with metrics populated. This ensures hparams are properly associated
        with the metrics in TensorBoard's hparams dashboard.
        
        Args:
            params: Hyperparameter dictionary
            metrics: Metrics dictionary (can be empty dict or None)
        """
        if metrics is None:
            metrics = {}
        
        # Call parent's log_hyperparams with both params and metrics
        # This properly writes hparams summary to TensorBoard
        TensorBoardLogger.log_hyperparams(self, params, metrics)
