from __future__ import annotations

from typing import Any

from pytorch_lightning.callbacks import RichProgressBar
from rich.progress import TextColumn, ProgressColumn, Task
from rich.text import Text


class StepRateColumn(ProgressColumn):
    """Renders the step rate (it/s divided by batch size)."""

    def __init__(self, batch_size: int, table_column: Any = None) -> None:
        super().__init__(table_column=table_column)
        self.batch_size = batch_size

    def render(self, task: Task) -> Text:
        """Show the step rate."""
        speed = task.finished_speed or task.speed
        if speed is None or self.batch_size == 0:
            return Text("- step/s", style="progress.data.speed")
        
        step_rate = speed * self.batch_size
        return Text(f"{step_rate:.2f} step/s", style="progress.data.speed")


class CustomRichProgressBar(RichProgressBar):
    """Custom Rich progress bar that includes a step/s column."""

    def __init__(self, batch_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def configure_columns(self, trainer: Any) -> list[Any]:
        """Add the StepRateColumn to the default columns."""
        columns = super().configure_columns(trainer)
        # Find where to insert. Usually we want it near the it/s (ProcessingSpeedColumn)
        # For simplicity, we'll append it or insert it before the metrics.
        columns.insert(-1, StepRateColumn(batch_size=self.batch_size))
        return columns
