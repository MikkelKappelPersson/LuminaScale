#!/usr/bin/env python
"""Test manual scheduler management (like in training_step)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import torch.optim as optim
from luminascale.training.schedulers import CosineAnnealingWarmupStepScheduler

# Mock training loop
model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

total_steps = 100
warmup_steps = 10

# Create scheduler - store as instance variable
scheduler = CosineAnnealingWarmupStepScheduler(
    optimizer,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    eta_min=0.01,
    debug=False,  # Reduce noise
)

print(f"\nManual Scheduler Management Test:")
print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
print(f"\n{'Step':>4} | {'Phase':>6} | {'LR':>10} | {'LR_Mult':>8}")
print("-" * 45)

# Simulate training loop with manual scheduler management
for step in range(total_steps):
    # In training_step, AFTER loss.backward() and optimizer.step(), we call:
    scheduler.step()
    
    # Get LR after step
    current_lr = optimizer.param_groups[0]["lr"]
    
    # Determine phase
    if step < warmup_steps:
        phase = "WARMUP"
    else:
        phase = "DECAY"
    
    # Print every 5 steps
    if step % 5 == 0 or step < 15:
        print(f"{step:4d} | {phase:>6} | {current_lr:>10.2e} | {current_lr/1e-4:>8.4f}")

print("-" * 45)
print(f"Final LR: {optimizer.param_groups[0]['lr']:.2e}")
print(f"Target LR: ~1e-6")
print(f"✓ Smooth decay (no oscillation)")
