#!/usr/bin/env python
"""Test scheduler to identify the oscillation issue."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import torch.optim as optim
from luminascale.training.schedulers import CosineAnnealingWarmupStepScheduler

# Create a simple model and optimizer
model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Setup scheduler
total_steps = 200  # Simulate 200 training steps
warmup_steps = max(10, int(total_steps * 0.1))  # 20 steps warmup

print(f"\nScheduler Configuration:")
print(f"  Base LR: 1e-4")
print(f"  Total Steps: {total_steps}")
print(f"  Warmup Steps: {warmup_steps}")
print(f"  Decay Steps: {total_steps - warmup_steps}")
print(f"  Eta Min: 0.01 (target 1e-6)")
print()

scheduler = CosineAnnealingWarmupStepScheduler(
    optimizer,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    eta_min=0.01,
    debug=True,
)

print("\n" + "="*80)
print("Testing step-by-step execution:")
print("="*80)

# Simulate training loop
for step in range(total_steps):
    # Get learning rate BEFORE step
    lr_before = optimizer.param_groups[0]["lr"]
    
    # Call scheduler.step() to advance to next LR
    scheduler.step()
    
    # Get learning rate AFTER step
    lr_after = optimizer.param_groups[0]["lr"]
    
    # Print detailed output every 10 steps
    if step % 10 == 0 or step < 25 or step > total_steps - 25:
        print(f"Step {step:3d}: LR_before={lr_before:.2e} → LR_after={lr_after:.2e}")

print("\n" + "="*80)
print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
print(f"Expected final LR: ~1e-6")
print("="*80)
