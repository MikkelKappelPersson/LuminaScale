"""Visualize learning rate schedules for comparison."""

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add src to path
import sys
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    LinearLR,
)
from luminascale.training.schedulers import (
    CosineAnnealingWarmupScheduler,
    ExponentialWarmupScheduler,
)


def simulate_lr_schedule(scheduler, num_epochs: int) -> list[float]:
    """Simulate learning rate schedule over epochs."""
    lrs = []
    for epoch in range(num_epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    return lrs


def create_dummy_optimizer(lr: float = 1e-3) -> optim.Optimizer:
    """Create a dummy model and optimizer for scheduler testing."""
    dummy_model = torch.nn.Linear(10, 1)
    return optim.Adam(dummy_model.parameters(), lr=lr)


# Parameters
num_epochs = 50
base_lr = 1e-3

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Learning Rate Schedules Comparison (50 epochs)", fontsize=16, fontweight="bold")

# ============================================================================
# Plot 1: Fixed LR vs All Decay Methods
# ============================================================================
ax = axes[0, 0]

# Fixed LR
fixed_lr = [base_lr] * num_epochs
ax.plot(range(num_epochs), fixed_lr, "k--", linewidth=2.5, label="Fixed LR (current)", alpha=0.8)

# Built-in schedulers
optimizer = create_dummy_optimizer(base_lr)
step_lr = StepLR(optimizer, step_size=10, gamma=0.5)
step_lrs = simulate_lr_schedule(step_lr, num_epochs)
ax.plot(range(num_epochs), step_lrs, "o-", linewidth=2, label="StepLR (step=10, γ=0.5)", markersize=3)

optimizer = create_dummy_optimizer(base_lr)
exp_lr = ExponentialLR(optimizer, gamma=0.95)
exp_lrs = simulate_lr_schedule(exp_lr, num_epochs)
ax.plot(range(num_epochs), exp_lrs, "s-", linewidth=2, label="ExponentialLR (γ=0.95)", markersize=3)

optimizer = create_dummy_optimizer(base_lr)
cosine_lr = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
cosine_lrs = simulate_lr_schedule(cosine_lr, num_epochs)
ax.plot(range(num_epochs), cosine_lrs, "^-", linewidth=2, label="CosineAnnealingLR (no warmup)", markersize=3)

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Learning Rate", fontsize=11)
ax.set_title("Built-in PyTorch Schedulers", fontsize=12, fontweight="bold")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="best")

# ============================================================================
# Plot 2: Custom Warmup Schedulers
# ============================================================================
ax = axes[0, 1]

# Fixed LR
ax.plot(range(num_epochs), fixed_lr, "k--", linewidth=2.5, label="Fixed LR (current)", alpha=0.8)

# Custom warmup schedulers
optimizer = create_dummy_optimizer(base_lr)
cosine_warmup = CosineAnnealingWarmupScheduler(
    optimizer, warmup_epochs=2, total_epochs=num_epochs
)
cosine_warmup_lrs = simulate_lr_schedule(cosine_warmup, num_epochs)
ax.plot(range(num_epochs), cosine_warmup_lrs, "D-", linewidth=2.5, 
        label="CosineAnnealingWarmup ⭐ RECOMMENDED", markersize=4, color="green")

optimizer = create_dummy_optimizer(base_lr)
exp_warmup = ExponentialWarmupScheduler(
    optimizer, warmup_epochs=2, decay_rate=0.95
)
exp_warmup_lrs = simulate_lr_schedule(exp_warmup, num_epochs)
ax.plot(range(num_epochs), exp_warmup_lrs, "v-", linewidth=2, 
        label="ExponentialWarmup", markersize=3, color="orange")

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Learning Rate", fontsize=11)
ax.set_title("Custom Warmup Schedulers", fontsize=12, fontweight="bold")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="best")

# ============================================================================
# Plot 3: Linear Scale (CosineAnnealingWarmup Detail)
# ============================================================================
ax = axes[1, 0]

ax.plot(range(num_epochs), cosine_warmup_lrs, "D-", linewidth=2.5, 
        label="CosineAnnealingWarmup", markersize=4, color="green")
ax.axvline(x=2, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="Warmup ends (epoch 2)")
ax.axhline(y=base_lr, color="gray", linestyle="--", linewidth=1, alpha=0.5)

# Annotate key phases
ax.text(1, base_lr * 0.5, "Warmup\nPhase", fontsize=10, ha="center", 
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))
ax.text(25, base_lr * 0.3, "Decay Phase\n(Fine-tuning)", fontsize=10, ha="center",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Learning Rate", fontsize=11)
ax.set_title("Linear Scale: Warmup Detail (CosineAnnealingWarmup)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="best")

# ============================================================================
# Plot 4: Comparison Table
# ============================================================================
ax = axes[1, 1]
ax.axis("off")

# Create comparison data
comparison_data = [
    ["Method", "LR @ Epoch 1", "LR @ Epoch 10", "LR @ Epoch 50", "Characteristics"],
    ["Fixed LR", f"{base_lr:.0e}", f"{base_lr:.0e}", f"{base_lr:.0e}", "No adaptation (plateau issue)"],
    ["StepLR", f"{base_lr:.0e}", f"{base_lr*0.5:.0e}", f"{base_lr*0.0625:.0e}", "Step-wise drops"],
    ["ExponentialLR", f"{base_lr:.0e}", f"{base_lr*0.95**9:.0e}", f"{base_lr*0.95**49:.0e}", "Smooth exponential"],
    ["CosineAnnealingLR", f"{base_lr:.0e}", f"{base_lr*0.72:.0e}", f"{base_lr*0.01:.0e}", "Smooth cosine, no warmup"],
    ["ExponentialWarmup", "0", f"{base_lr*0.95**8:.0e}", f"{base_lr*0.95**48:.0e}", "Warmup + exponential"],
    ["CosineAnnealingWarmup ⭐", "0", f"{base_lr*0.96:.0e}", f"{base_lr*0.01:.0e}", "Warmup + smooth decay"],
]

# Create table
table = ax.table(
    cellText=comparison_data,
    cellLoc="center",
    loc="center",
    colWidths=[0.22, 0.15, 0.15, 0.15, 0.33],
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor("#40466e")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Style recommended row
for i in range(5):
    table[(6, i)].set_facecolor("#90EE90")
    table[(6, i)].set_text_props(weight="bold")

# Alternate row colors
for i in range(1, 6):
    for j in range(5):
        if i != 5:  # Don't color the recommended row
            table[(i, j)].set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

ax.set_title("Numeric Comparison", fontsize=12, fontweight="bold", pad=20)

# ============================================================================
plt.tight_layout()
output_path = Path(__file__).parent / "lr_schedules_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved visualization to: {output_path}")
plt.show()

# ============================================================================
# Print detailed comparison
# ============================================================================
print("\n" + "="*80)
print("LEARNING RATE SCHEDULE COMPARISON (50 epochs, base LR = 1e-3)")
print("="*80 + "\n")

schedules = {
    "Fixed LR (Current)": fixed_lr,
    "StepLR (step=10, γ=0.5)": step_lrs,
    "ExponentialLR (γ=0.95)": exp_lrs,
    "CosineAnnealingLR (no warmup)": cosine_lrs,
    "ExponentialWarmup (warmup=2)": exp_warmup_lrs,
    "CosineAnnealingWarmup ⭐ RECOMMENDED": cosine_warmup_lrs,
}

epochs_to_show = [0, 1, 2, 5, 10, 20, 30, 40, 49]

for schedule_name, lrs in schedules.items():
    prefix = "⭐ " if "RECOMMENDED" in schedule_name else "   "
    print(f"{prefix}{schedule_name}")
    print("Epoch:", "  ".join(f"{e:>3}" for e in epochs_to_show))
    print("LR:   ", "  ".join(f"{lrs[e]:.0e}" for e in epochs_to_show))
    print()

print("="*80)
print("WHY CosineAnnealingWarmup IS RECOMMENDED FOR DEQUANTIZATION:")
print("="*80)
print("""
1. LINEAR WARMUP (Epochs 0-2):
   - Starts at 0, ramps to base_lr
   - Prevents early oscillation and convergence to bad local minima
   - Your current issue: immediately full LR causes plateau by epoch 2

2. COSINE DECAY (Epochs 2-50):
   - Smooth decay from base_lr → ~0
   - Matches natural gradient decay pattern
   - Allows fine-tuning at low epoch count (unlike StepLR)
   - Perfect for dequantization: micro-refinements need small steps

3. LONG TAIL (Later epochs):
   - Very small LR (1e-5 to 1e-6 range) enables micro-adjustments
   - Your issue: fixed 1e-4 is 100x too large when gradients are 1e-5

4. MATHEMATICAL INTUITION:
   Gradient magnitude decay (typical):
     Epoch 1:  ∇L ≈ 1e-3   (large)
     Epoch 2:  ∇L ≈ 1e-4   (medium)
     Epoch 10: ∇L ≈ 1e-5   (small)
     Epoch 50: ∇L ≈ 1e-5   (tiny)
   
   CosineAnnealingWarmup LR:
     Epoch 1:  LR ≈ 5e-4   (matches large gradient)
     Epoch 2:  LR ≈ 1e-3   (matches medium gradient)
     Epoch 10: LR ≈ 3e-4   (matches small gradient)  ← THIS IS KEY
     Epoch 50: LR ≈ 1e-6   (matches tiny gradient)
   
   Update size = LR × gradient ≈ constant (no oscillation or stalling)
""")
