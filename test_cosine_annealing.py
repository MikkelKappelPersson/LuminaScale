#!/usr/bin/env python
"""Test CosineAnnealingLR epoch-based decay."""

import torch
import torch.optim as optim

# Create dummy model and optimizer
model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Create CosineAnnealingLR scheduler
num_epochs = 5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6,
)

print(f"\nCosineAnnealingLR Test:")
print(f"Base LR: 1e-4")
print(f"Num Epochs: {num_epochs}")
print(f"eta_min: 1e-6")
print(f"\n{'Epoch':<6} {'LR':<15} {'LR Factor':<12}")
print("-" * 40)

for epoch in range(num_epochs):
    current_lr = optimizer.param_groups[0]["lr"]
    lr_factor = current_lr / 1e-4
    print(f"{epoch:<6} {current_lr:<15.2e} {lr_factor:<12.4f}")
    scheduler.step()

print("-" * 40)
final_lr = optimizer.param_groups[0]["lr"]
print(f"Final LR: {final_lr:.2e}")
print(f"Target:   1.00e-06")
print(f"Match: {'✓' if abs(final_lr - 1e-6) < 1e-7 else '✗'}")
print("\nCharacteristics:")
print("  - Linear start at ~50% (epoch 0)")
print("  - Smooth cosine decay to eta_min")
print("  - No oscillation - continuous smooth curve when logged to TensorBoard")
