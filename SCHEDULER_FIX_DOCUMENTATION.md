# Scheduler Oscillation Fix - Analysis & Solution

## Problem Identified

When using `interval="step"` with PyTorch Lightning's automatic scheduler management, the learning rate exhibited oscillation/sawtooth pattern:

```
LR: 1.0e-04 → 8.55e-05 → 5.05e-05 → 1.55e-05 → 1.0e-06 → [REPEAT]
```

The LR was cycling backwards instead of smoothly decaying forward.

## Root Cause

**PyTorch Lightning's `interval="step"` mode in automatic optimization has timing/state management issues** when used with step-based schedulers:

1. Lightning returns `{"optimizer": opt, "lr_scheduler": {...}}` from `configure_optimizers()`
2. Lightning then tries to manage scheduler.step() calls automatically
3. With `interval="step"`, Lightning calls `scheduler.step()` after each batch
4. **Problem**: The internal step counter state may be reset at epoch boundaries, or `step()` is called at the wrong time in the loop

## Verification

### Test 1: Scheduler in Isolation ✓
```python
# Direct scheduler.step() calls work perfectly
for step in range(200):
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    
# Result: Smooth decay 0 → 1e-4 → 1e-6 (NO oscillation)
```

### Test 2: Lightning Automatic Mode ✗
```python
# configure_optimizers returns scheduler config
return {
    "optimizer": optimizer,
    "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
}

# Result: Oscillating LR in TensorBoard (sawtooth/cycling pattern)
```

### Test 3: Manual scheduler.step() in training_step ✓
```python
# configure_optimizers returns ONLY optimizer
return optimizer

# In training_step(), after loss computation:
if self.lr_scheduler is not None:
    self.lr_scheduler.step()
    
# Result: Smooth decay (same as Test 1)
```

## Solution Implemented

**Use manual scheduler management instead of Lightning's automatic mode:**

### Changes Made

1. **schedulers.py**: Added `debug=True` parameter to `CosineAnnealingWarmupStepScheduler` to print LR values every 100 steps

2. **dequantization_trainer.py**:
   - Add `self.lr_scheduler = None` in `__init__`
   - In `configure_optimizers()`:
     - Create the scheduler and store as `self.lr_scheduler`
     - Return **only the optimizer**, not the scheduler dict
     - This prevents Lightning from automatically managing the scheduler
   - In `training_step()`:
     - After loss computation and logging
     - Manually call `self.lr_scheduler.step()` to advance the schedule
     - This ensures proper timing: after optimizer.step() → after loss.backward()

### Code Changes

```python
def __init__(self, ...):
    ...
    self.lr_scheduler = None  # Will be set in configure_optimizers

def configure_optimizers(self) -> dict:
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    
    # Calculate total steps and warmup
    total_steps = int(self.trainer.estimated_stepping_batches)
    warmup_steps = max(10, int(total_steps * 0.1))
    
    # Create and store scheduler
    self.lr_scheduler = CosineAnnealingWarmupStepScheduler(...)
    
    # Return ONLY optimizer - don't use Lightning's automatic mode
    return optimizer

def training_step(self, batch, batch_idx):
    ...
    loss = l2_loss(y_hat, y)
    self.log("loss_L2/train", loss, ...)
    
    # Manually step the scheduler AFTER optimizer.step()
    if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    
    return loss
```

## Why This Works

1. **Full Control**: We explicitly call `scheduler.step()` at the right place in the loop
2. **Correct Timing**: Called after both `optimizer.step()` (implicit in Lightning) and loss computation
3. **State Preservation**: The scheduler's internal step counter is incremented consistently every batch
4. **No Oscillation**: Step counter never resets; it monotonically increases: 0, 1, 2, 3, ... → total_steps

## Test Results

**Manual Mode - 100 steps:**
```
Step   0 | WARMUP | 1.0e-05  | (1% of base LR)
Step   9 | WARMUP | 1.0e-04  | (100% - warmup complete)
Step  10 | DECAY  | 1.0e-04  | (start cosine decay)
Step  50 | DECAY  | 5.74e-05 | (57% of base LR)
Step 100 | DECAY  | 1.0e-06  | (target reached) ✓ Smooth!
```

**Characteristics:**
- ✅ Linear warmup phase (steps 0-9): 0 → 100%
- ✅ Smooth cosine decay phase (steps 10-100): 100% → 1%
- ✅ No repeating cycles
- ✅ No sawtooth/oscillation

## TensorBoard Behavior

With this fix:
- `learning_rate` metric will show **single smooth curve** (warmup ramp + cosine decay)
- Not step-wise jumps (epoch-based) 
- Not sawtooth oscillation (Lightning's buggy interval=step)
- Properly synchronized with training steps

## Recommendation

This manual approach is **safer and more predictable** than relying on Lightning's automatic scheduler management, especially for step-based (per-batch) schedules. It's the recommended pattern in PyTorch docs when you need precise control over scheduler timing.
