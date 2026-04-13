# Hyperparameter Tuning Guide for Dequantization-Net

## Problem with Loss-Based Tuning

When you change **loss weights** in your configuration:

```yaml
loss:
  l1_weight: 1.0
  charbonnier_weight: 0.05
  grad_match_weight: 0.5
```

...the absolute `loss_total/train` value changes **independently of actual quality improvement**. 

**Example:**
- Run A: `charbonnier_weight: 0.05` → `loss_total: 2.847`
- Run B: `charbonnier_weight: 0.10` → `loss_total: 3.124`

Is Run B actually worse? **Not necessarily!** The higher loss is just because you weighted Charbonnier twice as heavily.

## Solution: Use PSNR for Weight-Independent Tuning

**PSNR (Peak Signal-to-Noise Ratio)** measures reconstruction quality and is **completely independent** of loss weights:

$$\text{PSNR} = 20 \log_{10}\left(\frac{\text{max\_val}}{\sqrt{\text{MSE}}}\right) \text{ [dB]}$$

### Why PSNR:
✅ **Weight-independent**: Doesn't change just because you adjust loss weights  
✅ **Directly measures quality**: Higher PSNR = better reconstruction  
✅ **Fair comparison**: Compare runs across all weight configurations with confidence  
✅ **Standard metric**: Industry-standard for image reconstruction quality  

## What Changed

### 1. DequantizationTrainer (`src/luminascale/training/dequantization_trainer.py`)

Added `compute_psnr()` function and PSNR logging in `training_step`:

```python
# Log weight-independent metrics (PSNR, SSIM) for fair hparam comparison
psnr_val = compute_psnr(y_hat, y)
self.log("metric_psnr/train", psnr_val, prog_bar=False, sync_dist=True)
```

### 2. HparamsMetricsCallback (`scripts/train_dequantization_net.py`)

Updated `_get_metrics_dict()` to prefer PSNR over loss:

```python
# Prefer PSNR (weight-independent) over loss for hparam tuning
metric_keys = [
    "metric_psnr/train",     # Weight-independent metric
    "loss_total/train",      # Fallback
    "loss_L1/train",         # Further fallback
]
```

### 3. TensorBoard Logging

Run your training and view TensorBoard:

```bash
tensorboard --logdir outputs/logs/
```

You'll see:
- **metric_psnr/train**: Weight-independent quality metric ✅
- **loss_total/train**: Weighted loss (for reference)
- **loss_L1/train, loss_Charbonnier/train, loss_EdgeAware/train**: Individual components

## Hyperparameter Tuning Workflow

### Step 1: Baseline Run
```bash
python scripts/train_dequantization_net.py \
  --config-name=default \
  epochs=10
```
→ Record baseline PSNR from TensorBoard

### Step 2: Tune Loss Weights
```bash
# Try different charbonnier_weight
python scripts/train_dequantization_net.py \
  --config-name=default \
  loss.charbonnier_weight=0.10 \
  epochs=10
```

### Step 3: Compare PSNR (Not Loss!)
In TensorBoard, compare:
- **metric_psnr/train** across configurations
- **NOT** loss_total/train (weight-dependent!)

Example comparison table:
| Config | PSNR (dB) | loss_total | Status |
|--------|-----------|-----------|--------|
| charbonnier=0.05 | **24.8** | 2.847 | Baseline |
| charbonnier=0.10 | **25.1** | 3.124 | ✅ Better |
| charbonnier=0.20 | **24.9** | 3.401 | Worse |

→ Pick `charbonnier=0.10` (highest PSNR)

## PSNR Interpretation

| PSNR Range | Quality |
|-----------|---------|
| > 40 dB | Excellent (imperceptible difference) |
| 30-40 dB | Good (minor visible differences) |
| 20-30 dB | Fair (noticeable differences) |
| < 20 dB | Poor (significant degradation) |

For **dequantization**, aim for **PSNR > 28-30 dB** in training.

## Advanced: Multi-Metric Tuning

For comprehensive evaluation, also track:

- **SSIM** (Structural Similarity): Perceptual quality
- **ΔE** (Delta E): Color space fidelity (important for ACES)

These will be added in future updates.

## FAQ

**Q: My PSNR is improving but loss is increasing. Is that bad?**  
A: No! If you increased `grad_match_weight`, the loss will increase while PSNR (actual quality) improves. That's the whole point of using PSNR.

**Q: Should I still monitor loss?**  
A: Yes! Monitor loss to ensure training is stable and decreasing. But use **PSNR for final tuning decisions**.

**Q: What if PSNR isn't logged?**  
A: Check that your trainer is latest version with `compute_psnr()`. Falls back to `loss_total/train` if missing.

**Q: How do I compare with my old loss-based runs?**  
A: Old runs logged `loss_total/train`. Use **PSNR for all future comparisons** and ignore old loss values.

---

**Last Updated:** April 13, 2026  
**Status:** ✅ Implemented in DequantizationTrainer and HparamsMetricsCallback
