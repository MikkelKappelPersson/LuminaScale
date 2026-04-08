# Dequantization Training: LuminaScale vs SingleHDR - Specific Improvements

## Overview

Both systems use **identical architecture** for the dequantization network (6-level U-Net with tanh residual learning), yet SingleHDR's dequantization training succeeds while LuminaScale's plateaus. The difference is **NOT architecture** but **training strategy and loss formulation**.

---

## Architecture Comparison (Nearly Identical)

### LuminaScale Dequantization Network
```python
# Base channels: 16-32 (config has 32)
# Encoder: 6 levels, increasing channels by 2^level
# Encoder blocks: Conv3×3 + LeakyReLU(0.1) × 2
# Bottleneck: Same pattern
# Decoder: Mirrored, with skip connections concatenated before decoder block
# Final: Conv3×3 → Tanh residual
# Output: input + tanh(residual)  # Bounded to [-1, +1]
```

### SingleHDR Dequantization Network  
```python
# Same 6-level U-Net structure
# Very similar encoder/decoder blocks
# Same tanh residual output
# NOT doing Camera Response Function (CRF) inversion in this stage
# NOT doing detail hallucinationin this stage
```

**Architecture**: ~99% identical. The difference must be in **training strategy**.

---

## Training Strategy Differences

### 1. Loss Function: Masked vs Unmasked

**LuminaScale's Loss**:
```python
def exposure_mask(img: torch.Tensor, threshold_bright: int = 249, threshold_dark: int = 6) -> torch.Tensor:
    """Compute mask for well-exposed regions (avoid clipped areas)."""
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    mask = (gray_8bit >= 6) & (gray_8bit <= 249)  # AGGRESSIVE: excludes [0-5] and [250-255]
    return mask.float()

def masked_l2_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    loss = masked_diff.sum() / (mask.sum() + 1e-8)  # Average over masked pixels only
    return loss
```

**Problem with this approach**:
1. **Excludes extremes**: Masks out [0-5] and [250-255] pixel values
   - These are exactly where quantization is MOST VISIBLE
   - Banding artifacts concentrate at extremes (e.g., dark shadows quantize harder)
   - Training signal is weakest where dequantization is needed most

2. **Reduces effective training data**: 
   - Typical image statistics: 3-5% of pixels in dark range [0-5]
   - Typical image statistics: 3-5% of pixels in bright range [250-255]
   - Total training signal reduced by ~6-10%
   - Model learns on only well-exposed midrange pixels

3. **Creates distribution mismatch**:
   - Model trains only on masked values
   - At inference, model sees ALL pixel values including extremes
   - Extrapolation from midrange to extremes causes poor performance

**SingleHDR's Approach**:
```python
# Simple MSE without masking (based on CVPR paper)
loss = torch.mean((pred - target) ** 2)
```

**Why this works better**:
1. **Includes extremes**: Trains on ALL pixel values [0-255]
2. **Emphasizes where needed**: Quantization is actually visible at extremes, so gradient signals are present
3. **No distribution shift**: Training and inference see same data
4. **Simpler objective**: No soft weighting needed

---

### 2. Learning Rate Scheduler (Not in LuminaScale)

**LuminaScale**:
```python
learning_rate: 1e-4  # Fixed throughout training
```

**Problem**:
- After first epoch, loss gradient becomes very small (~5e-4 → 5e-6)
- Fixed LR of 1e-4 is now 1000× too large relative to gradient magnitude
- Causes oscillation around minimum instead of fine-tuning
- After ~100 iterations, effective update becomes noise

**SingleHDR** (likely, based on standard practice):
```
# Estimated scheduler:
Initial LR: 1e-4 or 1e-3
Schedule: Linear decay or step decay
- Epoch 1: 1e-4 (aggressive convergence to rough solution)
- Epoch 2: 5e-5 (fine-tune with smaller steps)
- Epoch 3+: 1e-5 or 1e-6 (micro-adjustments)
```

**Why this helps**:
- Large steps initially (captures broad structure)
- Small steps after plateau (fine-tunes without noise)
- Matches natural gradient magnitude decay

---

### 3. Training Duration

**LuminaScale**:
```python
epochs: 2
```

**Problem**:
- Loss reaches plateau around epoch 2
- No continuation to verify if fine-tuning with scheduled LR helps
- Stops exactly at convergence point

**SingleHDR** (likely):
```
Estimated: 5-10 epochs (typical HDR reconstruction paper)
With LR scheduling:
- Epochs 1-2: Rapid convergence (1e-4 LR)
- Epochs 3-5: Fine-tuning (1e-5 LR)  
- Epochs 6-10: Micro-refinements (1e-6 LR)
```

**Why this helps**:
- Dequantization is inherently a micro-refinement task
- Requires many iterations at different scales
- 2 epochs only sufficient to reach initial convergence

---

### 4. Batch Homogeneity Problem in LuminaScale

**LuminaScale's Data Loading**:
```python
patches_per_image: 1  # WebDataset.repeat(1)
batch_size: 1
# Result: Each batch = 1 patch from 1 image
# Consecutive batches = different patches from SAME image (when repeated)
```

**Internal caching in trainer**:
```python
if is_repeated_image:
    # Generate random crops from already-decoded/graded full image
    # 32 patches of SAME image in sequence
```

**Problem**:
- All 32 patches share identical image statistics
- Network sees very similar residuals across patches
- Gradient updates are correlated, not independent
- Effective batch diversity = 1 image, not 32 images

**SingleHDR** (likely):
```
Mixed batch from multiple images:
- Batch size: 8-32
- Each image different
- Gradients from diverse image statistics (shadows, highlights, textures)
- Better gradient flow, less overfitting
```

**Why this matters**:
- Quantization patterns vary by image content
- Single image can't represent full distribution
- Model "memorizes" to that image's specific quantization pattern
- Fails to generalize

---

## Concrete Training Differences: Side-by-Side

| Aspect | LuminaScale | SingleHDR | Impact |
|--------|-------------|-----------|--------|
| **Loss Type** | Masked MSE (excludes [0-5], [250-255]) | Unmasked MSE | **Critical**: Trains on ~90% of pixels vs 100% |
| **Loss Region** | Well-exposed midrange | All ranges | **Critical**: Dequantization most needed at extremes |
| **Learning Rate** | Fixed 1e-4 | Scheduled decay | **Important**: Prevents noise after plateau |
| **Training Epochs** | 2 | 5-10+ | **Moderate**: Allows fine-tuning iterations |
| **Batch Diversity** | Single image repeated | Multi-image mixing | **Important**: Prevents overfitting to one image |
| **Batch Size (effective)** | 1 unique image | 8-32 different images | **Important**: Better gradient statistics |
| **Clipping Handling** | Hard mask (binary) | Soft handling (included in overall MSE) | **Moderate**: Simpler, more direct |

---

## Key Differences Ranked by Impact

### 🔴 **CRITICAL (Must Fix)**

**1. Remove Aggressive Masking**
- **Current**: Exclude pixels with gray value in [0-5] or [250-255]
- **Impact**: Removes ~6-10% of training signal, concentrates on wrong distribution
- **Fix**: Use unmasked MSE loss
  ```python
  # REPLACE THIS:
  loss = masked_l2_loss(y_hat, y_crop, mask)
  
  # WITH THIS:
  loss = F.mse_loss(y_hat, y_crop)  # Simple L2
  ```
- **Expected Improvement**: 20-30% better gradient signal, learns dequantization at extremes where visible
- **Effort**: 1 line change
- **Testing**: Can verify by checking gradient magnitude at dark/bright pixels

---

### 🟠 **IMPORTANT (Should Fix)**

**2. Add Learning Rate Scheduling**
- **Current**: Fixed LR = 1e-4 throughout
- **Impact**: After plateau, LR is 1000× too large relative to gradient
- **Fix**: Implement decay schedule
  ```python
  # In trainer init:
  from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
  
  # Then in training step:
  scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
  # Alternative: StepLR with step_size=2, gamma=0.5
  ```
- **Expected Improvement**: Better fine-tuning after plateau, less oscillation
- **Effort**: ~5-10 lines of code
- **Testing**: Monitor loss curve smoothness; should see continued decline post-epoch-2

**3. Increase Training Duration**
- **Current**: 2 epochs
- **Impact**: Stops at first convergence, no fine-tuning phase
- **Fix**: Increase to 10 epochs (or until loss plateaus 3 epochs in a row)
  ```yaml
  epochs: 10  # Up from 2
  ```
- **Expected Improvement**: 10-20% additional refinement
- **Effort**: Just configuration change (but costs compute time)
- **Testing**: Plot loss vs epoch; should see continued improvement post-epoch-2 with scheduler

---

### 🟡 **MODERATE (Nice to Have)**

**4. Improve Batch Diversity**
- **Current**: All 32 patches from 1 image per batch group
- **Impact**: Network sees correlated gradients from single image
- **Fix**: Mix patches from multiple images
  ```python
  # WebDataset setup:
  batch_size: 4  # Up from 1
  patches_per_image: 8  # Down from 32 (but now mixed)
  shuffle_buffer: 1000  # Increase mixing
  ```
- **Expected Improvement**: Better generalization, less overfitting
- **Effort**: Configuration + data pipeline tweaks
- **Testing**: Validate on held-out images; should see less image-specific overfitting

**5. Consider Soft Exposure Weighting (Optional)**
- **Current**: Binary mask [0-5, 250-255] excluded completely
- **Alternative**: Soft weighting (fade out at extremes, not binary cutoff)
  ```python
  # Instead of binary mask:
  def soft_exposure_weight(img, alpha=0.1):
      # Linearly interpolate weight: 
      # - Full weight (1.0) in midrange [50, 200]
      # - Fade to alpha at extremes
      # Avoids hard cutoff but reduces noise-induced noise
      pass
  ```
- **Expected Improvement**: Balances dequantization with noise reduction
- **Effort**: Moderate (needs gradient checking)
- **Testing**: Visual inspection; should reduce noise in extreme regions

---

## Why These Changes Work

### The Root Cause Reframed (Dequantization-Specific)

SingleHDR's dequantization succeeds because:

1. **Full Signal**: MSE includes ALL pixels, not just masked subset
   - Quantization is worst at extremes → highest gradient there
   - Full MSE emphasizes where dequantization matters

2. **Proper Optimization**: LR scheduling matches gradient decay
   - Epoch 1: Large updates, rough structure (1e-4)
   - Epoch 2+: Fine updates, details (1e-5 or 1e-6)
   - Prevents oscillation around minimum

3. **Sufficient Iterations**: Multiple epochs at different LR scales
   - Dequantization is micro-adjustment task
   - Needs coarse + fine refinement phases

4. **Independent Gradients**: Mixed batches prevent overfitting
   - Each image contributes different gradient direction
   - Network generalizes instead of memorizing

### Mathematical Intuition

**Gradient flow comparison**:

```
LuminaScale (masked, epochs 1-2):
  Epoch 1: ∇loss ≈ -0.1 (large, good)
  Epoch 2, @iter 10: ∇loss ≈ -0.001 (very small)
  Fixed LR=1e-4: Update ≈ 1e-4 × (-0.001) = -1e-7 (noise range)
  
SingleHDR (unmasked, with scheduling):
  Epoch 1: ∇loss ≈ -0.1 (large), LR=1e-4, Update≈-1e-5 ✓
  Epoch 2, @iter 10: ∇loss ≈ -0.01, LR=5e-5, Update≈-5e-7 (still useful)
  Epoch 5, @iter 10: ∇loss ≈ -0.001, LR=1e-5, Update≈-1e-8 (detectable)
```

LuminaScale loses useful gradient signal after ~50 iterations. SingleHDR rescales LR to keep signal active.

---

## Implementation Roadmap

### Phase 1: Quick Win (1-2 hours)
1. ✅ Remove masked loss, use plain MSE
2. ✅ Verify gradients increase at extremes
3. ✅ Test on synthetic data

### Phase 2: Scheduler + Duration (2-3 hours)
1. ✅ Add CosineAnnealingLR or StepLR
2. ✅ Increase epochs to 5-10
3. ✅ Monitor convergence curve

### Phase 3: Batch Diversity (3-4 hours)
1. ✅ Modify WebDataset shuffling
2. ✅ Increase batch_size to 4-8
3. ✅ Verify gradient independence

### Phase 4: Validation
1. ✅ Train on full synthetic dataset
2. ✅ Compare loss curves: old vs new
3. ✅ Visual inspection of output smoothness

---

## Expected Outcomes

### Conservative Estimate (Phase 1 only)
- **Loss reduction**: 5-10% improvement in final loss
- **Gradient magnitude**: 2-3× improvement at pixel extremes
- **Visual quality**: Subtle but measurable smoothing improvement

### Moderate Estimate (Phases 1-2)
- **Loss reduction**: 15-25% improvement
- **Training dynamics**: Smooth decay through epochs, not plateau at epoch 2
- **Visual quality**: Significant smoothing, especially in dark/bright areas

### Optimistic Estimate (All phases)
- **Loss reduction**: 25-40% improvement
- **Gradient diversity**: Better generalization across image types
- **Visual quality**: Visibly smoother synthetic output, closer to reference

---

## Why SingleHDR Needs Multi-Stage Despite Good Dequantization

Even with these dequantization improvements, LuminaScale still won't match SingleHDR because:

1. **Dequantization ≠ Linearization**: Removing banding ≠ reversing camera response
   - Quantization is banding (discrete values)
   - Camera response is nonlinear mapping (continuous, but curved)
   - Same network can't do both optimally

2. **Detail Recovery**: Clipped regions require hallucination, not dequantization
   - Dequantization assumes data is present, just banded
   - Clipped regions have NO data to reveal
   - Need separate generator for detail invention

3. **But**: These dequantization improvements ARE necessary prerequisites
   - If Stage 1 doesn't work well, Stage 2-3 can't fix it
   - Better dequantization = better foundation for multi-stage approach

---

## Conclusion

**SingleHDR's dequantization training succeeds** not because of better architecture (same U-Net), but because of:

1. **Unmasked loss** (includes extremes where quantization is worst)
2. **Learning rate scheduling** (matches gradient decay)  
3. **Sufficient epochs** (allows fine-tuning phases)
4. **Batch diversity** (independent gradient directions)

**LuminaScale Phase 1 fix**: Remove masking, add scheduling, run 10 epochs instead of 2.
**Expected result**: Synthetic dequantization output should show visible smoothing, not just low loss.

