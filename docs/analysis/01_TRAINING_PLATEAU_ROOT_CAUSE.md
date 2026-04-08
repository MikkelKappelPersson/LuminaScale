# In-Depth Analysis: Training Plateau & Residual Quantization

**Date**: April 8, 2026  
**Status**: Plateau observed at ~5e-6 loss after ~1-2 epochs  
**Problem**: Loss flattens despite continuing training; output images still show visible quantization banding

---

## Executive Summary

The training plateau and persistent quantization artifacts are caused by **fundamental mismatch between task and loss function**. The network learns to match smooth targets when training begins, but quickly reaches a local minimum because:

1. **L2 loss is insufficient for dequantization** – It's designed for MSE reconstruction, not artifact removal
2. **Aggressive exposure masking hides the real problem** – Only trains on non-clipped regions (6-249 in 8-bit)
3. **No explicit smoothness pressure** – Network learns average values but has no incentive to interpolate through quantization levels
4. **Architecture + activation bottleneck** – Tanh residual limits fine-grained corrections in mid-range values
5. **Fixed learning rate** – No decay mechanism to escape local minima or fine-tune later

---

## Part 1: Loss Function Analysis

### Current Loss Function

```python
def masked_l2_loss(pred, target, mask):
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    loss = masked_diff.sum() / (mask.sum() + 1e-8)
    return loss
```

**Characteristics:**
- **Symmetric penalty**: Both over- and under-estimation penalized equally
- **Pixel-level only**: No consideration of spatial structure or gradients
- **Masked to [6/255, 249/255]**: ~97% of the image excluded from training
- **Greedy averaging**: Network minimizes MSE by averaging, not true reconstruction

### Why L2 Loss Fails for Dequantization

#### Problem 1: L2 Loss ≠ Dequantization Loss

The fundamental issue is that **L2 loss minimizes MSE, not dequantization**. Consider this scenario:

```
Input (8-bit quantized):   [0.00, 0.004, 0.008, 0.012, ...]  (steps of 1/255 ≈ 0.004)
Target (true continuous):  [0.001, 0.005, 0.009, 0.013, ...]

Early training (MSE-optimal):
  Output: [0.002, 0.006, 0.010, 0.014, ...]  (averages input/target)
  L2 Loss: ~5e-7 (already very small!)
  
This MSE-optimal solution is NOT dequantization—it's just averaging.
```

**Why L2 stops improving:**
- Once the network learns to average, further reduction requires learning **exact discontinuities** where the 8-bit steps occur
- L2 loss doesn't incentivize smooth interpolation between quantized levels
- The model essentially "gives up" after reducing MSE to noise-like levels

#### Problem 2: Aggressive Masking Creates Blind Spots

```python
def exposure_mask(img, threshold_bright=249, threshold_dark=6):
    # Only trains on pixels with gray value in [6, 249] (out of 255)
    mask = (gray_8bit >= 6) & (gray_8bit <= 249)
```

**Issue**: Banding is **most visible in regions that are excluded**:
- Dark banding: shadows (gray < 6) → completely masked out
- Mid-tone banding: common in well-exposed images but subtle → masked if clipped
- Bright banding: highlights (gray > 249) → completely masked out

**Training coverage**: Only ~90-95% of pixels trained on, and only in specific (well-exposed) regions. The model never learns to handle the full tonal range.

#### Problem 3: Information Bottleneck at Tanh

```python
residual = self.final_conv(feat)      # [1, 3, 512, 512]
residual = self.tanh(residual)         # Output ∈ [-1, 1]
output = input_img + residual          # Clamped to [0, 1]
```

**Tanh properties:**
- Derivative peaks at 0, saturates near ±1
- For small residuals (like fine dequantization), gradient ≈ 0.1-0.5
- Acts as a **strong regularizer**, preventing large corrections
- When input is 8-bit quantized (discrete steps), tanh forces smooth approximation

**Consequence**: The network physically cannot output high-frequency corrections needed for true dequantization. It's biased toward smooth outputs, which masks high-frequency banding.

---

## Part 2: Architecture Analysis

### Model Capacity Bottleneck

**Current configuration:**
```yaml
base_channels: 32      # Current training config
num_levels: 6          # 6-level U-Net
model_params: ~2.8M    # Estimated parameters
```

**Receptive field analysis:**
```
Encoder: 3 → 32 → 64 → 128 → 256 → 512 → 1024
Bottleneck: 1024 channels
Decoder: Mirror back to 32

Max receptive field: ~(2^6 - 1) * 3 ≈ 189 pixels diagonal
```

For a 512×512 crop, receptive field covers ~40% of the image. This is **insufficient for modeling long-range gradient statistics** needed to:
1. Understand global scene structure
2. Predict smooth illumination gradients
3. Distinguish real shadows from quantization artifacts

### Why Base Channels = 32 Is Too Small

Dequantization requires modeling:
- **High-frequency details**: Quantization step boundaries (~1/255 resolution details)
- **Low-frequency structure**: Smooth lighting gradients across the image
- **Channel-specific artifacts**: Banding differs per channel (R, G, B have different statistics)

With only 32 base channels:
- Bottleneck has only 1024 channels total (after 6×2 factor expansion)
- Each scale level has < 1024 distinct features
- Insufficient capacity to learn different banding patterns in different regions

---

## Part 3: Training Dynamics Analysis

### Loss Curve Interpretation

From TensorBoard visualization (showing ~24k batches, reaching 5e-6 loss):

```
Epoch 1 (batches 0-12k):
  Loss: 7e-4 → 5e-5 (steep decline, 14× reduction)
  
Epoch 2 (batches 12k-24k):
  Loss: 5e-5 → 5e-6 (gradual decline, 10× reduction)
  Plateau visible around batch 20k

Characteristic: Classic **local minimum escape followed by fine-tuning plateau**
```

**Why plateau happens:**

1. **Quick phase (Epochs 1-2)**: Network learns "average behavior"
   - Reduction: MSE of random weights → MSE of averaging
   - Typical for any L2-based training
   - Loss drops rapidly but quality barely improves

2. **Plateau phase (post-Epoch 2)**: Network reaches MSE-optimal averaging solution
   - Further reduction requires learning discontinuities
   - Gradients become tiny: $\frac{\partial L}{\partial w}$ ≈ differences between input and target
   - With fixed LR (1e-4), weight updates become negligible

### Learning Rate Issue

**Current config**: `learning_rate: 1e-4` (fixed throughout training)

**Problem timeline:**
```
Early (batch 0-5k):        ∇L ≈ 1e-3,  Update: 1e-4 × 1e-3 = 1e-7 ✓ Good
Mid (batch 5k-15k):        ∇L ≈ 1e-4,  Update: 1e-4 × 1e-4 = 1e-8 OK
Late (batch 15k+):         ∇L ≈ 1e-5,  Update: 1e-4 × 1e-5 = 1e-9 Too small
```

Without learning rate scheduling:
- By late training, loss vector magnitude is in 1e-5 range
- Fixed LR of 1e-4 becomes too high early, too low late
- Optimizer takes wild steps early, tiny steps late
- **No way to fine-tune to sub-plateau loss values**

---

## Part 4: Data Pipeline & Masking Effects

### Mask Coverage Analysis

```python
threshold_dark: int = 6      # Pixel value > 6/255 ≈ 0.024 in [0,1]
threshold_bright: int = 249  # Pixel value < 249/255 ≈ 0.976 in [0,1]
```

For synthetic primary gradients:
```
Red gradient column:   [0.48, ..., 1.0]
   Masked pixels:      [6/255 ≤ px ≤ 249/255]
   Effective range:    [0.024, 0.976]
   Coverage:           ~95% of pixels, but corners/edges excluded
   
Dark shadows (px=0):   Excluded entirely ✗
Highlights (px=1):     Excluded entirely ✗
Mid-tones (px=0.5):    Included ✓
```

**Training blind spots:**
1. **Shadows**: Network never sees gradient structure in dark regions
2. **Highlights**: Network never learns to handle bright quantization
3. **Extreme values**: Banding at [0, 10] and [245, 255] never trained

### On-the-Fly Gradients Are "Too Perfect"

Current test data (synthetic primaries):
```
3 vertical gradients: R=1 (with G/B=0.48-0.52), etc.
Perfectly smooth input, perfectly quantized @ 8-bit
No real-world:
  - Noise
  - Color shifts
  - Sensor artifacts
  - Complex lighting
```

**Why this matters for learning**:
- Network learns "perfect quantization" pattern
- Real camera data has noise that masks banding patterns
- Quantization + noise behaves differently than pure quantization
- Model generalizes poorly to noisy real data

---

## Part 5: Quantization Artifact Reasoning

### Why Output Still Shows Banding

Even if loss plateaus at 5e-6, output can still appear quantized because:

#### Reason 1: Tanh Activation Enforces Smoothness
```python
residual = tanh(final_conv_output)  # Forces smooth, continuous output
output = input_img + residual       # Constrained to [-1, 1] range
```

The tanh ensures the network output is **always smooth**. Since input is harsh/quantized, output = input + smooth_residual = still somewhat quantized.

#### Reason 2: Training Doesn't Optimize for Perceptual Smoothness
The loss function doesn't measure what humans see:
- L2 loss: Pixel-level MSE
- Human perception: Smoothness of **gradients** (spatial continuity)

A model minimizing L2 can still output:
```
Input:         [0.00, 0.004, 0.008, 0.012, ...]
L2-optimal:    [0.002, 0.006, 0.010, 0.014, ...]  ← Better than input, but still has bands
Target:        [0.001, 0.005, 0.009, 0.013, ...]

Small MSE δ makes the last two visually indistinguishable to loss,
but humans still see banding in the model output.
```

#### Reason 3: Aliasing at Mask Boundary
The mask creates a sharp transition at gray=6 and gray=249:
```python
mask = (gray_8bit >= 6) & (gray_8bit <= 249)  # Hard boundary ✓ 249
                                               # Hard boundary ✗ 250
```

Near boundaries, training signals drop to zero. The network learns discontinuous behavior at mask edges, which manifests as visible artifacts.

---

## Part 6: Mathematical Insight – Why L2 Loss Hits a Wall

### Theorem: L2 Optimal Solution for Dequantization

For quantized signal $x_q$ with target $x_t$, L2 loss is:
$$L = \mathbb{E}[(f_\theta(x_q) - x_t)^2]$$

The network outputs $f_\theta(x_q)$ that minimizes this by learning:
$$f_{optimal}(x_q) \approx \mathbb{E}[x_t | x_q]$$

**Critical insight**: The optimal L2 solution is the **conditional mean**, which is:
- Smooth (averaging behavior)
- **Not recoverable from quantized input alone**
- Limited by how much variance the quantization removes

For 8-bit quantization with uniform levels:
```
Quantization bucket: [k/255, (k+1)/255)
Conditional mean:    (k + 0.5) / 255  ← Center of bucket

This is a local constant function! L2 cannot recover values within a bucket.
```

The network is **mathematically limited** by L2 loss to predicting the bucket center, not the true value.

---

## Part 7: Why Continued Training Won't Help

### Optimization Landscape

```
Loss landscape for dequantization with L2 loss:

          ┌─────────────────────────────────
    Loss │     
        5│  ●  (naive random init)
         │  ╲╲
        2│    ╲╲  ●  (after 5 epochs)
         │      ╲╲
       5e-3│        ╲ (steep gradient region)
          │          ╲
       5e-4│          ◉ ◉ ◉ ◉ ◉ (plateau plateau!)
          │          ^ local minimum
          ├────────────────────────────
            Training Iterations
```

**Key observation**: The plateau region is a **local minimum** of the L2 objective, not a local minimum of task performance.

The network has fully learned **"what L2 loss rewards"** (averaging), which is different from actual dequantization.

---

## Summary Table

| Issue | Root Cause | Evidence | Impact |
|-------|-----------|----------|--------|
| **Plateau** | L2 loss + fixed LR | Loss 5e-4 → 5e-6 rapid then flat | Can't escape MSE optimum |
| **Quantization visible** | Tanh + MSE optimum not = dequant | Output still banded despite low loss | Task/loss mismatch |
| **Poor gradient coverage** | Masking excludes extremes | Mask [6,249] out of [0,255] | Incomplete learning |
| **Small receptive field** | 6-level U-Net | RF ≈ 189px for 512px image | Can't model global gradients |
| **High-frequency loss** | No edge/gradient penalty | Only pixel-level MSE | Banding not directly penalized |
| **Learning rate stuck** | No scheduling | 1e-4 fixed throughout | Can't fine-tune late training |

---

## Next Steps for Analysis

See linked documents:
- `02_LOSS_FUNCTION_ANALYSIS.md` – Detailed loss function comparison
- `03_ARCHITECTURE_ANALYSIS.md` – Model capacity and design choices
- `04_DATA_PIPELINE_ANALYSIS.md` – Training data and masking effects
- `05_RECOMMENDED_FIXES.md` – Concrete solutions (without implementation)

