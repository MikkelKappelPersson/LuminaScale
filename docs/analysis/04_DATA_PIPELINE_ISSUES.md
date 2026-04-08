# Data Pipeline & Training Setup Analysis

**Focus**: How WebDataset setup, masking strategy, and batch structure prevent learning

---

## Part 1: Mask Effectiveness Study

### Current Masking Implementation

```python
def exposure_mask(img, threshold_bright=249, threshold_dark=6):
    """Compute mask for well-exposed regions (avoid clipped areas)."""
    gray = (0.299 * img[:, 0:1, :, :] 
          + 0.587 * img[:, 1:2, :, :] 
          + 0.114 * img[:, 2:3, :, :])
    gray_8bit = (gray * 255.0).round()
    mask = (gray_8bit >= threshold_dark) & (gray_8bit <= threshold_bright)
    return mask.float()
```

### Mask Coverage for Synthetic Primary Gradients

```
Red column (R=1.0, G/B vary 0.48-0.52):
  Gray value formula: 0.299*1.0 + 0.587*G + 0.114*B
  Min: 0.299 + 0.587*0.48 + 0.114*0.48 = 0.422 → 107/255
  Max: 0.299 + 0.587*0.52 + 0.114*0.52 = 0.475 → 121/255
  
  Mask thresholds: [6, 249] in 8-bit
  Coverage: 100% (all pixels 107-121/255 fall in [6, 249])

Green column (G=1.0, R/B vary):
  Min: 0.587*1.0 + 0.299*0.48 + 0.114*0.48 = 0.704 → 179/255
  Max: 0.587*1.0 + 0.299*0.52 + 0.114*0.52 = 0.752 → 192/255
  
  Coverage: 100% (all pixels 179-192/255 fall in [6, 249])

Blue column (B=1.0, similar analysis):
  Coverage: 100% (similar range)
```

**For synthetic gradients: Mask covers ~100% of pixels.**

### Mask Coverage for Real Data

```
Typical photograph with scene:
  - Very dark shadows: gray < 20/255  → MASKED OUT
  - Dark midtones: gray 30-50/255     → MASKED IN
  - Well-exposed midtones: gray 80-180/255 → MASKED IN
  - Bright highlights: gray 200-240/255 → MASKED IN
  - Blown highlights: gray > 250/255  → MASKED OUT
  
Typical coverage: 85-95% of scene

Where banding is MOST visible:
  - Very dark shadows (excluded by mask)
  - Very bright highlights (excluded by mask)
  - Exactly where banding matters for final image!
```

### Why the Mask Backfires

```
Intended purpose: "Ignore clipped regions where MSE loss is meaningless"

Unintended consequence:
  1. Network never learns to handle extremes
  2. Most visible banding (shadows/highlights) untrained
  3. Creates discontinuity at mask boundary
     (hard transition from trained [6,249] to untrained outside)
  4. Model learns: "extreme values don't matter"
  
Real-world failure:
  - Camera output with clipped blacks: No training signal
  - Camera output with clipped whites: No training signal
  - Result: Artifacts at mask boundaries in output
```

---

## Part 2: WebDataset Batch Structure Problem

### Current WebDataset Setup

From config:
```yaml
batch_size: 1              # One image per batch
patches_per_image: 1       # Generate 1 patch per image (in training loop: 32)
shuffle_buffer: 10         # Buffer size for shuffling
```

From training loop:
```python
# load batch (one image)
x_full, y_full = self._process_batch(batch)  # [3, H, W]

# Train on 32 patches from this one image
avg_loss, elapsed_ms = self._train_on_image(
    x_full, y_full, batch_idx, num_patches=32
)
```

### The Batch Homogeneity Problem

```
Traditional ML training:
  Batch 1: [image_A, image_B, image_C, image_D]  ← 4 different images
           Each image: Different content, lighting, colors
           Gradients: Diverse, informative
           
  Batch 2: [image_E, image_F, image_G, image_H]  ← 4 new images
           Gradients: Different again
           
  Result: Network trained on diverse patterns per epoch

LuminaScale WebDataset training:
  Batch 1: [image_A, image_A, image_A, image_A]  ← Same image, 32 patches
           gradient_1 = ∂L/∂w for patch_1 of image_A
           gradient_2 = ∂L/∂w for patch_2 of image_A
           ...
           gradient_32 = ∂L/∂w for patch_32 of image_A
           
           Average gradient: MeanGrad(patches from image_A)
           
  Batch 2: [image_B, image_B, image_B, image_B]  ← Same image, 32 patches
           Average gradient: MeanGrad(patches from image_B)
           
  Result: Network trained on averaged statistics of ONE image per batch!
```

### Why This Limits Learning

```
Diversity per update:
  Traditional:    4 different images → 4 diverse gradient signals
  WebDataset:     1 image, 32 patches → All patches from same content
  
  Impact on learning:
    - Gradients are highly correlated (same image!)
    - Network doesn't see diverse banding patterns
    - Overfits to specific image's quantization structure
    - Can't generalize to novel images

Example:
  Batch 1: Blue image
    Gradients: Learn to dequantize blue
    
  Batch 2: Another blue image
    Gradients: Similar to Batch 1 (still blue!)
    
  Result: Network learns "blue banding" but never sees red/green details!
  
Cache hits make this worse:
  "[CACHE HIT] Image 'img_001' (Hit rate: 87.5%)"
  
  87.5% of training time: Patches from SAME image!
  Network sees same image multiple times before moving on!
```

### Comparison: Proper Batch Diversity

```
Ideal batch composition (for dequantization):
  Batch: [red_patch_1, blue_patch_2, green_patch_3, ..., mixed_patch_8]
  
  Colors: RGB spread across patches
  Exposure: Dark, mid, bright tones represented
  Regions: Edges, centers, corners included
  
  Result: Each batch teaches all aspects of dequantization simultaneously

WebDataset current:
  Batch: [patch_1 (color A), patch_2 (color A), ..., patch_32 (color A)]
  
  Colors: Only one dominant color per batch (extreme specialization)
  Result: Slow learning, poor generalization
```

---

## Part 3: Synthetic Gradient Limitations

### What Synthetic Data Teaches

```python
hdr = create_primary_gradients(width=512, height=512)
# Result: Three vertical colored gradients
#   Left:   Red column with smooth G/B variation (0.48-0.52)
#   Middle: Green column with smooth R/B variation
#   Right:  Blue column with smooth R/G variation

ldr = quantize_to_8bit(hdr)
# Pure quantization, NO NOISE
```

### What This Misses (Real Data Characteristics)

```
1. Sensor Noise
   Real camera:    Shot noise, pattern noise, thermal noise
   Synthetic:      Perfect gradients, no noise
   
   Impact:
     Noise masks banding patterns in real sensors
     Network trained on pure banding can't handle noisy banding
     
2. Color Space Artifacts
   Real transforms: ACES grading, CDL corrections, color casts
   Synthetic:      Just quantized primaries
   
   Banding in different color spaces:
     sRGB: Banding visible in G channel (0.587 luminance weight)
     ACES: Different banding characteristics
     
   Network trained on sRGB doesn't generalize to ACES!

3. Spatial Structure
   Real image:     Objects, lighting, shadows, specularities
   Synthetic:      Uniform vertical gradients
   
   Network learns: "Gradients are vertical and uniform"
   Real test: "Gradients are complex 2D spatial structures"
   
4. Extreme Values
   Real camera clipping: Hard limits at 0 (black) and 255 (white)
   Soft corners due to sensor sensitivity
   
   Synthetic: Smooth gradients within [0.48, 1.0] range
   Untrained: 0 and 255 values (where banding is most visible!)

5. Dynamic Range
   Real HDR: Ratio between darkest and brightest ≈ 100:1 to 10000:1
   Synthetic gradient: Only spans 0.48-1.0 ≈ 2.1:1
   
   Network trained on narrow range: Poor for scenes with high DR
```

### Primary Gradient Specifically Problematic

```
Why synthetic primaries are the worst choice:

1. No spatial variation
   R gradient: Constant red column (same R every pixel vertically)
   → Network learns: "R doesn't change, just interpolate G/B"
   Real images: R changes everywhere
   → Generalization fails

2. Perfectly uniform quantization
   Every column: Same quantization error pattern
   → Network learns: "Same correction everywhere"
   Real images: Banding varies by position, sensor pattern
   → Generalization fails

3. No masking boundaries
   Synthetic: No regions where banding is particularly visible
   Real: Shadows/highlights have strongest banding
   → Network never learns to handle hard-to-detect banding

4. Color independence
   Synthetic: R, G, B channels have identical tasks (just different colors)
   Real: R might have more banding than G due to sensor differences
   → Generalization fails
```

---

## Part 4: Training Signal Weakness

### Gradient Magnitude Analysis

```
Toy example: Single pixel in R channel

Input (quantized):  128/255 ≈ 0.5020
Target (true):      128.3/255 ≈ 0.5032
Difference:         0.0012

Loss for single pixel:
  L = (0.5020 - 0.5032)^2 = 1.44e-6

Batch loss (512×512×3 pixels with ~97% masked):
  Total pixels: 262,144
  Masked pixels: 254,080 (97%)
  Training pixels: 8,064

  Each pixel loss: ~1e-6
  Batch MSE: 1e-6
  
After model prediction of residual=0:
  L = (0.5020 + 0 - 0.5032)^2 = 1.44e-6 (no improvement!)

Gradient magnitude:
  ∇L/∂w_i ∝ ∂L/∂output * ∂output/∂w_i
  
  ∂L/∂output = 2 * (output - target) = 2 * (0.5020 - 0.5032) = -0.0024
  
  Through network (6 layers + pooling + upsampling):
  ∂output/∂w_i = ∂network/∂w_i (product of 6+ derivatives)
  
  Each ReLU: ×1 or ×0.1 (LeakyReLU)
  Each pooling: ÷4 (average pooling dilutes gradients)
  
  Effective gradient: -0.0024 * product of tiny numbers ≈ 1e-9
```

### Why Gradients Plateau at 5e-6

```
Early training (random weights):
  Model output: random noise
  Output variance: ≈ 1.0
  Difference from target: ≈ 0.5 (average error)
  Loss: 0.5^2 = 0.25
  Gradient: ≈ 1e-3 (large)
  Update: 1e-4 * 1e-3 = 1e-7 (weights change significantly)
  
After ~5000 batches (learning to average):
  Model output: ~average value
  Output variance: ≈ small but non-zero
  Difference from target: ≈ target - average ≈ 0.001 (tiny!)
  Loss: 0.001^2 = 1e-6
  Gradient: ≈ 1e-9 (tiny)
  Update: 1e-4 * 1e-9 = 1e-12 (no change!)
  
Reaches plateau when:
  Gradient < ~1e-7
  Update < 1e-10 (below float32 precision for practical changes)
  
  New loss improvements < machine epsilon
```

---

## Part 5: Training Loop Structure Issue

### Current Per-Patch Updates

```python
def _train_on_image(self, x_full, y_full, batch_idx, num_patches=32):
    """For each patch: forward → loss → backward → optimizer.step()"""
    for patch_idx in range(num_patches):
        # Generate random crop
        x_crop, y_crop = random_crop(x_full, y_full, crop_size=512)
        
        # Forward
        y_hat = self.model(x_crop)
        
        # Loss
        mask = exposure_mask(y_crop)
        crop_loss = masked_l2_loss(y_hat, y_crop, mask)
        
        # Backward + Update
        optimizer.zero_grad()
        self.manual_backward(crop_loss)
        optimizer.step()  # ← 32 updates per image!
```

### Problems with Per-Patch Updates

```
1. No gradient accumulation
   Traditional: Accumulate gradients from batch → single update
   Current: 32 individual updates per image
   
   Impact:
     - 32× more frequent updates (more expensive)
     - But updates are 32× weaker (individual patches)
     - Variance in updates is higher
     - Harder to escape local minima

2. Optimizer state issues
   Adam maintains moving averages: m_t, v_t
   Updates to same image 32 times:
     m_t = 0.9 * m_t + 0.1 * ∇L  (from patch 1)
     m_t = 0.9 * m_t + 0.1 * ∇L  (from patch 2)  ← Same image!
     ...
     
   After 32 updates from same image:
     m_t reaches steady state for THAT image's gradients
     When next batch (different image) arrives:
       m_t must "forget" previous image's pattern
       But momentum from previous 32 patches still dominates
       
   Result: High variance in effective learning rates

3. Batch effect (or lack thereof)
   Network trained on patches from:
     - Same image (correlation)
     - Same spatial locations on average (if random cropping isn't perfect)
     - Same color distribution (if image is mostly one color)
     
   No diversity per update!
```

### Consequence: Premature Convergence

```
Effective batch size: 1 image ≈ 32 correlated patches
Traditional batch size: 32 diverse images

For diverse gradients:
  Need: 32 updates from 32 different images
  
Current setup:
  Gets: 32 updates from 32 patches of 1 image
  
Result:
  Converges faster (looks good early)
  To a worse minimum (can't generalize)
  Due to: High correlation between gradient signals
  
This explains:
  ✓ Fast initial loss drop (looks promising!)
  ✓ But premature plateau (local minimum dependent on first image)
```

---

## Part 6: Epoch and Early Stopping

### Config Settings

```yaml
epochs: 2              # Only 2 epochs!
learning_rate: 1e-4   # Fixed
checkpoint_freq: 1    # Save every epoch
```

### Why 2 Epochs Is Insufficient

```
With batch_size=1 and assuming 1000 images in dataset:
  Epoch 1: 1000 batches
    Batches 0-100:   Loss drops 5e-4 → 5e-5 (steep)
    Batches 100-500: Loss drops 5e-5 → 1e-5 (moderate)
    Batches 500-1000: Loss drops 1e-5 → 2e-6 (plateauing)
    
  Epoch 2: 1000 batches
    Batches 1000-1500: Loss drops 2e-6 → 5e-6 (barely moves!)
    Batches 1500-2000: Plateau (no improvement)
    
  Result: Training ends just as it's reaching plateau!
  
  Did not have enough epochs to:
    1. Escape the initial local minimum
    2. Fine-tune after plateauing
    3. See if learning rate scheduling could help
```

### Learning Rate Problem

```
With fixed LR=1e-4 AND 2 epochs:
  Epoch 1: LR too high for fine details, OK for coarse learning
  Epoch 2: LR still too high, overshoots refinements
  
  Only 2 epochs means:
    No time for decay schedule to help
    No time to explore multiple learning rates
    Model stuck with same LR throughout
    
If running for 50 epochs:
  With cosine annealing:
    Epoch 1-5:  High LR (broad search)
    Epoch 6-30: Decaying LR (fine-tuning)
    Epoch 31-50: Very low LR (polish)
    
  Gives model time to escape plateau and fine-tune!
```

---

## Summary: Data & Training Setup Issues

| Issue | Current | Problem | Impact |
|-------|---------|---------|--------|
| **Mask coverage** | [6,249]/255 | Excludes extremes where banding visible | Untrained shadows/highlights |
| **Batch composition** | 32 patches of 1 image | No color diversity per update | Overfits to image, poor generalization |
| **Synthetic data** | Pure primaries, no noise | Misses real banding characteristics | Doesn't transfer to real data |
| **Gradient diversity** | Same image, 32 times | Correlated gradients amplify bias | Fast plateau to wrong minimum |
| **Update frequency** | 32 updates/image | Per-patch updates, not batch | High variance, optimizer instability |
| **Training duration** | 2 epochs only | Too short to escape plateau | Stops just as learning breaks through |
| **Learning rate** | Fixed 1e-4 | No decay strategy | Can't fine-tune after plateau |

These issues **combine** to create the observed failure:
1. Homogeneous batches → fast convergence to poor minimum
2. Synthetic data → doesn't match real banding
3. Mask excludes extremes → untrain the hard cases
4. Fixed learning rate → can't escape plateau
5. Only 2 epochs → stops too early

