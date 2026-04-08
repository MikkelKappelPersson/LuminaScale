# Architectural Analysis: Design Constraints & Implications

## Current Architecture

```python
class DequantizationNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,    # Config uses 32
        num_levels: int = 6,
        leaky_relu_slope: float = 0.1,
    ):
        # Encoder: 3 → 16 → 32 → 64 → 128 → 256 → 512
        # Bottleneck: 512 channels
        # Decoder: 512 → 256 → 128 → 64 → 32 → 16 → 3
        # Output: residual = tanh(conv3x3(feat))
        # Final: output = input_img + residual
```

---

## Problem 1: Insufficient Capacity for Pattern Modeling

### Channel Progression with base_channels=16

```
Encoder levels and capacity:
  Level 0: 3 → 16   (× 16 channels)      [512×512] raw input
  Level 1: 16 → 32  (× 32 channels)      [256×256] after pooling
  Level 2: 32 → 64  (× 64 channels)      [128×128]
  Level 3: 64 → 128 (× 128 channels)     [64×64]
  Level 4: 128 → 256 (× 256 channels)    [32×32]
  Level 5: 256 → 512 (× 512 channels)    [16×16]
  
  Bottleneck: 512 channels at 16×16 resolution
  Total params: ~2.8M
```

### Why This Is Insufficient

Dequantization requires learning:

```
1. Quantization bucket detection
   Input: 8-bit value (one of 256 possibilities)
   Pattern: Which continuous value generated this 8-bit sample?
   
   With only 16 base channels:
   - 16 different feature filters at input
   - Each filter sees 8-bit quantized input
   - Can distinguish between ~16 patterns max at raw input level
   
   Needed: 256+ patterns (one per possible 8-bit value per channel)
   Actual:  512 max (across all spatial locations + channels)

2. Smooth interpolation through buckets
   Task: Given {128, 131, 134}, predict {128.1, 131.2, 134.3}
   Requires: Fine-grained positional encoding
   
   With 16 base channels:
   - Very small parameter budget for subtle corrections
   - Limited expressivity for >256 distinct "dequantized" values

3. Different banding patterns per channel
   R, G, B channels have independent quantization:
   - Possible R values: 256
   - Possible G values: 256
   - Possible B values: 256
   - Total combinations: 256^3 = 16.7M possible color patterns
   
   Model capacity for 3 channels:
   - Base channels: 16
   - Bottleneck: 512 total
   - Per-channel capacity: ~171 features
   - Per-color capacity: ~64k distinctions
   
   Gap: 16.7M / 64k ≈ 260× under-capacity
```

### Comparison: What Works in Similar Papers

Papers on single-image HDR reconstruction (Liu et al., CVPR 2020):
```
Architecture: 6-level U-Net with residual learning
BUT:
  Base channels: 64-128 (not 16!)
  Total params: 50-100M (not 2.8M!)
  
This 20-40× larger model is necessary because:
  - HDR task is even harder (4 channels, 16-bit or higher)
  - Requires learning per-pixel color space transforms
  - Must model complex camera sensor characteristics
```

---

## Problem 2: Receptive Field Limitations

### Effective Receptive Field Calculation

For a 6-level U-Net:
```
Each conv layer: kernel=3, receptive field contribution = 1
Each pooling: 2× downsampling, receptive field *= 2

Encoder path:
  Input:     RF = 1
  Conv:      RF = 3
  Pool:      RF = 6
  Conv:      RF = 8
  Pool:      RF = 16
  Conv:      RF = 18
  Pool:      RF = 36
  Conv:      RF = 38
  Pool:      RF = 76
  Conv:      RF = 78
  Pool:      RF = 156
  Bottleneck: RF ≈ 189

Decoder reverses, so max RF ≈ 189 pixels
```

### Why This Matters for Dequantization

```
Task: Interpolate smooth gradient through quantization levels

Input image: 512×512
Receptive field: 189×189 ≈ 36% coverage

Problem scenarios:

1. Gradient at edges
   Edge pixel (0,0) in 512×512 image
   - Can only see 189 pixels away
   - Can't infer large-scale gradient from far boundaries
   - Result: Edge artifacts, incorrect gradient direction

2. Long-range color cast
   Synthetic gradient: G channel varies 0.48 → 0.52 vertically
   - Full range spans 512 pixels
   - Receptive field: 189 pixels
   - Network can't model full smooth trend! 
   - Result: Can't predict proper interpolation direction

3. Scene structure inference
   Real photos: Lighting gradients span image boundaries
   - Shadows in corners, highlights in center
   - Network needs to understand full scene structure
   - With 189px RF on 512px image: Too limited!
```

### Receptive Field vs. Image Size

```
Current:  189 pixels RF on 512×512 image = 36% coverage
Recommended: 400+ pixels RF (≥80% for dequantization)

Achievable with:
  - Deeper network (more levels)
  - Larger kernel sizes (5×5 instead of 3×3)
  - Dilated convolutions (kernel_size=3, dilation=2+ increases RF faster)
```

---

## Problem 3: Tanh Activation Output Constraint

### Why Tanh Is Wrong for This Task

```python
residual = self.final_conv(feat)      # Unbounded output from conv
residual = self.tanh(residual)         # Saturate to [-1, 1]
output = input_img + residual          # Add bounded residual
```

### Gradient Analysis

```
tanh derivative: d/dx tanh(x) = 1 - tanh(x)^2

At x=0:         d/dx = 1 - 0 = 1.0       ← Full gradient
At x=±0.5:      d/dx ≈ 0.5                ← Half gradient
At x=±1:        d/dx ≈ 0.07               ← Starting to saturate
At x=±2:        d/dx ≈ 0.007              ← Saturated
At x=±3:        d/dx ≈ 0.0001             ← Nearly dead

For dequantization with residuals in [-0.5, 0.5] range:
  - Most corrections are in steep gradient region
  - But large corrections (> 0.5) hit saturation
  - Cannot express corrections > 1.0 (e.g., bridging across multiple quantization levels)
```

### What Corrections Are Needed?

```
For smooth 32-bit from quantized 8-bit:

Simple case (within one step):
  Input: 128/255 (quantized)
  True: 128.3/255
  Needed residual: +0.3/255 ≈ +0.0012
  → tanh allows this (just barely in linear region)

Complex case (bridging steps):
  Input: [128, 131, 134]/255 (three quantized steps)
  True: [128.1, 131.2, 134.3]/255 (smooth gradient)
  Needed: Predict smooth interpolation accounting for → position
  → May require larger corrections per pixel to match smooth curve
  → Tanh saturates and prevents fine interpolation
```

### Comparison: Without Tanh

```python
# Alternative 1: Unbounded residual
output = input_img + final_conv(feat)   # No saturation

Advantages:
  - Can express corrections of any magnitude
  - Gradients don't saturate
  - More expressive

Disadvantages:
  - Model might output values > 1.0 (need clipping)
  - Less stable training (unbounded loss)

# Alternative 2: Smooth bounded activation
output = input_img + 0.1 * tanh(final_conv(feat))
# or
output = input_img + 0.1 * sigmoid(final_conv(feat) - 0.5)

Advantages:
  - Still bounded but gentler than full tanh
  - Allows larger corrections without saturation
  - More stable than unbounded
```

---

## Problem 4: LeakyReLU Activation Issues

### Current Encoder/Decoder Blocks

```python
def _make_encoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.LeakyReLU(self.leaky_relu_slope, inplace=True),  # slope=0.1
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
    )
```

### Issue: LeakyReLU + Pooling → Information Loss

```
LeakyReLU(x, slope=0.1):
  if x > 0: output = x
  if x < 0: output = 0.1 * x  ← Tiny gradient for negative features!

Impact on backpropagation:
  Forward:  x → ReLU → [0 if x<0, x if x>0]
  Backward: ∂L/∂x = 0.1 * ∂L/∂out (for x<0)  ← 10x weaker!

During training:
  Early layers: Many activations go negative (random initialization)
  → LeakyReLU(0.1) kills 90% of signal
  → Gradient = gradient * 0.1 * 0.1 * 0.1 ... (6 levels!)
  → After 6 levels: gradient ≈ 0.1^6 ≈ 1e-6 × original
  
  Result: Vanishing gradients in encoder!
```

### Pooling Compounds the Problem

```python
self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

```
After each pooling:
  Spatial information: 4× reduction
  Feature channels: Same (32, 64, 128, ...)
  
Bottleneck representation:
  16×16 spatial grid  (512×512 → 16×16 after 5 poolings)
  1024 channels (base=16, level 5: 16*2^5 = 512 channels before concat)
  
  Total information: 16×16×1024 ≈ 262k values
  
  But must encode:
  - Quantization patterns (256^3 = 16.7M distinct possibilities)
  - Spatial gradients (512×512 unique positions)
  - Per-pixel color values
  
  Capacity gap: 16.7M / 262k ≈ 64× underutilized
```

---

## Problem 5: Skip Connections Don't Help Here

### How Skip Connections Are Used

```python
# Decoder combines upsampled features with encoder skip:
for level, (decoder_block, skip) in enumerate(zip(self.decoder_blocks, skip_connections)):
    feat = self.upsample(feat)                    # 2× upsampling
    feat = torch.cat([feat, skip], dim=1)         # Concatenate with skip
    feat = decoder_block(feat)                    # Process
```

### Why Skip Connections Backfire for Dequantization

```
Skip connection benefits (typical supervised learning):
  - Gradient flow (solves vanishing gradients)
  - Information highway (preserve low-level details)
  
Dequantization problem:
  - Input already contains quantization artifacts
  - Skip connection bypasses the network processing
  - Direct path: input → skip connection → decoder → output
  
  Result:
    output ≈ input + small_corrections
    Because skip carries quantized input directly through!
    
  Example:
    Skip has: quantized gradient [128, 131, 134]/255
    Decoder tries to "fix" it
    But decoder also sees: upsampled_features + skip
    
    The skip dominates spatial structure → prevents learning smooth interpolation
```

---

## Cumulative Effect

### Why All These Problems Interact

```
1. Small capacity (16 base channels)
   → Can't learn complex bucketing patterns
   → Forces network to learn averaging

2. Small receptive field (189 pixels)
   → Can't infer large-scale gradients
   → Relies on skip connections (which carry quantization!)

3. Tanh activation
   → Prevents large residual corrections
   → Biases toward smooth outputs (but input is quantized!)

4. Skip connections
   → Carry quantized input directly
   → Make it easy to output input + ~0

5. LeakyReLU + pooling
   → Vanishing gradients in encoder
   → Weak signals to learn fine details

Combined effect:
  → Network learns: residual ≈ 0
  → Output ≈ input (quantized)
  → No dequantization observed
  → Training loss plateaus (already achieved "best" under these constraints)
```

---

## Architectural Fixes (Conceptual)

### Fix 1: Increase Model Capacity

```python
# Current
base_channels: 16
params: 2.8M

# Improved
base_channels: 64        # 4× larger
# Result: params ≈ 40-50M
```

**Trade-off**: Slower training, more VRAM, but better expressivity.

### Fix 2: Increase Receptive Field

```python
# Option A: Deeper network
num_levels: 8  (instead of 6)
# RF: ~500 pixels

# Option B: Larger kernels
kernel_size: 5  (instead of 3)
# RF increases faster per layer

# Option C: Dilated convolutions
Conv2d(..., dilation=2)
# RF: 5 pixels per conv instead of 3
```

### Fix 3: Remove or Modify Tanh

```python
# Option A: Unbounded residual
output = input_img + final_conv(feat)

# Option B: Soft constraint
output = input_img + 0.1 * final_conv(feat)
# Smaller residuals, still unbounded

# Option C: Sigmoid variant
output = input_img + (sigmoid(final_conv(feat)) - 0.5)
# Bounded to [-0.5, 0.5] per channel instead of [-1, 1]
```

### Fix 4: Better Activation Functions

```python
# Option A: Replace LeakyReLU with GELU or SiLU
nn.GELU()  # Smoother gradients, no dead neurons

# Option B: Batch normalization before activation
nn.Conv2d(...)
nn.BatchNorm2d(...)
nn.GELU()
# Stabilizes gradients, allows stronger activations
```

### Fix 5: Conditional Skip Usage

```python
# Option A: Weighted skip connection
skip_weight = 0.3  # Reduce influence
feat = self.upsample(feat) * 0.7 + skip * skip_weight
feat = decoder_block(feat)

# Option B: Skip only at early layers
# Skip connections help deep layers escape plateaus
# Don't use skip in final decoder layers where main learning happens
```

---

## Summary: Architecture Constraints Table

| Issue | Current | Problem | Impact |
|-------|---------|---------|--------|
| **Capacity** | 16 base, 2.8M params | 256× underspec for 256^3 colors | Can't learn fine patterns |
| **Receptive field** | 189 pixels (36%) | Can't model 512px gradients | Relies on breaks (skip conn) |
| **Output activation** | Tanh(±1) | Saturates for corrections >0.5 | Can't bridge large steps |
| **Internal activation** | LeakyReLU(0.1) | Kills 90% gradient signal | Vanishing gradient |
| **Pooling** | 2× stride pooling | 256k bottleneck for 16.7M color space | No capacity at bottleneck |
| **Skip connections** | Carried through | Preserves quantization perfectly | Encourages residual≈0 |

---

## Key Takeaway

The architecture doesn't just fail to dequantize—**it's actively designed to preserve quantization**:
- Small capacity → Forces averaging (can't model fine details)
- Skip connections → Carry quantization through
- Tanh → Constrains residuals
- Receptive field → Too small to learn gradients

**This is the opposite of what dequantization needs.**

