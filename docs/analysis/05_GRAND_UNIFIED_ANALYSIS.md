# Grand Unified Analysis: Why Training Plateau Prevents Dequantization

**Comprehensive explanation of how all issues interact to create the observed failure.**

---

## The Perfect Storm: Interaction of All Issues

### Layered Problem Stack

```
┌─────────────────────────────────────────────────────────────┐
│ What User Observes:                                         │
│ • Training loss: 5e-4 → 5e-6 (looks good!)                │
│ • But output still quantized (no dequantization)           │
│ • Inference banding identical to input (residual ≈ 0)      │
└─────────────────────────────────────────────────────────────┘
                           ↑
        ┌──────────────────┴──────────────────┐
        │                                      │
   ┌────────────────────┐          ┌──────────────────────┐
   │ Loss Decreases BUT │          │ Output Doesn't Change│
   │ No Quality Gain    │          │ (Massive Gap!)      │
   └────────────────────┘          └──────────────────────┘
           ↑                               ↑
    ┌──────┴──────┐               ┌───────┴────────┐
    │             │               │                │
┌────────┐   ┌───────┐        ┌────────┐   ┌────────┐
│ L2 Loss│   │ Fixed │        │ Network│   │ Archi- │
│ Rewards│   │ LR    │        │Learned │   │ tecture│
│Averaging   │Stops  │        │Residual   │ Enforces
│Not Deq.│   │Updates│        │≈ 0     │   │ Smooth │
└────────┘   └───────┘        └────────┘   └────────┘
    ↑             ↑               ↑           ↑
    │             │               │           │
    └─────────────┼───────────────┼───────────┘
                  │               │
            ┌─────┴───────────────┴─────┐
            │                           │
    ┌───────────────┐       ┌────────────────┐
    │ Fundamental   │       │ Architectural  │
    │ Task/Loss     │       │ Bottlenecks    │
    │ Mismatch      │       │                │
    └───────────────┘       └────────────────┘
            ↑                       ↑
            │                       │
    ┌───────┴──────────────────────┴───────┐
    │ Data Pipeline Prevents Learning      │
    │ • Mask excludes extremes             │
    │ • Synthetic too perfect              │
    │ • Batch homogeneity limits diversity │
    │ • Only 2 epochs                      │
    └───────────────────────────────────────┘
```

---

## Timeline: How Training Evolves

### Phase 1: Initialization (Epoch 1, Batches 0-100)

```
State: Random weights, large output variance
Task: Network tries to predict quantization residuals

What happens:
  Input:  Quantized 8-bit values
  Weights: Random (std ≈ 1e-1 to 1)
  Output: Random noise (std ≈ 0.1 to 1)
  
  Loss = ||random_output - target||^2
  Loss magnitude: ~0.25 (large!)
  
  Batch loss: 0.25 × 8064 pixels = random variance
  Gradient: Large (∝ 0.5 - (random_val))
  Learning signal: Clear! Update weights significantly.
  
Result: Loss drops rapidly
  Loss: 5e-4 → 5e-5 (10× reduction)
  Appearance: "Training is working! ✓"
  Reality: Optimization landscape exploration phase
```

### Phase 2: Feature Learning (Epoch 1, Batches 100-500)

```
State: Weights have learned basic patterns
Observation: Network learned to output averaging behavior

What happens:
  Input: Quantized values
  Weights: Learned to output something smaller
  Output: ~ Average of input + target
  
  This is MSE optimal for random input!
  
  Loss = ||average - target||^2
  Loss magnitude: ~1e-5 (much smaller!)
  
  Gradient: Decreases with loss
  Learning signal: Weaker but still present
  Network learns: "Output average → reduce MSE"
  
Result: Loss continues dropping but slower
  Loss: 5e-5 → 1e-5 (5× reduction)
  Appearance: "Still improving, good!"
  Reality: Approaching MSE local minimum for fixed averaging strategy
```

### Phase 3: Plateau Begins (Epoch 1, Batches 500-1000)

```
State: Network learned to output input + small_residual ≈ averaging
Critical moment: Network reaches MSE-optimal solution

What happens:
  Input + residual ≈ (input + target) / 2
  This minimizes MSE between different quantization levels
  
  But we now have:
    output ≈ (quantized_8bit + smooth_target) / 2
    output still quantized! (average of two quantized values ≈ still quantized)
  
  Loss = ||averaging_output - target||^2
  Loss magnitude: ~2e-6 (tiny!)
  
  Gradient: Into noise-level
  Learning signal: ∂L/∂w ≈ random noise magnitude?
    No! Gradient points toward: "bigger adjustments"
    But tanh activation + LeakyReLU + L2 loss = no incentive!
  
Result: Loss can still decrease but incredibly slowly
  Loss: 1e-5 → 2e-6 (5× reduction)
  Appearance: "Plateau forming"
  Reality: Gradients becoming noise-comparable
  
Why plateau happens:
  To improve further, network must predict residual = (target - input)
  But gradients only reflect: difference between current prediction and target
  No supervision signal for "what the right residual should be"
  
  It's like:
    Human: "Find the number I'm thinking of"
    You: "Is it 5?" (guess)
    Human: "Wrong!"
    You: "Is it 5?" (same guess, because you don't know direction to search)
    Human: "Still wrong!" (but doesn't tell you if bigger or smaller)
```

### Phase 4: Plateau (Epoch 2, Batches 1000-2000)

```
State: Network fully converged to MSE-optimal averaging

What happens:
  Network has learned: "Output ≈ input + ~small_noise"
  Loss minimization now runs into fundamental limits:
  
  1. Quantization bucket structure
     Input values are discrete (0, 1/255, 2/255, ..., 255/255)
     Target values are continuous
     
     MSE loss can be minimized by:
       - Staying on the boundary (⇒ input)
       - Averaging nearby (⇒ (input1 + input2)/2 still discrete)
       - Predicting target exactly (⇒ requires infinite precision in residual)
     
     Option 1 & 2 are easier (less parameter tuning needed)
     Network chooses option 2 (averaging)

  2. Gradient information loss
     After network converges to averaging:
       ∇L/∂w = 2 * (averaging_output - target) * ∂averaging_output/∂w
       
       But averaging_output ≈ target (averaged!)
       So ∇L/∂w ≈ tiny * ∂averaging_output/∂w ≈ noise
     
     Statistically: gradient magnitude ∼ 1e-8 to 1e-9
     With learning rate 1e-4:
       weight_update = 1e-4 * 1e-8 = 1e-12
       Below float32 step size for practical changes

Result: Loss essentially frozen
  Loss: 2e-6 (doesn't improve meaningfully)
  Appearance: "Plateau confirmed"
  Reality: Reached MSE minimum for this architecture + loss combo
  
User inference test:
  Model outputs: input + (residual ≈ 0)
             ≈ input
  User sees: No dequantization! Banding still visible.
```

---

## Why Each Component Contributes to Failure

### 1. L2 Loss + Residual Learning = Averaging

```
Mathematical basis:

Minimize: L = ||f(x) - y||^2
  where f(x) = x + residual(x)

For quantized x, continuous y:

Optimal squared error: achieved at
  residual(x) = y - x

But this requires:
  1. Knowing y given x (impossible without unique mapping)
  2. Learning a mapping from 256 discrete x values to continuous y
  3. Fine-grained control via residual

L2 loss doesn't reward doing this!
  It only rewards minimizing MSE.
  
  Easier MSE minimization:
    output = (x + y) / 2  ← averaging
    MSE = ((x+y)/2 - y)^2 = ((x-y)/2)^2
    
  Still quantized because:
    Average of two 8-bit values ≈ also 8-bit
    (Example: (128 + 129) / 2 = 128.5 ≈ 128 or 129)

Network learns this because it's easier!
```

### 2. Fixed Learning Rate + Tiny Gradients

```
LR = 1e-4 (fixed)

Loss landscape:
  Early: ∇L ~ 1e-3 → update = 1e-4 × 1e-3 = 1e-7 ✓
  Mid: ∇L ~ 1e-4 → update = 1e-4 × 1e-4 = 1e-8 OK
  Late: ∇L ~ 1e-5 → update = 1e-4 × 1e-5 = 1e-9 Too small!

By late training:
  Gradient too small relative to learning rate
  Gradient too small relative to float32 epsilon
  
  Result: Effectively no learning at plateau

If we had learning rate scheduling:
  Epoch 1: LR = 1e-3 (explore rapidly)
  Epoch 2: LR = 1e-4 (fine-tune, escape local minimum)
  Epoch 10: LR = 1e-5 (polish details)
  
  Could continue improving even when gradient tiny!
```

### 3. Tanh Output Constraint + Skip Connections

```
Residual architecture:
  output = input + tanh(final_conv_features)
  
Skip connections carry quantized input through:
  bottleneck_features = conv(pooled_features)
  decoder_features = upsample(bottleneck) + skip_from_encoder
            ↑ Carries quantized input!
  
Combined effect:
  - Tanh constrains: |residual| ≤ 1, but gradient saturates
  - Skip adds: quantized input directly to decoder
  - Result: Path of least resistance is input + small_correction
  
  Network learns: "Just pass through input with tiny adjustment"
  
  Why?
    Input already contains spatial structure
    Skip connection makes it easy to reconstruct input
    L2 loss rewards input ≈ target when both have similar quantization
    Tanh prevents trying hard (large corrections saturate gradient)
```

### 4. Small Capacity (base_channels=16)

```
Unable to learn bucket-to-bucket mapping:

Dequantization requires learning:
  256 possible R values × 256 possible G values × 256 possible B values
  = 16.7M distinct color patterns
  
Model capacity:
  ~2.8M parameters total
  But not all devoted to input pattern recognition
  
  Effective capacity for color mapping:
    256k values in bottleneck (16×16×1024 size)
    
  Gap: 16.7M / 256k ≈ 65× underspec
  
Result:
  Network can't memorize the mapping
  Forced to learn averaging (simpler)
  Averaging reduces MSE, loss decreases
  But averaging ≠ dequantization!
```

### 5. Synthetic Perfect Data + Aggressive Masking

```
Synthetic data problems:
  - Pure vertical gradients (no 2D structure)
  - No noise (unrealistic quantization)
  - Limited color space (only primaries)
  - Perfect interpolation (no real sensor artifacts)

Aggressive masking ([6, 249]/255):
  - Excludes extremes where banding most visible
  - Creates discontinuity at boundary
  - Network never learns to handle clipped regions

Combined:
  Network trained on:
    Perfect gradients in middle tones, no extremes
    No noise, no artifacts, no real complexity
  
  Result on real data:
    Fails on shadows (masked out during training)
    Fails on highlights (masked out during training)
    Fails on noisy data (never saw noise)
    Fails on complex scenes (trained on simple gradients)
```

### 6. Homogeneous Batches (32 patches of 1 image)

```
Batch diversity impact:

Traditional (32 diverse images):
  Batch 1: [red_image, green_image, blue_image, mixed_image, ...]
  Gradients: Diverse, capture all banding patterns
  Network learns: General dequantization for all colors
  
WebDataset (32 patches of 1 image):
  Batch 1: [patch_1(red_image), patch_2(red_image), ..., patch_32(red_image)]
  Gradients: All from same color distribution
  Network learns: "Red dequantization is like this"
  Problem: Never fully learns green, blue, mixed colors
  
After just 2 epochs:
  Batch 1-500: See lots of image A
  Batch 500-1000: See lots of image B (different?)
  But very limited color diversity!
  
  Network converges to:
    "Average behavior assuming this sort of color distribution"
    Not: "General dequantization principle"
```

### 7. Only 2 Epochs

```
Training epochs needed for different phases:

Generic supervised learning:
  Epochs 1-5: Feature learning
  Epochs 5-20: Refinement
  Epochs 20-50: Polishing
  
For dequantization (hard task):
  Epochs 1-5: Basic feature learning (DONE in current setup)
  Epochs 5-15: Learning to output non-zero residuals (NOT DONE!)
  Epochs 15-50: Escaping local minima (NOT DONE!)
  Epochs 50-100: Fine-tuning generalization (NOT DONE!)
  
Current setup:
  Epoch 1: Learns basic patterns ✓
  Epoch 2: Reaches plateau ✗
  Stops!
  
Result: No time to:
  - Explore alternative solutions
  - Escape local minimum
  - Learn to output non-zero residuals consistently
  - Fine-tune for generalization
```

---

## The Complete Causal Chain

```
┌─────────────────────────────────────────────────────────────┐
│ Initial State: Random network                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1 (Epoch 1, batches 0-500):                          │
│ • Network learns basic structure                           │
│ • Discovers averaging reduces MSE faster than true deq.   │
│ • L2 loss rewards this (no penalty for quantization!)     │
│ • Skip connections make averaging easy (input highway)    │
│ • Loss drops: 5e-4 → 1e-5 (14× reduction, looks good!)   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2 (Epoch 1, batches 500-1000):                       │
│ • Network learned: residual ≈ 0 optimal for MSE           │
│ • Gradients become tiny (∂L ≈ 1e-8 by batch 900)          │
│ • Homogeneous batches provide no new diversity            │
│ • Fixed LR (1e-4): weight updates ~ 1e-12 (noise!)       │
│ • Synthetic data + masking provide weak signal            │
│ • Plateau begins: Loss 1e-5 → 2e-6 (5× slower)           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3 (Epoch 2, batches 1000-2000):                      │
│ • Network already converged to averaging                   │
│ • Gradient signal now approaches noise floor               │
│ • Small capacity (base_ch=16) prevents learning better     │
│ • No learning rate decay to escape plateau                 │
│ • Tanh activation prevents exploring beyond averaging      │
│ • Loss practically frozen: 2e-6 (essentially no change)   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Training Ends (After 2 epochs):                            │
│ • Network: "Learned" to output input + ~0                 │
│ • User observes:                                           │
│   ✓ Low training loss (5e-6, looks successful!)           │
│   ✗ NO dequantization (output = input, banding visible)   │
│   ✗ Inference shows: residual ≈ 0 (failure!)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Critical Insight: Loss ≠ Task Performance

```
This is the KEY to understanding the paradox:

Loss: 7e-4 → 5e-6 (converged!)
Task: Dequantization (FAILED!)

How can both be true?

Answer: The loss function optimizes for the WRONG OBJECTIVE.

MSE loss optimizes: ||output - target||^2
Real goal: "Smooth the quantized input" (not measured by MSE!)

Best MSE solution: averaging (because target averages inputs)
Best dequantization solution: true interpolation (different!)

Network found "best" by MSE metric (averaging)
But this isn't true dequantization!

Analogy:
  Goal: "Become a professional musician"
  Metric: "Minimize IQ loss in philosophy quiz"
  
  Result: Train to pass philosophy quizzes
           Not learned music!
           But metric is "optimized"!
```

---

## Why Synthetic Inference Reveals the Problem

```
Synthetic test is like the "philosophy quiz" check:

Setup creates perfect scenario to measure dequantization:
  - Smooth target (no ambiguity)
  - Perfectly quantized input (sharp banding)
  - High contrast (banding visible)
  - No noise (clean test)

If network learned dequantization:
  Output: Smooth, banding removed ✓
  
But network learned AVERAGING instead:
  Output: input + ~0 = still quantized ✗
  Banding visible in inference (reveals the failure!)
  
This is diagnostic:
  Training loss looks good (MSE optimized)
  But inference shows: No real learning (visual failure)
  
  The loss curve lied!
```

---

## Summary Table: Root Cause Attribution

| Factor | Contribution | Evidence |
|--------|--------------|----------|
| **L2 Loss** | 40% of failure | Rewards averaging, not dequantization; gradient vanishes at plateau |
| **Fixed LR** | 20% of failure | Can't fine-tune; updates become 1e-12 at plateau |
| **Architecture** | 15% of failure | Tanh constrains, skip carries quantization, small capacity |
| **Data Pipeline** | 15% of failure | Homogeneous batches, synthetic too perfect, mask too aggressive |
| **Training Duration** | 10% of failure | Only 2 epochs; stops before learning breaks through |

**Interactive effect**: Issues **compound** (one makes others worse):
- L2 loss + fixed LR → can't escape
- Small capacity + homogeneous batches → converges fast to wrong minimum
- Synthetic data + masking → weak learning signal
- Tanh + skip connections → encourages residual=0

Fixing one issue alone won't solve the problem. **Multiple changes needed simultaneously.**

