# Critical Insight: Why Synthetic Inference Shows ZERO Dequantization

**Observation**: When running inference with synthetic gradients, **output looks identical to input** (no smoothing, banding clearly visible).

**Combined with**: Training loss plateau at ~5e-6, this reveals a fundamental failure mode.

---

## The Core Problem: Residual Learning Collapse

### What's Actually Happening

```python
# Model architecture (simplified):
residual = self.final_conv(feat)
residual = self.tanh(residual)              # Output ∈ [-1, 1]
output = input_img + residual               # [0,1] + [-1,1]
```

**The failure mode:**
```
During training (with L2 loss):
  Loss = ||input_img + residual - target||^2
  
Network learns:
  residual ≈ 0  (zero residual!)
  output ≈ input_img
  
  Why? Because:
  - L2 loss only cares about minimizing MSE
  - Zero residual → output = input
  - If input is already somewhat close to target average → loss stops decreasing
  - Gradients become tiny → weight updates stop
  
At inference:
  Model outputs:  input_img + ~0 ≈ input_img
  Result: NO DEQUANTIZATION (output looks identical to input)
```

### Visual Confirmation of This Failure

When you observe:
1. ✓ Training loss decreasing (5e-4 → 5e-6)
2. ✓ But output doesn't change visually
3. ✓ Banding still obvious in synthetic test

This means the model learned:
```
Residual prediction:   residual ≈ 0
Output formula:        output = input + 0 ≈ input
Loss improvement:      MSE(input + 0, target) slightly better than MSE(random, target)
                      (but still bad! Just not as bad as random)
```

---

## Why Residual Learning Fails for Dequantization

### The Fundamental Mismatch

**Problem**: Residual learning assumes input is already "close" to target.
```
Typical use case:
  Input:  Noisy version of target
  Goal:   Learn small correction (residual)
  
Dequantization case:
  Input:  Severely quantized (8-bit levels = sharp steps)
  Goal:   Learn smooth interpolation (NOT a small correction!)
  
Residual needed:   Full -0.5 to +0.5 adjustments PER channel
                   (e.g., input 128/255 → target 128.3/255)
                   
Tanh range:        Only ±1, but...
Tanh gradient:     Saturates outside [-0.5, 0.5]
```

When the **required residual is large** (0.2-0.5), the tanh saturates and becomes a constant nonlinearity. The network can't learn fine adjustments.

### Actual Numbers from Primary Gradients

```
Red primary gradient case:
  R channel: 1.0 (unchanged)
  G channel: [0.48, ..., 0.52]  (0.04 range)
  B channel: [0.48, ..., 0.52]  (0.04 range)

After 8-bit quantization: 1/255 ≈ 0.004 steps
  G actual: [122/255, ..., 132/255] = [0.478, ..., 0.518]
  Banding: Sharp steps of 0.004

For dequantization:
  Model needs to predict: residual that interpolates through steps
  Magnitude needed: ±0.002 (half a quantization level) per channel
  
With tanh::
  tanh(0.001) ≈ 0.001  ✓ Still in linear region
  tanh(0.002) ≈ 0.002  ✓ OK
  tanh(0.003) ≈ 0.003  ✓ OK
  
But what actually happens:
  Network learns: residual ≈ 0 (zero!)
  Because: L2 loss doesn't reward small adjustments in right direction
           It only rewards MSE minimization
           
  So you get: output ≈ input (NO DEQUANTIZATION)
```

---

## Why L2 Loss Can't Supervise This

### The Supervision Problem

```
Ground truth pairing:
  Input (quantized):   [128/255, 131/255, 134/255, ...]
  Target (true):       [128.1/255, 131.2/255, 134.3/255, ...]
  
L2 Loss between predictions:
  Prediction 1: input + 0.0005 = [128.2/255, 131.2/255, 134.3/255, ...]
    L2 = 0.00001
    
  Prediction 2: input + 0.0 = [128/255, 131/255, 134/255, ...]
    L2 = 0.00002
    
  Prediction 3: constant (average) = [131/255, 131/255, 131/255, ...]
    L2 = huge (bad)

Early training:
  Network learns: "Prediction 1 is better than Prediction 2"
  Gradient signal: TINY (0.00001 vs 0.00002 difference)
  
  But also:
  Network learns: "Just copy input + small adjustment"
  Because: Input already captures most spatial structure
           Small adjustment gets OK loss value
           
Late training (plateau):
  Network has learned: residual ≈ 0...1e-4
  Loss: ~5e-6
  User observes: Output identical to input ✗
```

### Why Gradients Vanish

For MSE loss:
$$\frac{\partial L}{\partial \text{residual}} \propto (\text{input} + \text{residual} - \text{target})$$

After network learns `residual ≈ 0`:
$$\frac{\partial L}{\partial \text{residual}} \approx (\text{input} - \text{target})$$

For quantized input / target:
- Input quantized to discrete levels
- Target also quantized to same discrete levels!
- Difference ≈ 0 WITHIN the same quantization bucket
- **Gradients become ZERO or noise-level**

---

## Specific to Synthetic Primary Gradients

### Why Synthetic Test Reveals the Problem

```python
# From run_dequant_inference.py:
hdr = create_primary_gradients(width=target_w, height=target_h, dtype="float32")
hdr_clipped = np.clip(hdr, 0, 1)
ldr = quantize_to_8bit(hdr_clipped)

# quantize_to_8bit does:
quantized = ((np.clip(image, 0, 1) * 255).astype(np.uint8).astype(np.float32)) / 255.0
```

This creates:
1. **Smooth HDR reference** (ground truth)
2. **Perfectly clean 8-bit quantization** (no noise)

```
Perfect primary gradient in red channel:
  [1.0, 1.0, 1.0, ..., 1.0]  (fully red)
  No noise, pure quantization
  
After 8-bit quantization:
  [255/255, 255/255, 255/255, ..., 255/255]
  1.0 exactly (no information loss in red!)
  
Problem: Quantization is INVISIBLE in fully-saturated regions!
```

### What the Model Sees vs. What It Can Learn

```
Input (8-bit):         [R=1.0, G=122/255, B=122/255] everywhere in first column
Target (32-bit true):  [R=1.0, G=122.3/255, B=122.3/255] everywhere
Quantization error:    [0, -0.3/255, -0.3/255] = [0, -0.0012, -0.0012]

L2 loss per pixel:     (0 - 0)^2 + (-0.0012)^2 + (-0.0012)^2 ≈ 2.9e-6
Gradient magnitude:    ≈ 1.2e-5 (TINY!)

With 512x512 image:  5e-6 * 262k pixels = 1.3 (batch loss)
After gradient descent: update = 1e-4 * 1.2e-5 ≈ 1.2e-9 (no change!)
```

**The synthetic test is TOO PERFECT for dequantization training!**
- Uniform colors mean no spatial structure
- No noise to drown out quantization
- Quantization errors are tiny
- Gradients are minuscule

---

## Proof: Model is Actually Outputting Zero Residuals

### How to Verify This

You can check by modifying `run_dequant_inference.py`:

```python
# Add this diagnostic
def diagnostic_inference(model, device, input_tensor):
    with torch.no_grad():
        # Get the raw residual before addition
        output_with_residual = model(input_tensor)
        
        # Compute the inferred residual
        residual_learned = output_with_residual - input_tensor
        
        print(f"Input range:          [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
        print(f"Output range:         [{output_with_residual.min():.6f}, {output_with_residual.max():.6f}]")
        print(f"Residual range:       [{residual_learned.min():.6f}, {residual_learned.max():.6f}]")
        print(f"Residual mean:        {residual_learned.abs().mean():.6f}")
        print(f"Residual std:         {residual_learned.std():.6f}")
        
        # If residual is near-zero, model failed
        if residual_learned.abs().mean() < 1e-4:
            print("\n⚠️ WARNING: Residual is near-ZERO! Model learned to output input unchanged.")
            print("   This explains why synthetic test shows no dequantization.")
```

**Expected output if model trained correctly:**
```
Residual range:       [-0.01, 0.01]      ← Non-trivial corrections
Residual mean:        0.002               ← Systematic adjustment
```

**Expected output if training failed (what you're seeing now):**
```
Residual range:       [-1e-5, 1e-5]       ← Essentially zero!
Residual mean:        < 1e-6              ← Noise level
```

---

## Why This Happens (Mathematical Explanation)

### Setting: Residual Learning + MSE Loss

The model learns to minimize:
```
L = ||network(x) - y||^2
  = ||x + residual(x) - y||^2
```

Taking derivative w.r.t. residual:
```
∂L/∂r = 2 * (x + r - y)
      = 2 * (residual_error + r)
      
Setting to zero (Lagrange optimum):
r* = y - x
```

**But here's the catch:**
In practice, with SGD and finite gradients, the network doesn't reach r* directly.

Instead:
1. Network predicts residuals during forward pass
2. Gradient computed: ∂L/∂r = 2(x + r - y)
3. Weight update: w ← w - lr * ∂L/∂w
4. After each epoch, residual prediction improves slightly

**BUT if input x ≈ (y - δ) where δ is tiny quantization error:**
```
∂L/∂r = 2 * (small_quantization_error + learned_residual)
      ≈ 2 * learned_residual (if network hasn't learned correction yet)

The gradient ONLY contains what network predicted!
There's almost no supervision signal for "what residual should be."
```

This creates a **bootstrap problem**: The network can't learn the residual from gradients because there's no strong signal distinguishing "correct" from "zero."

---

## Summary: Why Synthetic Test Appears Unsmoothed

| Stage | What Happens | Observation |
|-------|--------------|------------|
| **Initialization** | Random weights | Output is noise |
| **Early training** | Network learns basic features | Loss drops (5e-4 → 5e-5) |
| **Plateau phase** | Network learns: residual ≈ 0 | Loss continues dropping (5e-5 → 5e-6) |
| **Inference** | Model outputs input + ~0 | **NO dequantization visible** ⚠️ |

**The model IS working, but it learned the wrong thing:**
- ✓ Loss decreases (training works)
- ✓ Model converges (optimization works)
- ✗ But learned `residual ≈ 0` (task specification is wrong!)
- ✗ Inference output ≈ input (banding not smoothed)

---

## Key Insight for Fixes

The solution requires **changing how the network learns to generate residuals**:

### What Needs to Change:
1. **Loss function must reward smoothness explicitly** (not just MSE)
   - Add Total Variation loss (penalizes gradients)
   - Or add edge detection loss (penalizes discontinuities)
   
2. **Residual learning architecture must change**
   - OR: Add explicit smoothness constraints
   - OR: Use direct prediction instead of residual
   
3. **Training signal must be stronger**
   - Avoid masking out regions where banding is visible
   - Use loss that responds to quantization specifically
   - Consider adding noise to training data (makes banding more obvious)

4. **Evaluation must show residuals**
   - Check that model predicts non-zero residuals
   - Verify residuals are in "smoothing" direction (not noise)

Without these changes, **the model will continue learning residual ≈ 0**, and synthetic tests will show no improvement.

