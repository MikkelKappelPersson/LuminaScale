# Color Transform Libraries Quick Reference Table

**Date**: April 5, 2026 | **Project**: LuminaScale

## Library Comparison Matrix

| Feature | **Kornia** | **scikit-image** | **OCIO (Current)** | **Colour Science** | **torchvision** |
|---------|:----------:|:---------------:|:------------------:|:------------------:|:---------------:|
| **Backend** | PyTorch | NumPy | OpenGL/CUDA | NumPy | PyTorch |
| **GPU Support** | ✅ Native CUDA | ❌ CPU only | ✅ GPU (OpenGL) | ❌ CPU only | ⚠️ Limited |
| **Differentiable** | ✅ Yes | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Type Hints** | ✅ Full | ⚠️ Partial | ✅ Partial | ⚠️ Minimal | ✅ Yes |
| **ACES Support** | ❌ No | ❌ No | ✅✅✅ Full | ⚠️ Basic | ❌ No |
| **Batch Optimized** | ✅ Yes | ✅ Yes | ⚠️ Via LUTs | ⚠️ Limited | ✅ Yes |
| **Tone Curves** | ⚠️ Limited | ⚠️ Basic | ✅ Via LUT | ✅ Yes | ❌ No |
| **3D LUT Support** | ❌ No | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Color Metrics** | ✅ ΔE | ✅ ΔE multiple | ❌ No | ✅ Multiple | ❌ No |
| **Dependencies** | PyTorch | NumPy only | OpenGL/EGL | NumPy | PyTorch/Vision |
| **Documentation** | ✅ Excellent | ✅ Very good | ✅ Good | ⚠️ Academic | ✅ Good |
| **License** | Apache 2.0 | BSD | BSD | BSD | BSD |

---

## Color Space Coverage Comparison

### RGB-based Conversions

| Space | **Kornia** | **scikit-image** | **OCIO** | **Colour Science** |
|-------|:----------:|:---------------:|:--------:|:------------------:|
| RGB ↔ HSV | ✅ | ✅ | ✅ | ✅ |
| RGB ↔ LAB | ✅ | ✅ | ✅ | ✅ |
| RGB ↔ XYZ | ⚠️ Via custom | ✅ | ✅ | ✅ |
| RGB ↔ YUV | ✅ | ✅ | ✅ | ✅ |
| sRGB OETF | ✅ | ⚠️ Via formula | ✅ | ✅ |
| ACES2065-1 | ❌ | ❌ | ✅✅✅ | ⚠️ Partial |
| ACEScg / AP1 | ❌ | ❌ | ✅✅✅ | ⚠️ Partial |
| ACES RRT | ❌ | ❌ | ✅✅✅ | ❌ |
| ACES ODT | ❌ | ❌ | ✅✅✅ | ❌ |

---

## Performance Comparison

| Metric | **Kornia** | **scikit-image** | **OCIO** | **PyTorch Hybrid** | **Notes** |
|--------|:----------:|:---------------:|:--------:|:------------------:|-----------|
| RGB↔LAB (1024×1024) | ~1-2ms | ~50-100ms | N/A | ~1ms | GPU timing |
| ACES2065→sRGB | ❌ Unsupported | ❌ Unsupported | ~8-11ms | ~3-5ms | Estimated |
| Memory (RGB buffer) | 12MB (GPU) | 12MB (CPU) | 12MB + 5MB LUT | 12MB (GPU) | For 1024×1024 |
| Startup (first call) | ~50ms | ~20ms | ~100-115ms | ~50ms | Shader compile |

---

## Feature Matrix - ACES Specific

| ACES Component | **Support in Kornia** | **Support in OCIO** | **PyTorch Hybrid** |
|----------------|:--------------------:|:------------------:|:------------------:|
| **AP0 color space** | ❌ Custom needed | ✅ Native | ✅ Via matrices |
| **AP1 color space** | ❌ Custom needed | ✅ Native | ✅ Via matrices |
| **RRT (Rendering Transform)** | ❌ Not available | ✅ Full | ⚠️ LUT-based |
| **ODT (Output Display)** | ❌ Not available | ✅ Full | ⚠️ LUT-based |
| **Tone mapping** | ❌ None | ✅ Via LUT | ⚠️ Power-law approx |
| **Gamut compression** | ❌ None | ✅ Via LUT | ⚠️ Approximate |
| **HDR support** | ⚠️ Manual | ✅ Full | ✅ Manual |
| **Multiple display configs** | ❌ No | ✅ Yes (100-4000 nits) | ⚠️ Manual |

---

## When to Use Each Library

### **Use Kornia When:**
- ✅ Building pure PyTorch color pipelines
- ✅ Need differentiable transforms for learning
- ✅ Want TorchScript deployment
- ✅ Working with standard color spaces (RGB, LAB, YUV)
- ✅ Need color metrics (ΔE computation)
- ❌ **NOT** for ACES-specific work without custom implementation

**Example**: 
```python
import kornia.color as K
import torch

rgb = torch.rand(3, 256, 256).cuda()
lab = K.rgb_to_lab(rgb)  # Differentiable!
```

---

### **Use scikit-image When:**
- ✅ Need reference implementations for validation
- ✅ Working offline (CPU is acceptable)
- ✅ Implementing color science algorithms
- ✅ Want well-documented, simple formulas
- ❌ **Not** for real-time or GPU-accelerated pipelines

**Example**:
```python
import skimage.color as skc
import numpy as np

rgb = np.random.rand(256, 256, 3)
lab = skc.rgb2lab(rgb)  # For reference
```

---

### **Use OCIO (Current LuminaScale) When:**
- ✅ Need **exact ACES specification compliance**
- ✅ Production rendering with multiple display configs
- ✅ Inference-only (no backprop needed)
- ✅ Maximum accuracy required
- ✅ Non-differentiable transforms acceptable
- ❌ **Not** for learnable/differentiable pipelines

**Current LuminaScale**:
```python
from luminascale.utils.io import aces_to_display_gpu

srgb_32bit, srgb_8bit = aces_to_display_gpu(
    aces_tensor,
    input_cs="ACES2065-1",
    display="sRGB - Display"
)
```

---

### **Use PyTorch Hybrid (Recommended for Training) When:**
- ✅ Need speed + accuracy balance
- ✅ Want differentiable transforms
- ✅ Training neural networks
- ✅ Can accept ~95% ACES accuracy
- ✅ GPU bandwidth critical

**Proposed Implementation**:
```python
class PyTorchACESTransform(nn.Module):
    def forward(self, aces2065_1):
        # Matrix-based (differentiable)
        ap1 = torch.matmul(AP0_TO_AP1_MATRIX, aces2065_1)
        
        # LUT-based (from OCIO)
        ap1_toned = self.tone_lut_interpolate(ap1)
        
        # Matrix-based again
        srgb = torch.matmul(AP1_TO_RGB_MATRIX, ap1_toned)
        return srgb_oetf(srgb)  # Gamma encoding
```

---

## Hardcoded Matrices Summary

### Available in Public Spec:

1. **AP0 ↔ AP1** (3×3 matrices)
   - Status: ✅ Documented in ACES spec
   - Accuracy: Industrial standard

2. **XYZ ↔ RGB** (Rec.709) (3×3 matrices)
   - Status: ✅ Public spec (CIEXYZ standard)
   - Accuracy: Standardized

3. **sRGB OETF/EOTF** (parameters)
   - Status: ✅ IEC 61966-2-1 standard
   - Parameters: γ=2.4, α=1.055, β=0.055

4. **RRT/ODT Tone Curves** (LUT-based)
   - Status: ⚠️ In OCIO config files
   - Format: CommonLUT Format (CLF)

### Rec.709 Luma Coefficients (Already in LuminaScale):
```
Y = 0.2126*R + 0.7152*G + 0.0722*B  (BT.709)
```

---

## Integration Roadmap for LuminaScale

### Phase 1: Research Validation ✅ DONE
- Identify libraries and their capabilities
- Analyze ACES math requirements
- Compare implementations

### Phase 2: Hybrid Implementation (Proposed)
- [ ] Extract LUT data from OCIO config
- [ ] Implement PyTorch matrix transforms
- [ ] Add 3D LUT interpolation layer
- [ ] Benchmark vs. OCIO reference
- [ ] Integrate into training pipeline

### Phase 3: Pure PyTorch (Optional)
- [ ] Develop learnable tone curves
- [ ] Fine-tune to datasets
- [ ] Measure accuracy loss vs. OCIO

### Phase 4: Production Deployment
- [ ] Keep OCIO for validation/inference
- [ ] Use PyTorch for training
- [ ] Document differences from spec
- [ ] Unit tests against OCIO reference

---

## Code Snippets for Reference

### Kornia: RGB to LAB
```python
import kornia.color as K
import torch

# Shape: [B, C, H, W] where C=3 (RGB)
rgb = torch.rand(8, 3, 256, 256).cuda()

# Get LAB
lab = K.rgb_to_lab(rgb)  # [8, 3, 256, 256]

# Compute ΔE
lab_ref = torch.rand(8, 3, 256, 256).cuda()
delta_e = K.rgb_to_lab(lab - lab_ref).norm(dim=1)  # Approximate
```

### scikit-image: RGB to LAB
```python
import skimage.color as skc
import numpy as np

# Shape: [H, W, 3]
rgb = np.random.rand(256, 256, 3)

# Get LAB
lab = skc.rgb2lab(rgb)

# Compute ΔE via formula
from skimage.color import deltaE_ciede2000
de = deltaE_ciede2000(lab1, lab2)  # Color difference
```

### OCIO (Current): ACES to sRGB
```python
from luminascale.utils.io import aces_to_display_gpu
import torch

aces = torch.rand(1024, 1024, 3).cuda()
srgb_32, srgb_8 = aces_to_display_gpu(aces)
```

### PyTorch Hybrid (Proposed)
```python
import torch
from torch import nn

class ACESTransform(nn.Module):
    def __init__(self):
        # Load from ACES spec
        self.ap0_to_ap1 = torch.tensor([
            [0.695202, 0.140679, 0.164119],
            [0.044794, 0.859671, 0.095534],
            [-0.005481, 0.004869, 1.000612]
        ], device='cuda')
        
        # Load tone LUT from OCIO (one-time)
        self.tone_lut = self._load_lut_from_ocio()
    
    def forward(self, aces2065_1):
        # AP0 → AP1 via matrix
        ap1 = torch.matmul(self.ap0_to_ap1, 
                          aces2065_1.reshape(-1, 3, 1)).reshape(-1, 3)
        
        # Tone mapping via LUT (differentiable interpolation)
        ap1_toned = self._3d_lut_interpolate(ap1, self.tone_lut)
        
        return ap1_toned
    
    def _3d_lut_interpolate(self, rgb, lut_3d):
        # Normalize RGB to [0, 1]
        rgb_norm = torch.clamp(rgb, 0, 1)
        # Trilinear interpolation...
        return interpolated_result
```

---

## Questions for Next Steps

1. **What is the accuracy tolerance for blind ACES normalization?**
   - Can we use 95% of OCIO accuracy?
   - Or do we need 99.9% match?

2. **Should tone curves be learnable or fixed?**
   - Learnable: More flexibility, more parameters
   - Fixed: Matches spec, simpler, faster

3. **Is differentiability through color space critical?**
   - For blind normalization: YES (need gradients)
   - For inference only: NO (OCIO is fine)

4. **What's the target latency for training?**
   - Current: 8-11ms per image (OCIO)
   - Target: 3-5ms per image (Hybrid)?
   - Or: <2ms per image (Pure PyTorch)?

---

**Document prepared for**: LuminaScale development team  
**Maintained by**: Research & Development  
**Last updated**: April 5, 2026
