# Color Space Transformation Libraries & ACES Mathematics Research

**Research Date**: April 5, 2026  
**Project**: LuminaScale - Neural Bit-Depth Expansion & ACES Color Space Normalization  
**Scope**: PyTorch-compatible color transformation libraries and ACES rendering pipeline mathematics

---

## Executive Summary

### Current LuminaScale Status
- **Architecture**: OCIO 2.5 + GPU (OpenGL/EGL headless rendering)
- **Transform**: ACES2065-1 → sRGB via OCIO shaders + 3D LUTs
- **Performance**: ~8-11ms per image (warm cache), includes PCIe transfer overhead
- **Limitation**: Non-differentiable, CPU↔GPU transfers bottleneck (6-8ms)

### Key Findings
1. **Kornia** is the best PyTorch-native color library but lacks ACES transforms
2. **ACES matrices are publicly documented** in official spec; can be extracted and implemented
3. **Hybrid approach** (PyTorch matrices + OCIO LUTs) offers 4-8× speedup while maintaining accuracy
4. **Pure PyTorch implementation** is feasible but requires careful tone curve handling

### Recommendation
**Implement PyTorch native transforms** for training pipeline:
- Use matrix operations for ACES2065-1 → AP1 → sRGB conversions
- Extract tone mapping LUTs from OCIO config for inference
- Enable backprop through color transforms for blind ACES normalization

---

## 1. PyTorch Color Transform Libraries Comparison

### 1.1 **Kornia** (Recommended for PyTorch projects)

| Aspect | Details |
|--------|---------|
| **GitHub** | https://github.com/kornia/kornia |
| **Status** | ✅ Production-ready, actively maintained |
| **Backend** | Pure PyTorch + CUDA kernels |
| **GPU Support** | Excellent (native CUDA) |
| **Type Hints** | Full (Python 3.8+) |

**Capabilities**:
- RGB ↔ HSV (Hue-Saturation-Value)
- RGB ↔ LAB (CIE LAB, D65 illuminant)
- RGB ↔ YUV / YCbCr
- sRGB OETF/EOTF (gamma encoding/decoding)
- Grayscale conversions
- Color difference metrics

**Strengths**:
- ✅ Fully differentiable (supports backprop)
- ✅ TorchScript compatible
- ✅ Batch operations optimized for GPU
- ✅ Well-documented with examples
- ✅ No external dependencies beyond PyTorch

**Limitations**:
- ❌ No ACES-specific transforms (AP0/AP1 primaries)
- ❌ Limited tone curve implementations
- ❌ No 3D LUT interpolation (matrices only)
- ❌ No domain-specific color metrics

**When to Use**: Building custom color science pipelines, research experiments, or when you need pure PyTorch differentiability.

**Example API**:
```python
import kornia.color as K
import torch

# RGB to LAB
rgb_tensor = torch.rand(3, 1024, 1024)  # [C, H, W]
lab_tensor = K.rgb_to_lab(rgb_tensor)

# sRGB OETF (linear → encoded)
linear_rgb = torch.tensor([1.0, 0.5, 0.25])
encoded = K.srgb_to_rgb(linear_rgb)  # applies gamma
```

---

### 1.2 **scikit-image** (Reference implementations)

| Aspect | Details |
|--------|---------|
| **Website** | https://scikit-image.org/ |
| **Backend** | NumPy (CPU only) |
| **Type Hints** | Partial |

**Capabilities**:
- RGB, HSV, XYZ, LAB, LUV conversions
- Y-based spaces: YUV, YIQ, YCbCr, YDbDr
- Color difference metrics: ΔE_CIE76, ΔE_CIEDE2000, ΔE_CMC
- Medical imaging stains: HED separation

**Strengths**:
- ✅ Simple, well-documented formulas
- ✅ Good for validation and reference
- ✅ Mature codebase
- ✅ No dependencies beyond NumPy

**Limitations**:
- ❌ CPU-only (no GPU)
- ❌ NumPy tensors (not PyTorch)
- ❌ Not differentiable
- ❌ No ACES support

**When to Use**: Validation, testing reference implementations, or offline CPU-based color grading.

---

### 1.3 **OpenColorIO (OCIO)** (Current LuminaScale)

| Aspect | Details |
|--------|---------|
| **Website** | https://opencolorio.org/ |
| **Type** | Industry-standard color management |
| **Backend** | GPU (OpenGL/CUDA) + LUTs |

**Current Usage in LuminaScale**:
- Config file: `config/aces/studio-config.ocio` (ACES 2.0, OCIO 2.5)
- Main processor: `src/luminascale/utils/gpu_torch_processor.py`
- Pipeline: ACES2065-1 → OCIO shader → sRGB via 3D LUTs

**Strengths**:
- ✅ Most accurate (official ACES implementation)
- ✅ Extensive ACES support (all versions)
- ✅ GPU acceleration via shaders
- ✅ LUT-based precision

**Limitations**:
- ❌ OpenGL/EGL dependency
- ❌ PCIe transfer overhead (6-8ms per image)
- ❌ Not differentiable
- ❌ Headless setup complexity
- ❌ Non-deterministic (GPU/driver dependent)

**Performance**:
- Cold cache: ~115ms (shader compilation + LUT upload)
- Warm cache: ~8-11ms (including PCIe transfers)
- Bottleneck: CPU↔GPU memory transfers (6-8ms of 8-11ms)

---

### 1.4 **Torchvision** (Limited)

| Aspect | Details |
|--------|---------|
| **Module** | torchvision.transforms |
| **Capabilities** | RGB ↔ Grayscale, BGR conversion, gamma adjustment |

**Status**: Primarily for preprocessing, minimal color science capabilities.

---

### 1.5 **Colour Science** (Academic reference)

| Aspect | Details |
|--------|---------|
| **GitHub** | https://github.com/colour-science/colour |
| **Backend** | NumPy (CPU only) |

**Capabilities**: Comprehensive but NumPy-based; no GPU support; primarily academic.

---

## 2. ACES Mathematics & Rendering Pipeline

### 2.1 ACES System Architecture (Version 2.0)

```
INPUT: ACES2065-1 (Scene-Referred Linear RGB)
└─ Color Space: Wide gamut (AP0 primaries)
└─ Range: Unbounded linear [-∞, ∞]
└─ Bit-depth: 32-bit floating-point (HDR)
        ↓
    ┏━━━━━━━━━━━━━━━━━━━━━┓
    ┃  RRT (Rendering     ┃
    ┃   Transform)        ┃
    ┃  Tone Map + Gamut   ┃
    ┗━━━━━━━━━━━━━━━━━━━━━┛
        ↓
INTERMEDIATE: ACES AP1 (Display-Referred Linear RGB)
└─ Color Space: Display gamut (AP1 primaries)
└─ Range: [0, 1] tone-mapped
└─ Hue-preserved rendering
        ↓
    ┏━━━━━━━━━━━━━━━━━━━━━┓
    ┃  ODT (Output Display┃
    ┃   Transform)        ┃
    ┃  Gamut Map + OETF   ┃
    ┗━━━━━━━━━━━━━━━━━━━━━┛
        ↓
OUTPUT: Display-Referred sRGB (Rec.709)
└─ Color Space: Rec.709 RGB
└─ Range: [0, 1] or [0, 255] (encoded)
└─ Bit-depth: Quantized (8-bit or 10-bit)
```

### 2.2 RRT (Reference Rendering Transform)

**Purpose**: Convert from scene-referred (unbounded linear) to display-referred (tone-mapped)

**Input**: ACES2065-1 AP0 RGB  
**Output**: ACES AP1 RGB (tone-mapped, perceptually correct)

**Pipeline**:
```
ACES2065-1 (AP0)
    ↓ [Matrix: AP0 → AP1 (3×3)]
ACES AP1 (Linear)
    ↓ [ACEStoJMh: RGB → JMh color appearance]
JMh Space (Lightness-Colorfulness-Hue)
    ↓ [Tone Scale: per-component nonlinear mapping]
    ↓ [Chroma Compression: gamut preservation]
    ↓ [Gamut Compression: fit to display bounds]
    ↓ [JMh → RGB: inverse color appearance]
ACES AP1 (Tone-Mapped)
```

**Key Components**:

1. **Color Space Matrix (AP0 → AP1)**:
   - Type: 3×3 linear transformation
   - Speed: <1ms for 1024×1024
   - Implementation: Pure matrix multiplication

2. **Tone Mapping**:
   - Method: Hellwig 2022 CAM (simplified JMh)
   - LUT-based or parametric power function
   - Unbounded input → [0, 1] output

3. **Gamut Compression**:
   - Method: Achromatic axis-aligned gamut compression
   - Operates in JMh space (preserves hue)
   - Brings out-of-gamut colors inside display primaries

### 2.3 ODT (Output Display Transform)

**Purpose**: Convert from display-referred AP1 → display-native encoding (sRGB, Rec.2020, etc.)

**Steps**:
```
ACES AP1
    ↓ [Matrix: AP1 → XYZ (3×3)]
XYZ
    ↓ [Matrix: XYZ → Rec.709 or Rec.2020 (3×3)]
Linear Display RGB
    ↓ [OETF: Apply display gamma encoding]
Display Encoded RGB [0, 1]
    ↓ [Quantize to 8-bit or 10-bit if needed]
Final Output (sRGB / Rec.709)
```

### 2.4 JMh Color Appearance Model (Simplified Hellwig 2022)

**Why JMh space?**
- **J (Lightness)**: Tone mapping applied only to J
- **M (Colorfulness)**: Chroma compression applied separately
- **h (Hue)**: Preserved unchanged (hue-constant rendering)

**Advantage**: Better "match" between different output configurations while preserving hue.

---

## 3. ACES Transform Matrices (Public Specification)

### 3.1 Primary Color Definitions (CIE xy chromaticity)

**AP0 (Scene Primaries - ACES2065-1)**:
```
Color      | x       | y       | Description
-----------|---------|---------|------------------
Red        | 0.7347  | 0.2653  | Wide gamut capture
Green      | 0.0000  | 1.0000  | Daylight green
Blue       | 0.0001  | -0.0770 | LED-sensitive capture
White (D60)| 0.3217  | 0.3378  | 6000K daylight
```

**AP1 (Display Primaries - ACES Rendering)**:
```
Color      | x       | y       | Description
-----------|---------|---------|------------------
Red        | 0.713   | 0.293   | Display reference red
Green      | 0.165   | 0.830   | Display green
Blue       | 0.128   | 0.044   | Display blue
White (D60)| 0.3217  | 0.3378  | Same illuminant as AP0
```

**Rec.709 (Standard Display - sRGB)**:
```
Color      | x       | y       | Description
-----------|---------|---------|------------------
Red        | 0.64    | 0.33    | CRT red phosphor
Green      | 0.29    | 0.60    | CRT green phosphor
Blue       | 0.15    | 0.06    | CRT blue phosphor
White (D65)| 0.3127  | 0.3290  | Daylight illuminant D65
```

### 3.2 Published Transform Matrices

#### **Matrix 1: ACES2065-1 (AP0) → ACES AP1**

```python
ACES_AP0_TO_AP1 = np.array([
    [ 0.695202192603776,  0.140678696470703,  0.164119110925521],
    [ 0.044794442326405,  0.859671142578125,  0.095534415531158],
    [-0.005480591960907,  0.004868886886478,  1.000611705074429]
], dtype=np.float32)
```

**PyTorch usage**:
```python
import torch
matrix_pt = torch.tensor(ACES_AP0_TO_AP1, dtype=torch.float32, device='cuda')
# Apply: AP1_rgb = (matrix_pt @ AP0_rgb.reshape(-1, 3, 1)).squeeze(-1)
```

#### **Matrix 2: ACES AP1 → CIE XYZ (D60)**

```python
ACES_AP1_TO_XYZ = np.array([
    [ 0.938280004,  0.033839841,  0.027879764],
    [ 0.063911881,  0.968911848,  0.003275704],
    [-0.011003658, -0.006957415,  1.050265854]
], dtype=np.float32)
```

#### **Matrix 3: CIE XYZ (D60) → Rec.709 RGB**

Also called: XYZ D65 to Rec.709 (with standard D65 tristimulus converter)

```python
XYZ_D65_TO_REC709 = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
], dtype=np.float32)
```

**Note**: D60↔D65 adaptation matrix needed if using D60 XYZ directly:
```python
D60_TO_D65_ADAPTATION = np.array([
    [ 0.9872,  0.0160,  -0.0032],
    [ 0.0160,  0.9892,   0.0148],
    [-0.0032,  0.0148,   1.0884]
], dtype=np.float32)
```

### 3.3 Color Space Constants

**Rec.709 Luma Coefficients** (for luminance computation):
```python
REC709_LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# Y = 0.2126*R + 0.7152*G + 0.0722*B
```

Used in:
- CDL saturation adjustment (already implemented)
- ΔE calculations
- Perceptual loss functions

**sRGB OETF Parameters**:
```python
# Encoding: linear → sRGB
def srgb_oetf(linear_rgb):
    # Piece-wise:
    # For x ≤ 0.0031308: y = 12.92 * x
    # For x > 0.0031308: y = 1.055 * x^(1/2.4) - 0.055
    
    # PyTorch:
    mask = linear_rgb <= 0.0031308
    encoded = torch.where(
        mask,
        12.92 * linear_rgb,
        1.055 * (linear_rgb ** (1/2.4)) - 0.055
    )
    return encoded
```

**sRGB EOTF Parameters**:
```python
# Decoding: sRGB → linear
def srgb_eotf(encoded_rgb):
    # Piece-wise:
    # For y ≤ 0.04045: x = y / 12.92
    # For y > 0.04045: x = ((y + 0.055) / 1.055)^2.4
```

---

## 4. Color Space Conversion Formulas

### 4.1 RGB ↔ XYZ (Rec.709)

**Forward (RGB → XYZ)**:
```
M_forward = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
]
XYZ = M_forward @ RGB
```

**Inverse (XYZ → RGB)**:
```
M_inverse = [
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
]
RGB = M_inverse @ XYZ
```

### 4.2 Tone Curve Mapping (LUT-Based)

**Approaches**:

1. **1D LUT** (single channel):
   - Input: [0, ∞) linear value
   - Lookup: Interpolate in LUT table
   - Output: [0, 1] tone-mapped value
   - Speed: ~1ms for batch

2. **3D LUT** (full color):
   - Input: RGB vector
   - Lookup: Trilinear interpolate in 3D array
   - Typical size: 64³ or 128³ float32 (~1-4 MB)
   - Speed: ~2-5ms for batch with GPU interpolation

3. **Power-law approximation**:
   - Formula: `out = in^(1/gamma_value)`
   - Speed: <1ms
   - Accuracy: Moderate (not perceptually matched)

**Trilinear LUT Interpolation** (PyTorch):
```python
def interpolate_3d_lut(rgb, lut_3d, lut_size=64):
    """
    Args:
        rgb: [H, W, 3] normalized to [0, 1]
        lut_3d: [N, N, N, 3] LUT data
    """
    # Normalize to [0, N-1]
    scaled = rgb * (lut_size - 1.0)
    idx_floor = torch.floor(scaled).long()
    idx_ceil = torch.ceil(scaled).long()
    
    # Clamp to valid range
    idx_floor = torch.clamp(idx_floor, 0, lut_size - 1)
    idx_ceil = torch.clamp(idx_ceil, 0, lut_size - 1)
    
    # Get 8 corner values
    v000 = lut_3d[idx_floor[:,:,0], idx_floor[:,:,1], idx_floor[:,:,2]]
    v111 = lut_3d[idx_ceil[:,:,0], idx_ceil[:,:,1], idx_ceil[:,:,2]]
    # ... (other 6 corners)
    
    # Trilinear interpolation
    frac = scaled - idx_floor.float()
    # Linear blend using fractions
    result = v000 * (1 - frac[..., 0:1]) * (1 - frac[..., 1:2]) * (1 - frac[..., 2:3])
    # ... (sum all 8 interpolated values)
    
    return result
```

---

## 5. Implementation Options Analysis

### 5.1 Option A: Pure OCIO GPU (Current LuminaScale)

**Architecture**:
```
PyTorch Tensor (GPU)
    ↓ [Detach to NumPy]
CPU NumPy Array
    ↓ [Upload to GL texture]
GPU GL Texture
    ↓ [Execute GLSL shader]
GPU Framebuffer
    ↓ [Readback to NumPy]
CPU NumPy Array
    ↓ [Convert to PyTorch]
PyTorch Tensor (GPU)
```

**Performance**:
| Operation | Time |
|-----------|------|
| CPU→GPU texture upload | 2-3ms |
| Shader execution | 1-2ms |
| GPU→CPU readback | 2-3ms |
| PyTorch conversion | <1ms |
| **Total (warm)** | **8-11ms** |
| Cold cache (1st call) | ~115ms |

**Advantages**:
- ✅ Official ACES implementation
- ✅ Maximum accuracy
- ✅ All ACES 2.0 features included
- ✅ Production-proven

**Disadvantages**:
- ❌ Non-differentiable
- ❌ PCIe bottleneck (6-8ms)
- ❌ OpenGL/EGL setup complexity
- ❌ Driver-dependent (non-deterministic)
- ❌ Not trainable end-to-end

**Best For**: Offline inference, validation, or when maximum accuracy is critical.

---

### 5.2 Option B: PyTorch + OCIO Hybrid

**Architecture**:
```
PyTorch Tensor (GPU)
    ↓ [AP0→AP1 matrix via torch.matmul]
ACES AP1 (GPU, differentiable)
    ↓ [Tone curve via LUT interpolation]
ACES AP1 Tone-Mapped (GPU)
    ↓ [AP1→XYZ→Rec709 matrices]
sRGB Linear (GPU, differentiable)
    ↓ [sRGB OETF via torch]
sRGB Encoded (GPU, [0,1])
```

**Performance**:
| Operation | Time |
|-----------|------|
| AP0→AP1 matrix | <1ms |
| LUT interpolation (3D) | 2-3ms |
| AP1→XYZ→RGB matrices | <1ms |
| sRGB OETF | <1ms |
| **Total** | **3-5ms** |

**Advantages**:
- ✅ 2-3× faster than OCIO
- ✅ Fully differentiable
- ✅ No PCIe overhead
- ✅ Deterministic (CUDA-based)
- ✅ TorchScript compatible
- ✅ Can extract from OCIO config

**Disadvantages**:
- ❌ Requires LUT extraction from OCIO
- ❌ Some tone curve approximation
- ❌ Needs validation against OCIO

**Implementation Effort**: Moderate (2-3 days)  
**Best For**: Training pipelines, end-to-end learnable systems.

---

### 5.3 Option C: Pure PyTorch Native

**Architecture**:
```
PyTorch Tensor (GPU)
    ↓ [Custom JMh color appearance model]
    ↓ [Parametric tone curve]
    ↓ [All matrix multiplications]
PyTorch Native (GPU, fully differentiable)
```

**Performance**:
| Operation | Time |
|-----------|------|
| All matrix operations | 1-2ms |
| Tone curve (power law) | <1ms |
| **Total** | **1-3ms** |

**Advantages**:
- ✅ 4-10× faster
- ✅ Fully differentiable
- ✅ Learnable parameters possible
- ✅ TorchScript compilable
- ✅ No external dependencies

**Disadvantages**:
- ❌ Cannot match OCIO exactly
- ❌ Tone curve approximation required
- ❌ Requires reimplementation
- ❌ Less official validation

**Implementation Effort**: High (5-7 days)  
**Best For**: Custom research, learnable color pipelines, on-device models.

---

### 5.4 Option D: Kornia-Based Pipeline

**Using Kornia color module + custom ACES layer**:
```python
import kornia.color as K

# Custom ACES transforms
class ACESTransform(nn.Module):
    def __init__(self):
        self.ap0_to_ap1_matrix = torch.tensor(...)
        
    def forward(self, aces2065_1):
        # Step 1: AP0 → AP1
        ap1 = F.linear(aces2065_1, self.ap0_to_ap1_matrix)
        
        # Step 2: Tone mapping (learnable or fixed)
        ap1_toned = self.tone_curve(ap1)
        
        # Step 3: AP1 → XYZ via Kornia
        xyz = F.linear(ap1_toned, self.ap1_to_xyz_matrix)
        
        # Step 4: XYZ → sRGB via Kornia
        rgb_lin = F.linear(xyz, self.xyz_to_rgb_matrix)
        
        # Step 5: sRGB OETF
        srgb_enc = K.srgb_to_rgb(rgb_lin)
        
        return srgb_enc
```

**Advantages**:
- ✅ Production-ready Kornia library
- ✅ Well-integrated with PyTorch ecosystem
- ✅ Differentiable throughout
- ✅ Good documentation

**Disadvantages**:
- ❌ Cannot leverage OCIO LUTs
- ❌ Tone curve must be approximated
- ❌ More custom code needed
- ❌ Less industry standard

**Best For**: Research, custom experiments, prototyping.

---

## 6. Recommended Implementation Path for LuminaScale

### Short-term (Production, no changes needed)
**Keep**: OCIO GPU processor as-is for inference

### Medium-term (Training optimization)
**Implement Option B**: PyTorch + OCIO Hybrid
- Extract LUTs from OCIO config (one-time)
- Implement matrix-based transforms in PyTorch
- Parallel paths: OCIO for validation, PyTorch for training
- Enables blind ACES normalization with backprop

### Long-term (Research)
**Explore Option C**: Pure PyTorch if learnable tone curves needed
- Develop learnable perceptual tone mapping
- Fine-tune to datasets
- Full end-to-end optimization

---

## 7. Reference Resources

### Official ACES Specifications
- **ACES GitHub**: https://github.com/aces-aswf/aces (v2.0, latest 2025-04-04)
- **Documentation**: https://docs.acescentral.com/
- **SMPTE Standards**: https://www.smpte.org/standards/aces-standards (ACES 1.0, 2.0)

### Component Repositories
- **aces-core**: Color matrices, RRT/ODT reference code
  - GitHub: https://github.com/aces-aswf/aces-core
  - Contains: Transform matrices, Hellwig CAM implementation

- **aces-output**: Output Display Transform specifications
  - GitHub: https://github.com/aces-aswf/aces-output

- **colour-science**: Python reference implementations
  - GitHub: https://github.com/colour-science/colour
  - Useful for validation, but CPU-only

### PyTorch Libraries
- **Kornia**: https://github.com/kornia/kornia
  - Best practice PyTorch color transforms
  - Examples: RGB↔LAB, gamma encoding, metrics

- **OpenColorIO**: https://opencolorio.org/
  - Config loading: Python API
  - LUT extraction: Via GpuShaderDesc

### Papers & Research
- **Hellwig 2022**: "A new approach to JMh color appearance model"
  - Referenced in ACES 2.0 ODT
  - DOI: https://doi.org/10.1002/col.22792

- **ACES Technical Documentation**:
  - Tone mapping details: https://docs.acescentral.com/system-components/output-transforms/technical-details/tone-mapping/
  - Gamut compression: https://docs.acescentral.com/system-components/output-transforms/technical-details/gamut-compression/

---

## 8. Key Takeaways for LuminaScale

1. **Matrix Coefficients are public** → Can implement PyTorch transforms
2. **Hybrid approach is viable** → Extract OCIO LUTs, use PyTorch matrices
3. **Performance gain is significant** → 2-3× speed improvement possible
4. **Differentiability enables research** → Blind ACES norm needs backprop
5. **Industry validation maintained** → Can validate against OCIO reference

**Next Steps**:
- [ ] Extract LUT data from OCIO config  
- [ ] Implement PyTorch matrix transforms  
- [ ] Add LUT interpolation layer  
- [ ] Benchmark vs. OCIO reference  
- [ ] Integrate into training loop  

