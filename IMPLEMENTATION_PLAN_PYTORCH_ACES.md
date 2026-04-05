# Implementation Plan: PyTorch Native ACES Color Transforms

**Date**: April 5, 2026  
**Status**: Research Complete → Ready for Implementation  
**Recommendation**: Implement Hybrid Approach (Option B)  
**Estimated Effort**: 1-2 weeks for core + testing  
**Performance Gain**: 2-3× speedup, Fully Differentiable ✅

---

## Executive Summary

This plan transitions LuminaScale from OCIO OpenGL rendering to PyTorch-native color transforms, removing HPC/EGL complexity while improving training speed and enabling backpropagation through color transforms.

### Why This Matters
- **Current**: 8-11ms per image (OCIO GPU) + EGL/OpenGL HPC headaches
- **Proposed**: 3-5ms per image (PyTorch Hybrid) + pure CUDA, no OpenGL needed
- **Training Impact**: 2-3× faster data loading, differentiable colors for future learnable transforms
- **HPC**: Eliminates EGL/OpenGL problems entirely

### Hybrid Approach Breakdown
```
ACES2065-1 (input)
    ↓
[Pure PyTorch Matrices] ← Fast (< 1ms)
    ↓ AP0→AP1, AP1→XYZ, XYZ→Rec.709
[LUT Interpolation] ← Moderate (2-3ms)
    ↓ RRT tone curve, gamut compression, sRGB OETF
sRGB (output)
```

---

## Phase 1: Core Implementation (Week 1)

### 1.1 Create New Module Structure

**File to Create**: `src/luminascale/utils/pytorch_aces_transformer.py`

**Imports & Dependencies**:
```python
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path
```

**Classes to Implement**:
1. `ACESMatrices` - Immutable class holding all transformation matrices
2. `ACESColorTransformer` - Main transform engine
3. `LUTInterpolator` - Efficient tone curve LUT lookup
4. `ACESAPproximation` - Fast approximation option (optional)

### 1.2 Matrix Definition Layer

**What to implement**:
- AP0 ↔ AP1 matrix (3×3)
- AP1 ↔ XYZ matrix (3×3)
- XYZ ↔ Rec.709 matrix (3×3)
- Inverse matrices for validation
- Matrix pre-computation on device (CUDA)

**Location**: `pytorch_aces_transformer.py` lines 20-80

**Code Skeleton**:
```python
class ACESMatrices:
    """Immutable ACES color transformation matrices."""
    
    # All matrices from ACES spec (copy from ACES_IMPLEMENTATION_MATHEMATICS.md)
    M_AP0_TO_AP1 = torch.tensor([...], dtype=torch.float32)
    M_AP1_TO_XYZ = torch.tensor([...], dtype=torch.float32)
    M_XYZ_TO_REC709 = torch.tensor([...], dtype=torch.float32)
    
    @classmethod
    def to_device(cls, device: str) -> dict:
        """Move all matrices to target device."""
        return {name: mat.to(device) for name, mat in vars(cls).items()}
```

### 1.3 LUT Extraction from OCIO

**What to implement**:
- Function to read OCIO config and extract RRT/ODT tone curves
- Evaluate tone curves at regular intervals (e.g., 4096 points)
- Store as 1D LUT tensor
- Implement lookup + linear interpolation

**Location**: `pytorch_aces_transformer.py` lines 100-200

**Inputs**:
- OCIO config path: `config/aces/studio-config.ocio`
- OCIO library API calls

**Outputs**:
- Python dict with LUTs: `{"rrt_tone_curve": tensor, "odt_curve": tensor}`

**Key Function Signature**:
```python
def extract_luts_from_ocio(config_path: str) -> dict[str, torch.Tensor]:
    """Extract RRT/ODT LUTs from OCIO config to PyTorch tensors."""
```

### 1.4 Core Transform Pipeline

**Main class: `ACESColorTransformer`**

**Methods to implement**:
1. `__init__(device='cuda', use_approximation=False)` - Initialize matrices & LUTs
2. `ap0_to_ap1(aces_tensor)` - Matrix multiply
3. `apply_rrt(ap1_tensor)` - Tone mapping via LUT
4. `apply_odt(linear_rec709)` - Output device transform (sRGB encode + dequantize)
5. `aces_to_srgb_32f(aces_tensor)` - Full pipeline → float32 [0, 1]
6. `aces_to_srgb_8u(aces_tensor)` - Full pipeline → uint8 [0, 255]

**API Signature**:
```python
class ACESColorTransformer:
    def __init__(self, device: str = 'cuda', use_lut: bool = True):
        """Initialize transformer with matrices and LUTs on device."""
        
    def aces_to_srgb_32f(self, aces_tensor: torch.Tensor) -> torch.Tensor:
        """[H, W, 3] ACES → [H, W, 3] sRGB float32 on GPU."""
        
    def aces_to_srgb_8u(self, aces_tensor: torch.Tensor) -> torch.Tensor:
        """[H, W, 3] ACES → [H, W, 3] sRGB uint8 on GPU."""
```

### 1.5 LUT Interpolation Utility

**Helper class: `LUTInterpolator`**

**Methods**:
1. `__init__(lut_tensor)` - Store LUT
2. `lookup_nearest(x)` - Nearest neighbor (fast)
3. `lookup_linear(x)` - Linear interpolation (better accuracy)
4. `lookup_cubic(x)` - Cubic interpolation (slow, probably not needed)

**Implementation note**: Use `torch.searchsorted()` for fast quantile lookups on CUDA.

---

## Phase 2: Testing & Validation (Days 4-5)

### 2.1 Unit Tests

**File**: `tests/test_pytorch_aces_transformer.py`

**Test cases**:
1. **Matrix validity**: Determinant ≈ 1.0, invertibility
2. **Color space conversions**: Round-trip error < 1e-5 (numerical precision)
3. **LUT lookups**: Monotonicity, out-of-range handling
4. **Pipeline accuracy**: Compare vs OCIO reference (< 2% RMSE)
5. **Tensor shapes**: [H,W,3], [B,H,W,3], batch processing
6. **Device compatibility**: CPU and CUDA execution
7. **Type compatibility**: float32, float16 (for training)
8. **Performance**: Latency benchmarks < 5ms for 1024²

**Reference data**:
- Use 10 images from `dataset/test/` (or create synthetic images)
- Run through both OCIO and PyTorch, compare pixel values
- Compute PSNR, SSIM, ΔE metrics

### 2.2 Integration Tests

**Test**: Replace `gpu_torch_processor.apply_ocio_torch()` with PyTorch version

1. Update `dataset_pair_generator.py` to use new transformer
2. Run 100-image batch through training data pipeline
3. Verify output shape, value ranges, gradient flow
4. Compare timing: old vs new

### 2.3 Accuracy Validation

**Comparison matrix**:
```
Input: ACES2065-1 [H, W, 3]
    ↓
OCIO reference (baseline)
    ↓
PyTorch Hybrid (test)
    ↓
Metrics:
  - PSNR: > 30 dB (excellent, near-indistinguishable)
  - SSIM: > 0.95 (structural similarity)
  - ΔE (cie2000): < 0.5 (human imperceptible, good for color)
  - Max error: < 1/255 per channel
```

---

## Phase 3: Integration with Existing Code (Days 5-7)

### 3.1 Update `io.py`

**Function**: `aces_to_display_gpu()`

**Current behavior**:
```python
def aces_to_display_gpu(aces_tensor, ...):
    processor = GPUTorchProcessor(headless=True)
    srgb_32bit, srgb_8bit = processor.apply_ocio_torch(aces_tensor, ...)
    processor.cleanup()
    return srgb_32bit, srgb_8bit
```

**New behavior**:
```python
def aces_to_display_gpu(aces_tensor, use_pytorch=True, ...):
    if use_pytorch:
        transformer = ACESColorTransformer(device=aces_tensor.device)
        srgb_32bit = transformer.aces_to_srgb_32f(aces_tensor)
        srgb_8bit = transformer.aces_to_srgb_8u(aces_tensor)
    else:
        # Fall back to OCIO (for validation)
        processor = GPUTorchProcessor(headless=True)
        srgb_32bit, srgb_8bit = processor.apply_ocio_torch(aces_tensor, ...)
        processor.cleanup()
    return srgb_32bit, srgb_8bit
```

**Key Point**: Add `use_pytorch` flag to allow parallel validation

### 3.2 Update `dataset_pair_generator.py`

**Change**:
- Replace `self.ocio_processor.apply_ocio_torch()` calls with `self.pytorch_transformer.aces_to_srgb_32f()`

**Location**: ~line 130 (load_aces_and_transform method)

### 3.3 Update `scripts/generate_on_the_fly_dataset.py`

**Change**:
- Initialize `ACESColorTransformer` instead of `GPUTorchProcessor`
- Update in `OnTheFlyACESDataset.__init__()`

### 3.4 Optional: Keep OCIO as Fallback

**Benefits**:
- Validate PyTorch transform accuracy
- A/B testing
- Gradual migration
- Industry compliance

**Implementation**:
- Add config option to `configs/*.yaml`: `use_pytorch_aces: true`
- Wrap transformer initialization with try/except
- Fall back to OCIO if any error

---

## Phase 4: Performance Optimization (Days 7-8)

### 4.1 Profiling & Bottleneck Identification

**Tools**: 
- `torch.profiler` for GPU profiling
- `time.perf_counter()` for CPU timing

**Profiling Points**:
```python
with torch.profiler.profile(...) as prof:
    transformer.aces_to_srgb_32f(large_batch)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 4.2 Optimization Targets

1. **Matrix multiplies**: Can batch AP0→AP1 with other operations
2. **LUT lookups**: Pre-allocate interpolation tensors
3. **Memory layout**: Ensure CHW vs HWC consistency
4. **Quantization**: Keep float32 for accuracy (no float16 here)

### 4.3 Approximation Options (Optional)

If further speedup needed:
- **Fast path**: Skip gamut compression (saves ~1ms, < 1% error)
- **Reduced LUT resolution**: 1024 points instead of 4096 (saves 0.5ms)
- **Vectorized processing**: Process multiple images per GPU call

---

## Phase 5: Documentation & Handoff (Day 8-9)

### 5.1 Code Documentation

**In-code docstrings**:
- Full mathematical explanation (reference to ACES_IMPLEMENTATION_MATHEMATICS.md)
- Input/output tensor specifications
- Device handling (CPU vs CUDA)
- Performance characteristics

**Example**:
```python
def aces_to_srgb_32f(self, aces_tensor: torch.Tensor) -> torch.Tensor:
    """Transform ACES2065-1 to sRGB display space.
    
    Implements the full ACES rendering pipeline:
    1. AP0 → AP1 matrix transform
    2. Tone mapping (RRT) via LUT interpolation
    3. AP1 → XYZ → Rec.709 matrix chain
    4. sRGB OETF (gamma encode)
    
    Args:
        aces_tensor: [H, W, 3] or [B, H, W, 3] float32 tensor on CUDA
                     Values unbounded, in ACES2065-1 color space
    
    Returns:
        srgb_32f: Same shape as input, float32 [0, 1]
                  Rec.709 sRGB linear (before OETF)
    
    Timing: ~3-5ms per 1024×1024 image on RTX4090
    Differentiable: ✅ Yes (enables backprop)
    """
```

### 5.2 User Guide

**For trainers**:
- How to enable PyTorch ACES transforms
- Configuration options
- Validation workflow
- Performance tuning

**For researchers**:
- Mathematical breakdown (reference ACES_IMPLEMENTATION_MATHEMATICS.md)
- How to extract LUTs from other OCIO configs
- How to modify tone curves for experiments

---

## File Structure

### New Files to Create

```
src/luminascale/utils/
  ├── pytorch_aces_transformer.py      (400-500 lines)
  │   ├── ACESMatrices
  │   ├── LUTInterpolator
  │   └── ACESColorTransformer
  │
tests/
  └── test_pytorch_aces_transformer.py (300-400 lines)
       ├── test_matrix_validity
       ├── test_color_conversions
       ├── test_accuracy_vs_ocio
       ├── test_performance
       └── test_integration

docs/
  └── PYTORCH_ACES_USAGE.md             (Reference guide)
```

### Files to Modify

```
src/luminascale/utils/
  ├── io.py                          (add pytorch flag to aces_to_display_gpu)
  └── dataset_pair_generator.py      (replace ocio_processor with pytorch transformer)

scripts/
  └── generate_on_the_fly_dataset.py (init transformer instead of ocio processor)

configs/
  └── *.yaml                         (add use_pytorch_aces: true flag)
```

### Files to Keep Unchanged

```
src/luminascale/utils/
  ├── gpu_torch_processor.py         (keep as fallback/reference)
  ├── gpu_cdl_processor.py           (unrelated, keep as-is)
  └── look_generator.py              (unrelated, keep as-is)
```

---

## Implementation Checklist

### Week 1: Core Implementation
- [ ] **Day 1-2**: Create `pytorch_aces_transformer.py`
  - [ ] Implement `ACESMatrices` class
  - [ ] Implement `LUTInterpolator` class
  - [ ] Implement `ACESColorTransformer` core methods
  
- [ ] **Day 3**: LUT extraction
  - [ ] Implement `extract_luts_from_ocio()`
  - [ ] Test LUT validity (monotonicity, range checks)
  
- [ ] **Day 4**: Testing & validation
  - [ ] Create unit tests (`tests/test_pytorch_aces_transformer.py`)
  - [ ] Run accuracy comparison vs OCIO (< 2% error)
  - [ ] Performance benchmarks
  
- [ ] **Day 5**: Integration
  - [ ] Update `io.py` with `use_pytorch` flag
  - [ ] Update `dataset_pair_generator.py`
  - [ ] Update `generate_on_the_fly_dataset.py`
  - [ ] Integration tests on real data

### Week 2: Polish & Optimization
- [ ] **Day 6-7**: Optimization & refinement
  - [ ] Profile with `torch.profiler`
  - [ ] Optimize matrix ops, LUT lookups
  - [ ] Test on various GPU sizes/images
  
- [ ] **Day 8-9**: Documentation & final validation
  - [ ] Docstrings & inline comments
  - [ ] Write PYTORCH_ACES_USAGE.md
  - [ ] Final validation tests
  
- [ ] **Day 10** (optional): Approximation modes
  - [ ] Implement fast approx (no gamut compression)
  - [ ] Document trade-offs

---

## Success Criteria

### Functional
- ✅ Transforms ACES→sRGB identically to OCIO (< 2% RMSE)
- ✅ Supports batch processing [B, H, W, 3]
- ✅ Works on CPU and CUDA devices
- ✅ Gradients flow properly for backprop
- ✅ Handles out-of-range values gracefully

### Performance
- ✅ Latency < 5ms per 1024×1024 image
- ✅ Memory < 2GB for batch size 4
- ✅ No PCIe transfers (pure GPU)
- ✅ 2-3× speedup vs OCIO

### Quality
- ✅ PSNR > 30 dB vs OCIO reference
- ✅ SSIM > 0.95 vs OCIO reference
- ✅ ΔE (cie2000) < 0.5 vs OCIO reference
- ✅ Zero visual differences to human eye

### HPC
- ✅ No OpenGL/EGL required
- ✅ Pure CUDA execution
- ✅ Works in Singularity container
- ✅ No display/X11 dependencies

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Accuracy gap > 2% | Low | Keep OCIO fallback, validate early |
| LUT extraction bugs | Medium | Test with multiple OCIO configs |
| Performance degradation | Low | Profile frequently, optimize iteratively |
| Integration issues | Medium | Phased integration with feature flags |
| Numerical instability (float32) | Low | Have float64 option for fallback |

---

## Timeline Summary

```
Week 1 (Mon-Fri):
  Days 1-2: Core module creation
  Days 3:   LUT extraction & testing
  Days 4-5: Unit + integration tests
  
Week 2 (Mon-Wed):
  Days 6-7: Performance optimization
  Days 8-9: Documentation
  Day 10:   Buffer/contingency
  
Total: ~10 days engineering (1.5 weeks calendar)
```

---

## Next Steps: Decision Points

### Proceed with Implementation?

**Decision Tree**:
1. **Immediate (implement)**: If training data loading is bottleneck
2. **Deferred (6 months)**: If current OCIO performance is acceptable
3. **Fallback**: Keep OCIO always available as reference

### Questions for Team

1. Is HPC/EGL causing practical problems RIGHT NOW?
2. Is training speed a bottleneck (data loading vs model)?
3. Do we need differentiable color transforms for learnable tones in Phase 3?

**Recommendation**: Implement now because:
- ✅ Removes OpenGL/EGL complexity (HPC benefit)
- ✅ 2-3× faster data loading (training benefit)
- ✅ Enables future learnable color transforms
- ✅ Only ~1.5 weeks effort
- ✅ Fully backward compatible (keep OCIO fallback)

---

**Document Version**: 1.0  
**Date**: April 5, 2026  
**Status**: Ready for Engineering  
**Next Action**: Approve → Begin Phase 1 implementation
