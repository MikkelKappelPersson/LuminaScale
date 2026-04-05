# LuminaScale: OCIO & OpenGL Codebase Map

## Overview
This document maps all files involved in OCIO (OpenColorIO), OpenGL, ACES color space operations, and GPU rendering throughout the LuminaScale codebase.

---

## 1. Core GPU/OCIO/OpenGL Files

### [src/luminascale/utils/gpu_torch_processor.py](src/luminascale/utils/gpu_torch_processor.py)
**Primary GPU-accelerated ACES-to-display color transform engine**

**Key Components:**
- **PyOpenColorIO (OCIO) Integration**: Imports `PyOpenColorIO as ocio`
- **OpenGL/EGL Rendering**: Full headless GPU rendering pipeline
  - EGL context initialization for headless environments
  - OpenGL 4.0 shader program compilation and linking
  - Framebuffer and texture management
- **GLSL Shaders**: Vertex and fragment shaders with OCIO-generated transforms
- **ACES Transform**: GPU-accelerated `apply_ocio_torch()` method
  - Converts ACES2065-1 → sRGB (or other display targets)
  - Applies RRT (Rendering Reference Transform) + ODT (Output Device Transform)
  - Generates OpenGL shaders from OCIO processor

**Major Functions:**
- `GPUTorchProcessor.__init__()` - Initializes EGL/OpenGL context
- `_initialize_egl()` - Headless GPU setup (EGL_DEFAULT_DISPLAY, pbuffer surfaces)
- `_build_shader_program()` - Compiles GLSL vertex/fragment shaders
- `_allocate_ocio_tex()` - Allocates GPU textures for OCIO 3D LUTs (color lookup tables)
- `apply_ocio_torch()` - Main transform entry point (tensor in, tensor out)
- `_setup_framebuffer()` - Offscreen render target setup
- `_setup_quad_geometry()` - Screen-aligned quad for fullscreen render

**GPU Operations:**
- Texture upload: EXR data → GPU VRAM
- Shader execution: OCIO transform on each pixel (GL_TRIANGLES render)
- Result readback: `glReadPixels()` RGB32F → float32 tensor
- Quantization: float32 → uint8 LDR on GPU

**Dependencies:**
- `PyOpenColorIO` (OCIO library)
- `OpenGL` (GL), `OpenGL.EGL` (headless rendering)
- PyTorch (CUDA tensors)

**Performance Notes:**
- Shader caching by (input_cs, display, view) tuple
- Headless EGL allows HPC/Slurm environments without X11
- All I/O and transforms occur on GPU (zero CPU math bias)

---

### [src/luminascale/utils/io.py](src/luminascale/utils/io.py)
**Image I/O and color space conversion utilities**

**Functions:**
- `read_exr(path)` - Reads EXR files using OpenImageIO (OIIO)
- `write_exr(path, array)` - Writes float32 tensors to EXR
- `oiio_aces_to_display(path)` - CPU-based ACES→sRGB conversion via OIIO
  - Uses `oiio.ImageBufAlgo.colorconvert(buf, "aces", "sRGB")`
  - Depends on OCIO config from environment (`OCIO` env var)
- `aces_to_display_gpu()` - GPU-accelerated wrapper
  - Instantiates `GPUTorchProcessor`
  - Calls `processor.apply_ocio_torch()` with ACES2065-1 input
  - Returns (float32, uint8) tensor pair on GPU
- `image_to_tensor(image_path)` - Multi-format image loader
  - Detects bit-depth (8/10/12/16-bit) and normalizes to [0,1]
  - Supports EXR (via OIIO) and PIL formats
- `convert_to_aces(raw_dir, aces_dir)` - RAW→ACES pipeline
  - Uses `rawtoaces` command-line tool (external dependency)
  - Batch processes camera RAW files to ACES format

**Color Space Pipeline:**
- Input: RAW (color matrices) → ACES2065-1 (linear, scene-referred)
- Processing: CDL grading, OCIO transforms
- Output: sRGB (display-referred, gamma-encoded)

---

### [src/luminascale/utils/gpu_cdl_processor.py](src/luminascale/utils/gpu_cdl_processor.py)
**GPU-accelerated CDL (Color Decision List) processor**

**Purpose:**
Applies per-pixel color grading (CDL) transforms on GPU to generate synthetic "looks" during on-the-fly dataset generation.

**Core Function:**
- `apply_cdl_gpu(image, cdl_params)` - Applies CDL formula on GPU
  - **CDL Formula**: `Output = (Input × Slope + Offset) ^ Power`
  - Then: Luma-weighted saturation blend
  - Fully parallelized via PyTorch broadcast operations
  - Output stays on GPU [H, W, 3] float32

**CDL Parameters (from `look_generator.py`):**
- `slope` - Multiply (highlights/gain)
- `offset` - Add (shadows/lift)
- `power` - Gamma (contrast/punchiness)
- `saturation` - Luma-weighted saturation blend (1.0 = neutral)

**Usage:**
- Generates random looks per image during training
- Avoids CPU bottleneck by keeping tensors on GPU
- Integrates into data pipeline before quantization

---

### [src/luminascale/utils/look_generator.py](src/luminascale/utils/look_generator.py)
**Generative color grading framework**

**Key Types:**
- `CDLParameters` dataclass
  - Fields: slope, offset, power, saturation
  - Method: `to_cdl_xml()` - Generates CDL XML for OCIO integration

**Functions:**
- `random_cdl(variance=0.2, sat_range=(0.4, 1.6))` - Random look generation
  - Master + variance approach per channel
  - Creates "cross-processed" color casts
  - Extreme looks for augmented training data

**OCIO Integration:**
- CDL XML can be used in OCIO config as named transforms
- Allows looks to be applied in OCIO pipeline before final render

---

## 2. Dataset Generation & Preprocessing

### [scripts/bake_dataset.py](scripts/bake_dataset.py)
**Pre-bake EXR→PNG/EXR pipeline to eliminate CPU bottleneck**

**Purpose:**
Before training, converts raw ACES EXR files to display-referred LDR (8-bit PNG or 32-bit EXR) with optional CDL looks applied.

**Workflow:**
1. Load ACES EXR (ACES2065-1, linear, float32)
2. For ~50% of images: Apply random CDL look (GPU)
3. Convert ACES → sRGB using GPU OCIO pipeline
4. Quantize to 8-bit and save PNG (or 32-bit float EXR)

**Functions:**
- Uses `ocio_aces_to_display()` and `ocio_aces_to_srgb_with_look()`
- Applies looks from `get_single_random_look()`
- Outputs: 8-bit sRGB PNG files (fast I/O during training)

**OCIO Config:**
```python
os.environ["OCIO"] = str(project_root / "config" / "aces" / "studio-config.ocio")
```

---

### [scripts/quality_filtered_aces_conversion.py](scripts/quality_filtered_aces_conversion.py)
**Post-process ACES images with quality filtering**

**Purpose:**
Takes ACES EXRs and applies OCIO display conversion with optional quality checks.

---

## 3. Training & Inference

### [src/luminascale/training/dequantization_trainer.py](src/luminascale/training/dequantization_trainer.py)
**PyTorch Lightning trainer for bit-depth expansion model**

**GPU Pipeline Comment:**
```python
# Initialize GPU pipeline (ACES load → CDL → OCIO)
```

**Key Features:**
- Uses PyTorch CUDA device placement
- Dataloader integrates on-the-fly GPU CDL + OCIO transforms
- Training loop: batch to GPU, forward pass, backward pass
- Uses Lightning multi-GPU support (`accelerator: gpu` in config)

---

### [src/luminascale/utils/dequantization_inference.py](src/luminascale/utils/dequantization_inference.py)
**Model inference utilities**

**GPU Functions:**
- `run_inference_on_batch()` - Runs model on [B, 3, H, W] tensor on GPU
- `run_inference_on_single_image()` - Single image inference with unsqueeze
- `infer_dataset_with_comparison()` - Batch inference with metric computation

---

## 4. Configuration & Environment

### [config/aces/studio-config.ocio](config/aces/studio-config.ocio)
**OCIO Configuration File (SMPTE Academy)**

**Contents:**
- Color space definitions (ACES, sRGB, display spaces)
- Named views and displays
- Look definitions (CDL, LUT-based)
- Display rendering transforms (RRT, ODT)

**Usage:**
- Set via `os.environ["OCIO"]` before loading OCIO
- Defines available displays and views
- Example view: "ACES 2.0 - SDR 100 nits (Rec.709)"

---

## 5. Testing & Validation

### [test_aces_render_comparison.py](test_aces_render_comparison.py)
**Validation script comparing OIIO vs GPU ACES transforms**

**Purpose:**
Verifies that CPU OIIO rendering and GPU OpenGL rendering produce identical results.

**Key Functions:**
- `aces_to_display_old()` - CPU reference (uses `oiio.ImageBufAlgo.ociodisplay()`)
- `aces_to_display_current()` - GPU version (from `io.py`)
- `compare_renders()` - Pixel-wise difference analysis

**Delta-E Analysis:**
- Computes max/mean pixel differences
- Flags pixels with diff > 1e-5 as divergent
- Useful for debugging OpenGL/OCIO integration

---

### [notebooks/aces_degradation_testing.ipynb](notebooks/aces_degradation_testing.ipynb)
**Interactive ACES transformation testing**

**Experiments:**
- ACES image degradation (quantization artifacts)
- Look application and chaining
- Visual quality validation
- Uses both CPU and GPU renderers

---

### [notebooks/verify_gpu_renderer_orientation.ipynb](notebooks/verify_gpu_renderer_orientation.ipynb)
**GPU renderer image orientation validation**

**Verifies:**
- Correct OpenGL texture coordinate mapping
- No Y-axis flip or rotation
- Pixel-perfect alignment with CPU renderer

---

## 6. Dependencies Summary

### Direct OCIO/OpenGL Libraries:
| Library | Version | Usage |
|---------|---------|-------|
| `PyOpenColorIO` | Latest | OCIO processor, shader extraction, config management |
| `PyOpenGL` | ≥3.1.10 | OpenGL context, shader compilation, texture/FB operations |
| `PyOpenGL-accelerate` | ≥3.1.10 | OpenGL acceleration (optional but recommended) |
| `OpenImageIO (OIIO)` | Latest | EXR I/O, CPU ACES conversion fallback |

### Defined in [pixi.toml](pixi.toml):
```toml
pyopengl = ">=3.1.10, <4"
pyopengl-accelerate = ">=3.1.10, <4"
openimageio = "*"
pyopencolorio = "*"
```

### Environment Setup:
```bash
# Set OCIO config path (must be set before importing OCIO-dependent code)
export OCIO=/path/to/config/aces/studio-config.ocio
```

---

## 7. ACES Color Space Pipeline

### Transformation Flow:

```
RAW Camera Input
    ↓ [rawtoaces command]
ACES2065-1 (Linear, Scene-Referred, Float32)
    ↓ [GPU: CDL Optional]
ACES2065-1 with Grade Applied
    ↓ [GPU: OCIO RRT+ODT via OpenGL Shader]
sRGB (Linear)
    ↓ [CPU: Quantization & Gamma]
sRGB (Display-Referred, 8-bit PNG or 32-bit EXR)
    ↓ [Model: Dequantization]
Estimated ACES (for validation)
```

### Color Spaces Used:
- **ACES2065-1**: Academy Color Encoding System, linear, float32 [0, ∞)
- **sRGB-Linear**: Linear RGB before gamma encoding
- **sRGB**: Display-referred gamma-encoded [0, 1] or [0, 255]

### Transforms:
- **RRT (Rendering Reference Transform)**: ACES2065-1 → OCES (output color encoding space)
- **ODT (Output Device Transform)**: OCES → sRGB (Rec.709 gamma)
- **CDL (Color Decision List)**: Per-channel slope/offset/power/saturation grading

---

## 8. GPU/CUDA Operations

### Device Placement:
```python
# Config files (configs/default.yaml, configs/hpc_slurm.yaml)
device: cuda
accelerator: gpu  # PyTorch Lightning multi-GPU
```

### Tensor Operations on GPU:
- **gpu_torch_processor.py**:
  - EXR→GPU texture upload
  - OCIO shader execution (parallel pixel-wise)
  - Result readback (glReadPixels → GPU tensor)
- **gpu_cdl_processor.py**:
  - CDL formula: `(Input × Slope + Offset) ^ Power` (PyTorch broadcast)
  - Saturation blending (luma-weighted)
- **dequantization_inference.py**:
  - Model forward pass on GPU batches
  - Tensor-to-tensor without CPU roundtrips

### Headless GPU Rendering (HPC):
- **EGL Context**: `EGL_DEFAULT_DISPLAY`, pbuffer surfaces
- **No X11 Required**: Can run on compute nodes without display server
- **Multi-GPU Support**: PyTorch Lightning distributed training via Slurm

---

## 9. File Dependencies Graph

```
gpu_torch_processor.py
├── PyOpenColorIO (OCIO transforms)
├── OpenGL + OpenGL.EGL (GPU rendering)
└── PyTorch (CUDA tensors, interop)

io.py
├── gpu_torch_processor (GPU ACES transform)
├── OpenImageIO (EXR I/O, CPU fallback)
└── PyTorch (tensor conversion)

gpu_cdl_processor.py
├── look_generator.py (CDL params)
├── OpenImageIO (EXR loading)
└── PyTorch (GPU CDL ops)

bake_dataset.py
├── io.py (ocio_aces_to_display, ocio_aces_to_srgb_with_look)
├── look_generator.py (random looks)
└── OpenImageIO + PIL (output)

training/dequantization_trainer.py
├── io.py (image loading)
├── gpu_cdl_processor.py (on-the-fly CDL)
└── PyTorch Lightning (GPU training)

config/aces/studio-config.ocio
└── Referenced by: all OCIO-dependent modules via os.environ["OCIO"]
```

---

## 10. Key Implementation Notes

### Shader Caching Strategy
```python
# In gpu_torch_processor.py
cache_key = (input_cs, display, view)  # e.g., ("ACES2065-1", "sRGB - Display", "ACES 2.0 - SDR 100 nits")
if cache_key not in self._shader_cache:
    # Compile and cache
```
→ Avoids recompiling identical transforms

### EGL Initialization Fallbacks
```python
# Try multiple display options:
1. EGL_DEFAULT_DISPLAY (standard)
2. eglGetDisplay(0) (numbered display)
3. eglGetDisplay(None) (null display)
```
→ Handles various Linux/HPC environments (NVIDIA driver quirks)

### GPU Memory Management
- Input texture deleted immediately after shader execution
- LUT textures cached across transforms
- Framebuffer reused for same resolution

### Zero-CPU-Math Approach
Per **docs/spec: data pipeline.md**:
- EXR load → GPU texture (I/O on GPU)
- CDL computation → GPU (PyTorch)
- OCIO shader → GPU (OpenGL)
- Quantization → GPU (PyTorch, optional)
- Output tensor ready for model

→ CPU only fetches next image while GPU processes previous pipeline

---

## Summary Table

| File | Lines | Role | OCIO | OpenGL | ACES | GPU |
|------|-------|------|------|--------|------|-----|
| gpu_torch_processor.py | ~900 | Core GPU renderer | ✓ | ✓ | ✓ | ✓ |
| io.py | 256 | I/O & color convert | ✓ | - | ✓ | ✓ |
| gpu_cdl_processor.py | ~200 | GPU color grader | - | - | ✓ | ✓ |
| look_generator.py | ~150 | Look generation | ✓ | - | - | - |
| bake_dataset.py | ~150 | Dataset pre-process | ✓ | - | ✓ | ✓ |
| dequantization_trainer.py | ~250 | Training loop | - | - | - | ✓ |
| dequantization_inference.py | ~120 | Inference utils | - | - | - | ✓ |
| test_aces_render_comparison.py | ~100 | Validation | ✓ | - | ✓ | - |
| config/aces/studio-config.ocio | ~500 | OCIO config | ✓ | - | ✓ | - |

---

**Last Updated:** April 5, 2026 | **Version:** 1.0
