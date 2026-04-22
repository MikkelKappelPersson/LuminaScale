# LuminaScale: Comprehensive OCIO Usage Analysis

**Last Updated**: April 5, 2026 | **Scope**: Complete ACES→sRGB color pipeline analysis

---

## 1. FILES USING PyOpenColorIO

### Main Implementation Files
| File | Purpose | Usage Pattern |
|------|---------|---------------|
| `src/luminascale/utils/gpu_torch_processor.py` | GPU-accelerated OCIO transforms via EGL headless rendering | `import PyOpenColorIO as ocio` |
| `src/luminascale/utils/io.py` | CPU and GPU ACES-to-display conversion functions | Uses `GPUTorchProcessor` wrapper |
| `src/luminascale/utils/dataset_pair_generator.py` | ACES loading + OCIO transforms for training data | Initializes `GPUTorchProcessor(headless=True)` |

### Usage in Scripts & Notebooks
| Location | Role |
|----------|------|
| `scripts/train_dequant_net.py` | Sets `OCIO` env var before training; Dataset uses OCIO transforms |
| `scripts/generate_on_the_fly_dataset.py` | On-the-fly dataset generation with OCIO transforms |
| `scripts/bake_dataset.py` | Batch ACES→sRGB conversion; imports `ocio_aces_to_display` + `ocio_aces_to_srgb_with_look` |
| `scripts/run_dequant_inference.py` | Post-processing via `oiio_aces_to_display` |
| `test_aces_render_comparison.py` | Comparison of OIIO vs GPU rendering implementations |
| **Notebooks** | `verify_gpu_renderer_orientation.ipynb`, `on_the_fly_data_test.ipynb`, `dequantization_inference.ipynb`, `aces_degradation_testing.ipynb` |

### Configuration
| File | Content |
|------|---------|
| `config/aces/studio-config.ocio` | ACES 2.0 + OCIO 2.5 + ColorSpaces 4.0.0 studio configuration |

---

## 2. FUNCTIONS APPLYING OCIO TRANSFORMS

### GPU-Accelerated (Recommended)

#### `GPUTorchProcessor.apply_ocio_torch()`
**Location**: [src/luminascale/utils/gpu_torch_processor.py#L634-L780](src/luminascale/utils/gpu_torch_processor.py#L634-L780)

```python
def apply_ocio_torch(
    self,
    aces_tensor: torch.Tensor,  # [H, W, 3] float32 on CUDA
    input_cs: str = "ACES2065-1",
    display: str = "sRGB - Display",
    view: str = "ACES 2.0 - SDR 100 nits (Rec.709)",
) -> Tuple[torch.Tensor, torch.Tensor]
```

**Implementation Steps**:
1. Query OCIO config: `config.getProcessor(input_cs, display, view, TRANSFORM_DIR_FORWARD)`
2. Extract GPU processor: `processor.getDefaultGPUProcessor()`
3. Generate GLSL shader: `gpu_processor.extractGpuShaderInfo(shader_desc)`
4. Compile fragment shader with OCIO-generated code
5. Allocate GPU textures for 1D/2D/3D LUTs
6. Render input tensor via OpenGL framebuffer
7. Readback as float32 + uint8

**Returns**: `(srgb_32bit, srgb_8bit)` - both [H, W, 3] on GPU

**Key Features**:
- Uses EGL headless rendering (no X11 needed)
- **Shader caching** by (input_cs, display, view)
- Texture memory management for LUTs
- Automatic framebuffer resizing

---

#### `aces_to_display_gpu()`
**Location**: [src/luminascale/utils/io.py#L74-L106](src/luminascale/utils/io.py#L74-L106)

Wrapper around `GPUTorchProcessor.apply_ocio_torch()` with default parameters.

```python
def aces_to_display_gpu(
    aces_tensor: torch.Tensor,
    input_cs: str = "ACES2065-1",
    display: str = "sRGB - Display",
    view: str = "ACES 2.0 - SDR 100 nits (Rec.709)",
) -> tuple[torch.Tensor, torch.Tensor]
```

---

#### `DatasetPairGenerator.load_aces_and_transform()`
**Location**: [src/luminascale/utils/dataset_pair_generator.py#L80-L99](src/luminascale/utils/dataset_pair_generator.py#L80-L99)

Loads ACES from LMDB + applies OCIO transform in one call.

```python
def load_aces_and_transform(self, key: str) -> tuple[torch.Tensor, torch.Tensor]
```

---

#### `DatasetPairGenerator.load_aces_apply_cdl_and_transform()`
**Location**: [src/luminascale/utils/dataset_pair_generator.py#L101-L137](src/luminascale/utils/dataset_pair_generator.py#L101-L137)

**Full pipeline**: Load ACES → Apply CDL grading → OCIO transform

```python
def load_aces_apply_cdl_and_transform(
    self, key: str, cdl_params: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]
```

---

### CPU-Based (Fallback)

#### `oiio_aces_to_display()`
**Location**: [src/luminascale/utils/io.py#L56-L67](src/luminascale/utils/io.py#L56-L67)

CPU implementation using OpenImageIO (no GPU):

```python
def oiio_aces_to_display(path: Path | str) -> np.ndarray
```

Uses OpenImageIO's `colorconvert(buf, "aces", "sRGB")` which calls OCIO internally.

**Note**: Much slower than GPU version; uses environment `OCIO` config.

---

### Missing Implementation ⚠️

#### `ocio_aces_to_srgb_with_look()` 
**Status**: **NOT FOUND** (imported in `bake_dataset.py` but missing from `io.py`)

**Expected Signature**:
```python
def ocio_aces_to_srgb_with_look(
    path: Path | str, 
    look: CDLParameters
) -> np.ndarray
```

Should apply CDL look before OCIO transform. Currently causes ImportError in bake_dataset.py.

---

## 3. COLOR SPACE TRANSFORMS (Input/Output Pairs)

### Primary Transform Pipeline

| Stage | Input Space | Output Space | Transform Type | Implementation |
|-------|-------------|--------------|---|---|
| **Input** | ACES2065-1 (linear, wide gamut) | — | Scene-referred | EXR file |
| **1. Optional CDL** | ACES2065-1 | ACES2065-1 (graded) | Color grading (Slope/Offset/Power) | `GPUCDLProcessor` |
| **2. OCIO RRT** | ACES2065-1 | ACES AP1 | Reference Rendering Transform | OCIO shader |
| **3. OCIO ODT** | ACES AP1 | sRGB (display-linear) | Output Device Transform | OCIO shader |
| **4. sRGB OETF** | sRGB (linear) | sRGB (encoded) | Gamma encoding (Rec.709) | GPU framebuffer |
| **Final** | — | sRGB (gamma-corrected, narrow gamut) | Display-referred | Readback |

### Transform Configuration

**Default Transform Path**:
```
Input: ACES2065-1
Display: sRGB - Display
View: ACES 2.0 - SDR 100 nits (Rec.709)
Direction: FORWARD
```

**Parameters Extracted from OCIO Config**:
- OCIO Version: 2.5
- ACES Version: 2.0
- ColorSpaces Version: 4.0.0
- GPU Shader Language: GLSL 4.0

---

## 4. DEFAULT DISPLAY & VIEW SETTINGS

### From `config/aces/studio-config.ocio`

#### Active Displays
```yaml
active_displays: 
  - sRGB - Display
  - Display P3 - Display
  - Display P3 HDR - Display
  - Gamma 2.2 Rec.709 - Display
  - P3-D65 - Display
  - Rec.1886 Rec.709 - Display
  - Rec.2100-HLG - Display
  - Rec.2100-PQ - Display
  - ST2084-P3-D65 - Display
```

#### Default Display Transforms (Shared Views)

| View Name | Use Case | Nits | Primaries | Default |
|-----------|----------|------|-----------|---------|
| **ACES 2.0 - SDR 100 nits (Rec.709)** | Standard SDR | 100 | Rec.709 | ✅ |
| **ACES 2.0 - SDR 100 nits (P3 D65)** | SDR cinema | 100 | DCI P3 | — |
| **ACES 2.0 - HDR 500 nits (P3 D65)** | Dim HDR | 500 | DCI P3 | — |
| **ACES 2.0 - HDR 1000 nits (P3 D65)** | Bright HDR | 1000 | DCI P3 | — |
| **ACES 2.0 - HDR 2000 nits (P3 D65)** | Very bright HDR | 2000 | DCI P3 | — |
| **ACES 2.0 - HDR 4000 nits (P3 D65)** | Ultra-bright HDR | 4000 | DCI P3 | — |
| **ACES 2.0 - HDR 500 nits (Rec.2020)** | HDR broadcast | 500 | Rec.2020 | — |
| **Un-tone-mapped** | Linear reference | — | As-is | — |
| **Video (colorimetric)** | Video reference | — | As-is | — |

#### Luma Coefficients
```yaml
luma: [0.2126, 0.7152, 0.0722]  # BT.709 luma weights
```

#### Default View Transform
```yaml
default_view_transform: Un-tone-mapped
```

#### File Rules (Auto Color Space Assignment)
```yaml
file_rules:
  - !<Rule> {name: EXR, colorspace: ACES2065-1, pattern: "*", extension: exr}
  - !<Rule> {name: Movies, colorspace: Rec.1886 Rec.709 - Display, extension: [mp4, mov, mxf]}
  - !<Rule> {name: Default, colorspace: sRGB - Display}
```

---

## 5. OCIO CONFIGURATION LOADING

### Environment Variable Setup

**All entry points set this before importing OCIO modules**:

```python
ocio_config_path = project_root / "config" / "aces" / "studio-config.ocio"
if ocio_config_path.exists():
    os.environ["OCIO"] = str(ocio_config_path)
```

**Scripts using this pattern**:
- `train_dequant_net.py` (line 62-63)
- `bake_dataset.py` (line 33)
- `generate_on_the_fly_dataset.py` (line 36)

**Notebooks using this pattern**:
- `verify_gpu_renderer_orientation.ipynb`
- `on_the_fly_data_test.ipynb`
- `dequantization_inference.ipynb`
- `dequantization_inference_imagegen.ipynb`
- `dequantization_validation.ipynb`

### Config Loading in GPU Processor

```python
# In apply_ocio_torch()
config = ocio.GetCurrentConfig()  # Reads OCIO env var
processor = config.getProcessor(
    input_cs,  # e.g., "ACES2065-1"
    display,   # e.g., "sRGB - Display"
    view,      # e.g., "ACES 2.0 - SDR 100 nits (Rec.709)"
    ocio.TRANSFORM_DIR_FORWARD
)
```

---

## 6. GPU SHADER GENERATION PATTERNS

### Shader Compilation Pipeline

**Location**: [src/luminascale/utils/gpu_torch_processor.py#L280-L335](src/luminascale/utils/gpu_torch_processor.py#L280-L335)

#### Step 1: Generate GLSL Code from OCIO
```python
gpu_processor = processor.getDefaultGPUProcessor()
shader_desc = ocio.GpuShaderDesc.CreateShaderDesc(language=ocio.GPU_LANGUAGE_GLSL_4_0)
gpu_processor.extractGpuShaderInfo(shader_desc)
frag_src = GLSL_FRAG_OCIO_SRC_FMT.format(ocio_src=shader_desc.getShaderText())
```

#### Step 2: Compile Vertex Shader
```glsl
#version 400 core
uniform mat4 mvpMat;
in vec3 in_position;
in vec2 in_texCoord;
out vec2 vert_texCoord;

void main() {
    vert_texCoord = in_texCoord;
    gl_Position = mvpMat * vec4(in_position, 1.0);
}
```

#### Step 3: Build Fragment Shader Template
```glsl
#version 400 core
uniform sampler2D imageTex;
in vec2 vert_texCoord;
out vec4 frag_color;

{ocio_src}  // Injected OCIO GLSL code

void main() {
    vec4 inColor = texture(imageTex, vert_texCoord);
    vec4 outColor = OCIOMain(inColor);
    frag_color = outColor;
}
```

#### Step 4: Link Program
```python
program = GL.glCreateProgram()
GL.glAttachShader(program, vert)
GL.glAttachShader(program, frag)
GL.glBindAttribLocation(program, 0, "in_position")
GL.glBindAttribLocation(program, 1, "in_texCoord")
GL.glLinkProgram(program)
```

### Shader Program Caching

**Cache Key**: `(input_cs, display, view)`

```python
def _build_shader_program_cached(self, frag_src, input_cs, display, view) -> bool:
    cache_key = (input_cs, display, view)
    if cache_key in self._shader_cache:
        self._shader_program = self._shader_cache[cache_key]
        return True  # Cache hit
    # Compile and store
    success = self._build_shader_program(frag_src)
    if success:
        self._shader_cache[cache_key] = self._shader_program
    return success
```

**Cache Clearing**:
```python
def clear_shader_cache(self) -> None:
    for prog_id in self._shader_cache.values():
        GL.glDeleteProgram(prog_id)
    self._shader_cache.clear()
```

### Texture Allocation for LUTs

**Location**: [src/luminascale/utils/gpu_torch_processor.py#L544-L595](src/luminascale/utils/gpu_torch_processor.py#L544-L595)

```python
def _allocate_ocio_tex(self) -> bool:
    """Allocate GPU textures for OCIO lookup tables."""
    for i in range(self._ocio_shader_desc.getNumTextures()):
        tex_info = self._ocio_shader_desc.getTexture(i)
        tex_type = GL.GL_TEXTURE_1D if tex_info.textureDimensions == 1 else GL.GL_TEXTURE_2D
        
        # Create texture on GPU
        tex = GL.glGenTextures(1)
        GL.glBindTexture(tex_type, tex)
        GL.glTexImage2D(tex_type, 0, GL.GL_RGBA32F, tex_info.width, tex_info.height, ...)
        
        self._ocio_tex_ids.append((tex, tex_info.textureName, tex_info.samplerName, tex_type, tex_index))
    return True
```

**Texture Types**:
- 1D LUT: Traditional OCIO color LUTs (e.g., gamma curves)
- 2D/3D LUT: Complex transforms (e.g., tone-mapping for HDR)

### Texture Binding in Shader

```python
def _use_ocio_tex(self) -> None:
    """Bind all OCIO textures to shader."""
    for tex, tex_name, sampler_name, tex_type, tex_index in self._ocio_tex_ids:
        GL.glActiveTexture(GL.GL_TEXTURE0 + tex_index)
        GL.glBindTexture(tex_type, tex)
        sampler_loc = GL.glGetUniformLocation(self._shader_program, sampler_name)
        GL.glUniform1i(sampler_loc, tex_index)
```

---

## 7. CACHED TRANSFORM LOGIC

### Multi-Level Caching Strategy

#### Level 1: Shader Program Cache (Memory)
- **Key**: `(input_cs, display, view)`
- **Stores**: Compiled OpenGL program IDs
- **Hit Rate**: High for repeated transforms with same parameters
- **Location**: `self._shader_cache` dict

```python
# First call: COMPILE
if cache_key not in self._shader_cache:
    processor = config.getProcessor(input_cs, display, view, ...)
    gpu_processor = processor.getDefaultGPUProcessor()
    # ... shader compilation ...
    self._shader_cache[cache_key] = program_id

# Subsequent calls: REUSE
self._shader_program = self._shader_cache[cache_key]
GL.glUseProgram(self._shader_program)
```

#### Level 2: OCIO GPU Processor (OCIO Internal)
OCIO library caches extracted transforms internally via `getDefaultGPUProcessor()`.

#### Level 3: Texture Memory (GPU VRAM)
Once allocated, OCIO LUT textures stay in GPU memory until `_del_ocio_tex()` is called.

### Memory Management

**Input Texture Cleanup**:
```python
# After rendering
GL.glDeleteTextures([self._image_tex])
self._image_tex = None
```

**Full Cleanup**:
```python
def cleanup(self) -> None:
    self._del_ocio_tex()       # Delete LUT textures
    GL.glDeleteProgram(self._shader_program)
    GL.glDeleteVertexArrays(1, [self._vao])
    GL.glDeleteFramebuffers(1, [self._framebuffer])
    GL.glDeleteRenderbuffers(1, [self._renderbuffer])
    GL.glDeleteTextures([self._image_tex])
```

---

## 8. DATA FLOW: ACES INPUT → DISPLAY OUTPUT

### Complete End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                           │
└─────────────────────────────────────────────────────────────────────┘

1. LOAD STAGE (CPU)
   ├─ ACES EXR File (16-bit linear, ACES2065-1 color space)
   ├─ Read via OpenImageIO: oiio.ImageBuf(path)
   └─ Convert to PyTorch tensor [H, W, 3] float32

2. OPTIONAL: CDL GRADING (GPU CUDA)
   ├─ Input: ACES tensor [H, W, 3] on GPU
   ├─ Apply: (Input × Slope + Offset) ^ Power
   ├─ Apply: Saturation blend (luma-weighted)
   └─ Output: Graded ACES [H, W, 3] on GPU

3. OCIO TRANSFORM (GPU OpenGL)
   ├─ Stage 3a: Config & Processor Extraction
   │   ├─ Load config from OCIO env var
   │   ├─ Get processor: config.getProcessor(
   │   │   "ACES2065-1",                              # In
   │   │   "sRGB - Display",                          # Display
   │   │   "ACES 2.0 - SDR 100 nits (Rec.709)",      # View
   │   │   TRANSFORM_DIR_FORWARD
   │   └─ )
   │
   ├─ Stage 3b: Shader Code Generation
   │   ├─ gpu_processor = processor.getDefaultGPUProcessor()
   │   ├─ Extract GLSL 4.0 code via GpuShaderDesc
   │   ├─ Shader includes:
   │   │   ├─ RRT (Reference Rendering Transform)
   │   │   ├─ ODT (Output Device Transform Rec.709)
   │   │   └─ Optional user looks/CDL
   │   └─ Code injected into fragment shader template
   │
   ├─ Stage 3c: Shader Compilation & Caching
   │   ├─ Check cache[("ACES2065-1", "sRGB - Display", "SDR 100")]
   │   ├─ If miss: Compile vertex + fragment shader, link program
   │   ├─ If hit: Reuse cached program ID
   │   └─ Store in cache for future calls
   │
   ├─ Stage 3d: GPU Texture Allocation (LUTs)
   │   ├─ Extract LUT data from shader descriptor
   │   ├─ Allocate 1D/2D/3D textures on GPU
   │   ├─ Bind to texture units (GL_TEXTURE0, GL_TEXTURE1, ...)
   │   └─ Upload LUT data to VRAM
   │
   ├─ Stage 3e: Framebuffer Rendering
   │   ├─ Create input texture from ACES tensor
   │   ├─ Bind input to GL_TEXTURE0
   │   ├─ Set up output framebuffer (RGB32F)
   │   ├─ Render screen-aligned quad via glDrawElements()
   │   ├─ Execute fragment shader on every pixel:
   │   │   ├─ Sample ACES color from input texture
   │   │   ├─ Call OCIOMain(color) [OCIO GLSL function]
   │   │   ├─ Calculate RRT transform
   │   │   ├─ Sample LUTs via texture lookups
   │   │   ├─ Calculate ODT transform
   │   │   ├─ Apply sRGB encoding (Rec.709 gamma)
   │   │   └─ Write to output framebuffer
   │   └─ Synchronize: glFinish()
   │
   └─ Stage 3f: Readback to CPU & Return
       ├─ glReadPixels() → sRGB_linear [H, W, 3] float32
       ├─ Create uint8 version: (clip(sRGB) * 255).astype(uint8)
       ├─ Transfer both to GPU via torch.from_numpy()
       └─ Delete intermediate textures

4. OUTPUT DELIVERY (GPU → Training)
   ├─ sRGB_32bit: [H, W, 3] float32 on CUDA, values [0.0, 1.0]
   ├─ sRGB_8bit:  [H, W, 3] uint8 on CUDA, values [0, 255]
   └─ Both ready for training loss computation

5. MEMORY CLEANUP (GPU)
   ├─ Delete OCIO LUT textures (if not reused)
   ├─ Delete framebuffer & renderbuffer
   ├─ Delete vertex arrays & buffers
   └─ Return GPU memory to CUDA
```

### Key Data Structures

#### ACES Input Tensor
- **Shape**: [H, W, 3] (height, width, RGB channels)
- **Type**: float32
- **Device**: CUDA GPU
- **Color Space**: ACES2065-1 (linear, wide gamut)
- **Value Range**: [0.0, ∞) (can exceed 1.0 for bright values)

#### Intermediate: CDL-Graded ACES
- **Shape**: [H, W, 3]
- **Type**: float32
- **Device**: CUDA GPU
- **Color Space**: ACES2065-1 (post-grading)
- **Modifications applied**: Slope, Offset, Power, Saturation

#### Final sRGB Output (32-bit)
- **Shape**: [H, W, 3]
- **Type**: float32
- **Device**: CUDA GPU
- **Color Space**: sRGB (display-referred, gamma-encoded)
- **Value Range**: [0.0, 1.0] (clipped for display)

#### Final sRGB Output (8-bit)
- **Shape**: [H, W, 3]
- **Type**: uint8
- **Device**: CUDA GPU
- **Color Space**: sRGB (quantized)
- **Value Range**: [0, 255]

### Performance Characteristics

| Stage | Implementation | Time | Bottleneck |
|-------|---|---|---|
| Load EXR | OpenImageIO (CPU) | ~50-200ms (disk I/O) | Disk speed |
| CDL Grading | PyTorch CUDA | ~5-10ms | Negligible |
| OCIO Transform | GPU OpenGL | ~20-50ms | GPU throughput |
| Readback | glReadPixels | ~10-30ms | PCIe bandwidth |
| **Total** | — | **~85-290ms** per image | GPU-limited |

**Note**: Shader compilation (first call) adds ~100-300ms one-time cost.

---

## 9. KNOWN LIMITATIONS & ISSUES

### Critical Issues

1. **Missing Function**: `ocio_aces_to_srgb_with_look()`
   - **Impact**: `bake_dataset.py` line 94 will fail with ImportError
   - **Solution**: Needs implementation in `io.py`
   - **Expected Behavior**: Apply CDL look + OCIO transform in one call

2. **EGL Context Fragility**
   - **Issue**: EGL initialization can fail on different HPC environments
   - **Workaround**: Multiple fallback attempts in `_initialize_egl()`
   - **Location**: [gpu_torch_processor.py#L160-L200](src/luminascale/utils/gpu_torch_processor.py#L160-L200)

### Design Limitations

1. **One Context Per Processor**
   - Single EGL context per `GPUTorchProcessor` instance
   - Cannot share across processes (multi-GPU requires separate instantiation)

2. **Shader Compilation Not Parallelized**
   - First call with new (input_cs, display, view) triplet incurs ~100-300ms
   - No concurrent shader compilation

3. **Fixed Headless Resolution**
   - EGL pbuffer surface created at 1024×1024
   - Framebuffer resizes dynamically, but initial allocation is fixed

4. **No Multi-GPU Load Balancing**
   - Each `GPUTorchProcessor` binds to single GL context
   - Multiple processors don't automatically distribute

### Color Accuracy Notes

1. **Look Application Order**
   - Current pipeline: ACES → CDL (look) → RRT → ODT
   - CDL applied in ACES2065-1 space (scene-referred)
   - Matches ACES 2.0 standard but differs from ACES 1.0

2. **Gamma Encoding**
   - sRGB OETF (Rec.709) applied by ODT
   - Output is presumed to be viewed on standard monitoring setup

3. **Reference Rendering Transform (RRT)**
   - Baked into OCIO config; cannot be customized per-image
   - Uses ACES 2.0 reference tonemap

---

## 10. SUMMARY TABLE

| Category | Details |
|---|---|
| **Primary OCIO File** | `config/aces/studio-config.ocio` (OCIO 2.5, ACES 2.0) |
| **GPU Implementation** | `GPUTorchProcessor` with EGL headless rendering + GLSL 4.0 |
| **CPU Fallback** | `oiio_aces_to_display()` using OpenImageIO |
| **Input Color Space** | ACES2065-1 (linear, wide gamut) |
| **Output Color Space** | sRGB (gamma-encoded, narrow gamut) |
| **Default Transform** | ACES 2.0 - SDR 100 nits (Rec.709) view |
| **Shader Caching** | By (input_cs, display, view) tuple |
| **Texture Caching** | OCIO LUT textures cached in GPU VRAM |
| **Optional CDL** | Slope/Offset/Power/Saturation grading via `GPUCDLProcessor` |
| **Pipeline Speed** | ~85-290ms per image (GPU-limited) |
| **Known Issues** | Missing `ocio_aces_to_srgb_with_look()` function |

---

## 11. ACES CONVERSION FLOW DIAGRAM

```
RAW Image (Camera)
    ↓
    └──→ [rawtoaces] → ACES2065-1 EXR (16-bit linear, wide gamut)
                           ↓
                    [Load to GPU Tensor]
                           ↓
         ┌─────────────────────────────────────┐
         │    Optional CDL Color Grading       │
         │  (Slope/Offset/Power/Saturation)    │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │    OCIO Transform Pipeline          │
         ├─────────────────────────────────────┤
         │ 1. Load config from OCIO env var    │
         │ 2. Generate GLSL shader code        │
         │ 3. Compile & cache shader program   │
         │ 4. Allocate GPU textures for LUTs   │
         │ 5. Render via OpenGL framebuffer    │
         │    - Execute RRT (tone-map)         │
         │    - Execute ODT (to display)       │
         │    - Apply sRGB OETF (gamma)        │
         │ 6. Readback to torch tensors        │
         └─────────────────────────────────────┘
                           ↓
              sRGB Display-Referred Output
              ├─ 32-bit float [0.0, 1.0]
              └─ 8-bit quantized [0, 255]
                           ↓
                  Training Data Ready
              (Bit-depth expansion, color matching)
```

---

**End of Analysis** | Generated: 2026-04-05

