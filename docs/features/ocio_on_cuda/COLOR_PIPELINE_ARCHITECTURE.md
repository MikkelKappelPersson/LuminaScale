# LuminaScale Color Pipeline - Visual Architecture

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING DATA FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌──────────────────┐
                            │  Raw EXR Files   │
                            │ (ACES2065-1 HDR) │
                            │  32-bit float    │
                            └─────────┬────────┘
                                      │
                                      ↓
                    ┌─────────────────────────────────┐
                    │  LUT PACKING (bake_dataset.py)  │
                    │                                 │
                    │ For each EXR:                   │
                    │ 1. [Optional] Apply CDL look    │
                    │ 2. OCIO transform to sRGB       │
                    │ 3. Quantize to uint8            │
                    │ 4. Save as PNG or EXR           │
                    └──────────────┬──────────────────┘
                                   │
                                   ↓
                    ┌─────────────────────────────────┐
                    │  LMDB Packing (pack_lmdb.py)    │
                    │                                 │
                    │ Binary layout:                  │
                    │ [Header (12B)][HDR][LDR]        │
                    │  H,W,C | float32 | uint8        │
                    └──────────────┬──────────────────┘
                                   │
                                   ↓ (40-100x speedup)
                    ┌─────────────────────────────────┐
                    │   LMDB Dataset File             │
                    │   (Uncompressed, Read-only)     │
                    │   Size: ~1-100 GB               │
                    └──────────────┬──────────────────┘
                                   │
          ┌────────────────────────┴───────────────────────┐
          │                                                │
    ┌─────▼──────────────────────────────────────────────────────────┐
    │                   TRAINING EPOCH LOOP                          │
    │                                                                │
    │  ┌────────────────────────────────────────────────────────┐   │
    │  │ DataLoader                                             │   │
    │  │                                                        │   │
    │  │  num_workers=4      (CPU I/O threads)                │   │
    │  │  batch_size=16      (GPU batch)                      │   │
    │  │  pin_memory=True    (DMA-enabled host pinning)       │   │
    │  │                                                        │   │
    │  │  For each batch:                                      │   │
    │  │  ├─ Read bytes from LMDB (CPU, non-blocking)         │   │
    │  │  ├─ Parse header (H, W, C) from first 12 bytes        │   │
    │  │  ├─ Create numpy array from binary buffer             │   │
    │  │  ├─ Convert to torch tensor (still CPU)               │   │
    │  │  └─ Transfer to GPU (.to(device, non_blocking=True))  │   │
    │  └────────────────┬─────────────────────────────────────┘   │
    │                   │ [B, H, W, 3] on CUDA                     │
    │                   ↓                                          │
    │  ┌────────────────────────────────────────────────────────┐  │
    │  │ GPU Processing Phase                                   │  │
    │  │                                                        │  │
    │  │ DatasetPairGenerator.load_aces_apply_cdl_and_   │     │  │
    │  │    transform()                                  │     │  │
    │  │                                                 │     │  │
    │  │ ┌─────────────────────────────────────────┐    │ ← – ┼  │
    │  │ │ 1. CDL Grading (Optional)               │    │     │  │
    │  │ │    gpu_cdl_processor.apply_cdl_gpu()    │    │     │  │
    │  │ │    - Load randop CDL params             │    │     │  │
    │  │ │    - Input × Slope + Offset             │    │     │  │
    │  │ │    - (result) ^ Power                   │    │     │  │
    │  │ │    - Luma-weighted saturation           │    │     │  │
    │  │ │    → Output: [B, H, W, 3] graded ACES   │    │     │  │
    │  │ └──────────────┬──────────────────────────┘    │     │  │
    │  │                ↓                               │     │  │
    │  │ ┌─────────────────────────────────────────┐    │     │  │
    │  │ │ 2. OCIO Transform (EGL/OpenGL)          │    │     │  │
    │  │ │    gpu_torch_processor.apply_ocio_torch()   │     │  │
    │  │ │                                         │    │     │  │
    │  │ │ ┌──────────────────────────────────┐   │    │     │  │
    │  │ │ │ A. Setup (First time only)       │   │    │     │  │
    │  │ │ │  - Query OCIO config             │   │    │     │  │
    │  │ │ │  - Generate GLSL shader code     │   │    │     │  │
    │  │ │ │  - Compile vertex/fragment       │   │    │     │  │
    │  │ │ │  - Cache by transform params     │   │ ← – ┼  │
    │  │ │ └──────────────────────────────────┘   │    │     │  │
    │  │ │                                         │    │     │  │
    │  │ │ ┌──────────────────────────────────┐   │    │     │  │
    │  │ │ │ B. Upload (Every batch)          │   │    │ ← – ┼  │
    │  │ │ │  - CUDA tensor → numpy (GPU→CPU) │   │    │     │  │
    │  │ │ │  - numpy → GL_TEXTURE_2D (CPU→GPU)  │    │     │  │
    │  │ │ │  - Upload 1D/2D/3D LUTs (cold)   │   │ ← – ┼  │
    │  │ │ │  → Texture in GPU VRAM            │   │    │     │  │
    │  │ │ └──────────────────────────────────┘   │    │     │  │
    │  │ │                                         │    │     │  │
    │  │ │ ┌──────────────────────────────────┐   │    │     │  │
    │  │ │ │ C. Render (Every batch)          │   │    │     │  │
    │  │ │ │  - Bind framebuffer              │   │    │     │  │
    │  │ │ │  - Render quad via shader        │   │    │     │  │
    │  │ │ │  - Fragment runs: OCIOMain()     │   │    │     │  │
    │  │ │ │  - Output: [B, H, W, 3] RGB32F   │   │    │     │  │
    │  │ │ └──────────────────────────────────┘   │    │     │  │
    │  │ │                                         │    │     │  │
    │  │ │ ┌──────────────────────────────────┐   │    │     │  │
    │  │ │ │ D. Readback & Quantize           │   │    │ ← – ┼  │
    │  │ │ │  - glReadPixels() [GPU STALL]    │   │    │     │  │
    │  │ │ │  - GPU→CPU transfer (PCIe)       │   │    │     │  │
    │  │ │ │  - Float32 → uint8 (×255)        │   │    │     │  │
    │  │ │ │  - Copy back to GPU              │   │ ← – ┼  │
    │  │ │ │  → Output: float32 + uint8       │   │    │     │  │
    │  │ │ └──────────────────────────────────┘   │    │     │  │
    │  │ │                                         │    │     │  │
    │  │ └─────────────────────────────────────────┘    │     │  │
    │  │                                                 │     │  │
    │  │ Return: srgb_32bit [B, H, W, 3], srgb_8bit [...]  │  │
    │  └────────────────┬─────────────────────────────────┘  │  │
    │                   │                                     │  │
    │                   ↓ Both on GPU, ready for BDE network │  │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │ BDE Network Forward Pass                           │  │
    │  │                                                    │  │
    │  │ Input:     srgb_8bit [degraded 8-bit]              │  │
    │  │ Reference: srgb_32bit [target 32-bit full-res]     │  │
    │  │            aces_graded [reference ACES]            │  │
    │  │                                                    │  │
    │  │ Loss computation:                                  │  │
    │  │ 1. Perceptual loss (VGG features)                  │  │
    │  │ 2. Pixel-space MSE                                 │  │
    │  │ 3. Color accuracy loss (ΔE in ACES space)          │  │
    │  │ 4. Consistency loss (reference ACES)               │  │
    │  │                                                    │  │
    │  │ Backpropagation & optimizer step                   │  │
    │  └────────────────────────────────────────────────────┘  │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
                          ↑ (repeat per batch)
                          │ (repeat per epoch)
```

## OpenGL Rendering Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OCIO GPU TRANSFORM PIPELINE                             │
│                      (apply_ocio_torch details)                             │
└─────────────────────────────────────────────────────────────────────────────┘

Input Tensor [H, W, 3] on CUDA
        │
        ↓
┌───────────────────┐
│ EGL Initialization│  (First time only)
│                   │
│ eglGetDisplay()   │──────┐ Multiple fallback paths:
│ eglInitialize()   │      ├─ EGL_DEFAULT_DISPLAY
│ eglCreateContext()│      ├─ eglGetDisplay(0)
│ eglMakeCurrent()  │      └─ eglGetDisplay(None)
│                   │
│ Context Version: 4.0
│ Surface: 1024×1024 Pbuffer (offscreen)
└───────┬───────────┘
        │
        ↓ (First transform only)
┌───────────────────────────────────┐
│ Shader Generation & Compilation   │
│                                   │
│ 1. Query OCIO Config              │
│    ocio.GetCurrentConfig()        │
│                                   │
│ 2. Get Processor                  │
│    input_cs="ACES2065-1"          │
│    display="sRGB - Display"       │
│    view="ACES 2.0 - ..."          │
│    processor = config.getProcessor│
│                                   │
│ 3. Create GpuShaderDesc           │
│    shader_desc = ocio.GpuShaderDesc  │
│      .CreateShaderDesc(           │
│      language=GLSL_4_0)           │
│                                   │
│ 4. Extract GPU Shader Info        │
│    gpu_processor.extractGpuShaderInfo│
│      (shader_desc)                │
│                                   │
│ 5. Get GLSL Code                  │
│    glsl_text = shader_desc        │
│      .getShaderText()             │
│                                   │
│    Content: matrix ops, LUT lookups,
│             color transforms,
│             gamma encoding       │
│                                   │
│ 6. Compile Shaders                │
│    glCompileShader(VERTEX)        │ Fixed quad
│    glCompileShader(FRAGMENT)      │ + OCIO code
│    glLinkProgram()                │
│                                   │
│ 7. Cache                          │
│    _shader_cache[(input_cs,       │
│      display, view)] = program_id │
└───────┬───────────────────────────┘
        │
        ↓ (Every frame)
┌───────────────────────────────────┐
│ Texture & Geometry Setup          │
│                                   │
│ 1. Convert Tensor → Numpy         │
│    aces_tensor.cpu().numpy()      │
│    [CUDA → CPU, GPU→CPU transfer] │
│                                   │
│ 2. Create Input Texture           │
│    GL_TEXTURE_2D (RGB32F)         │
│    glTexImage2D(...,              │
│      W, H, GL_RGB, GL_FLOAT, data)│
│    [CPU → GPU transfer, PCIe]     │
│                                   │
│ 3. Allocate LUT Textures          │
│    For each 3D LUT:               │
│    ├─ GL_TEXTURE_3D (64³-128³)    │
│    ├─ glTexImage3D(...)           │
│    └─ Bind to sampler uniform     │
│                                   │
│    For each 1D/2D LUT:            │
│    ├─ GL_TEXTURE_{1D,2D}          │
│    └─ glTexImage{1,2}D(...)       │
│                                   │
│ 4. Setup Framebuffer              │
│    glGenFramebuffers()            │
│    Attach RGB32F color target     │
│    Attach DEPTH32F depth buffer   │
│                                   │
│ 5. Setup Screen Quad              │
│    glGenVertexArrays()            │
│    6 vertices forming 2 triangles  │
│    Tex coords: [0,1] × [0,1]      │
│    Positions: NDC [-1,1] × [-1,1] │
└───────┬───────────────────────────┘
        │
        ↓
┌───────────────────────────────────┐
│ Rendering                         │
│                                   │
│ glBindFramebuffer(..., fbo)       │
│ glBindVertexArray(vao)            │
│ glViewport(0, 0, W, H)            │
│                                   │
│ glUseProgram(shader_program)      │
│                                   │
│ Bind Textures:                    │
│ ├─ glActiveTexture(GL_TEXTURE0)   │
│ ├─ glBindTexture(GL_TEXTURE_2D)   │ Input image
│ ├─ glActiveTexture(GL_TEXTURE1)   │
│ ├─ glBindTexture(GL_TEXTURE_3D)   │ OCIO LUT 1
│ └─ (repeat for all LUTs)          │
│                                   │
│ Set Uniforms:                     │
│ ├─ MVP matrix (identity)          │
│ ├─ Texture samplers               │
│ └─ OCIO-specific uniforms          │
│                                   │
│ glDrawElements(6, ...)  ← Render   │
│ glFinish()              ← GPU sync │
│                                   │
│ [Fragment Shader runs on GPU]     │
│ For each pixel:                   │
│   inColor = texture(imageTex, uv) │
│   (runs OCIO color math)          │
│   outColor = OCIOMain(inColor)    │
│ ═══════════════════════════════════│
│ Result written to framebuffer     │
└───────┬───────────────────────────┘
        │
        ↓
┌───────────────────────────────────┐
│ Readback & Quantization           │
│                                   │
│ glReadBuffer(COLOR_ATTACHMENT0)   │
│                                   │
│ glReadPixels(0, 0, W, H,          │
│   GL_RGB, GL_FLOAT, buf)          │
│                                   │
│ [GPU ↔ CPU, SYNCHRONOUS STALL]    │
│ CPU waits for GPU to flush        │
│ PCIe transfer happens             │
│ [GPU→CPU transfer]                │
│                                   │
│ res_32f = np.frombuffer(buf,      │
│   dtype=np.float32).reshape(...)  │
│                                   │
│ Quantize to uint8:                │
│ ├─ Clip to [0.0, 1.0]             │
│ ├─ Scale: × 255                   │
│ ├─ Round                          │
│ └─ Cast to uint8                  │
│ res_8u = (res_32f × 255)          │
│   .astype(uint8)                  │
│                                   │
│ Convert back to GPU tensors:      │
│ ├─ numpy → torch on CPU           │
│ ├─ .to(device) [CPU→GPU]          │
│ └─ Return (float32, uint8)        │
└───────┬───────────────────────────┘
        │
        ↓
Output Tensors on GPU:
 ├─ srgb_32bit [H, W, 3] FLOAT32
 └─ srgb_8bit [H, W, 3] UINT8
```

## Memory Layout Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   GPU MEMORY (VRAM) STATE                       │
│                                                                 │
│                   [During Rendering]                           │
│                                                                 │
│  ┌──────────────────────┐                                      │
│  │ Shader Program(s)    │ ~50 KB (loaded once)                │
│  │ ├─ VS (quad)         │                                      │
│  │ └─ FS (OCIO code)    │                                      │
│  └──────────────────────┘                                      │
│                                                                 │
│  ┌──────────────────────┐                                      │
│  │ LUT Textures         │ 10-50 MB (warm cache)               │
│  │ ├─ 3D LUT (64-128³)  │ ~30 MB  per image                  │
│  │ ├─ 1D LUT (gamma)    │ ~1 KB                               │
│  │ └─ 2D LUT (tone map) │ ~1-10 MB                            │
│  └──────────────────────┘                                      │
│                                                                 │
│  ┌──────────────────────┐                                      │
│  │ Input Texture        │ 12 MB (RGB32F)                      │
│  │ [H, W, 3] float32    │ 1024² × 3 × 4 bytes                │
│  └──────────────────────┘                                      │
│                                                                 │
│  ┌──────────────────────┐                                      │
│  │ Output Texture       │ 12 MB (RGB32F)                      │
│  │ [H, W, 3] float32    │ 1024² × 3 × 4 bytes                │
│  └──────────────────────┘                                      │
│                                                                 │
│  ┌──────────────────────┐                                      │
│  │ Framebuffer          │ 16 MB (FBO)                         │
│  │ ├─ Color target      │ 12 MB RGB32F                        │
│  │ └─ Depth buffer      │  4 MB DEPTH32F                      │
│  └──────────────────────┘                                      │
│                                                                 │
│  ┌──────────────────────┐                                      │
│  │ Vertex/Index Buffers │ ~1 KB (fixed quad)                 │
│  │ (VAO)                │                                      │
│  └──────────────────────┘                                      │
│                                                                 │
│  ────────────────────────────────                              │
│  Total per 1024² image: ~63 MB                                  │
│                                                                 │
│  PyTorch CUDA memory (separate):                                │
│  ├─ Input batch [B, H, W, 3]  = B × 12 MB                     │
│  ├─ Output batch [B, H, W, 3] = B × 12 MB                     │
│  ├─ Model weights              = 50-200 MB                     │
│  └─ Activations/gradients      = 100-500 MB (variable)        │
│                                                                 │
│  ────────────────────────────────                              │
│  Total (B=16, batch):                                           │
│  ~(63 + 24×16 + 300) MB = ~687 MB for full pipeline            │
│  (Typical GPU: 12-40 GB available)                              │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Timeline (per batch of 8 images)

```
t=0ms     ├─ CPU Load Start (DataLoader)
          │  LMDB sequential read (8 × 12MB = 96MB)
          │  Minimal processing (just convert to tensor)
t~20ms    ├─ CPU Load Complete → GPU Transfer Start  
          │  .to(device, non_blocking=True)
          │  8 images × 12MB = 96MB over PCIe
          │  PCIe Gen3 × 4: ~4GB/s → ~24ms
          │
t~44ms    ├─ Transfer Complete → GPU Processing Start
          │  ┌─ Image 1 ────────┐
          │  │ Upload: 3ms      │
          │  │ CDL: 1ms         │
          │  │ Render: 2ms      │
          │  │ Readback: 3ms    │
          │  │ ─────────────────│
          │  │ Total: 9ms       │
          │  └──────────────────┘
          │  ┌─ Image 2 (overlapped) ──┐
          │  │ Can start during 1's     │
          │  │ PCIe readback            │
          │  │ ~1ms parallel overhead   │
          │  └──────────────────────────┘
          │  × 8 images sequentially
          │  (Some parallelism possible with async)
          │
t~116ms   ├─ GPU Processing Complete
          │  (8 images × 9ms warm = 72ms)
          │  + transfer overhead ~24ms
          │
t~116ms   ├─ Results Ready on GPU
          │  Can immediately start training forward pass
          │
          ├─ Next batch loads asynchronously
          │
```

## Bottleneck Waterfall

```
Warm Run (8 images, 1024×1024 each)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CPU→GPU Transfer    ▓▓▓▓▓  6ms (PCIe bandwidth limited)  ← PRIMARY
                    6ms                                    ← 50-60% of latency
                    
GL Upload           ▓▓▓   3ms (redundant with transfer)
Shader Bind         ▓     0.5ms
LUT Bind            ▓     0.5ms
Render              ▓▓    1-2ms
GPU Stall/Readback  ▓▓    2-3ms                          ← SECONDARY
                    2-3ms                                  ← 20-30% of latency
CPU post-processing ▓     1ms
GPU→CPU Transfer    ▓▓▓   3ms (PCIe readback)

Data Copy overhead  ▓     1ms
─────────────────────────────
TOTAL WARM:         ~8-11ms per image
                    ~65-90ms per 8-image batch
                    ≈ 90-130 images/sec throughput


Cold Run (first call, new transform)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EGL Setup           ▓▓▓▓▓▓▓▓▓▓  50-150ms  (one-time)

Shader Compile      ▓▓▓▓▓▓▓▓▓▓▓  100-200ms  ← DOMINANT
                    100-200ms

LUT Upload (first)  ▓▓▓▓▓        10-20ms

─ Then all warm + overhead items ─

─────────────────────────────
TOTAL COLD:         ~170-380ms  (first batch only)
                    ≈ 2-6 images/sec
```

---

**Key Insight**: The pipeline is **I/O bound**, not compute bound. The GPU shader
execution is fast (~2-5ms), but PCIe transfers dominate (~6-8ms of 10ms total).

Improvement opportunities require either:
1. CUDA↔OpenGL interop (eliminate format conversion)
2. Batch processing (amortize setup overhead)
3. Async readback (PBO + fences)
