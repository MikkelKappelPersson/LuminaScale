# Spec: ACES2065-1 Color Space Mapper

## Purpose
The ACES2065-1 Color Space Mapper is the "second head" of the LuminaScale pipeline. Its primary objective is to map images from arbitrary display-referred color spaces (sRGB, Rec.709, etc.) into the scene-referred linear ACES2065-1 (AP0) color space.

## Architecture: Hybrid Neural LUT & Local Refiner
To achieve state-of-the-art (SOTA) performance with high efficiency, the mapper utilizes a dual-head architecture.

### 1. Global Fitting Head: Image-Adaptive 3D LUT with SFT
- **Mechanism (Weight Predictor)**: A **Spatial-Frequency Transformer (SFT)** analyzes the input image to predict weights for a bank of basis 3D LUTs.
  - **Spatial Domain**: Captures global lighting, objects, and scene context.
  - **Frequency Domain**: Analyzes textures and noise patterns to better identify the source ISP "intent" (e.g., whether noise indicates a boosted dark scene).
- **Function**: Handles the heavy lifting of gamut primary shifts and tone mapping inversion.
- **Benefits**: Superior long-range dependency modeling compared to standard CNNs; more accurate "Many-to-One" mapping by "looking through" varying photo-finishing effects.

### 2. Local Refinement Head: Laplacian Reconstruction
- **Mechanism**: An **Adaptive Laplacian Pyramid** reconstruction pipeline.
- **Function**: Uses a **PPB (Pixel-wise Progressive Block)** or **HFBlock** (High-Frequency Block) to handle the residual reconstruction of the image. It fuses the global tone-mapped result from the LUTs with high-frequency details from the source image.
- **Benefits**: Ensures resolution-independent detail preservation and artifact suppression (e.g., Halo Loss) that a global LUT cannot provide.
- **Reference**: Defined in LLF-LUT via `PPB.py` (Laplacian decomposition/reconstruction) and coordinated in `LLF_LUT.py`.

## Integration & Framework
- **Framework**: PyTorch Lightning
- **Logger**: Aim (integrated via `AimLogger`)
- **Telemetry**: Track $\Delta E_{ITP}$, PSNR, and training throughput (samples/sec).

## Optimization & Loss Functions
- **Primary Loss**: Charbonnier Loss (for robustness against outliers in linear light).
- **Color Accuracy**: $\Delta E_{ITP}$ (ICtCp-based) to ensure perceptual uniformity in high dynamic range.
- **Structural Integrity**: Gradient matching loss to preserve edge information during bit-depth expansion.

## Training Strategy
- **Many-to-One**: Train on diverse sRGB renditions of the same ACES ground truth to ensure invariance to source camera metadata.
- **Datasets**: 
  - MIT-Adobe FiveK (Expert C)
  - PPR10K (Skin tone fidelity)
- **Precision**: 16-bit mixed precision (BF16 or FP16) to handle the wide dynamic range of ACES linear light.

## Implementation Details & References
- **Reference Repository**: A copy of the LLF-LUT repository is located at `/run/media/mikkelkp/MKP01/med8_project/refs/LLF-LUT`.
  - The choice of **Charbonnier Loss** (as seen in `utils/loss.py` of the reference) is motivated by its robustness in handling outlier pixels in linear light reconstruction.
  - **Mandatory Attribution**: Any code derived from or inspired by this repository must include a clear reference in the docstrings (e.g., `Based on LLF-LUT (Zeng et al./Wang et al.)`).

## Modular Model Structure
To ensure clarity and maintainability, the architecture is decomposed as follows. All model files are located directly in `src/luminascale/models/` for consistency.

### 1. `SpatialFrequencyTransformer` (SFT)
- **Path**: [src/luminascale/models/spatial_frequency_transformer.py](src/luminascale/models/spatial_frequency_transformer.py)
- **Function**: The weight prediction head based on `Spatial_Transformer.py` in the LLF-LUT reference.

### 2. `Adaptive3DLUT`
- **Path**: [src/luminascale/models/adaptive_3d_lut.py](src/luminascale/models/adaptive_3d_lut.py)
- **Function**: Implements the learnable bank of 3D LUTs based on `LUT.py` in the LLF-LUT reference.

### 3. `LocalRefinementHead` (PPB & Laplacian Engine)
- **Path**: [src/luminascale/models/local_refiner.py](src/luminascale/models/local_refiner.py)
- **Function**: The spatially-aware compensation head based on the **Pixel-wise Progressive Block (PPB)** and **High-Frequency Block (HFBlock)** logic from LLF-LUT.
- **Components**:
  - `Lap_Pyramid_Conv`: Gaussian/Laplacian decomposition and reconstruction.
  - `HFBlock`: Residual processing head using `remapping` logic to fuse global LUT results with high-frequency residuals.
  - `PPB`: High-level wrapper that incorporates an **edge map** (via Canny) for edge-aware refinement.
- **Reference**: Ported and adapted from `PPB.py` in the LLF-LUT reference.

### 4. `ACESMapper` (Integrator)
- **Path**: [src/luminascale/models/aces_mapper.py](src/luminascale/models/aces_mapper.py)
- **Function**: The top-level integrator (counterpart to `DequantNet`). Coordinates the SFT, LUT, and Local Refinement heads.

## Architecture Configuration
- **Module Name**: `ACESMapper`
- **Trainer Path**: [src/luminascale/training/aces_trainer.py](src/luminascale/training/aces_trainer.py)
- **Config**: [configs/task/mapper.yaml](configs/task/mapper.yaml)
