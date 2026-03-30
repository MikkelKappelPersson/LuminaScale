# Project Plan: AI-Restorative Color & Bit-Depth Model

**Version:** 2.0  
**Last Modified:** 2026-03-30  
**Status:** Sequential Two-Stage Architecture

## 1. Project Objective

Build a Machine Learning model capable of processing 8-bit, "arbitrarily stylized," or AI-generated images and transforming them into a clean, 16-bit-equivalent **ACES2065-1** color space. The model must sequentially solve **color space normalization** followed by **structural bit-depth expansion**, inspired by the HDR reconstruction pipeline in *"Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline"* (arXiv:2004.01179).

## 2. Model Architecture: Sequential Two-Stage Cascade

The model decomposes the problem into two independent, sequentially-trained stages before end-to-end joint fine-tuning. This approach isolates bit-depth expansion from color conversion, enabling cleaner learning and better error isolation.

### Stage 1: Dequantization Net (sRGB → sRGB)

- **Architecture:** Deep U-Net with skip connections
  
- **Operation:** Maps 8-bit sRGB images (quantized, with compression artifacts) into smooth 32-bit sRGB equivalents.
  
- **Training Data:** 8-bit sRGB inputs paired against 32-bit sRGB ground truth.
  
- **Job:** Learn to reverse quantization (banding) artifacts and smooth gradients within sRGB color space.
  
- **Output:** 32-bit sRGB images (clean, smooth gradients, no banding).
  
- **Loss:** MSE on sRGB values, with optional gradient penalty for banding suppression.
  
- **Normalization:** Input: `/255.0` → [0,1], Target: `/1.0` → [0,1]
  

### Stage 2: Color Conversion Net (sRGB → ACES)

- **Architecture:** U-Net with residual connections.
  
- **Operation:** Transforms 32-bit sRGB images (from Stage 1 output or ground truth) into 32-bit ACES2065-1 linear space.
  
- **Training Data:** 32-bit sRGB (ground truth, smooth) paired against 32-bit ACES ground truth.
  
- **Job:** Learn the colorimetric transformation from sRGB color space to ACES, including exposure/brightness adjustment.
  
- **Output:** 32-bit ACES images (correct color space, full linear range).
  
- **Loss:** MSE on ACES values, with optional perceptual loss in display space.
  
- **Normalization:** Input: `/1.0` → [0,1], Target: `/6.0` → [0,1]
  

### Stage 3: Joint Fine-Tuning (End-to-End)

- **Operation:** Both networks trained simultaneously on full pipeline: 8-bit sRGB input → [Stage 1 Dequant] → [Stage 2 Color Convert] → 32-bit ACES output.
  
- **Purpose:** Reduce error accumulation; Stage 1 learns to output smooth sRGB in a way that facilitates Stage 2's color conversion.
  
- **Loss:** Weighted combination:
  
  $$L_{total} = \lambda_{deq} \cdot L_{MSE,deq} + \lambda_{color} \cdot L_{MSE,color} + \lambda_p \cdot L_{perceptual}$$
  
  Suggested weights: $\lambda_{deq} = 1.0$, $\lambda_{color} = 1.0$, $\lambda_p = 0.001$

## 3. Dataset Strategy: LMDB-Based Sequential Training

The model is trained on paired LMDB data preprocessed by the `pack_lmdb.py` utility, containing 8-bit sRGB images and corresponding 32-bit ACES ground truth.

**Dataset Location:** `/mnt/MKP01/med8_project/LuminaScale/dataset/training_data.lmdb`

### Training Data Structure (LMDB)

The training dataset is packed into LMDB format with the following key structure:

```python
{
  "ldr": np.uint8 array [0, 255],       # 8-bit sRGB (quantized, arbitrary looks)
  "hdr": np.float32 array [0, 6.0],     # 32-bit ACES (normalized by 6.0)
}
```

### Stage 1: Dequantization Pre-Training (sRGB → sRGB)

- **Input:** `ldr` from LMDB (8-bit sRGB, quantized)
  
- **Target:** 32-bit sRGB ground truth (one-time generated from 8-bit by linear upsampling or inverse tone-mapping)
  
- **Purpose:** Train network to reverse quantization artifacts and smooth banding within sRGB.
  
- **Note:** 32-bit sRGB ground truth should be generated once, either by:
  1. Inverse tone-mapping of sRGB → assumed linear, then smoothed, or
  2. Using sRGB as the reference and upsampling to 32-bit float with careful handling of gamma
  

### Stage 2: Color Conversion Pre-Training (sRGB → ACES)

- **Input:** 32-bit sRGB (ground truth, smooth from Stage 1 training data)
  
- **Target:** `hdr` from LMDB (32-bit ACES ground truth)
  
- **Purpose:** Train network to map from sRGB color space to ACES, learning color correction and exposure adjustment.
  

### Stage 3: Joint Fine-Tuning Data Flow

- **Input:** `ldr` from LMDB (8-bit sRGB)
  
- **Stage 1 Output:** Smoothed 32-bit sRGB
  
- **Stage 2 Output:** Final 32-bit ACES
  
- **Target:** `hdr` from LMDB (32-bit ACES ground truth)
  

## 4. Loss Functions & Training Logic

### Stage 1: Dequantization Pre-Training

- **Pixel Loss ($L_{MSE}$):** Mean squared error against 32-bit sRGB target.
  
- **Gradient Loss (optional, $L_{Grad}$):** Penalizes sharp steps in smooth areas to suppress banding artifacts.
  

### Stage 2: Color Conversion Pre-Training

- **Pixel Loss ($L_{MSE}$):** Mean squared error against target ACES values.
  
- **Perceptual Loss (optional, $L_p$):** Feature-space similarity (e.g., VGG on display-referenced space) to preserve visual quality.
  

### Stage 3: Joint Fine-Tuning

- **Dequant Loss ($L_{deq}$):** MSE between Stage 1 output (32-bit sRGB) and ground truth smooth sRGB.
  
- **Color Loss ($L_{color}$):** MSE between Stage 2 output and ground truth ACES.
  
- **Perceptual Loss ($L_p$):** Feature-space similarity to prevent hallucination.
  
- **Total Loss:**
  $$L_{total} = \lambda_{deq} \cdot L_{deq} + \lambda_{color} \cdot L_{color} + \lambda_p \cdot L_p$$

## 5. Implementation Roadmap

1. **✓ Dataset Preparation:** LMDB with 8-bit sRGB and 32-bit ACES pairs created via `pack_lmdb.py`.
   
2. **Stage 1 Dataset Gen:** Generate 32-bit sRGB ground truth from 8-bit sRGB inputs. Create Stage 1 training pairs.
   
3. **→ Stage 1 Training:** Train Dequantization Net on 8-bit sRGB → 32-bit sRGB task. Validate convergence.
   
4. **Stage 2 Training:** Train Color Conversion Net on 32-bit sRGB → 32-bit ACES task.
   
5. **Joint Fine-Tuning:** Load both checkpoints and train end-to-end with weighted loss combination.
   
6. **Validation & Deployment:** Test on held-out real-world 8-bit sRGB images; export inference pipeline.
