# Research: Bit-Depth Expansion (BDE) & Gradient Smoothing

## Overview
This document explores the challenges of bit-depth expansion (BDE), specifically focusing on gradient smoothing and artifact reduction. Common issues like "false contouring" or "banding" occur when low-bit-depth (LBD) images are displayed in high-bit-depth (HBD) spaces.

## Key Papers & Insights

### 1. BitNet: Learning-based bit-depth expansion (Byun et al., 2018)
- **Problem**: Traditional methods suffer from inaccurate smoothing or loss of detail.
- **Insight**: Highlights the importance of "dense gradient flow". The network needs to understand the underlying continuous signal despite the discrete quantization steps.
- **Applicability**: Suggests that our current U-Net might need better gradient-aware loss functions rather than just L2.

### 2. Deep Reconstruction of Least Significant Bits (Zhao et al., 2019)
- **Concept**: Instead of predicting the whole HBD image, the network specifically focuses on reconstructing the "Missing LSBs".
- **Insight**: Training on the LSB residual helps in avoiding the vanishing gradient problem and focuses the model on the quantization error itself.
- **Applicability**: This aligns with our "residual learning" approach but suggests we should evaluate if our residual is effectively recovering the lost bits.

### 3. Bit-depth Enhancement via CNN (Liu et al., 2017)
- **Challenge**: Identifies "contouring effects" in smooth gradient areas as a primary quality degrader.
- **Insight**: Early deep learning methods struggled with over-smoothing.
- **Applicability**: Confirms that smoothing gradients is a known hard problem in BDE.

### 4. BE-CALF: Bit-depth enhancement by concatenating all level features (Liu et al., 2019)
- **Concept**: Uses a multi-scale approach to capture both global context (for gradients) and local details.
- **Applicability**: Our 6-level U-Net does concatenate features, but we might need to verify if the "global" information is reaching the output effectively to handle wide gradients.

---

## Analysis of Gradient Smoothing Failure
Based on the project analysis ([01_TRAINING_PLATEAU_ROOT_CAUSE.md](../analysis/01_TRAINING_PLATEAU_ROOT_CAUSE.md)), our model plateaus because:
1. **L2 Loss doesn't penalize banding**: MSE is minimized by a "step-like" average which still contains artifacts.
2. **Missing Local Smoothness Constraint**: There is no explicit reward for predicting a linear gradient between two quantized steps.
3. **Information Bottleneck**: The Tanh activation might be regularizing the output too much for fine-grained LSB recovery.

## Proposed Strategy: Gradient-Aware Training
- **Total Variation (TV) Loss**: Penalize sharp jumps in the residual where the target is smooth.
- **Gradient Difference Loss**: Compare $\nabla Pred$ with $\nabla Target$.
- **Data Augmentation (Blur)**: Smoothing the target to force the model to learn the "original" continuous signal (see [dataset_blur_strategies.md](dataset_blur_strategies.md)).
