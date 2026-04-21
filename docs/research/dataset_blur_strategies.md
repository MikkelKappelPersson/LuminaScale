# Research: Dataset Blur & Preprocessing for Dequantization

## Concept: Learning from "Clean" Gradients
The user's hypothesis is that applying blur to the dataset could remove grain/noise and create more natural gradients for the model to learn from. This effectively shifts the task from "dequantization of noisy data" to "interpolation of clean signals".

## Theoretical Basis

### 1. Cascaded Diffusion Models (Ho et al., 2022)
- **Insight**: Uses "blurring augmentation" to help models learn structure before fine details.
- **Applicability**: By blurring the HBD target (and possibly the LBD input), we can focus the model on the low-frequency gradient reconstruction first.

### 2. Multi-stage Progressive Image Restoration (Zamir et al., 2021)
- **Concept**: Progressive training (small patches/small blur $\rightarrow$ full resolution/no blur).
- **Applicability**: We could start training on blurred data to get the gradient logic right, then "anneal" the blur to introduce details.

### 3. Blurring as a "Denoising" step for Ground Truth
- In cinematic workflows, grain is often added *after* the linear gradient.
- If our HBD targets have sensor noise/grain, the quantization of that noise creates "chatter" in the banding.
- Blurring the target (HBD) during training creates a "super-smooth" target that doesn't exist in nature but represents the *ideal* mathematical gradient.

## Implementation Ideas

### Strategy A: Gaussian Blur on Ground Truth (Target Only)
- **Mechanism**: Apply a light Gaussian blur ($\sigma \in [0.5, 1.5]$) to the HBD target before calculating loss.
- **Pros**: Removes grain, provides a perfectly smooth target for the model to "reach" for.
- **Cons**: Might cause the model to lose sharpness in high-frequency regions.

### Strategy B: Progressive Blur (Curriculum Learning)
- **Epoch 0-5**: $\sigma=2.0$ (Learn core gradients)
- **Epoch 5-10**: $\sigma=1.0$ (Refine edges)
- **Epoch 10+**: $\sigma=0.0$ (Fine-tune on real data)
- **Effect**: Prevents the model from getting "stuck" in the L2-optimal step-function local minimum.

### Strategy C: Frequency-Aware Loss
- Instead of blurring the image, calculate loss in the frequency domain or use a **Multiscale Structural Similarity (MS-SSIM)** loss.
- SSIM is inherently more sensitive to "structure" (like gradients) than L2.

## Conclusion & Recommendation
Applying blur to the training targets is a solid strategy to combat the "averaging" behavior of L2 loss. It forces the network to find the continuous function that underlies the quantized steps without being distracted by high-frequency sensor noise.

**Recommended Action**: Implement a `target_blur` parameter in the training pipeline or as an on-the-fly augmentation in the dataset loader.
