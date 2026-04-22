# **Advanced Computational Strategies for Neural Image Dequantization and 32-Bit Radiance Reconstruction**

The transition from standard 8-bit integer image representations to high-fidelity 32-bit floating-point radiance maps represents a fundamental shift in the landscape of computational photography and computer vision. Historically, the digital imaging pipeline has been constrained by the limitations of consumer hardware, which necessitated the compression of high-dynamic-range scene information into the 256 discrete levels afforded by 8-bit encoding.1 While this bit depth is sufficient for basic perceptual tasks under standard viewing conditions, it introduces irreversible information loss through quantization, manifesting as visually disruptive artifacts like color banding, false contouring, and the loss of fine texture in extreme highlight or shadow regions.4  
The challenge of dequantization—restoring the missing precision and dynamic range of a scene-linear 32-bit signal from an 8-bit source—is inherently ill-posed. Multiple high-precision values are collapsed into a single integer during quantization, requiring a model to not only interpolate missing values but also "hallucinate" information that was never recorded.1 Achieving state-of-the-art results in this domain requires a synthesis of deep generative modeling, physically grounded prior knowledge, and sophisticated training regimes that align the reconstructed signals with both human perceptual systems and the underlying physics of light.9

## **Theoretical Framework of Discretization and Quantization Artifacts**

Quantization is the process of mapping a large set of input values to a smaller, discrete set. In the context of 8-bit imaging, this involves partitioning the intensity range into 256 bins. The mathematical degradation model for an 8-bit image $I\_{LBD}$ generated from a 32-bit high-bit-depth source $I\_{HBD}$ can be expressed as:

$$I\_{LBD}(x) \= \\mathcal{Q}\_8(\\Gamma\_{CRF}(I\_{HBD}(x) \\cdot \\Delta t) \+ \\eta)$$  
Here, $\\Delta t$ is the exposure time, $\\Gamma\_{CRF}$ represents the non-linear camera response function (CRF) that mimics the aesthetic of specific sensors, $\\eta$ is the additive sensor noise, and $\\mathcal{Q}\_8$ is the 8-bit quantization operator.13 The information lost during the application of $\\mathcal{Q}\_8$ is known as quantization noise. Unlike Gaussian noise, quantization noise is signal-dependent and highly structured, often appearing as distinct "staircase" steps in regions of smooth gradients, such as clear skies or sunset transitions.4  
The human visual system is disproportionately sensitive to these discrete steps due to the Mach band effect, a lateral inhibition mechanism in the retina that enhances the perceived contrast at the boundaries of different intensity levels.6 This physiological phenomenon makes even small quantization errors visually objectionable, especially on modern displays that support high-dynamic-range (HDR) and wide-color-gamut (WCG) standards.2 Consequently, a dequantization model must reconstruct a continuous intensity manifold that eliminates these false contours while maintaining the integrity of the original 8-bit base data.2

| Representation | Data Type | Value Range | Dynamic Range stops | Applications |
| :---- | :---- | :---- | :---- | :---- |
| Standard (SDR) | 8-bit Integer | 0 to 255 | \~5-6 | Web content, legacy TV |
| Deep Color | 10-bit Integer | 0 to 1023 | \~10-12 | HDR10, modern gaming |
| High Precision | 16-bit Integer | 0 to 65,535 | \~14-16 | RAW processing, TIFF |
| Scene Linear | 32-bit Float | $-\\infty$ to $+\\infty$ | Infinite (Theoretical) | Physical light modeling |

The shift from 8-bit to 32-bit float is not merely an increase in precision but a transition from a non-linear, perceptually encoded domain to a linear radiance domain where intensity values are proportional to the number of photons hitting the sensor.3 This linear space is critical for accurate light transport modeling, compositing, and professional post-production workflows.1

## **Architectural Paradigms for Neural Dequantization**

The modeling of dequantization has evolved from simple linear interpolation to complex deep learning architectures capable of understanding scene context.

### **Convolutional and Residual Architectures**

Early attempts at bit-depth expansion utilized deep convolutional neural networks (CNNs) to learn a direct mapping between 8-bit and higher-precision images. Models such as BE-CNN and the subsequent BE-CALF (Bit-Depth Enhancement by Concatenating All Level Features) focused on residual learning, where the network predicts the difference between the low-bit input and the high-bit ground truth.2 These models established the importance of multi-scale feature extraction, as dequantization requires both local information to preserve edges and global information to smooth out large-scale banding artifacts.16  
A refined approach emerged with the Bit Restoration Network (BRNet), which introduced the concept of learning a weighting map.2 Instead of generating high-precision pixels from scratch, BRNet predicts a per-pixel weight $\\alpha$ that modulates the expansion within a "rational range" defined by the 8-bit input. This ensures that the reconstructed 32-bit values are constrained to the interval $\[V\_8, V\_8 \+ 1)$ in the integer-scaled space, preventing the model from over-modifying the established signal and leading to more stable training.2

### **Transformer-Based Global Modeling**

The inherent limitation of CNNs is their local receptive field, which makes it difficult to distinguish between legitimate high-frequency textures and low-frequency quantization bands.15 Vision Transformers (ViTs) and their derivatives address this by employing self-attention mechanisms that can model long-range dependencies across the entire image.24  
The Multiscale Recurrent Fusion Transformer (MRFT) and the Arbitrary Bitwise Coefficient (ABCD) network represent the current state-of-the-art in transformer-based dequantization.16 These models use hierarchical attention to "borrow" information from similar, well-defined textures in other parts of the image to guide the reconstruction of heavily quantized regions. For example, if one part of a sky gradient is sharp and another is banded, the transformer can align the features of the sharp region to inform the de-banding of the artifact-ridden area.16

### **Wavelet-State Space Models (WaveMamba)**

An emerging trend in 2024 and 2025 is the use of frequency-domain modeling, specifically the Wavelet State Space Model (WaveMamba).15 This approach decomposes the image using a Discrete Wavelet Transform (DWT) into low-frequency and high-frequency sub-bands. Since banding artifacts primarily reside in the low-frequency smooth areas, the model can apply aggressive dequantization to these regions while using state-space modeling (inspired by Mamba architectures) to maintain the linear complexity and high-frequency structural integrity of edges and textures.15 This dual-domain processing allows for a more nuanced balance between noise suppression and detail preservation than is possible in raw pixel space.15

## **Generative Modeling and Radiance Hallucination**

While CNNs and Transformers excel at signal restoration, they often fail to recover information in clipped regions, such as blown-out highlights where the 8-bit value is 255 but the actual radiance could be hundreds of times higher.9 This requires generative models capable of hallucinating plausible textures and radiance levels based on learned priors of the natural world.9

### **The Role of Generative Adversarial Networks**

GAN-based models, such as DAGAN (Deep Attentive GAN) and TSGAN (Two-Stream GAN), introduce a discriminator that evaluates the perceptual realism of the reconstructed image.2 By training the generator to fool the discriminator, the model learns to produce sharp textures and natural gradients that avoid the "plasticky" or overly smooth look of pure MSE-based reconstruction.2 However, GANs are notoriously difficult to train, often suffering from mode collapse where they produce a limited variety of outputs, or introducing hallucinatory artifacts that do not align with the physical light of the scene.30

### **Latent Diffusion Models (LDMs) and Gain Maps**

The most sophisticated approach as of 2025 involves Latent Diffusion Models (LDMs). These models define a process of gradually adding noise to data and then learning to reverse that noise.9 When applied to dequantization, LDMs provide superior perceptual priors, allowing the model to reconstruct highly complex textures and specular highlights that are statistically consistent with real-world HDR data.9  
A critical breakthrough in this area is the "Gain Map" paradigm used in models like GMODiff.10 Directly mapping 8-bit integers to 32-bit floats is difficult because of the massive disparity in scale and the sparse distribution of radiance values. Instead, the model is trained to predict an 8-bit (or higher) "Gain Map" $G$ that acts as a multiplier for the original 8-bit LDR image $I\_{LDR}$:

$$I\_{HDR} \= I\_{LDR} \\cdot \\exp(\\log(\\text{Gain}\_{max}) \\cdot G)$$  
This approach decomposes the problem into a reliable base component (the input) and a learned refinement component (the gain map).9 By reformulating dequantization as gain map estimation, models can leverage pre-trained 8-bit LDMs (like Stable Diffusion) without needing to redesign the latent space or retrain the Variational Autoencoder (VAE).9

| Model | Architecture | Innovation | Best Use Case |
| :---- | :---- | :---- | :---- |
| **BRNet** | CNN | Learning weighting maps rather than pixels | Stable bit-depth expansion |
| **MRFT** | Transformer | Recurrent fusion of multi-scale tokens | Global consistency, high res |
| **WaveMamba** | SSM \+ DWT | Frequency-aware state space modeling | Efficient de-banding |
| **GMODiff** | LDM | One-step gain map refinement | Perceptual HDR reconstruction |
| **LumaFlux** | DiT | Physically-guided adaptive diffusion | Universal inverse tone mapping |

## **Physically-Guided Diffusion Transformers: LumaFlux**

The current benchmark for 8-bit to 32-bit dequantization and inverse tone mapping is the LumaFlux model.11 LumaFlux adapts a large-scale Diffusion Transformer (DiT) backbone to perform universal dequantization across varied styles and camera pipelines.11 It introduces three synergistic modules that move beyond pure statistical learning into the realm of physical light modeling:

1. **Physically-Guided Adaptation (PGA)**: This module injects physical descriptors—including luminance, spatial gradients, and spectral frequency cues—into the transformer's attention mechanism via low-rank residuals.11 By making the attention blocks aware of the luminance distribution, the model can adjust its processing for deep shadows versus bright highlights.11  
2. **Perceptual Cross-Modulation (PCM)**: Utilizing conditioning from frozen vision encoders like SigLIP, the PCM module stabilizes chroma and texture.11 This prevents the "semantic drift" common in pure generative models, where the dequantized output might change the identity of an object or its color saturation incorrectly.11  
3. **Rational-Quadratic Spline (RQS) Tone-Field Decoder**: Instead of a traditional convolutional decoder that might introduce new artifacts, LumaFlux uses an RQS decoder to reconstruct smooth, interpretable radiance fields.12 This provides the final 32-bit floating-point precision with mathematical guarantees of smoothness in gradient regions.12

LumaFlux also emphasizes latent manifold alignment through LogC3 encoding.42 By mapping unbounded 32-bit radiance into a range that matches the VAE's native 8-bit SDR manifold via logarithmic curves, the model preserves highlight fidelity that would otherwise be clamped or distorted in the latent space.42

## **Training Data Engineering and Synthesis**

The efficacy of a dequantization model is heavily dependent on the quality and diversity of its training data. Because paired 8-bit and 32-bit images are not naturally occurring in large quantities, sophisticated synthesis pipelines are required.44

### **Synthetic Data Generation Pipelines**

A robust training pipeline must simulate the entire camera ISP (Image Signal Processor) in reverse. High-bit-depth images, often from the Adobe MIT5K dataset or professional RAW libraries, serve as the ground truth.45 The synthesis of the corresponding 8-bit inputs involves several critical steps:

* **Exposure Variation**: Each 32-bit image is sampled at multiple virtual exposure levels ($\\text{EV} \\in \\{-4, 0, \+4\\}$) to ensure the model learns to reconstruct both overexposed and underexposed areas.20  
* **Noise Injection**: Physically plausible noise, including photon shot noise and electronic readout noise, is added to the signal before quantization.49 This forces the model to learn the difference between sensor noise and quantization steps.15  
* **Dithering**: Adding a small amount of random noise (e.g., uniform or Gaussian) before the 8-bit rounding operation simulates the stochastic nature of real-world quantization and prevents the model from overfitting to "perfectly" truncated integer steps.6  
* **CRF Application**: The linear radiance is passed through various non-linear camera response functions to simulate the aesthetic profiles of different camera brands (Canon, Nikon, Sony), ensuring the model generalizes across devices.13

### **Key Datasets for Radiance Restoration**

| Dataset | Type | resolution | Key Feature | Primary Application |
| :---- | :---- | :---- | :---- | :---- |
| **MIT5K** | Real (RAW) | \~12MP | High bit-depth DSLR images | Base ground truth |
| **GTA-HDR** | Synthetic | 512x512 | 40k pairs from game engine | Large-scale training |
| **LIVE-TMHDR** | Real (Video) | 4K | Expert-tone-mapped sequences | Temporal consistency |
| **RealRaw-HDR** | Real (RAW) | Various | Actual sensor quantization data | Noise/BDE modeling |
| **Sintel** | Synthetic | Various | Computer-generated sequences | Test/Benchmarking |

The GTA-HDR dataset is particularly notable for providing a "clean" 32-bit radiance ground truth that is impossible to capture with physical sensors due to optical lens flare and sensor blooming.28 By using synthetic data from high-end rendering engines, models can learn the theoretical limit of radiance reconstruction.47

## **Mathematical Foundations of the Training Objective**

Training an 8-bit to 32-bit model using standard loss functions like Mean Squared Error (MSE) is problematic. In the radiance domain, intensity values can span six orders of magnitude (from $10^{-3}$ to $10^{3}$ $cd/m^2$). A loss function applied in linear space will be dominated by bright pixels, causing the model to completely ignore details in the shadows.49

### **Perceptual Radiance Compression ($\\mu$-law)**

To balance the gradient contributions from all luminance levels, SOTA models compute the loss in a perceptually compressed domain.8 The most common approach is the $\\mu$-law compression:

$$L \= \\left\\| \\frac{\\log(1 \+ \\mu \\hat{I}\_{HDR})}{\\log(1 \+ \\mu)} \- \\frac{\\log(1 \+ \\mu I\_{HDR})}{\\log(1 \+ \\mu)} \\right\\|\_1$$  
where $\\mu$ is typically set between 255 and 5000\.8 This logarithmic scaling ensures that the model pays equal attention to a 0.01% change in shadow intensity and a 0.01% change in highlight intensity.56

### **Multi-Term Loss Functions**

State-of-the-art dequantization models employ a weighted combination of multiple loss terms to address different aspects of the reconstruction 37:

1. **Reconstruction Loss ($L\_{rec}$)**: Usually $L\_1$ or Huber loss in the $\\mu$-law or log-domain to ensure numerical accuracy.8  
2. **Perceptual Loss ($L\_{perc}$)**: Based on deep feature differences extracted from a pre-trained VGG19 or SigLIP network.37 This term is crucial for restoring textures that are perceptually important but numerically small.61  
3. **Adversarial Loss ($L\_{adv}$)**: A GAN-based loss term that penalizes the model for producing outputs that lack the high-frequency statistical characteristics of real 32-bit images.22  
4. **Total Variation Loss ($L\_{TV}$)**: Encourages piecewise smoothness to mitigate residual banding artifacts and high-frequency noise.55  
5. **Reliability Mask Loss**: In models like GMODiff, a binary cross-entropy loss is used to train a segmentation head that predicts which regions of the 8-bit input are "unreliable" (e.g., saturated), allowing the model to focus its generative power on those specific areas.37

## **Practical Implementation and Numerical Stability**

Creating a 32-bit dequantization model in frameworks like PyTorch requires rigorous attention to numerical stability and data types.65

### **Precision and Mixed-Precision Training**

While the input is 8-bit (typically normalized to $$), all internal model weights and intermediate feature maps should be maintained in 32-bit floating point (torch.float32) to prevent rounding errors that would re-introduce banding.65 However, the use of bfloat16 (Brain Floating Point) is often preferred over float16 for mixed-precision training.66 bfloat16 has the same dynamic range as float32 (8-bit exponent), which is essential for handling the large range of values in HDR radiance, whereas standard float16 (5-bit exponent) frequently leads to overflow and "NaN" gradients during radiance reconstruction.66

### **The OpenEXR Pipeline**

The standard file format for 32-bit floating-point images is OpenEXR.3 In a PyTorch-based training loop, the OpenEXR Python module should be used to load and save data to ensure that no intermediate clamping occurs.71 Unlike PNG or JPEG, which clamp values to 0-255 or 0-1, OpenEXR stores true linear light values that can exceed 1.0 (for highlights) and can even be negative (for certain color primaries).3

Python

\# Optimal PyTorch Radiance Reconstruction Output Handling  
import torch  
import OpenEXR  
import Imath

def tensor\_to\_exr(tensor, filename):  
    \# Ensure tensor is (C, H, W) and in float32  
    img \= tensor.detach().cpu().permute(1, 2, 0).numpy()  
    header \= OpenEXR.Header(img.shape, img.shape)  
    half\_chan \= Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))  
    header\['channels'\] \= dict()  
    exr \= OpenEXR.OutputFile(filename, header)  
    exr.writePixels({'R': img\[:,:,0\].tobytes(),   
                     'G': img\[:,:,1\].tobytes(),   
                     'B': img\[:,:,2\].tobytes()})  
    exr.close()

### **Quantization-Aware Optimization**

To improve the model's robustness to different 8-bit compression levels, "fake quantization" layers can be inserted into the training pipeline.72 During the forward pass, these layers simulate the rounding and clipping of the input, while during the backward pass, the Straight-Through Estimator (STE) is used to bypass the zero-gradient problem of the rounding operator, allowing the model to adapt its weights to the specific noise patterns of quantization.74

## **Benchmarking and Evaluation Metrics**

The evaluation of 8-bit to 32-bit dequantization cannot rely on standard LDR metrics like PSNR or SSIM. These metrics treat all pixels equally and do not account for the non-linear human perception of brightness or the extreme range of HDR values.76

### **HDR-Specific Perceptual Metrics**

1. **HDR-VDP-3 (High Dynamic Range Visual Difference Predictor)**: This is the most comprehensive metric for radiance restoration.76 It incorporates a complete model of the early human visual system, including optical scattering in the eye, non-linear photoreceptor response, and local luminance adaptation.16 It outputs a probability-of-detection map for artifacts and a Just Objectionable Difference (JOD) score, which correlates highly with human subjective quality ratings.11  
2. **PU-PSNR and PU-SSIM**: These metrics apply a Perceptually Uniform (PU) transformation—designed to mimic the human eye's luminance response across its entire functional range (from $10^{-5}$ to $10^{10}$ $cd/m^2$)—to both the reconstructed and ground truth radiance images before calculating standard scores.9 This prevents the score from being biased toward highlight errors.76  
3. **$\\Delta E\_{ITP}$**: Standardized in Rec. 2100, this metric measures color differences specifically for high-luminance, wide-color-gamut signals.11 It is more sensitive to color shifts in the dequantized output than standard Lab-based color metrics.11  
4. **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures the perceptual distance in a deep feature space.37 While not HDR-specific, LPIPS is exceptionally good at identifying whether the model has successfully restored natural textures or if it has introduced generative "hallucinations" that look artificial.37

### **Comparative Performance Benchmarks**

| Method | PSNR (PU) | HDR-VDP-3 (JOD) | Inference Time | Perceptual Quality |
| :---- | :---- | :---- | :---- | :---- |
| **Deep SR-ITM** | 26.59 | 6.92 | \~0.05s | Low (Blurry highlights) |
| **ICTCPNet** | 33.12 | 8.90 | \~0.08s | Medium (Some banding) |
| **LEDiff** | 32.25 | 5.32 | \~40.0s | High (Detailed) |
| **GMODiff** | 34.80 | 9.20 | \~0.89s | SOTA (Precise) |
| **LumaFlux** | 35.34 | 9.45 | \~0.15s | SOTA (Universal) |

Recent results from the AIM 2025 and NTIRE 2024 challenges indicate that hybrid approaches—using a fast regression model to handle the base dequantization and a one-step diffusion model for texture refinement—provide the best trade-off between numerical fidelity and perceptual realism.10

## **Hardware Acceleration and Edge Deployment**

While 32-bit floating-point precision is necessary for professional workflows, the computational cost of dequantization can be prohibitive for mobile devices. Recent research, such as the QuartDepth framework, focuses on optimizing these models for Application-Specific Integrated Circuits (ASICs) and mobile NPUs.79

### **Activation Polishing and Compensation**

To deploy a dequantization model on resource-limited hardware, it is often necessary to quantize the *model's* weights and activations to 4-bit or 8-bit precision.67 This creates a paradox: a quantized model performing dequantization.79 To mitigate the performance drop, techniques like LogNP activation polishing are used to transform the skewed, outlier-heavy distribution of depth and radiance features into a more quantization-friendly, normalized distribution.79

### **Kernel Fusion and Instruction Optimization**

For real-time 32-bit dequantization in camera viewfinders, kernel fusion is employed.74 By combining the non-linear CRF inversion, the radiance reconstruction, and the final tone mapping into a single computational pass, developers can significantly reduce memory bandwidth bottlenecks.74 This is particularly effective on modern NVIDIA GPUs using TensorRT, which can fuse layers to maximize the throughput of 8-bit Tensor Cores while maintaining a 32-bit accumulator for final radiance output.74

## **Conclusion and Future Outlook**

The development of a machine learning model for dequantization from 8-bit to 32-bit images has matured from simple image processing to a complex synthesis of physical radiance modeling and generative artificial intelligence. The current state-of-the-art approach favors a multi-stage architecture: a high-capacity regression network (often a Transformer or SSM) to handle the global de-banding and CRF inversion, followed by a physically-guided one-step diffusion stage to hallucinate specular highlights and fine textures.9  
The "best" approach for a new model implementation involves:

1. **Architecture**: A Diffusion Transformer (DiT) backbone, such as the one used in LumaFlux, which allows for global context and parameter-efficient adaptation through LoRA.11  
2. **Training Paradigm**: A Gain Map-driven approach that anchors the model to the original LDR data while predicting the radiance expansion ratio.9  
3. **Data Strategy**: A combination of real-world RAW data (MIT5K) and large-scale synthetic radiance maps (GTA-HDR) to provide a diverse range of exposure and noise profiles.45  
4. **Loss Metrics**: Perceptual optimization in the compressed $\\mu$-law domain, evaluated using HDR-specific metrics like HDR-VDP-3 to ensure alignment with human vision.8

As displays continue to push the boundaries of peak brightness and color saturation, the need for these neural radiance restoration models will only grow. Future research is likely to focus on temporal coherence for video dequantization and zero-shot generalization to handle the myriad camera response functions and compression artifacts found in wild user-generated content.15 By treating dequantization not as a discrete problem but as a continuous radiance manifold estimation task, neural networks can effectively bridge the gap between 8-bit legacy content and the 32-bit future of digital imaging.

#### **Citerede værker**

1. Bit Depth — Siril 1.5.0 documentation, tilgået april 22, 2026, [https://siril.readthedocs.io/en/latest/file-formats/Bit-depth.html](https://siril.readthedocs.io/en/latest/file-formats/Bit-depth.html)  
2. Learning Weighting Map for Bit-Depth Expansion within a Rational Range \- arXiv, tilgået april 22, 2026, [https://arxiv.org/pdf/2204.12039](https://arxiv.org/pdf/2204.12039)  
3. High Dynamic Range (HDR) Images FAQ, tilgået april 22, 2026, [https://www.hdrsoft.com/resources/dri.html](https://www.hdrsoft.com/resources/dri.html)  
4. tilgået april 22, 2026, [https://arxiv.org/pdf/2110.08569\#:\~:text=Performance%20evaluation%20shows%20that%20deep,methods%20both%20quantitatively%20and%20visually.\&text=Banding%20or%20false%20contour%20artifacts,and%20are%20caused%20by%20quantization.](https://arxiv.org/pdf/2110.08569#:~:text=Performance%20evaluation%20shows%20that%20deep,methods%20both%20quantitatively%20and%20visually.&text=Banding%20or%20false%20contour%20artifacts,and%20are%20caused%20by%20quantization.)  
5. Deep Image Debanding, tilgået april 22, 2026, [https://arxiv.org/abs/2110.08569](https://arxiv.org/abs/2110.08569)  
6. Adaptive Debanding Filter \- Laboratory for Image and Video Engineering, tilgået april 22, 2026, [https://www.live.ece.utexas.edu/publications/2020/SPL2020\_AdaDeband.pdf](https://www.live.ece.utexas.edu/publications/2020/SPL2020_AdaDeband.pdf)  
7. Image rendering bit depth \- pIXELsHAM, tilgået april 22, 2026, [https://www.pixelsham.com/2023/05/29/image-rendering-bit-depth/](https://www.pixelsham.com/2023/05/29/image-rendering-bit-depth/)  
8. Single-Image HDR Reconstruction With Task-Specific Network Based on Channel Adaptive RDN \- CVF Open Access, tilgået april 22, 2026, [https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Chen\_Single-Image\_HDR\_Reconstruction\_With\_Task-Specific\_Network\_Based\_on\_Channel\_Adaptive\_CVPRW\_2021\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Chen_Single-Image_HDR_Reconstruction_With_Task-Specific_Network_Based_on_Channel_Adaptive_CVPRW_2021_paper.pdf)  
9. GMODiff: One-Step Gain Map Refinement with Diffusion Priors for HDR Reconstruction, tilgået april 22, 2026, [https://arxiv.org/html/2512.16357v1](https://arxiv.org/html/2512.16357v1)  
10. GMODiff: One-Step Gain Map Refinement with Diffusion Priors for Efficient HDR Reconstruction \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2512.16357v2](https://arxiv.org/html/2512.16357v2)  
11. LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers, tilgået april 22, 2026, [https://arxiv.org/html/2604.02787v1](https://arxiv.org/html/2604.02787v1)  
12. \[2604.02787\] LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers \- arXiv, tilgået april 22, 2026, [https://arxiv.org/abs/2604.02787](https://arxiv.org/abs/2604.02787)  
13. How To Cheat With Metrics in Single-Image HDR Reconstruction \- CVF Open Access, tilgået april 22, 2026, [https://openaccess.thecvf.com/content/ICCV2021W/LCI/papers/Eilertsen\_How\_To\_Cheat\_With\_Metrics\_in\_Single-Image\_HDR\_Reconstruction\_ICCVW\_2021\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021W/LCI/papers/Eilertsen_How_To_Cheat_With_Metrics_in_Single-Image_HDR_Reconstruction_ICCVW_2021_paper.pdf)  
14. How to cheat with metrics in single-image HDR reconstruction \- Department of Computer Science and Technology |, tilgået april 22, 2026, [https://www.cl.cam.ac.uk/\~rkm38/pdfs/eilertsen2021\_si\_hdr\_quality.pdf](https://www.cl.cam.ac.uk/~rkm38/pdfs/eilertsen2021_si_hdr_quality.pdf)  
15. Guiding WaveMamba with Frequency Maps for Image Debanding This work has been supported by the UKRI MyWorld Strength in Places Programme (SIPF00006/1). \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2508.11331v1](https://arxiv.org/html/2508.11331v1)  
16. Latent Space Embedding for Bit-Depth Enhancement: Synergize With Super-Resolution \- IEEE Xplore, tilgået april 22, 2026, [https://ieeexplore.ieee.org/iel8/76/11475579/11223713.pdf](https://ieeexplore.ieee.org/iel8/76/11475579/11223713.pdf)  
17. Learning Weighting Map for Bit-Depth Expansion within a Rational Range \- GitHub, tilgået april 22, 2026, [https://github.com/yuqing-liu-dut/bit-depth-expansion](https://github.com/yuqing-liu-dut/bit-depth-expansion)  
18. Bit Depth Considerations \- Tech/Engineering \- Community \- ACESCentral, tilgået april 22, 2026, [https://community.acescentral.com/t/bit-depth-considerations/4751](https://community.acescentral.com/t/bit-depth-considerations/4751)  
19. Converting 8bit to 16bit or 32bit images out of Comfyui : r/StableDiffusion \- Reddit, tilgået april 22, 2026, [https://www.reddit.com/r/StableDiffusion/comments/18bf08j/converting\_8bit\_to\_16bit\_or\_32bit\_images\_out\_of/](https://www.reddit.com/r/StableDiffusion/comments/18bf08j/converting_8bit_to_16bit_or_32bit_images_out_of/)  
20. HDR Cookbook – Creating 32-bit HDRs the Right Way, tilgået april 22, 2026, [https://farbspiel.wordpress.com/2011/03/27/hdr-cookbook-creating-32-bit-hdrs-the-right-way/](https://farbspiel.wordpress.com/2011/03/27/hdr-cookbook-creating-32-bit-hdrs-the-right-way/)  
21. GitHub \- TJUMMG/BDE: Bit Depth Enhancement, tilgået april 22, 2026, [https://github.com/TJUMMG/BDE](https://github.com/TJUMMG/BDE)  
22. Loss Functions for Image Restoration with Neural Networks \- Research at NVIDIA, tilgået april 22, 2026, [https://research.nvidia.com/sites/default/files/pubs/2017-03\_Loss-Functions-for/NN\_ImgProc.pdf](https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf)  
23. Beyond Illumination: Fine-Grained Detail Preservation in Extreme Dark Image Restoration, tilgået april 22, 2026, [https://arxiv.org/html/2508.03336v1](https://arxiv.org/html/2508.03336v1)  
24. Understanding what the machines see: State-of-the-art computer vision at CVPR 2024, tilgået april 22, 2026, [https://www.qualcomm.com/news/onq/2024/06/understanding-what-the-machines-see-state-of-the-art-computer-vision-at-cvpr-2024](https://www.qualcomm.com/news/onq/2024/06/understanding-what-the-machines-see-state-of-the-art-computer-vision-at-cvpr-2024)  
25. NTIRE 2024 Challenge on Image Super-Resolution (×4): Methods and Results \- CVF Open Access, tilgået april 22, 2026, [https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen\_NTIRE\_2024\_Challenge\_on\_Image\_Super-Resolution\_x4\_Methods\_and\_Results\_CVPRW\_2024\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen_NTIRE_2024_Challenge_on_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2024_paper.pdf)  
26. NTIRE 2024 Challenge on Light Field Image Super-Resolution: Methods and Results \- CVF Open Access, tilgået april 22, 2026, [https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang\_NTIRE\_2024\_Challenge\_on\_Light\_Field\_Image\_Super-Resolution\_Methods\_and\_CVPRW\_2024\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf)  
27. liuzhen03/HDR-Transformer-PyTorch: The official PyTorch implementation of the ECCV 2022 paper: Ghost-free High Dynamic Range Imaging with Context-aware Transformer \- GitHub, tilgået april 22, 2026, [https://github.com/liuzhen03/HDR-Transformer-PyTorch](https://github.com/liuzhen03/HDR-Transformer-PyTorch)  
28. A Review Toward Deep Learning for High Dynamic Range Reconstruction \- MDPI, tilgået april 22, 2026, [https://www.mdpi.com/2076-3417/15/10/5339](https://www.mdpi.com/2076-3417/15/10/5339)  
29. AIM 2025 challenge on Inverse Tone Mapping Report: Methods and Results \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2508.13479v2](https://arxiv.org/html/2508.13479v2)  
30. HDR Reconstruction Boosting with Training-Free and Exposure-Consistent Diffusion \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2602.19706v1](https://arxiv.org/html/2602.19706v1)  
31. Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline | Request PDF \- ResearchGate, tilgået april 22, 2026, [https://www.researchgate.net/publication/343467657\_Single-Image\_HDR\_Reconstruction\_by\_Learning\_to\_Reverse\_the\_Camera\_Pipeline](https://www.researchgate.net/publication/343467657_Single-Image_HDR_Reconstruction_by_Learning_to_Reverse_the_Camera_Pipeline)  
32. GANs vs. Diffusion Models: In-Depth Comparison and Analysis \- Sapien, tilgået april 22, 2026, [https://www.sapien.io/blog/gans-vs-diffusion-models-a-comparative-analysis](https://www.sapien.io/blog/gans-vs-diffusion-models-a-comparative-analysis)  
33. Improved image reconstruction from brain activity through automatic image captioning \- PMC, tilgået april 22, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11811215/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11811215/)  
34. GANs vs. Diffusion Models: Putting AI to the test | Aurora Solar, tilgået april 22, 2026, [https://aurorasolar.com/blog/putting-ai-to-the-test-generative-adversarial-networks-vs-diffusion-models/](https://aurorasolar.com/blog/putting-ai-to-the-test-generative-adversarial-networks-vs-diffusion-models/)  
35. Synthetic Scientific Image Generation with VAE, GAN, and Diffusion Model Architectures \- PMC, tilgået april 22, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12387873/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12387873/)  
36. \[D\] What are the advantages of GANs over Diffusion Models in image generation? \- Reddit, tilgået april 22, 2026, [https://www.reddit.com/r/MachineLearning/comments/184j8c3/d\_what\_are\_the\_advantages\_of\_gans\_over\_diffusion/](https://www.reddit.com/r/MachineLearning/comments/184j8c3/d_what_are_the_advantages_of_gans_over_diffusion/)  
37. \[Literature Review\] GMODiff: One-Step Gain Map Refinement with Diffusion Priors for HDR Reconstruction \- Moonlight, tilgået april 22, 2026, [https://www.themoonlight.io/en/review/gmodiff-one-step-gain-map-refinement-with-diffusion-priors-for-hdr-reconstruction](https://www.themoonlight.io/en/review/gmodiff-one-step-gain-map-refinement-with-diffusion-priors-for-hdr-reconstruction)  
38. Decoding HDR Image Formats (I): Basic Concepts of Gainmap | JacksBlog, tilgået april 22, 2026, [https://jackchou00.com/en/posts/gainmap-image-intro/](https://jackchou00.com/en/posts/gainmap-image-intro/)  
39. Architectural Paradigms. (Left) Baseline Diffusion Transformer (DiT)... | Download Scientific Diagram \- ResearchGate, tilgået april 22, 2026, [https://www.researchgate.net/figure/Architectural-Paradigms-Left-Baseline-Diffusion-Transformer-DiT-architecture-where\_fig2\_403530124](https://www.researchgate.net/figure/Architectural-Paradigms-Left-Baseline-Diffusion-Transformer-DiT-architecture-where_fig2_403530124)  
40. Sigmoid Loss for Language Image Pre-Training \- ResearchGate, tilgået april 22, 2026, [https://www.researchgate.net/publication/377429802\_Sigmoid\_Loss\_for\_Language\_Image\_Pre-Training](https://www.researchgate.net/publication/377429802_Sigmoid_Loss_for_Language_Image_Pre-Training)  
41. LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers, tilgået april 22, 2026, [https://www.catalyzex.com/paper/lumaflux-lifting-8-bit-worlds-to-hdr-reality](https://www.catalyzex.com/paper/lumaflux-lifting-8-bit-worlds-to-hdr-reality)  
42. HDR Video Generation via Latent Alignment with Logarithmic Encoding \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2604.11788v1](https://arxiv.org/html/2604.11788v1)  
43. Paper page \- HDR Video Generation via Latent Alignment with Logarithmic Encoding, tilgået april 22, 2026, [https://huggingface.co/papers/2604.11788](https://huggingface.co/papers/2604.11788)  
44. arXiv:2503.20211v1 \[cs.CV\] 26 Mar 2025, tilgået april 22, 2026, [https://arxiv.org/pdf/2503.20211](https://arxiv.org/pdf/2503.20211)  
45. NTIRE 2025 Challenge on RAW Image Restoration and Super-Resolution \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2506.02197v1](https://arxiv.org/html/2506.02197v1)  
46. AIM 2025 challenge on Inverse Tone Mapping Report: Methods and Results | Request PDF, tilgået april 22, 2026, [https://www.researchgate.net/publication/394688233\_AIM\_2025\_challenge\_on\_Inverse\_Tone\_Mapping\_Report\_Methods\_and\_Results](https://www.researchgate.net/publication/394688233_AIM_2025_challenge_on_Inverse_Tone_Mapping_Report_Methods_and_Results)  
47. WACV 2025: Theory, Experiments, Dataset, and Code for our newly proposed LDR → HDR Deep Learning Dataset called GTA-HDR \- GitHub, tilgået april 22, 2026, [https://github.com/HrishavBakulBarua/GTA-HDR](https://github.com/HrishavBakulBarua/GTA-HDR)  
48. yurizzzzz/Bit-Depth\_Enhancement: Restore the low bit-depth images back to the high bit-depth images(Pytorch codes) \- GitHub, tilgået april 22, 2026, [https://github.com/yurizzzzz/Bit-Depth\_Enhancement](https://github.com/yurizzzzz/Bit-Depth_Enhancement)  
49. Training Neural Networks on RAW and HDR Images for Restoration Tasks \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2312.03640v2](https://arxiv.org/html/2312.03640v2)  
50. Learning to Generate Realistic Noisy Images via Pixel-level Noise-aware Adversarial Training, tilgået april 22, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2021/file/1a5b1e4daae265b790965a275b53ae50-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/1a5b1e4daae265b790965a275b53ae50-Paper.pdf)  
51. Rethinking Noise Synthesis and Modeling in Raw Denoising \- CVF Open Access, tilgået april 22, 2026, [https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang\_Rethinking\_Noise\_Synthesis\_and\_Modeling\_in\_Raw\_Denoising\_ICCV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Rethinking_Noise_Synthesis_and_Modeling_in_Raw_Denoising_ICCV_2021_paper.pdf)  
52. Efficient HDR Reconstruction from Real-World Raw Images \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2306.10311v6](https://arxiv.org/html/2306.10311v6)  
53. US11277543B1 \- Perceptual dithering for HDR video and images \- Google Patents, tilgået april 22, 2026, [https://patents.google.com/patent/US11277543B1/en](https://patents.google.com/patent/US11277543B1/en)  
54. Quantization & Dithering, tilgået april 22, 2026, [https://sites.google.com/view/ananyamukherjeehome/image-processing/quantization-dithering](https://sites.google.com/view/ananyamukherjeehome/image-processing/quantization-dithering)  
55. \[1511.08861\] Loss Functions for Image Restoration with Neural Networks \- ar5iv \- arXiv, tilgået april 22, 2026, [https://ar5iv.labs.arxiv.org/html/1511.08861](https://ar5iv.labs.arxiv.org/html/1511.08861)  
56. Mu-law algorithm \- Wikipedia, tilgået april 22, 2026, [https://en.wikipedia.org/wiki/Mu-law\_algorithm](https://en.wikipedia.org/wiki/Mu-law_algorithm)  
57. Reconstructing HDR Images using Non-Learning and Deep Learning Based Multi-exposure Image Synthesis Techniques \- Stanford University, tilgået april 22, 2026, [http://stanford.edu/class/ee367/Winter2022/report/zhang\_zhang\_report.pdf](http://stanford.edu/class/ee367/Winter2022/report/zhang_zhang_report.pdf)  
58. Deep-HdrReconstruction/loss.py at master \- GitHub, tilgået april 22, 2026, [https://github.com/marcelsan/Deep-HdrReconstruction/blob/master/loss.py](https://github.com/marcelsan/Deep-HdrReconstruction/blob/master/loss.py)  
59. Logarithmic Scaling of Loss Functions for Enhanced Self-Supervised Accelerated MRI Reconstruction \- PMC, tilgået april 22, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12691438/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12691438/)  
60. Image Correction via Deep Reciprocating HDR Transformation \- CVF Open Access, tilgået april 22, 2026, [https://openaccess.thecvf.com/content\_cvpr\_2018/papers/Yang\_Image\_Correction\_via\_CVPR\_2018\_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Image_Correction_via_CVPR_2018_paper.pdf)  
61. Guide to PyTorch Loss Functions \- Medium, tilgået april 22, 2026, [https://medium.com/biased-algorithms/guide-to-pytorch-loss-functions-90ab7ca85ec2](https://medium.com/biased-algorithms/guide-to-pytorch-loss-functions-90ab7ca85ec2)  
62. Loss Functions in Deep Learning: A Comprehensive Review \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2504.04242v1](https://arxiv.org/html/2504.04242v1)  
63. A Survey of Loss Functions in Deep Learning \- MDPI, tilgået april 22, 2026, [https://www.mdpi.com/2227-7390/13/15/2417](https://www.mdpi.com/2227-7390/13/15/2417)  
64. Impact of loss functions on the performance of a deep neural network designed to restore low-dose digital mammography \- PMC, tilgået april 22, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10267506/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10267506/)  
65. Understanding PyTorch — From Tensors to Training a Neural Network | by Vamsikd, tilgået april 22, 2026, [https://medium.com/@vamsikd219/understanding-pytorch-from-tensors-to-training-a-neural-network-652eb9844b86](https://medium.com/@vamsikd219/understanding-pytorch-from-tensors-to-training-a-neural-network-652eb9844b86)  
66. What Every User Should Know About Mixed Precision Training in PyTorch, tilgået april 22, 2026, [https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)  
67. Practical Quantization in PyTorch, tilgået april 22, 2026, [https://pytorch.org/blog/quantization-in-practice/](https://pytorch.org/blog/quantization-in-practice/)  
68. Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training \- NVIDIA Developer, tilgået april 22, 2026, [https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)  
69. Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks, tilgået april 22, 2026, [http://papers.neurips.cc/paper/8736-hybrid-8-bit-floating-point-hfp8-training-and-inference-for-deep-neural-networks.pdf](http://papers.neurips.cc/paper/8736-hybrid-8-bit-floating-point-hfp8-training-and-inference-for-deep-neural-networks.pdf)  
70. Chapter 26\. The OpenEXR Image File Format \- NVIDIA Developer, tilgået april 22, 2026, [https://developer.nvidia.com/gpugems/gpugems/part-iv-image-processing/chapter-26-openexr-image-file-format](https://developer.nvidia.com/gpugems/gpugems/part-iv-image-processing/chapter-26-openexr-image-file-format)  
71. The OpenEXR Python Module, tilgået april 22, 2026, [https://cary-ilm-openexr.readthedocs.io/en/stable/python.html](https://cary-ilm-openexr.readthedocs.io/en/stable/python.html)  
72. Quantization In PyTorch \- Meegle, tilgået april 22, 2026, [https://www.meegle.com/en\_us/topics/quantization/quantization-in-pytorch](https://www.meegle.com/en_us/topics/quantization/quantization-in-pytorch)  
73. Quantization-Aware Training (QAT): A step-by-step guide with PyTorch | Generative-AI, tilgået april 22, 2026, [https://wandb.ai/byyoung3/Generative-AI/reports/Quantization-Aware-Training-QAT-A-step-by-step-guide-with-PyTorch--VmlldzoxMTk2NTY2Mw](https://wandb.ai/byyoung3/Generative-AI/reports/Quantization-Aware-Training-QAT-A-step-by-step-guide-with-PyTorch--VmlldzoxMTk2NTY2Mw)  
74. Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT, tilgået april 22, 2026, [https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)  
75. How to Build Model Quantization \- OneUptime, tilgået april 22, 2026, [https://oneuptime.com/blog/post/2026-01-30-model-quantization/view](https://oneuptime.com/blog/post/2026-01-30-model-quantization/view)  
76. Perceptual Assessment and Optimization of HDR Image Rendering \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2310.12877v4](https://arxiv.org/html/2310.12877v4)  
77. Overview of High-Dynamic-Range Image Quality Assessment \- PMC \- NIH, tilgået april 22, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11508586/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11508586/)  
78. A Review of the Image Quality Metrics used in Image Generative Models \- Paperspace Blog, tilgået april 22, 2026, [https://blog.paperspace.com/review-metrics-image-synthesis-models/](https://blog.paperspace.com/review-metrics-image-synthesis-models/)  
79. QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge, tilgået april 22, 2026, [https://cvpr.thecvf.com/virtual/2025/poster/35224](https://cvpr.thecvf.com/virtual/2025/poster/35224)  
80. QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge \- CVPR 2025 Open Access Repository \- The Computer Vision Foundation, tilgået april 22, 2026, [https://openaccess.thecvf.com/content/CVPR2025/html/Shen\_QuartDepth\_Post-Training\_Quantization\_for\_Real-Time\_Depth\_Estimation\_on\_the\_Edge\_CVPR\_2025\_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Shen_QuartDepth_Post-Training_Quantization_for_Real-Time_Depth_Estimation_on_the_Edge_CVPR_2025_paper.html)  
81. QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2503.16709v1](https://arxiv.org/html/2503.16709v1)  
82. Clipping-Based Post Training 8-Bit Quantization of Convolution Neural Networks for Object Detection \- MDPI, tilgået april 22, 2026, [https://www.mdpi.com/2076-3417/12/23/12405](https://www.mdpi.com/2076-3417/12/23/12405)  
83. HDR Reconstruction Boosting with Training-Free and Exposure-Consistent Diffusion \- arXiv, tilgået april 22, 2026, [https://arxiv.org/abs/2602.19706](https://arxiv.org/abs/2602.19706)