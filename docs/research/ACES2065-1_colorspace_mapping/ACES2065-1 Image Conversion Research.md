# **Technical Framework for Arbitrary Color Space Mapping to ACES2065-1 Using Deep Neural Architectures**

The evolution of digital imaging and motion picture production has necessitated a shift from display-referred standards, such as sRGB and Rec.709, toward scene-referred encoding systems that preserve the full radiometric intent of a captured environment. The Academy Color Encoding System (ACES) represents the pinnacle of this movement, providing a standardized framework for color management that encompasses acquisition, grading, and long-term archival.1 Within this ecosystem, ACES2065-1 (utilizing the AP0 primary set) serves as the primary interchange and archival format, theoretically capable of representing all colors visible to the human eye in a linear, high-dynamic-range representation.1  
The transition from a standard 8-bit image to a 32-bit ACES2065-1 file is a dual-stage computational challenge. While dequantization restores the bit-depth and reduces quantization errors, the secondary stage—colorspace fitting—is far more complex. This process involves inverting the non-linear transformations, gamut clipping, and artistic tone mapping applied during the creation of standard dynamic range (SDR) images.4 To achieve state-of-the-art (SOTA) performance in this domain, a model must be capable of inferring the original scene-linear values from a compressed signal where much of the high-frequency radiometric information has been discarded or reshaped.

## **Foundations of the ACES Color System and Chromatic Standards**

A profound understanding of the ACES architecture is required to develop a fitting model that adheres to professional standards. ACES is not a single color space but a collection of transforms and spaces designed for specific pipeline stages. The target for this research, ACES2065-1, uses AP0 primaries which are wider than the physically realizable spectral locus to ensure no color information is lost during interchange.1  
In contrast, ACEScg utilizes the AP1 primary set, which is more tightly aligned with the colors found in natural scenes and modern digital cinema cameras, making it the preferred space for rendering and compositing.2 When training a machine learning model, the choice of internal representation is critical. While the final output must be linear ACES2065-1, the high dynamic range of linear light—often spanning several orders of magnitude—can lead to vanishing gradients and numerical instability during backpropagation.7 Consequently, SOTA models often perform calculations in logarithmic variants like ACEScc or ACEScct, which utilize the AP1 primaries but normalize the signal into a range more conducive to neural network optimization.3

| Color Space | Primary Set | Transfer Function | Typical Application |
| :---- | :---- | :---- | :---- |
| ACES2065-1 | AP0 | Linear | Interchange, Archival, Master Storage |
| ACEScg | AP1 | Linear | 3D Rendering, VFX Compositing |
| ACEScc | AP1 | Logarithmic | Technical Color Grading |
| ACEScct | AP1 | Logarithmic (Toe) | Artistic Grading (Film-like Feel) |
| sRGB | Rec.709 | Gamma 2.2 | Web, Standard Display Output |
| Rec.2100 PQ | BT.2020 | ST.2084 (Perceptual) | High Dynamic Range (HDR) Display |

The mapping of an "arbitrary" color space to ACES requires the inversion of the "image formation" pipeline. Standard images are display-referred, meaning their values are meaningful only in relation to a specific display's characteristics.1 A scene-referred ACES image, however, describes the light in the original scene. The model must therefore account for the white balance, the specific gamut mapping algorithm used by the source camera or software, and the tone mapping curve that compressed the scene's dynamic range into the 0-1 range of a standard file.4

## **State-of-the-Art Architectures in Color Space Mapping**

Current research into color space mapping and SDR-to-HDR conversion has moved beyond simple global matrices toward hybrid architectures that combine the efficiency of Lookup Tables (LUTs) with the expressive power of deep learning. The primary goal is to achieve content-adaptive transformations that can handle varying lighting conditions and source gamuts without human intervention.

### **Image-Adaptive 3D Lookup Tables**

The traditional approach to color correction relies on 3D LUTs, which map an input RGB triplet to an output RGB triplet via trilinear interpolation within a fixed lattice. While efficient, static LUTs cannot adapt to the specific needs of an individual image.9 The pioneering work by Zeng et al. introduced Image-Adaptive 3D LUTs, where a lightweight Convolutional Neural Network (CNN) analyzes a downsampled version of the input image to predict weights for a set of basis LUTs.9 These weights are used to fuse the basis LUTs into a single, image-specific transformation.  
This paradigm has been extended through mechanisms like AdaInt (Adaptive Intervals). Standard 3D LUTs utilize a uniform grid, which may over-sample areas of the gamut with little color information while under-sampling critical regions like skin tones or subtle gradients.10 AdaInt allows the network to predict non-uniform sampling intervals, concentrating the LUT's precision where it is most needed for the specific input.11

### **Spatially-Aware and Collaborative Transformations**

A significant limitation of 3D LUTs is their spatial invariance. Since a LUT operates only on the color value of a pixel, it cannot account for local context, such as shadow recovery in high-contrast scenes or the removal of compression artifacts.13 To address this, SOTA models like the Collaborative Transformations Framework (CoTF) utilize a dual-head approach.11  
The first head applies a global, image-adaptive 3D LUT to handle the heavy lifting of the colorspace and gamut shift. The second head is a pixel-wise transformation—typically a lightweight UNet or a multi-scale Laplacian pyramid—that acts as a local compensator.11 This secondary head refines the output of the LUT, restoring local textures and correcting spatially-varying exposure errors that a global map cannot reach.11

| Architectural Component | Function | SOTA Example | Performance Benefit |
| :---- | :---- | :---- | :---- |
| Global Fitting Head | Gamut and Tone Mapping | Adaptive 3D LUT | $O(1)$ complexity, color stability |
| Local Compensation Head | Texture and Detail Recovery | Laplacian Pyramid / UNet | Resolution-independent detail |
| Weight Predictor | Content-Aware Selection | SFT (Spatial-Frequency Transformer) | Global scene understanding |
| Grid Generator | Precision Optimization | AdaInt | Perceptual uniformity in gamut |

The LLF-LUT++ architecture represents a further refinement, integrating global and local operators through a closed-form Laplacian pyramid decomposition.14 This allows the model to process 4K resolution images in approximately 13.5 ms on a single GPU, making it suitable for real-time video applications while achieving significant gains in Peak Signal-to-Noise Ratio (PSNR) over traditional methods.14

## **Dataset Analysis and Training Methodologies**

The availability of 16,000 correctly converted ACES2065-1 images from the MIT-Adobe FiveK and PPR10K datasets provides a massive competitive advantage for training SOTA models. MIT-Adobe FiveK is renowned for its diverse scenes and professional retouches (Expert C is the common gold standard), while PPR10K offers a deep focus on human skin tones and professional portrait photography.16

### **The Many-to-One Training Strategy**

A critical challenge in "arbitrary" colorspace fitting is that the model does not know the source metadata (e.g., whether the input was sRGB, Rec.709, or a proprietary camera log). To build a model that is invariant to these factors, research suggests a "Many-to-One" training objective.18  
In this approach, a single ground-truth scene-referred anchor (the ACES EXR) is paired with multiple diverse sRGB "renditions" of the same scene. These renditions are generated by programmatically varying the virtual ISP parameters, such as white balance, contrast gains, and tone-curve shapes.18 By training on this heterogeneous set of inputs, the neural network learns to "look through" the photo-finishing effects to find the underlying radiometric truth of the scene.18

### **Data Augmentation and Pre-processing**

Given that the target is 32-bit linear light, data augmentation must be handled with care. Standard augmentations like random cropping and flipping are standard 19, but color-specific augmentations are more vital. Rebalancing strategies are often used to ensure the model performs equally well on "tail" classes, such as extreme low-light or ultra-saturated colors that are rare in standard datasets.20  
The input images are typically downsampled to a low resolution (e.g., $256 \\times 256$ or $512 \\times 512$) for the global weight predictor, while the local head processes patches or full-resolution data in a residual fashion.11 This allows the model to capture global tonal characteristics without the computational burden of passing high-resolution images through a heavy transformer or deep CNN.

## **Optimization Objectives and Loss Functions**

Traditional loss functions like Mean Squared Error (MSE) in RGB space are often inadequate for high-dynamic-range color science. In a linear ACES space, a difference of 0.1 in a bright highlight might be invisible, while a difference of 0.001 in a dark shadow could represent a massive shift in visibility.

### **Perceptual Uniformity and Delta E ITP**

For wide-gamut and HDR training, $\\Delta E\_{ITP}$ has emerged as the SOTA metric, surpassing the older CIEDE2000 standard. $\\Delta E\_{ITP}$ is based on the ICtCp color space, which was designed by Dolby Laboratories specifically for HDR and wide color gamut (WCG) technology.21 Unlike CIELAB-based metrics, $\\Delta E\_{ITP}$ provides excellent perceptual uniformity across three orders of magnitude of luminance ($0.1$ to $1000$ cd/m²).21  
The ICtCp space represents intensity (I), Tritan (Ct \- blue/yellow), and Protan (Cp \- red/green). The $\\Delta E\_{ITP}$ formula incorporates a scalar (typically 720\) to equate a result of 1.0 to a single Just Noticeable Difference (JND).21 Utilizing $\\Delta E\_{ITP}$ as a loss function ensures that the model optimizes for the errors that the human eye is most likely to perceive, rather than raw mathematical variance.

### **The Role of HDR-VDP-3 in Evaluation**

For the most exhaustive evaluation of a fitting model, the High-Dynamic-Range Visual-Difference-Predictor version 3 (HDR-VDP-3) is the industry benchmark. It is a multi-metric that simulates the optical and retinal pathways of the human eye, including glare, senile miosis, and contrast sensitivity.22  
HDR-VDP-3 offers several diagnostic "heads":

1. **Quality Head:** Predicts perceived quality degradation in Just-Objectionable-Differences (JOD). A degradation of 1 JOD implies that 75% of observers would notice a difference.23  
2. **Visibility Head:** Generates a probability map indicating which parts of the image contain visible artifacts.23  
3. **Contrast Distortion Head:** Specifically measures if the local contrast has been preserved, over-enhanced, or lost—a critical check for any tone mapping or colorspace fitting algorithm.23

| Metric | Domain | Best For | Threshold |
| :---- | :---- | :---- | :---- |
| PSNR | Signal | General Fidelity | High is better |
| $\\Delta E\_{ITP}$ | Perceptual | Color Accuracy (HDR) | \< 1.0 (Invisble) |
| HDR-VDP-3 (Quality) | Vision Science | Overall "Naturalness" | Higher JOD is better |
| HDR-VDP-3 (Visibility) | Vision Science | Artifact Detection | Lower probability is better |

## **Image Output vs. LUT Output: A Comparative Analysis**

The user's consideration of whether the model should output an image or a LUT is a central trade-off in ML-based color science. Both approaches have distinct advantages depending on the target application (e.g., real-time video playback vs. high-quality VFX archival).

### **Advantages and Constraints of Image Output**

Directly outputting a 32-bit EXR image allows the neural network to perform spatially-variant adjustments. This is essential for:

* **Dequantization Refinement:** While the user has a separate dequantization head, the fitting head can refine the local gradients to prevent banding in the high-dynamic-range space.12  
* **Artifact Suppression:** Deep CNNs can learn to distinguish between genuine scene detail and compression artifacts (halos, blockiness) that should not be expanded into the HDR space.24  
* **Highlight Reconstruction:** Spatially-aware models can "hallucinate" plausible textures in clipped regions by analyzing the surrounding context.12

However, the computational cost is high. Generating a full 4K frame directly through a neural network can take hundreds of milliseconds, even on high-end hardware, which precludes its use in live broadcast or real-time editing.14

### **Advantages and Constraints of LUT Output**

Predicting a LUT (or weights for a LUT bank) is the fastest possible approach. The neural network only needs to process a small, downsampled version of the image to determine the correct transformation.9 Once the LUT is generated, it can be applied to the high-resolution source with $O(1)$ complexity using standard GPU texture hardware.12  
The primary disadvantage is the lack of spatial awareness. A LUT applies the same transformation to a pixel regardless of whether it is in the sky or the deep shadows of a forest. This makes it impossible for a pure LUT-based model to correct local exposure errors or restore fine details.12

### **The SOTA Compromise: Neural LUTs and Spatially-Aware Fusing**

Modern solutions seek to bridge this gap. The "Neural LUT" approach compresses hundreds of high-quality professional LUTs into a tiny neural representation (0.25 MB) that can be reconstructed on the fly.25 This provides the efficiency of a LUT with the variety of a content-adaptive system.  
Furthermore, "Spatially-Aware LUTs" use a per-pixel weighting map. Instead of one LUT for the whole image, the model predicts a global LUT and a local "weight map" that determines how strongly that LUT should be applied to different regions.13 This allows for a degree of local adjustment without the full overhead of an image-generating CNN.

## **Advanced Architectural Features: Transformers and Flow Matching**

To push a model to SOTA levels on the PPR10K and FiveK benchmarks, several advanced ML components should be considered.

### **Spatial-Frequency Transformer (SFT) Weight Predictors**

In many-to-one fitting tasks, the model must understand the "intent" of the source image. Is it a night scene that was brightened in the camera, or a day scene that was darkened? Standard CNNs have a limited receptive field and may struggle with this global context. Transformers, through their self-attention mechanism, are far better at capturing long-range dependencies.14  
A Spatial-Frequency Transformer (SFT) analyzes the image in both the spatial domain (to see objects and lighting) and the frequency domain (to see textures and noise). This allows it to predict more accurate weights for the basis 3D LUTs, ensuring that the global colorspace shift is tailored to the specific scene content.14

### **Gradient-Preserving Flow Matching**

For the local restoration head, "Flow Matching" has emerged as a powerful alternative to traditional GANs or diffusion models. Flow-based methods model the transformation between the SDR distribution and the target ACES distribution as a continuous, invertible flow.12 This is particularly effective for image restoration because it is inherently stable and can handle complex, spatially-varying degradations that static LUTs cannot touch.12 FlowLUT, for example, integrates the efficiency of LUTs with iterative flow matching to restore fine structural details that are typically lost during the SDR-to-HDR conversion process.12

## **Practical Integration: FFMPEG and OpenColorIO Workflows**

For professional implementation, the model must integrate with existing industry tools. The OpenColorIO (OCIO) library is the standard for color management in VFX and animation. SOTA models should ideally export their global transformation as an OCIO-compatible LUT or a series of matrices and curves.1  
FFMPEG is often used for the batch processing of these assets. While FFMPEG is not traditionally suited for 32-bit floating-point color space conversions using its internal filters, it can be combined with OpenImageIO (oiiotools) for more accurate AP0-to-display piping.27 A model that can output its results directly into a .cube or .spimtx file allows professional colorists to audit the results in software like DaVinci Resolve or Nuke before final delivery.4

## **Benchmark Performance and Comparative Results**

Recent results on the MIT-Adobe FiveK and PPR10K datasets demonstrate the superiority of adaptive, spatially-aware methods. While PSNR is the primary metric in academic challenges like NTIRE, perceptual metrics like $\\Delta E\_{ab}$ and $\\Delta E\_{00}$ are closely watched for colorimetric accuracy.17

| Model | Dataset | PSNR (dB) ↑ | ΔEab​ ↓ | Complexity |
| :---- | :---- | :---- | :---- | :---- |
| HDRNet | PPR10K (Exp C) | 24.08 | 8.87 | Medium |
| 3D LUT (Adaptive) | PPR10K (Exp C) | 25.18 | 7.58 | Low |
| SepLUT | PPR10K (Exp C) | 25.59 | 7.51 | Low |
| AdaInt | PPR10K (Exp C) | 25.68 | 7.31 | Low |
| NamedCurves | PPR10K (Exp C) | 26.81\* | 6.48\* | High |
| LLF-LUT++ | MIT-Adobe FiveK | 28.50+ | \- | Medium |

\*NamedCurves uses color naming decomposition and Bezier curve estimation to simulate local editing.17  
These results indicate that the "Adaptive 3D LUT" approach provides a strong baseline, but incorporating spatial information (as in NamedCurves or LLF-LUT++) is necessary to achieve the final 1-2 dB of performance required for SOTA status.

## **Narrative Synthesis and Implementation Roadmap**

The objective of transforming an 8-bit image into a 32-bit ACES2065-1 master is a quest for radiometric truth. The research suggests that the "two-head" solution proposed by the user is precisely the right architectural path, provided the second head (fitting) is sufficiently sophisticated.

### **Phase 1: Global Chromatic Fitting**

The foundation of the model should be an image-adaptive LUT head. Using the 16,000 ACES images, the model should be trained to predict the weights for 3-5 basis 3D LUTs. This head handles the "fitting" from the arbitrary input gamut to the AP0 primaries. By utilizing a Spatial-Frequency Transformer (SFT) for the weight predictor, the model can make informed decisions based on the global scene context.14 To ensure the best performance in the HDR range, the training must be conducted in ACEScct and optimized using a $\\Delta E\_{ITP}$ loss function.6

### **Phase 2: Local Radiometric Compensation**

The secondary head should focus on "Local Tone Tuning." Drawing from the CoTF and LLF-LUT++ architectures, this head should operate as a residual module.10 It analyzes the output of the global LUT and the original dequantized 32-bit input to add or subtract light at the pixel level. This is where the model "un-does" local tone mapping decisions and reconstructs clipped highlights. A Laplacian pyramid approach is recommended here, as it allows for extremely efficient processing while preserving the sharp edge details critical for 4K and 8K masters.14

### **Phase 3: Many-to-One Robustness Training**

To make the model truly "metadata-agnostic," the training phase must incorporate the "Many-to-One" strategy.18 By synthetically generating hundreds of thousands of SDR variants for each of the 16,000 ground-truth images, the model becomes robust to different camera brands, software exports, and artistic "looks." This ensures that when an "arbitrary" image is input, the model can reliably find its way back to the ACES2065-1 reference.18

## **Conclusions and Technical Recommendations**

Building a SOTA model for ACES2065-1 conversion requires a meticulous balance of colorimetric accuracy and spatial restoration. The research highlights several inescapable conclusions for a professional-grade implementation.  
First, the output of the fitting head should be a hybrid: a predicted image-adaptive LUT for the global transform, combined with a high-resolution residual image for local refinement. This allows for real-time performance in production environments while maintaining the ability to produce high-fidelity archival masters.  
Second, the choice of loss functions must move beyond the standard ML toolkit. $\\Delta E\_{ITP}$ is non-negotiable for HDR accuracy, and HDR-VDP-3 must be used as the final arbiter of quality to ensure that the reconstructed images feel "natural" and lack visible artifacts.  
Third, the training pipeline should leverage the unique properties of the ACES ecosystem. Training in ACEScct provides the necessary numerical stability, while targeting ACES2065-1 ensures the longevity and interoperability of the results. By integrating these SOTA components—Adaptive LUTs, SFT weight predictors, Laplacian pyramid local heads, and Many-to-One robustness training—it is possible to create a model that not only fits arbitrary color spaces to the ACES standard but does so with the precision and nuance expected by professional colorists and cinematographers.

#### **Citerede værker**

1. Chapter 1.5: Academy Color Encoding System (ACES) \- Chris Brejon, tilgået april 22, 2026, [https://chrisbrejon.com/cg-cinematography/chapter-1-5-academy-color-encoding-system-aces/](https://chrisbrejon.com/cg-cinematography/chapter-1-5-academy-color-encoding-system-aces/)  
2. Motion Graphics \- Prolost, tilgået april 22, 2026, [https://prolost.com/blog/tag/Motion+Graphics](https://prolost.com/blog/tag/Motion+Graphics)  
3. Color Image Processing | PDF \- Scribd, tilgået april 22, 2026, [https://www.scribd.com/document/395744278/Color-Image-Processing](https://www.scribd.com/document/395744278/Color-Image-Processing)  
4. Custom ACES SDR to HDR Part 3 \- Dolby Vision Trims \- Mixing Light, tilgået april 22, 2026, [https://mixinglight.com/color-grading-tutorials/sdr-to-hdr-part-3-dolby-vision/](https://mixinglight.com/color-grading-tutorials/sdr-to-hdr-part-3-dolby-vision/)  
5. AgX vs ACES : r/vfx \- Reddit, tilgået april 22, 2026, [https://www.reddit.com/r/vfx/comments/16ue10g/agx\_vs\_aces/](https://www.reddit.com/r/vfx/comments/16ue10g/agx_vs_aces/)  
6. Fusion \- Blackmagic Design, tilgået april 22, 2026, [https://documents.blackmagicdesign.com/UserManuals/Fusion17\_Manual.pdf](https://documents.blackmagicdesign.com/UserManuals/Fusion17_Manual.pdf)  
7. ACES tonemapping curve SDR vs HDR : r/colorists \- Reddit, tilgået april 22, 2026, [https://www.reddit.com/r/colorists/comments/1cmd5ut/aces\_tonemapping\_curve\_sdr\_vs\_hdr/](https://www.reddit.com/r/colorists/comments/1cmd5ut/aces_tonemapping_curve_sdr_vs_hdr/)  
8. Rookie \- Setup ACES in Resolve with HDR Display \- Post (DI, Edit, Mastering) \- Community, tilgået april 22, 2026, [https://community.acescentral.com/t/rookie-setup-aces-in-resolve-with-hdr-display/3814](https://community.acescentral.com/t/rookie-setup-aces-in-resolve-with-hdr-display/3814)  
9. \[2303.09170\] NLUT: Neural-based 3D Lookup Tables for Video ..., tilgået april 22, 2026, [https://ar5iv.labs.arxiv.org/html/2303.09170](https://ar5iv.labs.arxiv.org/html/2303.09170)  
10. Learning Differential Pyramid Representation for Tone Mapping \- OpenReview, tilgået april 22, 2026, [https://openreview.net/pdf/baabf5e5c6709bfa3f12e3d7f19903edada786e4.pdf](https://openreview.net/pdf/baabf5e5c6709bfa3f12e3d7f19903edada786e4.pdf)  
11. CVPR Poster Real-Time Exposure Correction via Collaborative ..., tilgået april 22, 2026, [https://cvpr.thecvf.com/virtual/2024/poster/31065](https://cvpr.thecvf.com/virtual/2024/poster/31065)  
12. FlowLUT: Efficient Image Enhancement via Differentiable LUTs and Iterative Flow Matching, tilgået april 22, 2026, [https://arxiv.org/html/2509.23608v1](https://arxiv.org/html/2509.23608v1)  
13. Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2508.16121v1](https://arxiv.org/html/2508.16121v1)  
14. High-resolution Photo Enhancement in Real-time: A Laplacian Pyramid Network \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2510.11613v1](https://arxiv.org/html/2510.11613v1)  
15. High-Resolution Photo Enhancement in Real-Time: A Laplacian Pyramid Network, tilgået april 22, 2026, [https://www.computer.org/csdl/journal/tp/2026/03/11204685/2aPD0bSYDbq](https://www.computer.org/csdl/journal/tp/2026/03/11204685/2aPD0bSYDbq)  
16. Contrast-dependent saturation adjustment for outdoor image enhancement | Request PDF \- ResearchGate, tilgået april 22, 2026, [https://www.researchgate.net/publication/311320845\_Contrast-dependent\_saturation\_adjustment\_for\_outdoor\_image\_enhancement](https://www.researchgate.net/publication/311320845_Contrast-dependent_saturation_adjustment_for_outdoor_image_enhancement)  
17. NamedCurves: Learned Image Enhancement via Color Naming \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2407.09892v1](https://arxiv.org/html/2407.09892v1)  
18. RawGen: Learning Camera Raw Image Generation \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2604.00093v1](https://arxiv.org/html/2604.00093v1)  
19. How to Handle Images of Different Sizes in a Convolutional Neural Network \- Wandb, tilgået april 22, 2026, [https://wandb.ai/ayush-thakur/dl-question-bank/reports/How-to-Handle-Images-of-Different-Sizes-in-a-Convolutional-Neural-Network--VmlldzoyMDk3NzQ](https://wandb.ai/ayush-thakur/dl-question-bank/reports/How-to-Handle-Images-of-Different-Sizes-in-a-Convolutional-Neural-Network--VmlldzoyMDk3NzQ)  
20. A Systematic Review on Long-Tailed Learning \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2408.00483v1](https://arxiv.org/html/2408.00483v1)  
21. Color Distance and Delta E \- ColorAide Documentation, tilgået april 22, 2026, [https://facelessuser.github.io/coloraide/distance/](https://facelessuser.github.io/coloraide/distance/)  
22. (PDF) HDR-VDP-3: A multi-metric for predicting image differences, quality and contrast distortions in high dynamic range and regular content \- ResearchGate, tilgået april 22, 2026, [https://www.researchgate.net/publication/370295932\_HDR-VDP-3\_A\_multi-metric\_for\_predicting\_image\_differences\_quality\_and\_contrast\_distortions\_in\_high\_dynamic\_range\_and\_regular\_content](https://www.researchgate.net/publication/370295932_HDR-VDP-3_A_multi-metric_for_predicting_image_differences_quality_and_contrast_distortions_in_high_dynamic_range_and_regular_content)  
23. HDR-VDP-3: A multi-metric for predicting image differences ... \- arXiv, tilgået april 22, 2026, [https://arxiv.org/abs/2304.13625](https://arxiv.org/abs/2304.13625)  
24. Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement \- arXiv, tilgået april 22, 2026, [https://arxiv.org/html/2507.12135v1](https://arxiv.org/html/2507.12135v1)  
25. Efficient Neural Network Encoding for 3D Color Lookup Tables, tilgået april 22, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/33059/35214](https://ojs.aaai.org/index.php/AAAI/article/view/33059/35214)  
26. Experimental results on MIT-Adobe FiveK \[6\] dataset. \- ResearchGate, tilgået april 22, 2026, [https://www.researchgate.net/figure/Experimental-results-on-MIT-Adobe-FiveK-6-dataset\_tbl1\_360960708](https://www.researchgate.net/figure/Experimental-results-on-MIT-Adobe-FiveK-6-dataset_tbl1_360960708)  
27. FFMPEG / ACES workflow for SDR and HDR deliverables \- Community \- ACESCentral, tilgået april 22, 2026, [https://community.acescentral.com/t/ffmpeg-aces-workflow-for-sdr-and-hdr-deliverables/1498](https://community.acescentral.com/t/ffmpeg-aces-workflow-for-sdr-and-hdr-deliverables/1498)  
28. FFmpeg Filters Documentation, tilgået april 22, 2026, [https://ffmpeg.org/ffmpeg-filters.html](https://ffmpeg.org/ffmpeg-filters.html)