# .Project Plan: AI-Restorative Color & Bit-Depth Model

## 1\. Project Objective

Build a Machine Learning model capable of processing 8-bit, “arbitrarily stylized,” or AI-generated images and transforming them into a clean, 16-bit-equivalent **ACES2065-1** color space. The model must simultaneously solve for **structural quantization (banding)** and **global color normalization**.

## 2\. Model Architecture: The “Sequential Cascade” Strategy

Instead of a single-shot black box, the model uses a two-stage sequential approach to isolate structural repair from mathematical color mapping.

### Stage 0: Pre-Processing (Non-ML)

-   **Action:** Cast the input 8-bit integer image ($\[0, 255\]$) directly to **32-bit Float** ($\[0.0, 1.0\]$).
    
-   **Purpose:** Provides the “mathematical headroom” necessary for interpolation without rounding errors.
    

### Stage 1: Structural Head (BDE - Bit Depth Expansion)

-   **Architecture:** Spatial-aware network: UNet with skip connections \[\[
    
-   **Operation:** Analyzes local pixel neighborhoods to identify “stair-step” quantization patterns.
    
-   **Job:** Interpolates missing values to create smooth gradients while preserving edges.
    
-   **Output:** A “clean” 16-bit-equivalent image still in the source’s arbitrary/stylized color state.
    

### Stage 2: Global Head (Color Space Normalization)

-   **Architecture:** Global Context Encoder (Downsampled input) + $1 \\times 1$ Convolution / MLP.
    
-   **Operation:** Identifies the global “intent” of the image (e.g., “This is a landscape with extreme teal-shift”).
    
-   **Job:** Maps the “cleaned” pixels into the ACES2065-1gamut and linear gamma.
    
-   **Integration:** Uses a **Global Skip Connection** to ensure fine texture from Stage 1 is maintained during the color shift.
    

## 3\. Dataset Strategy: Procedural Degradation

The model will be trained on “Perfect” 16-bit ACES2065-1 data that has been synthetically “damaged” to simulate AI-generated and stylized 8-bit inputs.

### The “Ground Truth” (Target)

-   **Source:** High-quality 16-bit/32-bit floating point images.
    
-   **Format:** Linear ACES2065-1
    

### The “Input Generation” (Synthetic Damage)

To teach the model to handle “unknown” and “stylized” inputs, each training sample is created by applying the following random functions:

1.  **Arbitrary Color Warping:** \* Apply a random $3 \\times 3$ matrix with coefficients between $-0.5$ and $2.5$ to simulate extreme saturation and hue twists.
    
2.  **Stylized Contrast Curves:** \* Apply random **Sigmoid** or **Power** functions to simulate “crushed” blacks and “blown out” AI highlights.
    
3.  **AI Artifact Simulation:**
    
    -   Add low-frequency **Chroma Noise** (color blotches) to mimic AI diffusion errors.
        
    -   Apply a random **Unsharp Mask** to simulate the over-sharpening common in AI generators.
        
4.  **The 8-bit Break:**
    
    -   **Quantization:** Round values to 8-bit integers without dithering. This “bakes in” the banding that the BDE head must learn to solve.

## 4\. Loss Functions & Training Logic

To ensure the BDE head doesn’t just “blur” the image and the Color head doesn’t “hallucinate” colors, a compound loss is used:

-   **Pixel Loss (**$L\_{MSE}$**):** Accuracy check against the 16-bit ACES target.
    
-   **Gradient Loss (**$L\_{Grad}$**):** Penalizes sharp steps in smooth areas (the “Anti-Banding” enforcer).
    
-   **Perceptual Loss (**$L\_{VGG}$**):** Compares feature maps to ensure the image maintains “natural” textures and doesn’t look “plastic.”
    
-   **Style Loss (Gram Matrix):** Specifically helps the model ignore the “stylization” of the input and focus on the underlying structure.
    

## 5\. Implementation Roadmap

1.  **Data Gen Script:** Build the Python/PyTorch pipeline to generate “Damaged” 8-bit inputs from 16-bit sources on-the-fly.
    
2.  **Base Training:** Train the BDE head alone on a “Same-Color-Space” de-banding task.
    
3.  **Full Cascade Training:** Connect the Color head and train end-to-end with the ACES2065-1  target.
    
4.  **Refinement:** Fine-tune using only “AI-Generated” samples to sharpen the model’s ability to handle specific diffusion artifacts.