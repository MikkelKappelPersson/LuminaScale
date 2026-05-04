This specification outlines a **"Lean I/O, Heavy Compute"** architecture designed for multi-GPU HPC environments. By moving all mathematical operations (CDL, OCIO, Quantization) to the GPU, we bypass the CPU/Memory bottleneck, allowing the training to run at the speed of your PCIe bus and VRAM.

---

## Technical Specification: GPU-Native Synthetic Data Pipeline

### 1\. Phase 1: Minimalist CPU I/O (The Funnel)

The CPU’s only job is to fetch bytes and hand them off. No normalization, no casting, and no color math.

-   **Storage:** LMDB (Read-only, No-lock mode).
    
-   **Decoding:** Use `numpy.frombuffer` to map raw bytes directly to a NumPy array.
    
-   **Tensor Creation:** Convert to `torch.Tensor` while keeping the original data type (e.g., `float32` or `float16`).
    
-   **Dataloader Settings:** \* `num_workers`: Low to medium (e.g., 4-8 per GPU) to keep CPU cycles free for the OS and network fabric.
    
    -   `pin_memory=True`: Mandatory. This enables the use of **Direct Memory Access (DMA)** for faster transfers.
        

### 2\. Phase 2: The Transfer Bridge

We utilize asynchronous transfers to ensure the GPU never waits for the next batch.

-   **Non-Blocking Transfer:** Use `batch.to(device, non_blocking=True)`. This allows the CPU to start preparing the next batch while the current one is still moving over the PCIe bus.
    
-   **Precision Management:** If memory bandwidth is a concern, store ACES as `float16` in LMDB and cast to `float32` immediately upon landing on the GPU.
    

### 3\. Phase 3: GPU-Native Generation (The Engine)

Once the ACES reference is in VRAM, the generation happens in parallel across thousands of CUDA cores.

#### A. Random CDL Generation

Generate a tensor of SOP (Slope, Offset, Power) and Saturation values directly on the device:

-   **Formula:**
    
    $$Output = (Input \\times Slope + Offset)^{Power}$$
    
-   **Saturation:** Apply using a luma-weighted matrix based on your target primaries (Rec.709 or ACES).
    

#### B. Color Space Conversion (OCIO)

Call your GPU-accelerated OCIO kernels.

-   **Input:** CDL-graded ACES (Linear).
    
-   **Output:** Linear sRGB (32-bit float).
    

#### C. Bit-Depth Expansion (BDE) Prep

The 32-bit sRGB is branched to create the "degraded" input:

1.  **OETF:** Apply the sRGB transfer function (Gamma 2.2-ish curve).
    
2.  **Quantization:** \* Scale: $x \\times 255$
    
    -   Round/Clamp: `torch.clamp(torch.round(x), 0, 255)`
        
    -   Normalize: Divide by 255.0 to return to $\[0, 1\]$ float space.
        
3.  **Dithering (Optional):** Add random noise before quantization if your BDE net needs to learn de-dithering.
    

---

## 4\. Implementation Logic (PyTorch)

Python

```
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleACESDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        
    def __getitem__(self, index):
        # 1. Fetch raw bytes from LMDB (CPU)
        with self.env.begin(write=False) as txn:
            buf = txn.get(f"{index}".encode())
            
        # 2. Minimalist conversion (CPU)
        # Assuming data was saved as (C, H, W) float32
        arr = np.frombuffer(buf, dtype=np.float32).reshape(3, 1024, 1024)
        return torch.from_numpy(arr) # Return as-is

class GpuGenerator(torch.nn.Module):
    def __init__(self, ocio_renderer):
        super().__init__()
        self.ocio = ocio_renderer

    def forward(self, aces_batch):
        # All of this happens at lightning speed on VRAM
        params = self.get_random_cdl_params(aces_batch.size(0))
        
        # Apply CDL and OCIO
        graded_aces = self.apply_cdl(aces_batch, params)
        srgb_32bit_linear = self.ocio.to_srgb_linear(graded_aces)
        
        # Create 8-bit sRGB with OETF (The BDE Input)
        srgb_8bit = self.apply_srgb_oetf(srgb_32bit_linear)
        srgb_8bit = (srgb_8bit * 255).round().clamp(0, 255) / 255.0
        
        return {
            "input_8bit": srgb_8bit,        # BDE Input
            "ref_32bit_srgb": srgb_32bit_linear, # BDE Target
            "ref_aces": aces_batch          # Color Project Target
        }
```

---

## 5\. HPC Performance Guidelines

-   **Multi-Node Scaling:** Use `DistributedDataParallel` (DDP). Since each GPU has its own copy of the `GpuGenerator`, the only data traveling across the network is the ACES reference and the model gradients.
    
-   **LMDB Map Size:** Set `map_size` to at least the size of the database. On HPC, use `readahead=False` to prevent the OS from trying to cache the entire DB into RAM, which can crash nodes with limited memory.
    
-   **Validation:** Keep a small subset of "pre-baked" 8-bit images on disk. Use these to validate your on-the-fly generation logic during the first epoch to ensure your GPU quantization perfectly matches your ground truth expectations.
    

### Why this works:

By the time the CPU finishes fetching the next 32-bit image, the GPU has already processed the previous batch's CDL, OCIO, and Quantization multiple times over. The "Zero-Math" CPU approach ensures your high-end HPC GPUs are never waiting for the CPU to finish a `clip()` or `pow()` operation.
