This Markdown file summarizes the strategy for your bit-expansion project on an HPC. It focuses on using **LMDB** to store your paired **ACES (EXR)** and **sRGB (PNG)** data to eliminate GPU idle time.

---

# HPC Data Pipeline: Bit-Expansion Project

## 1. The Problem: I/O Bottleneck
* **Current State:** 6,000 EXRs (~100MB each) and 6,000 PNGs.
* **The Bottleneck:** CPU decompression takes **3 seconds per pair** (2s for EXR, 1s for PNG).
* **The Symptom:** GPUs sit idle while the CPU struggles to "unpack" ZIP/PIZ compression.
* **The HPC Factor:** Multi-GPU nodes amplify this; 8 GPUs require data faster than any CPU can decompress it.

---

## 2. The Solution: LMDB (Lightning Memory-Mapped Database)
Instead of 12,000 individual compressed files, we store everything in **one uncompressed binary database**.



### Why LMDB for this project?
* **Zero Decompression:** Data is stored as raw bytes. The CPU does zero math to load an image.
* **Concurrent Reads:** Designed for multiple processes (Multi-GPU) to read the same file without locking.
* **Resolution Flexibility:** Unlike a raw `.bin` memmap, LMDB handles different image resolutions easily.
* **HPC Friendly:** Uses only **one Inode**, preventing metadata strain on Lustre/GPFS filesystems.

---

## 3. Data Storage Estimate (Raw/Uncompressed)
Since we are bypassing compression to save CPU cycles, disk usage will increase.

| Resolution | EXR (Float32) | sRGB (Uint8) | Total per Pair | Total (6k images) |
| :--- | :--- | :--- | :--- | :--- |
| **4K** (3840x2160) | ~99.5 MB | ~24.9 MB | **~125 MB** | **~750 GB** |
| **8K** (7680x4320) | ~398 MB | ~99.5 MB | **~498 MB** | **~2.9 TB** |

> **Note:** Your 1TB allocation is sufficient for the entire 4K dataset, or a ~2,000 image subset of the 8K dataset.

---

## 4. Implementation Strategy

### Step A: The "Packer" Script
Run this once as a CPU job to create the database. It converts:
1.  **ACES (EXR)** $\rightarrow$ Raw `float32` array.
2.  **sRGB (PNG)** $\rightarrow$ Raw `uint8` array.
3.  **Store** $\rightarrow$ Both arrays as a single "blob" under one key.

### Step B: The HPC Job (`sbatch`)
1.  **Stage to Local SSD:** Copy the `.mdb` file from network storage to the node's local `/scratch` or `$TMPDIR`.
2.  **Train:** Use a PyTorch `Dataset` that opens the local LMDB.
3.  **Result:** GPU utilization should jump to **90-100%**.

---

## 5. Comparison of Approaches

| Method | Accuracy | GPU Speed | CPU Load | Disk Space |
| :--- | :--- | :--- | :--- | :--- |
| **Individual Files (Current)** | 100% | **Slow** | Very High | ~600 GB |
| **Pre-convert to .pt** | 100% | Fast | Low | ~1.5 TB+ |
| **LMDB (Raw Binary)** | **100%** | **Instant** | **Zero** | **~0.8 - 3 TB** |

---

### Next Step
**Would you like me to provide the Python code for the "Packer" script to create this LMDB from your two folders?**