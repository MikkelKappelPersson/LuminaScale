"""On-the-fly Dataset Generation with On-device GPU Processing.

Implements the "Lean I/O, Heavy Compute" architecture from the data pipeline spec.
- Phase 1: Minimalist CPU I/O from LMDB  
- Phase 2: Non-blocking GPU transfer
- Phase 3: GPU-native CDL generation + ACES to sRGB conversion

This is designed to be integrated into a PyTorch DataLoader for training datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple
import logging
import os
import pickle
import tempfile

import numpy as np
import torch
import lmdb
from tqdm import tqdm

# Add src to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Set OCIO environment
ocio_config_path = project_root / "config" / "aces" / "studio-config.ocio"
if ocio_config_path.exists():
    os.environ["OCIO"] = str(ocio_config_path)

from luminascale.utils.gpu_cdl_processor import GPUCDLProcessor
from luminascale.utils.gpu_torch_processor import GPUTorchProcessor
from luminascale.utils.look_generator import get_single_random_look
from luminascale.utils.io import aces_to_display_gpu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OnTheFlyACESDataset:
    """GPU-native on-the-fly dataset generation from LMDB ACES source.
    
    Phase 1 (CPU): Fetch raw ACES bytes from LMDB
    Phase 2 (GPU): Async transfer to VRAM
    Phase 3 (GPU): CDL grading + Display transform
    
    Usage in training loop:
        dataset = OnTheFlyACESDataset("path/to/aces.lmdb")
        for batch in dataset.iter_batches(batch_size=8):
            hdr_aces, ldr_srgb = batch  # Both on GPU, ready for training
    """

    def __init__(
        self,
        lmdb_aces_path: Path | str,
        batch_size: int = 8,
        cache_cdl_params: bool = True,
    ) -> None:
        """Initialize dataset.
        
        Args:
            lmdb_aces_path: Path to LMDB database containing ACES images.
            batch_size: Batch size for GPU processing.
            cache_cdl_params: If True, pre-generate random CDL params for all batches.
        """
        self.lmdb_path = Path(lmdb_aces_path)
        assert self.lmdb_path.exists(), f"LMDB not found: {self.lmdb_path}"

        self.batch_size = batch_size
        self.device = device
        
        # PyTorch ACES transformer (primary, with LUT for accuracy)
        from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer
        self.pytorch_transformer = ACESColorTransformer(device=device, use_lut=True)
        
        self.cdl_processor = GPUCDLProcessor(device=device)
        self.ocio_processor = GPUTorchProcessor(headless=True)

        # Open LMDB in read-only mode (no locking)
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
        )

        # Get dataset size
        with self.env.begin(write=False) as txn:
            self.size = txn.stat()["entries"]

        logger.info(f"Dataset initialized: {self.size} ACES images in LMDB")
        logger.info(f"Batch size: {batch_size}, Batches per epoch: {self.size // batch_size}")

        if cache_cdl_params:
            self._cache_random_cdl_params()

    def _cache_random_cdl_params(self) -> None:
        """Pre-generate random CDL params for faster iteration."""
        num_batches = (self.size // self.batch_size) + 1
        self.cdl_param_cache = [get_single_random_look() for _ in range(num_batches)]
        logger.info(f"Pre-cached {num_batches} random CDL parameter sets")

    def _load_aces_from_lmdb(self, index: int) -> np.ndarray:
        """Load single ACES image from LMDB (Phase 1: CPU I/O).
        
        Note: Now uses raw byte buffer reading instead of pickle.
        Layout: [Header (12B: H, W, C)][HDR Data][LDR Data]
        """
        with self.env.begin(write=False) as txn:
            if not hasattr(self, "_keys"):
                keys_buf = txn.get(b"__keys__")
                self._keys = pickle.loads(keys_buf)
            
            img_name = self._keys[index % len(self._keys)]
            buf = txn.get(img_name.encode("ascii"))
            if buf is None:
                raise KeyError(f"Key {img_name} not found in LMDB")

        # 1. Parse header (3 x int32 = 12 bytes)
        header = np.frombuffer(buf[:12], dtype=np.int32)
        H, W, C = int(header[0]), int(header[1]), int(header[2])
        
        # 2. Extract HDR (float32, 4 bytes/pixel)
        hdr_size = H * W * C * 4
        # We start after the header (index 12)
        hdr_arr = np.frombuffer(buf[12:12+hdr_size], dtype=np.float32).reshape(C, H, W)
        
        # Note: In on-the-fly generation, LDR is generated from HDR on GPU,
        # so we don't necessarily need to load the LDR bytes from the buffer 
        # unless it's used for validation or specific training modes.
        
        return hdr_arr

    def iter_batches(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Iterate over batches with async GPU transfer and on-device processing.
        
        Yields:
            (aces_batch: [B, H, W, 3] float32 on GPU,
             srgb_batch: [B, H, W, 3] float32 on GPU)
        """
        num_batches = self.size // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            batch_indices = range(start_idx, start_idx + self.batch_size)

            # Phase 1: CPU I/O - Load raw bytes
            aces_list = []
            for idx in batch_indices:
                try:
                    aces_arr = self._load_aces_from_lmdb(idx % self.size)
                    aces_list.append(torch.from_numpy(aces_arr.astype(np.float32)))
                except KeyError:
                    logger.warning(f"Failed to load index {idx}, skipping")
                    continue

            if not aces_list:
                continue

            # Phase 2: Non-blocking GPU transfer
            # Stack and transfer asynchronously
            # Shape: (B, C, H, W) -> permute to (B, H, W, C)
            aces_batch = torch.stack(aces_list, dim=0).to(
                self.device, non_blocking=True
            ).permute(0, 2, 3, 1)  # [B, H, W, 3]

            # Phase 3: GPU-native processing
            # Apply random CDL on GPU (per-image processing)
            cdl_params = self.cdl_param_cache[batch_idx % len(self.cdl_param_cache)]
            srgb_batch_list = []
            for i in range(aces_batch.shape[0]):
                # Apply CDL: [H, W, 3]
                graded_aces = self.cdl_processor.apply_cdl_gpu(
                    aces_batch[i], cdl_params
                )
                # Apply PyTorch ACES transform: [H, W, 3] -> [H, W, 3]
                srgb_32f = self.pytorch_transformer.aces_to_srgb_32f(graded_aces)
                srgb_batch_list.append(srgb_32f)

            srgb_batch = torch.stack(srgb_batch_list, dim=0)

            yield aces_batch, srgb_batch

    def cleanup(self) -> None:
        """Close LMDB and cleanup GPU resources."""
        self.env.close()
        self.cdl_processor.cleanup()
        self.ocio_processor.cleanup()
        logger.info("Dataset cleanup complete")


def generate_on_the_fly_dataset(
    lmdb_aces_path: Path | str,
    output_dir: Path | str,
    batch_size: int = 8,
    num_batches: int = 10,
    save_format: str = "tensor",
) -> None:
    """Generate and save on-the-fly dataset examples.
    
    Args:
        lmdb_aces_path: Path to ACES LMDB database.
        output_dir: Where to save example batches.
        batch_size: Batch size for GPU processing.
        num_batches: Number of batches to generate and save.
        save_format: "tensor" (PyTorch) or "numpy".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OnTheFlyACESDataset(lmdb_aces_path, batch_size=batch_size)

    for batch_idx, (aces_batch, srgb_batch) in enumerate(dataset.iter_batches()):
        if batch_idx >= num_batches:
            break

        # Move to CPU for saving
        aces_cpu = aces_batch.cpu().numpy()
        srgb_cpu = srgb_batch.cpu().numpy()

        if save_format == "tensor":
            torch.save(
                {"aces": aces_batch, "srgb": srgb_batch},
                output_dir / f"batch_{batch_idx:04d}.pt",
            )
        else:
            np.save(output_dir / f"batch_{batch_idx:04d}_aces.npy", aces_cpu)
            np.save(output_dir / f"batch_{batch_idx:04d}_srgb.npy", srgb_cpu)

        logger.info(f"Saved batch {batch_idx}: {aces_cpu.shape}")

    dataset.cleanup()
    logger.info(f"✅ Generated {num_batches} batches to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate on-the-fly dataset batches with GPU CDL processing"
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        required=True,
        help="Path to ACES LMDB database",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/on_the_fly_test",
        help="Output directory for generated batches",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for GPU processing",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to generate",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["tensor", "numpy"],
        default="tensor",
        help="Format for saving batches",
    )

    args = parser.parse_args()

    try:
        generate_on_the_fly_dataset(
            args.lmdb_path,
            args.output_dir,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            save_format=args.save_format,
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
