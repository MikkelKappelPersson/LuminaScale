#!/usr/bin/env python3
"""Quick benchmark to identify bottleneck in data loading."""
import time
import logging
import torch
from luminascale.training.dequantization_trainer import OnTheFlyBDEDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize dataset
    logger.info("Initializing dataset...")
    dataset = OnTheFlyBDEDataset(
        lmdb_path="/home/student.aau.dk/fs62fb/projects/LuminaScale/dataset/training_data.lmdb",
        patches_per_image=32,
        crop_size=512,
        device=torch.device("cuda:0"),
        world_size=1,
        rank=0,
    )
    
    logger.info(f"Dataset has {len(dataset)} samples")
    
    # Benchmark first 5 samples
    logger.info("\nBenchmarking __getitem__ calls:")
    for i in range(5):
        t_start = time.perf_counter()
        srgb_8u, srgb_32f = dataset[i]
        t_end = time.perf_counter()
        t_ms = (t_end - t_start) * 1000
        logger.info(f"  Sample {i}: {t_ms:.1f}ms - Shape: {srgb_8u.shape}, {srgb_32f.shape}")

if __name__ == "__main__":
    main()
