#!/usr/bin/env python3
"""Benchmark DataLoader to find bottleneck."""
import time
import logging
import torch
from torch.utils.data import DataLoader
from luminascale.training.dequantization_trainer import OnTheFlyBDEDataset

logging.basicConfig(level=logging.WARNING)
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
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"\nDataset: {len(dataset)} samples ({len(dataset)//32} images × 32 patches)")
    print(f"DataLoader: batch_size=4, num_batches={len(dataloader)}")
    print("\nBenchmarking first 50 batches:\n")
    
    batch_times = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 50:
            break
        
        t_start = time.perf_counter()
        # Simulate what training does
        srgb_8u, srgb_32f = batch
        t_end = time.perf_counter()
        t_ms = (t_end - t_start) * 1000
        batch_times.append(t_ms)
        
        if batch_idx < 20 or batch_idx % 10 == 0:
            print(f"  Batch {batch_idx:3d}: {t_ms:8.1f}ms - Shape: {srgb_8u.shape}")
    
    print(f"\nAverage batch time (ex. first 5): {sum(batch_times[5:]) / len(batch_times[5:]):.1f}ms")
    print(f"Average throughput: {1000 / (sum(batch_times[5:]) / len(batch_times[5:])) * 4:.1f} samples/sec")

if __name__ == "__main__":
    main()
