"""V3.5 verification: decode-in-workers ships torch tensors (shared-memory IPC)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from luminascale.data.wds_dataset import LuminaScaleWebDataset


def main():
    ds = LuminaScaleWebDataset(
        shard_path="dataset/shards/ACES2065-1/dev/shards/train",
        batch_size=1,
        shuffle_buffer=10,
        is_training=True,
        metadata_parquet="dataset/shards/ACES2065-1/dev/training_metadata.parquet",
        decode_in_workers=True,
        crop_size=2048,
    )
    n = 0
    for b in ds.get_loader(num_workers=2):
        item = b[0][0]
        print(
            f"batch {n}: pixels type={type(item.pixels).__name__} "
            f"tensor={isinstance(item.pixels, torch.Tensor)} dtype={item.pixels.dtype} "
            f"shape={tuple(item.pixels.shape)} decode_ms={item.meta['decode_ms']:.0f}"
        )
        n += 1
        if n >= 3:
            break
    print("V35 CHECK OK")


if __name__ == "__main__":
    main()
