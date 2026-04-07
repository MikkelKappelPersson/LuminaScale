# WebDataset Implementation Guide: Shards & Looping

WebDataset (WDS) solves the problem of POSIX filesystem overhead by streaming large `.tar` files. This is essential for the AAU AI Cloud (HPC) where small file I/O can be extremely slow over the network fabric.

## 1. Shard Generation (The "Bake" Step)

We will use `wds.TarWriter` to create shards. Each entry in the tarball represents one sample:
- `000001.exr`: The raw floating-point ACES data.
- `000001.json`: Local metadata (id, resolution, stats).

### Pseudo-code for Bake Script
```python
import webdataset as wds
import pyarrow.parquet as pq

# Load finalized metadata
metadata = pq.read_table("dataset/training_metadata.parquet").to_pandas()

with wds.ShardWriter("dataset/shards/%06d.tar", maxsize=3e9) as sink:
    for idx, row in metadata.iterrows():
        with open(row.source_path, "rb") as f:
            exr_data = f.read()
        
        sample = {
            "__key__": row.id,
            "exr": exr_data,
            "json": row.to_json()
        }
        sink.write(sample)
```

## 2. Training DataLoader

We use the `WebDataset` class to stream the shards.

### Integration with PyTorch
```python
import webdataset as wds

dataset = (
    wds.WebDataset("dataset/shards/{000000..000100}.tar")
    .shuffle(100) # Shuffle buffer size
    .decode("torch") # Use custom decoder for EXR (float32)
    .to_tuple("exr", "json")
    .batched(32)
)

loader = wds.WebLoader(dataset, num_workers=4)
```

## 3. Why This Is Better for LuminaScale
- **No Linear Slowdown**: Unlike LMDB, reading from a stream does not degrade over time. Shards are independent blocks.
- **Memory Efficiency**: Only the current shard being read (and the shuffle buffer) occupies significant RAM.
- **Portability**: Shards can be copied to local `/scratch` on different HPC nodes easily.
- **Elasticity**: Shards can be distributed across multi-node training clusters using `DistributedSampler`.
