# Dataset Migration Research: Formats and Scaling

This document captures the research into replacing the current LMDB-based dataset with a more scalable and robust architecture using Raw EXRs, WebDataset, and Parquet/SQLite for metadata.

## 1. Source Data: Raw EXRs
The project currently uses LMDB to store raw `float32` bytes. While fast for small values, LMDB's performance degrades when storing 250MB+ images due to memory-mapping overhead and OS page eviction thrashing.

### Why Raw EXR?
- **Domain Standard**: Industrial light and magic (ILM) format for HDR/Linear data.
- **Precision**: Supports `float16` (half) and `float32` natively.
- **Compression**: PIZ, ZIP, or B44 compression can significantly reduce disk footprint while maintaining fidelity.
- **Metadata**: Can store ACES metadata (primaries, white point) directly in headers.
- **Tooling**: Deep integration with OpenImageIO (OIIO) and PyOpenEXR.

### Constraints
- Scaling to 20,000+ files on a networked file system (HPC) can cause "Small File Problem" (metadata overhead on the filesystem).

## 2. Training Format: WebDataset (.tar shards)
WebDataset (WDS) is a library for accessing datasets stored as POSIX tar archives.

### Why WebDataset?
- **Sequential I/O**: Accesses data as a stream of large tar files (shards), which is optimal for HPC and cloud storage.
- **Zero Metadata Bottleneck**: Filesystem only sees a few large `.tar` files rather than millions of small EXRs.
- **PyTorch Integration**: Native `IterableDataset` support.
- **Shuffle & Split**: Provides high-performance shard-level and worker-level shuffling.

### Sharding Strategy
- Target shard size: 1GB to 5GB (approx. 4 to 20 raw EXRs per shard at 250MB/ea).
- Shards can be stored on high-speed scratch or streamed from network storage.

## 3. Metadata & Split Management: Parquet (Columnar)
Since the EXR source data is already curated and sorted, we skip the need for a relational staging database (SQLite) and move directly to a "Frozen" Parquet manifest.

### Why Parquet?
- **Speed**: Extremely fast for filtering/aggregating attributes (e.g., "all images with saturation > X").
- **Integration**: Works perfectly with HuggingFace datasets, Spark, and `pandas`.
- **Storage**: Highly compact, immutable format that is easy to version-control or ship with shards.
- **Efficiency**: Allows reading only the columns needed for training (e.g., `id`, `split`, `shard_id`) without loading the full table into RAM.

**Recommendation**: Use **Parquet** (`training_metadata.parquet`) as the single source for dataset splits and statistic-driven data selection.

## 4. Proposed Architecture
1. **Raw Storage**: Individual `.exr` files in a structured directory (Source of Truth).
2. **Metadata**: A single `metadata.parquet` file containing image stats (mean, std, saturation, resolution, file path).
3. **Training Shards**: A set of `.tar` files generated from the EXRs + metadata.
4. **DataLoader**: `wds.WebDataset` piping data directly to GPU-accelerated ACES/CDL pipeline.
