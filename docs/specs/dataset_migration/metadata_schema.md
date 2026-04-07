# Metadata Schema Specification: Parquet and Sharding

To manage 20,000+ high-resolution ACES images, we use a single Apache Parquet file for metadata. This provides high-throughput columnar access for filtering and dataset-wide statistics calculation.

## 1. Schema Definition

| Column | Type | Description |
|--------|------|-------------|
| `id` | `VARCHAR(64)` | Unique hash or UUID of the image. |
| `source_path` | `TEXT` | Absolute path to the original .exr file. |
| `shard_id` | `INTEGER` | The WebDataset tar shard containing this image. |
| `resolution` | `INT[2]` | [Width, Height] of the image. |
| `mean_rgb` | `FLOAT[3]` | Per-channel mean luminance (linear). |
| `std_rgb` | `FLOAT[3]` | Per-channel standard deviation. |
| `max_luma` | `FLOAT` | Maximum relative luminance (useful for finding clipped/HDR). |
| `saturation_index` | `FLOAT` | Average colorfulness (to filter monochromatic). |
| `contrast_index` | `FLOAT` | RMS contrast. |
| `split` | `ENUM` | `train` (80%), `val` (10%), `test` (10%). |
| `created_at` | `TIMESTAMP` | Record creation date. |

## 2. Split Strategy: 80/10/10
The dataset will be partitioned into Training, Validation, and Testing sets using a deterministic hash-based split to ensure consistency across different runs.

- **Training (80%)**: Primary data for model gradient updates.
- **Validation (10%)**: Used for hyperparameter tuning and early stopping.
- **Testing (10%)**: Final "unseen" evaluation.

## 3. Why Parquet Only?
Since the source EXR data is already standardized as **ACES2065-1 (Linear)** and curated, we skip calculating per-image stats (luminance/saturation) unless needed for specific architectural constraints. We move directly to a "Frozen" metadata table that primarily manages the split assignments and shard mapping.

### Performance Benefits
- **Columnar Efficiency**: Reading only the `split` column to filter for training samples is near-instant (<1s for 20k rows).
- **Zero-Copy Reads**: Integrates seamlessly with `pyarrow` and `pandas`.
- **Portability**: A single `.parquet` file can be shipped alongside the dataset shards.

## 3. Workflow
1.  **Extract & Analayze**: Scan curated EXR files and calculate statistics (Luma, Saturation, etc.).
2.  **Generate Table**: Build a `pandas.DataFrame` or `pyarrow.Table`.
3.  **Persist**: Save to `dataset/training_metadata.parquet`.
4.  **Bake**: Use the Parquet table as the manifest for `wds.TarWriter`.
