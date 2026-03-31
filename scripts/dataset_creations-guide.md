# Guide to create the full dataset

To create full dataset run
```
python scripts/quality_filtered_aces_conversion.py && python scripts/bake_dataset.py
```
## HDR exr

in order to run quality check and create aces EXR files, run:
````
python scripts/quality_filtered_aces_conversion.py
````

for pixi:
```
pixi run python scripts/quality_filtered_aces_conversion.py
```

Only some images:
```
pixi run python scripts/quality_filtered_aces_conversion.py --max-images=10
```

## LDR

To generate LDR sRGB dataset (using the **GPU-accelerated** pipeline) with a 50% neutral/graded split:

```bash
# Generate 8-bit PNGs (standard)
pixi run python scripts/bake_dataset.py

# Generate 32-bit Float EXRs (for high-fidelity LDR)
pixi run python scripts/bake_dataset.py --float32 --output-dir dataset/temp/srgb_32bit
```

### Why use GPU Baking?
The new bake script uses `ocio_aces_to_srgb_with_look` which leverages a headless EGL GPU renderer. This is significantly faster than the OIIO CPU implementation, especially when applying complex ACES 2.0 transforms and color looks.

## LMDB Packing (Performance Optimization)

To speed up training by 40-100x and maximize GPU utilization, pack the processed ACES and sRGB images into an uncompressed LMDB database. This eliminates the CPU bottleneck caused by decoding large EXR/PNG files.

**Note:** Ensure you have enough disk space (approx. 125MB per image pair).

```bash
pixi run python scripts/pack_lmdb.py --aces-dir dataset/temp/aces --srgb-dir dataset/temp/srgb_looks --output-path dataset/training_data.lmdb
```

After packing, update your configuration (e.g., `configs/test.yaml`) to point to the LMDB path:

```yaml
dataset:
  lmdb_path: dataset/training_data.lmdb
```
