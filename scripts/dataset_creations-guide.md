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

to generate LDR sRGB dataset with 50 with looks 50% neutral use
```
pixi run python scripts/bake_dataset.py
```

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
