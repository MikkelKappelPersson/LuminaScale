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
