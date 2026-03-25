# guide to create the full dataset

in order to run quality check and create aces EXR files, run:
````
python scripts/quality_filtered_aces_conversion.py
````

for pixi:
```
pixi run python scripts/quality_filtered_aces_conversion.py
```

only some images:
```
pixi run python scripts/quality_filtered_aces_conversion.py --max-images=10
```