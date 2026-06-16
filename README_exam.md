# LuminaScale Exam Image Inference

Run full inference (Dequantization → ACES Mapper) on exam images and save to the output directory:

```bash
pixi run python scripts/run_full_inference.py \
  --input /run/media/mikkelkp/MKP_T7_2TB/Exam/project_showcase/exam_images_input \
  --output /run/media/mikkelkp/MKP_T7_2TB/Exam/project_showcase/exam_images_output
```

This processes all images in the input directory and saves results (ACES EXR + sRGB JPG preview) to the output directory.
