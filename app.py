#!/usr/bin/env python3
"""Gradio app for LuminaScale full inference (dequant -> ACES mapper)."""

from __future__ import annotations

import os
import random
import logging
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from scripts import run_full_inference as rfi
from src.luminascale.utils.output_catalog import OutputCatalog


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference"
DEBUG_LOG_PATH = DEFAULT_OUTPUT_DIR / "gradio_debug.log"
OUTPUT_DISPLAY_ORDER: tuple[str, ...] = (
	"input",
	"dequant output",
	"full output",
	"full output(srgb)",
	"reference",
	"reference(srgb)",
)
DEFAULT_SLIDER_SELECTIONS: tuple[str, str] = ("full output(srgb)", "reference(srgb)")
DEFAULT_DOWNLOAD_SELECTIONS: tuple[str, ...] = ("input", "full output", "reference")


LOGGER = logging.getLogger("luminascale.gradio")
if not LOGGER.handlers:
	LOGGER.setLevel(logging.INFO)
	LOGGER.propagate = False
	_stream_handler = logging.StreamHandler()
	_stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
	LOGGER.addHandler(_stream_handler)
	_file_handler = logging.FileHandler(DEBUG_LOG_PATH, encoding="utf-8")
	_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
	LOGGER.addHandler(_file_handler)


def _parse_triplet(text: str, name: str) -> tuple[float, float, float]:
	parts = [part.strip() for part in text.split(",")]
	if len(parts) != 3:
		raise gr.Error(f"{name} must contain exactly 3 comma-separated values", duration=5)
	values = (float(parts[0]), float(parts[1]), float(parts[2]))
	return values


def _save_preview_png(image_chw: torch.Tensor, destination: Path) -> Path:
	destination.parent.mkdir(parents=True, exist_ok=True)
	image_np = torch.clamp(image_chw, 0.0, 1.0).permute(1, 2, 0).numpy()
	image_u8 = (image_np * 255.0).round().astype("uint8")
	Image.fromarray(image_u8).save(destination)
	return destination


def _preview_path_for_input(source_path: str) -> str:
	resolved_source_path = _normalize_gradio_image_path(source_path).strip()
	if not resolved_source_path:
		LOGGER.info("Skipping preview generation because source path is empty")
		return ""
	LOGGER.info("Generating preview for source=%s", resolved_source_path)
	preview_dir = DEFAULT_OUTPUT_DIR / "gradio_previews"
	preview_dir.mkdir(parents=True, exist_ok=True)
	preview_path = preview_dir / f"{Path(resolved_source_path).stem}_preview.png"
	preview_tensor = rfi.image_to_tensor(Path(resolved_source_path))
	_save_preview_png(preview_tensor, preview_path)
	LOGGER.info("Preview saved to %s", preview_path)
	return str(preview_path)


def _normalize_gradio_image_path(value: object) -> str:
	if value is None:
		return ""
	if isinstance(value, str):
		return value
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		image_value = value.get("image")
		if isinstance(image_value, dict):
			path_value = image_value.get("path")
			if isinstance(path_value, str):
				return path_value
		path_value = value.get("path")
		if isinstance(path_value, str):
			return path_value
	return str(value)


def _catalog_display_names(outputs_catalog: dict[str, str]) -> list[str]:
	"""Return the available output display names in a stable, preferred order."""
	return [name for name in OUTPUT_DISPLAY_ORDER if name in outputs_catalog]


def _catalog_download_defaults(outputs_catalog: dict[str, str]) -> list[str]:
	"""Return the preferred default download selections that exist in the catalog."""
	return [name for name in DEFAULT_DOWNLOAD_SELECTIONS if name in outputs_catalog]


def _catalog_slider_defaults(outputs_catalog: dict[str, str]) -> tuple[str, str]:
	"""Choose valid default dropdown values for the slider from the current catalog."""
	available_names = _catalog_display_names(outputs_catalog)
	if not available_names:
		return "", ""

	left_default, right_default = DEFAULT_SLIDER_SELECTIONS
	left_value = left_default if left_default in outputs_catalog else available_names[0]
	if right_default in outputs_catalog:
		right_value = right_default
	elif len(available_names) > 1:
		right_value = available_names[1]
	else:
		right_value = available_names[0]
	return left_value, right_value


def _sync_output_controls(outputs_catalog: dict[str, str]) -> tuple[object, object, tuple[str, str], object]:
	"""Update dropdown choices/defaults and refresh the image slider after inference."""
	choices = _catalog_display_names(outputs_catalog)
	left_value, right_value = _catalog_slider_defaults(outputs_catalog)
	slider_images = _get_slider_images(left_value, right_value, outputs_catalog)
	download_defaults = _catalog_download_defaults(outputs_catalog)
	return (
		gr.update(choices=choices, value=left_value),
		gr.update(choices=choices, value=right_value),
		slider_images,
		gr.update(choices=choices, value=download_defaults),
	)


def run_pipeline(

	input_path: str,
	output_dir: str,
	output_name: str,
	save_plot: bool,
	input_is_aces: bool,
	seed: int,
	crop_size: int,
	keep_padding: bool,
	max_side: int,
	# Defaulted internal parameters (removed from UI)
	dequant_output_name: str="",
	save_dequant: bool=False,
	mapper_checkpoint: str="outputs/training/mapper/20260425_231537/checkpoints/aces-mapper-20260425_231537-epoch=09.ckpt",
	dequant_checkpoint: str="outputs/training/dequant/20260422_120606_L1=1.0_L2=0.0_CB=1.0_EA=0.0_TV-huber=0.0/checkpoints/last.ckpt",
	look_mode: str = "random",
	slope: str = "1.0,1.0,1.0",
	offset: str = "0.0,0.0,0.0",
	power: str = "1.0,1.0,1.0",
	saturation: float = 1.0,
	align_multiple: int = 64,
	num_luts: int = 3,
	lut_dim: int = 33,
	num_lap: int = 3,
	num_residual_blocks: int = 5,
	dequant_channels: int = 32,
	device_name: str = "cuda",
	amp_enabled: bool = False,
	amp_dtype_name: str = "fp16",
) -> tuple[str, str | None, str | None, str | None, dict[str, str]]:
	# --- Pre-inference validation (outside try-except to allow gr.Error visibility) ---
	input_image_path = Path(input_path).expanduser().resolve()
	if not input_image_path.exists():
		raise gr.Error(f"Input image does not exist: {input_image_path}", duration=5)

	mapper_ckpt = Path(mapper_checkpoint).expanduser().resolve()
	dequant_ckpt = Path(dequant_checkpoint).expanduser().resolve()
	if not mapper_ckpt.exists():
		raise gr.Error(f"Mapper checkpoint does not exist: {mapper_ckpt}", duration=5)
	if not dequant_ckpt.exists():
		raise gr.Error(f"Dequant checkpoint does not exist: {dequant_ckpt}", duration=5)

	if input_is_aces:
		if input_image_path.suffix.lower() != ".exr":
			raise gr.Error("input_is_aces requires an EXR input", duration=5)

	if look_mode == "manual":
		_parse_triplet(slope, "slope")
		_parse_triplet(offset, "offset")
		_parse_triplet(power, "power")

	try:
		LOGGER.info("Starting inference run")
		LOGGER.info("input=%s output_dir=%s", input_path, output_dir)
		LOGGER.info("mapper_checkpoint=%s", mapper_checkpoint)
		LOGGER.info("dequant_checkpoint=%s", dequant_checkpoint)
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

		if device_name == "cuda" and not torch.cuda.is_available():
			device_name = "cpu"

		device = torch.device(device_name)
		amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
		use_amp = bool(amp_enabled and device.type == "cuda")

		ocio_config = PROJECT_ROOT / "config" / "aces" / "studio-config.ocio"
		if ocio_config.exists():
			os.environ["OCIO"] = str(ocio_config)

		out_dir = Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_OUTPUT_DIR
		out_dir.mkdir(parents=True, exist_ok=True)

		# Initialize output catalog for tracking all inference outputs
		catalog = OutputCatalog(out_dir)

		input_stem = input_image_path.stem
		pred_name = output_name.strip() if output_name.strip() else f"{input_stem}_out.exr"
		deq_name = dequant_output_name.strip() if dequant_output_name.strip() else f"{input_stem}_dequant.exr"
		plot_name = f"{input_stem}_plot.png"
		preview_name = f"{input_stem}_mapper_preview.png"

		resolved_pred_aces_output = out_dir / pred_name
		resolved_dequant_output = out_dir / deq_name
		resolved_plot_output = out_dir / plot_name
		resolved_preview_output = out_dir / preview_name

		look = None
		dequant_input_override_chw: torch.Tensor | None = None
		clean_reference_chw: torch.Tensor | None = None
		dequant_reference_chw: torch.Tensor | None = None

		if input_is_aces:
			LOGGER.info("Preparing ACES reference input")
			look = rfi.build_look(look_mode, slope, offset, power, saturation)
			aces_chw = rfi.image_to_tensor(input_image_path).to(device=device, dtype=torch.float32)
			aces_hwc = aces_chw.permute(1, 2, 0)

			cdl_processor = rfi.GPUCDLProcessor(device=device)
			transformer = rfi.ACESColorTransformer(device=device, use_lut=True)

			clean_srgb_hwc = transformer.aces_to_srgb_32f(aces_hwc.unsqueeze(0)).squeeze(0)
			clean_srgb_hwc = torch.clamp(clean_srgb_hwc, 0.0, 1.0)
			clean_reference_chw = clean_srgb_hwc.permute(2, 0, 1).detach().cpu()

			aces_graded_hwc = cdl_processor.apply_cdl_gpu(aces_hwc, look)
			srgb_hwc = transformer.aces_to_srgb_32f(aces_graded_hwc.unsqueeze(0)).squeeze(0)
			srgb_hwc = torch.clamp(srgb_hwc, 0.0, 1.0)
			dequant_reference_chw = srgb_hwc.permute(2, 0, 1).detach().cpu()
			srgb_hwc = torch.round(srgb_hwc * 255.0) / 255.0
			dequant_input_override_chw = srgb_hwc.permute(2, 0, 1)

		if save_dequant:
			dequant_stage_output = resolved_dequant_output
			temporary_dequant_path: Path | None = None
		else:
			temporary_dequant_path = out_dir / f"{input_stem}_dequant_tmp.exr"
			dequant_stage_output = temporary_dequant_path

		dequant_model = rfi.load_dequant_model_from_checkpoint(
			checkpoint_path=dequant_ckpt,
			device=device,
			channels=dequant_channels,
		)
		LOGGER.info("Loaded dequant model")
		if use_amp:
			dequant_model = dequant_model.to(dtype=amp_dtype)

		LOGGER.info("Running dequant inference")
		_, dequant_input_chw_cpu, dequant_output_chw_cpu = rfi.run_dequant_inference(
			model=dequant_model,
			input_path=input_image_path,
			output_path=dequant_stage_output,
			device=device,
			align_multiple=align_multiple,
			amp_enabled=use_amp,
			amp_dtype=amp_dtype,
			quantize_input=False,
			input_chw_override=dequant_input_override_chw,
		)

		# Capture input and dequant output to catalog
		catalog.add_tensor_output(
			"input",
			dequant_input_chw_cpu,
			f"{input_stem}_input.jpg",
			quality=95,
		)
		catalog.add_tensor_output(
			"dequant output",
			dequant_output_chw_cpu,
			f"{input_stem}_dequant.jpg",
			quality=95,
		)
		LOGGER.info("Captured dequant stage outputs to catalog")

		mapper_model = rfi.load_model_from_checkpoint(
			checkpoint_path=mapper_ckpt,
			device=device,
			num_luts=num_luts,
			lut_dim=lut_dim,
			num_lap=num_lap,
			num_residual_blocks=num_residual_blocks,
		)
		LOGGER.info("Loaded mapper model")
		# Keep the mapper in float32: the spatial-frequency transformer uses
		# complex ops that fail under ComplexHalf autocast on CUDA.

		LOGGER.info("Running mapper inference")
		pred_aces_chw_cpu, mapper_srgb_chw_cpu = rfi.run_mapper_inference_on_srgb(
			model=mapper_model,
			input_srgb_chw=dequant_output_chw_cpu,
			crop_size=crop_size,
			align_multiple=align_multiple,
			max_side=max_side,
			keep_aligned_output=keep_padding,
			device=device,
		)

		# Capture full output (ACES EXR + sRGB JPG)
		catalog.add_aces_output(
			"full output",
			pred_aces_chw_cpu,
			f"{input_stem}_out.exr",
			f"{input_stem}_out_srgb.jpg",
			device,
			quality=95,
		)
		LOGGER.info("Captured mapper ACES output to catalog")

		rfi.write_exr(resolved_pred_aces_output, pred_aces_chw_cpu.numpy())
		LOGGER.info("Saved predicted ACES EXR to %s", resolved_pred_aces_output)
		_save_preview_png(mapper_srgb_chw_cpu, resolved_preview_output)
		LOGGER.info("Saved mapper preview to %s", resolved_preview_output)

		# Capture reference output if input_is_aces
		# Note: We save the clean (ungraded) reference ACES in EXR format
		if input_is_aces:
			# Reload ACES tensor for EXR export (reference should be the original clean ACES)
			aces_original_chw = rfi.image_to_tensor(input_image_path).to(device=device, dtype=torch.float32)
			catalog.add_reference_aces_output(
				aces_original_chw,
				f"{input_stem}_reference.exr",
				f"{input_stem}_reference_srgb.jpg",
				device,
				quality=95,
			)
			LOGGER.info("Captured reference ACES output to catalog")

		plot_path: str | None = None
		if save_plot and input_is_aces:
			LOGGER.info("Saving dashboard plot")
			if dequant_reference_chw is None:
				raise gr.Error("Expected dequant reference for dashboard", duration=5)
			reference_for_dashboard = clean_reference_chw if input_is_aces else None
			rfi.save_full_inference_dashboard(
				input_srgb_chw=dequant_input_chw_cpu,
				dequant_srgb_chw=dequant_output_chw_cpu,
				mapper_srgb_chw=mapper_srgb_chw_cpu,
				reference_srgb_chw=reference_for_dashboard,
				dequant_reference_srgb_chw=dequant_reference_chw,
				save_path=resolved_plot_output,
			)
			plot_path = str(resolved_plot_output)

		if (not save_dequant) and temporary_dequant_path is not None and temporary_dequant_path.exists():
			temporary_dequant_path.unlink()
			LOGGER.info("Removed temporary dequant file")

		summary_lines = [
			"Inference completed.",
			f"Input: {input_image_path}",
			f"Predicted ACES EXR: {resolved_pred_aces_output}",
			f"Mapper sRGB preview: {resolved_preview_output}",
			f"Dequant intermediate saved: {'yes' if save_dequant else 'no'}",
			f"Dashboard saved: {'yes' if plot_path else 'no'}",
			f"CDL look used: {'yes' if look is not None else 'no'}",
			f"Device: {device}",
			f"AMP: {'enabled' if use_amp else 'disabled'} ({amp_dtype_name})",
		]
		if save_plot and not input_is_aces:
			summary_lines.append("Dashboard was skipped because input_is_aces is disabled.")
		summary_lines.append(f"Debug log: {DEBUG_LOG_PATH}")

		dequant_path = str(resolved_dequant_output) if save_dequant else None
		outputs_dict = catalog.get_catalog()
		return "\n".join(summary_lines), str(resolved_pred_aces_output), dequant_path, plot_path, outputs_dict

	except Exception as exc:  # noqa: BLE001
		LOGGER.exception("Inference failed")
		details = "\n".join(
			[
				f"Inference failed: {exc}",
				"",
				"Traceback:",
				traceback.format_exc(),
			]
		)
		details += f"\nDebug log: {DEBUG_LOG_PATH}"
		return details, None, None, None, {}


def _warmup_pipeline(
	mapper_checkpoint: str,
	dequant_checkpoint: str,
	device_name: str = "cuda",
) -> None:
	"""Run a dummy pass to initialize CUDA kernels and prevent the first-run crash."""
	try:
		if device_name == "cuda" and not torch.cuda.is_available():
			return

		LOGGER.info("Warming up pipeline with dummy data...")
		device = torch.device(device_name)

		# 1. Load models
		mapper_ckpt = Path(mapper_checkpoint).expanduser().resolve()
		dequant_ckpt = Path(dequant_checkpoint).expanduser().resolve()
		if not mapper_ckpt.exists() or not dequant_ckpt.exists():
			LOGGER.warning("Warmup skipped: Checkpoints not found at default paths.")
			return

		dequant_model = rfi.load_dequant_model_from_checkpoint(dequant_ckpt, device, channels=32)
		mapper_model = rfi.load_model_from_checkpoint(
			mapper_ckpt,
			device,
			num_luts=3,
			lut_dim=33,
			num_lap=3,
			num_residual_blocks=5,
		)

		# 2. Run dummy dequant (128x128)
		dummy_input = torch.zeros((1, 3, 128, 128), device=device)
		with torch.no_grad():
			_ = dequant_model(dummy_input)
			LOGGER.info("Dequant warmup complete.")

			# 3. Run dummy mapper
			_ = mapper_model(dummy_input)
			LOGGER.info("Mapper warmup complete.")

		LOGGER.info("Pipeline warmup successful.")
	except Exception as e:
		LOGGER.warning("Warmup failed (this is usually non-fatal): %s", e)


def _generate_exr_thumbnail(exr_path: str) -> str:
	"""Generate a 256x256 JPEG thumbnail for an EXR file for gallery display."""
	exr_path_obj = Path(exr_path)
	thumbnail_dir = PROJECT_ROOT / "outputs" / "inference" / "gallery_thumbs"
	thumbnail_path = thumbnail_dir / f"{exr_path_obj.stem}_thumb.jpg"
	
	if thumbnail_path.exists():
		return str(thumbnail_path)
	
	try:
		thumbnail_dir.mkdir(parents=True, exist_ok=True)
		tensor = rfi.image_to_tensor(exr_path_obj)  # [C, H, W]
		
		# Resize to 256x256
		tensor_hwc = tensor.permute(1, 2, 0)  # [H, W, C]
		pil_image = Image.fromarray((torch.clamp(tensor_hwc, 0.0, 1.0) * 255.0).round().numpy().astype("uint8"))
		pil_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
		pil_image.save(thumbnail_path, "JPEG", quality=85)
		
		LOGGER.info("Generated 256x256 JPEG thumbnail for %s", exr_path)
		return str(thumbnail_path)
	except Exception as e:
		LOGGER.warning("Failed to generate thumbnail for %s: %s", exr_path, e)
		return exr_path


def _prepare_gallery_images() -> tuple[list[str], dict[str, str]]:
	"""Prepare gallery images, generating thumbnails for EXR files.
	
	Returns:
		Tuple of (gallery_image_paths, thumbnail_to_original_mapping)
	"""
	jpg_files = sorted(str(p) for p in Path("assets/imgs").glob("*.jpg"))
	exr_files = sorted(str(p) for p in Path("assets/imgs").glob("*.exr"))
	
	# Map thumbnails to original files
	thumbnail_to_original = {}
	exr_thumbs = []
	for exr in exr_files:
		thumb = _generate_exr_thumbnail(exr)
		exr_thumbs.append(thumb)
		thumbnail_to_original[thumb] = exr
	
	# Combine and sort by original path
	all_images = jpg_files + exr_thumbs
	return sorted(all_images), thumbnail_to_original


def _run_pipeline_from_ui(
	selected_input_path: str,
	output_name: str,
	save_plot: bool,
	input_is_aces: bool,
	seed: int,
	crop_size: int,
	keep_padding: bool,
	max_side: int,
) -> tuple[str | None, str | None, str | None, str | None, dict[str, str]]:
	resolved_input_path = selected_input_path.strip()
	LOGGER.info("UI selected input path=%s", resolved_input_path)
	if not resolved_input_path:
		raise gr.Error("Please select a gallery image or upload an image", duration=5)
	return run_pipeline(
		input_path=resolved_input_path,
		output_dir=output_dir,
		output_name=output_name,
		save_plot=save_plot,
		input_is_aces=input_is_aces,
		seed=seed,
		crop_size=crop_size,
		keep_padding=keep_padding,
		max_side=max_side,
	)




def _get_slider_images(
	dropdown1_value: str,
	dropdown2_value: str,
	outputs_catalog: dict[str, str],
) -> tuple[str, str]:
	"""Get image paths for slider based on dropdown selections.
	
	Args:
		dropdown1_value: Display name of first comparison image.
		dropdown2_value: Display name of second comparison image.
		outputs_catalog: Catalog dict from inference outputs.
		
	Returns:
		Tuple of (image_path_1, image_path_2) for slider display.
		Returns ("", "") if catalog is empty.
	"""
	if not outputs_catalog:
		LOGGER.warning("Outputs catalog is empty, cannot populate slider")
		return "", ""
	
	path1 = outputs_catalog.get(dropdown1_value, "")
	path2 = outputs_catalog.get(dropdown2_value, "")
	
	LOGGER.info("Slider images: %s -> %s, %s -> %s", dropdown1_value, path1, dropdown2_value, path2)
	return path1, path2


def _get_download_zip(
	selected_outputs: list[str],
	outputs_catalog: dict[str, str],
) -> str | None:
	"""Create a zip file with selected outputs.
	
	Args:
		selected_outputs: List of display names to include in zip.
		outputs_catalog: Catalog dict from inference outputs.
		
	Returns:
		Path to created zip file, or None if nothing selected.
	"""
	import zipfile
	
	if not selected_outputs or not outputs_catalog:
		LOGGER.warning("No outputs selected or catalog is empty")
		return None
	
	# Collect selected files
	files_to_zip = []
	for output_name in selected_outputs:
		if output_name in outputs_catalog:
			file_path = outputs_catalog[output_name]
			files_to_zip.append((output_name, file_path))
		else:
			LOGGER.warning("Selected output '%s' not found in catalog", output_name)
	
	if not files_to_zip:
		LOGGER.warning("No valid outputs found for zip")
		return None
	
	# Create zip file
	zip_dir = DEFAULT_OUTPUT_DIR / "downloads"
	zip_dir.mkdir(parents=True, exist_ok=True)
	
	import time
	zip_filename = f"luminascale_outputs_{int(time.time())}.zip"
	zip_path = zip_dir / zip_filename
	
	try:
		with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
			for display_name, file_path in files_to_zip:
				# Use display name + file extension as archive name
				file_obj = Path(file_path)
				archive_name = f"{display_name.replace('(', '').replace(')', '')}{file_obj.suffix}"
				LOGGER.info("Adding to zip: %s -> %s", file_path, archive_name)
				zf.write(file_path, arcname=archive_name)
		
		LOGGER.info("Created download zip: %s", zip_path)
		return str(zip_path)
	except Exception as e:
		LOGGER.exception("Failed to create zip file: %s", e)
		return None


example_images, _thumbnail_to_original = _prepare_gallery_images()

with gr.Blocks(title="LuminaScale Full Inference") as demo:
	gr.Markdown("# LuminaScale Gradio Inference")
	gr.Markdown("Run dequantization followed by ACES mapping using local checkpoints.")

	with gr.Row():
		with gr.Column(scale=1):
			gr.Markdown("## Select input image \n Upload an image or select from gallery.")
			selected_input_path = gr.State(None)

			input_upload = gr.Image(label="Upload input image", type="filepath", )
			input_gallery = gr.Gallery(example_images, label="Example inputs", columns=1, allow_preview=False)
		with gr.Column(scale=2):
			with gr.Column():
				gr.Markdown("## Input Image \n ")
				input_image = gr.Image(label="Selected input preview", type="filepath", interactive=False, value=None)
				gr.Markdown("## Params")
			
			with gr.Row():
				input_is_aces = gr.Checkbox(label="Input is ACES EXR", value=True)
				save_plot = gr.Checkbox(label="Save dashboard plot", value=True)
				keep_padding = gr.Checkbox(label="Keep alignment padding", value=False)
			
			with gr.Row():
				output_name = gr.Textbox(label="Output filename", value="")


			with gr.Row():
				seed = gr.Number(label="Seed", value=9, precision=0)
				crop_size = gr.Number(label="Crop size (0=disable)", value=0, precision=0)
				max_side = gr.Number(label="Max side (0=disable)", value=1024, precision=0)

			run_button = gr.Button("Run full inference", variant="primary")
	
	with gr.Column():
		gr.Markdown("## Output")
		with gr.Row(scale=1):
			slider_dropdown_1 = gr.Dropdown(value=DEFAULT_SLIDER_SELECTIONS[0], choices=list(OUTPUT_DISPLAY_ORDER), label="compare image 1", info="first image for comparison.")
			slider_dropdown_2 = gr.Dropdown(value=DEFAULT_SLIDER_SELECTIONS[1], choices=list(OUTPUT_DISPLAY_ORDER), label="compare image 2", info="second image for comparison.")
		img_slider = gr.ImageSlider(label="image slider", type="filepath")
		plot_output = gr.Image(label="Dashboard plot", type="filepath")
	
	with gr.Column():
		gr.Markdown("## Download")
		download_checkbox_group = gr.CheckboxGroup(value=list(DEFAULT_DOWNLOAD_SELECTIONS), choices=list(OUTPUT_DISPLAY_ORDER), label="Select downloads", info="Select which outputs to download.")
		download_button = gr.DownloadButton(label="Download selected outputs") 
	
	
	
	status_output = gr.Textbox(label="Status", lines=12)
	
	# State variable to store outputs catalog for use in callbacks
	outputs_catalog = gr.State({})
	
	def _set_selected_input(source_path: object) -> tuple[str, str]:
		LOGGER.info("Received UI selection payload type=%s value=%r", type(source_path).__name__, source_path)
		resolved_source_path = _normalize_gradio_image_path(source_path).strip()
		LOGGER.info("Normalized UI selection path=%s", resolved_source_path)
		if not resolved_source_path:
			raise gr.Error("No input image selected", duration=5)
		preview_path = _preview_path_for_input(resolved_source_path)
		LOGGER.info("Selection ready preview=%s input=%s", preview_path, resolved_source_path)
		return preview_path, resolved_source_path

	def _gallery_to_input_path(event: gr.SelectData) -> tuple[str, str]:
		selected_index = event.index
		LOGGER.info("Gallery selection event index=%r value_type=%s", selected_index, type(event.value).__name__)
		if selected_index is None:
			raise gr.Error("No gallery image selected", duration=5)
		selected_idx = int(selected_index)
		if not (0 <= selected_idx < len(example_images)):
			raise gr.Error("Gallery selection out of range", duration=5)
		selected_path = example_images[selected_idx]
		# Map thumbnail back to original EXR if needed
		if selected_path in _thumbnail_to_original:
			actual_input_path = _thumbnail_to_original[selected_path]
			LOGGER.info("Gallery resolved thumbnail=%s to original=%s", selected_path, actual_input_path)
		else:
			actual_input_path = selected_path
			LOGGER.info("Gallery resolved path=%s", actual_input_path)
		return _set_selected_input(actual_input_path)

	input_gallery.select(fn=_gallery_to_input_path, outputs=[input_image, selected_input_path])
	input_upload.select(fn=_set_selected_input, inputs=input_upload, outputs=[input_image, selected_input_path])
	input_upload.upload(fn=_set_selected_input, inputs=input_upload, outputs=[input_image, selected_input_path])

	output_dir = str(DEFAULT_OUTPUT_DIR)

	run_event = run_button.click(
		fn=_run_pipeline_from_ui,
		inputs=[
			selected_input_path,
			output_name,
			save_plot,
			input_is_aces,
			seed,
			crop_size,
			keep_padding,
			max_side,
		],
		outputs=[status_output, gr.File(visible=False), gr.File(visible=False), plot_output, outputs_catalog],
	)

	# Wire slider dropdown changes to update slider images
	def _update_slider(dropdown1: str, dropdown2: str, catalog: dict[str, str]) -> tuple[str, str]:
		"""Callback to update slider when dropdowns change."""
		return _get_slider_images(dropdown1, dropdown2, catalog)

	run_event.success(
		fn=_sync_output_controls,
		inputs=[outputs_catalog],
		outputs=[slider_dropdown_1, slider_dropdown_2, img_slider, download_checkbox_group],
	)

	slider_dropdown_1.change(
		fn=_update_slider,
		inputs=[slider_dropdown_1, slider_dropdown_2, outputs_catalog],
		outputs=[img_slider],
	)
	slider_dropdown_2.change(
		fn=_update_slider,
		inputs=[slider_dropdown_1, slider_dropdown_2, outputs_catalog],
		outputs=[img_slider],
	)

	# Wire download button - returns file path for download
	def _handle_download_request(selected: list[str], catalog: dict[str, str]) -> str | None:
		"""Wrapper to handle download request and return file path."""
		return _get_download_zip(selected, catalog)
	
	download_button.click(
		fn=_handle_download_request,
		inputs=[download_checkbox_group, outputs_catalog],
		outputs=download_button,
	)



if __name__ == "__main__":
	# Run warmup before launching the UI to ensure CUDA kernels are initialized
	_warmup_pipeline(
		mapper_checkpoint="outputs/training/mapper/20260425_231537/checkpoints/aces-mapper-20260425_231537-epoch=09.ckpt",
		dequant_checkpoint="outputs/training/dequant/20260422_120606_L1=1.0_L2=0.0_CB=1.0_EA=0.0_TV-huber=0.0/checkpoints/last.ckpt"
	)
	demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
