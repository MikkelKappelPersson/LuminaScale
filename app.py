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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference"
DEBUG_LOG_PATH = DEFAULT_OUTPUT_DIR / "gradio_debug.log"


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


def run_pipeline(

	input_path: str,
	output_dir: str,
	output_name: str,
	dequant_output_name: str,
	save_dequant: bool,
	save_plot: bool,
	input_is_aces: bool,
	seed: int,
	crop_size: int,
	keep_padding: bool,
	max_side: int,
	# Defaulted internal parameters (removed from UI)
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
) -> tuple[str, str | None, str | None, str | None]:
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

		rfi.write_exr(resolved_pred_aces_output, pred_aces_chw_cpu.numpy())
		LOGGER.info("Saved predicted ACES EXR to %s", resolved_pred_aces_output)
		_save_preview_png(mapper_srgb_chw_cpu, resolved_preview_output)
		LOGGER.info("Saved mapper preview to %s", resolved_preview_output)

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
		return "\n".join(summary_lines), str(resolved_pred_aces_output), dequant_path, plot_path

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
		return details, None, None, None


def _run_pipeline_from_ui(
	selected_input_path: str,
	output_dir: str,
	output_name: str,
	dequant_output_name: str,
	save_dequant: bool,
	save_plot: bool,
	input_is_aces: bool,
	seed: int,
	crop_size: int,
	keep_padding: bool,
	max_side: int,
) -> tuple[str, str | None, str | None, str | None]:
	resolved_input_path = selected_input_path.strip()
	LOGGER.info("UI selected input path=%s", resolved_input_path)
	if not resolved_input_path:
		raise gr.Error("Please select a gallery image or upload an image", duration=5)
	return run_pipeline(
		input_path=resolved_input_path,
		output_dir=output_dir,
		output_name=output_name,
		dequant_output_name=dequant_output_name,
		save_dequant=save_dequant,
		save_plot=save_plot,
		input_is_aces=input_is_aces,
		seed=seed,
		crop_size=crop_size,
		keep_padding=keep_padding,
		max_side=max_side,
	)


default_input = "dataset/full/aces/MIT-Adobe_5K_a0001-jmac_DSC1459.exr"
example_images = sorted(str(p) for p in Path("assets/imgs").glob("*.jpg"))

with gr.Blocks(title="LuminaScale Full Inference") as demo:
	gr.Markdown("# LuminaScale Gradio Inference")
	gr.Markdown("Run dequantization followed by ACES mapping using local checkpoints.")

	with gr.Row():
		with gr.Column(scale=1):
			gr.Markdown("## Select input image")
			selected_input_path = gr.State(default_input)

			input_gallery = gr.Gallery(example_images, label="Example inputs", columns=1, allow_preview=False)
			input_upload = gr.Image(label="Upload input image", type="filepath", )
		with gr.Column(scale=2):
			with gr.Column():
				gr.Markdown("## Input Image")
				input_image = gr.Image(label="Selected input preview", type="filepath", interactive=False, value=_preview_path_for_input(default_input))
				gr.Markdown("## Params")
			with gr.Row():
				output_dir = gr.Textbox(label="Output directory", value=str(DEFAULT_OUTPUT_DIR))
				output_name = gr.Textbox(label="Predicted ACES EXR filename", value="")
				dequant_output_name = gr.Textbox(label="Dequant EXR filename", value="")

			with gr.Row():
				input_is_aces = gr.Checkbox(label="Input is ACES EXR", value=True)
				save_dequant = gr.Checkbox(label="Save dequant intermediate", value=False)
				save_plot = gr.Checkbox(label="Save dashboard plot", value=True)
				keep_padding = gr.Checkbox(label="Keep alignment padding", value=False)

			with gr.Row():
				seed = gr.Number(label="Seed", value=9, precision=0)
				crop_size = gr.Number(label="Crop size", value=0, precision=0)
				max_side = gr.Number(label="Max side", value=1024, precision=0)

	run_button = gr.Button("Run full inference", variant="primary")

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
		LOGGER.info("Gallery resolved path=%s", selected_path)
		return _set_selected_input(selected_path)

	input_gallery.select(fn=_gallery_to_input_path, outputs=[input_image, selected_input_path])
	input_upload.select(fn=_set_selected_input, inputs=input_upload, outputs=[input_image, selected_input_path])
	input_upload.upload(fn=_set_selected_input, inputs=input_upload, outputs=[input_image, selected_input_path])

	status_output = gr.Textbox(label="Status", lines=12)
	debug_log_output = gr.Textbox(label="Debug log path", value=str(DEBUG_LOG_PATH), interactive=False)
	pred_exr_output = gr.Textbox(label="Predicted ACES EXR path")
	dequant_exr_output = gr.Textbox(label="Dequant EXR path (if saved)")
	plot_output = gr.Image(label="Dashboard plot", type="filepath")

	run_button.click(
		fn=_run_pipeline_from_ui,
		inputs=[
			selected_input_path,
			output_dir,
			output_name,
			dequant_output_name,
			save_dequant,
			save_plot,
			input_is_aces,
			seed,
			crop_size,
			keep_padding,
			max_side,
		],
		outputs=[status_output, pred_exr_output, dequant_exr_output, plot_output],
	)



if __name__ == "__main__":
	demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
