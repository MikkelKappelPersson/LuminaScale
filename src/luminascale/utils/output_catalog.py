"""Output catalog management for inference pipelines.

Manages creation, tracking, and cataloging of inference outputs including
EXR files and their sRGB/JPG previews for gallery and download functionality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import numpy as np
from PIL import Image

from . import io as io_utils


class OutputCatalog:
	"""Manages output file creation and metadata tracking.
	
	Handles saving tensors as images (JPG/EXR), converting ACES to sRGB,
	and maintaining a catalog of all outputs with display names and paths.
	"""
	
	def __init__(self, output_dir: Path | str):
		"""Initialize catalog with output directory.
		
		Args:
			output_dir: Directory where all outputs will be saved.
		"""
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.catalog: Dict[str, str] = {}
	
	def _tensor_to_jpg(
		self,
		tensor_chw: torch.Tensor,
		output_path: Path | str,
		quality: int = 95,
	) -> Path:
		"""Convert float32 tensor [C, H, W] to JPG file.
		
		Args:
			tensor_chw: Float32 tensor with shape [C, H, W], values in [0, 1].
			output_path: Path where JPG will be saved.
			quality: JPEG quality (1-100). Default 95 for high quality.
			
		Returns:
			Path to saved JPG file.
		"""
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		
		# Clamp to [0, 1] and convert to numpy
		tensor_np = torch.clamp(tensor_chw, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
		
		# Convert to uint8
		image_u8 = (tensor_np * 255.0).round().astype("uint8")
		
		# Save as JPG
		Image.fromarray(image_u8).save(output_path, "JPEG", quality=quality)
		return output_path
	
	def _aces_tensor_to_srgb_jpg(
		self,
		aces_tensor_chw: torch.Tensor,
		output_path: Path | str,
		device: torch.device,
		quality: int = 95,
	) -> Path:
		"""Convert ACES [C, H, W] tensor to sRGB JPG.
		
		Args:
			aces_tensor_chw: ACES2065-1 float32 tensor [C, H, W].
			output_path: Path where JPG will be saved.
			device: Torch device for computation.
			quality: JPEG quality (1-100). Default 95.
			
		Returns:
			Path to saved JPG file.
		"""
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		
		# Convert ACES [C, H, W] to sRGB using transformer
		# Import from the local module to get the transformer
		from .pytorch_aces_transformer import ACESColorTransformer
		
		aces_hwc = aces_tensor_chw.permute(1, 2, 0).to(device=device, dtype=torch.float32)
		transformer = ACESColorTransformer(device=device, use_lut=True)
		srgb_hwc = transformer.aces_to_srgb_32f(aces_hwc.unsqueeze(0)).squeeze(0)
		srgb_chw = srgb_hwc.permute(2, 0, 1).detach().cpu()
		
		# Save as JPG
		return self._tensor_to_jpg(srgb_chw, output_path, quality=quality)
	
	def add_tensor_output(
		self,
		name: str,
		tensor_chw: torch.Tensor,
		filename: str,
		quality: int = 95,
	) -> str:
		"""Add a tensor output (saved as JPG) to the catalog.
		
		Args:
			name: Display name (e.g., "input", "dequant output").
			tensor_chw: Float32 tensor [C, H, W] with values in [0, 1].
			filename: Filename to save as (e.g., "image_input.jpg").
			quality: JPEG quality (1-100). Default 95.
			
		Returns:
			Absolute path to saved file.
		"""
		output_path = self.output_dir / filename
		self._tensor_to_jpg(tensor_chw, output_path, quality=quality)
		self.catalog[name] = str(output_path)
		return str(output_path)
	
	def add_aces_output(
		self,
		name: str,
		aces_tensor_chw: torch.Tensor,
		exr_filename: str,
		srgb_jpg_filename: str,
		device: torch.device,
		quality: int = 95,
	) -> tuple[str, str]:
		"""Add an ACES output with both EXR and sRGB JPG variants.
		
		Args:
			name: Base display name (e.g., "full output").
			aces_tensor_chw: ACES2065-1 float32 tensor [C, H, W].
			exr_filename: Filename for EXR (e.g., "image_out.exr").
			srgb_jpg_filename: Filename for sRGB JPG (e.g., "image_out_srgb.jpg").
			device: Torch device for ACES→sRGB conversion.
			quality: JPEG quality for JPG variant.
			
		Returns:
			Tuple of (exr_path, srgb_jpg_path).
		"""
		# Save EXR
		exr_path = self.output_dir / exr_filename
		io_utils.write_exr(exr_path, aces_tensor_chw.cpu().numpy())
		self.catalog[name] = str(exr_path)
		
		# Save sRGB JPG variant
		srgb_jpg_path = self._aces_tensor_to_srgb_jpg(
			aces_tensor_chw, self.output_dir / srgb_jpg_filename, device, quality=quality
		)
		self.catalog[f"{name}(srgb)"] = str(srgb_jpg_path)
		
		return str(exr_path), str(srgb_jpg_path)
	
	def add_reference_aces_output(
		self,
		aces_tensor_chw: torch.Tensor,
		exr_filename: str,
		srgb_jpg_filename: str,
		device: torch.device,
		quality: int = 95,
	) -> tuple[str, str]:
		"""Add reference ACES output (EXR + sRGB JPG).
		
		Args:
			aces_tensor_chw: ACES2065-1 float32 tensor [C, H, W].
			exr_filename: Filename for EXR.
			srgb_jpg_filename: Filename for sRGB JPG.
			device: Torch device for conversion.
			quality: JPEG quality for JPG variant.
			
		Returns:
			Tuple of (exr_path, srgb_jpg_path).
		"""
		return self.add_aces_output(
			"reference",
			aces_tensor_chw,
			exr_filename,
			srgb_jpg_filename,
			device,
			quality=quality,
		)
	
	def get_catalog(self) -> dict[str, str]:
		"""Get the complete catalog mapping display names to file paths.
		
		Returns:
			Dictionary with format: {"display_name": "/absolute/path/to/file", ...}
		"""
		return self.catalog.copy()
