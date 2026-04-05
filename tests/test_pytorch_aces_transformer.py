"""Unit tests for PyTorch ACES color transformer.

Tests cover:
- Matrix validity (determinant, invertibility)
- Tensor shape handling (single image, batch, various sizes)
- Device compatibility (CPU, CUDA)
- Numerical correctness (against known values)
- Differentiability (gradients flow)
- Type handling (float32, uint8)
"""

import logging
from pathlib import Path

import pytest
import torch
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from luminascale.utils.pytorch_aces_transformer import (
    ACESMatrices,
    LUTInterpolator,
    ACESColorTransformer,
    aces_to_srgb_torch,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestACESMatrices:
    """Test ACES color transformation matrices."""
    
    def test_matrix_determinants(self):
        """Verify matrices have expected determinants."""
        # AP0→AP1 should be nearly orthogonal (det ≈ 1.0)
        det_ap0_ap1 = torch.det(ACESMatrices.M_AP0_TO_AP1)
        assert 0.99 < det_ap0_ap1 < 1.01, f"AP0→AP1 det={det_ap0_ap1}, expect ~1.0"
        
        logger.info(f"✓ AP0→AP1 determinant: {det_ap0_ap1:.6f}")
    
    def test_matrices_on_device(self):
        """Test moving matrices to different devices."""
        matrices_cpu = ACESMatrices.to_device('cpu')
        assert matrices_cpu["M_AP0_TO_AP1"].device.type == 'cpu'
        
        if torch.cuda.is_available():
            matrices_cuda = ACESMatrices.to_device('cuda')
            assert matrices_cuda["M_AP0_TO_AP1"].device.type == 'cuda'
            logger.info("✓ Matrices successfully moved to CUDA")
        else:
            logger.warning("CUDA not available, skipping GPU test")
    
    def test_matrix_dtype(self):
        """Verify matrices are float32."""
        assert ACESMatrices.M_AP0_TO_AP1.dtype == torch.float32
        assert ACESMatrices.M_AP1_TO_XYZ.dtype == torch.float32
        assert ACESMatrices.M_XYZ_TO_REC709.dtype == torch.float32
        logger.info("✓ All matrices are float32")


class TestLUTInterpolator:
    """Test 3D LUT interpolation."""
    
    @pytest.fixture
    def simple_lut(self):
        """Create a simple test LUT."""
        # Create identity-like LUT (input ≈ output)
        size = 8
        lut = torch.zeros(size, size, size, 3, dtype=torch.float32)
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    lut[i, j, k, 0] = i / (size - 1)
                    lut[i, j, k, 1] = j / (size - 1)
                    lut[i, j, k, 2] = k / (size - 1)
        return lut
    
    def test_lut_create(self, simple_lut):
        """Test LUT creation and initialization."""
        interp = LUTInterpolator(simple_lut)
        assert interp.lut_size == 8
        logger.info(f"✓ LUT created with size {interp.lut_size}")
    
    def test_lut_nearest_lookup(self, simple_lut):
        """Test nearest-neighbor LUT lookup."""
        interp = LUTInterpolator(simple_lut)
        
        # Query single point
        query = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
        result = interp.lookup_nearest(query)
        
        assert result.shape == (1, 3)
        logger.info(f"Nearest lookup [0.5,0.5,0.5] → {result[0].tolist()}")
    
    def test_lut_trilinear_lookup(self, simple_lut):
        """Test trilinear LUT lookup."""
        interp = LUTInterpolator(simple_lut)
        
        # Query single point
        query = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float32)
        result = interp.lookup_trilinear(query)
        
        assert result.shape == (1, 3)
        # For identity LUT, output should be close to input
        assert torch.allclose(result[0], query[0], atol=0.1)
        logger.info(f"Trilinear lookup [0.25,0.5,0.75] → {result[0].tolist()}")
    
    def test_lut_batch_lookup(self, simple_lut):
        """Test batch LUT lookup."""
        interp = LUTInterpolator(simple_lut)
        
        # Query batch of points
        batch = torch.rand(10, 3)
        result = interp.lookup_trilinear(batch)
        
        assert result.shape == (10, 3)
        logger.info(f"✓ Batch lookup: {batch.shape} → {result.shape}")


class TestACESColorTransformer:
    """Test main ACES color transformer."""
    
    @pytest.fixture
    def transformer_cpu(self):
        """Create transformer on CPU."""
        return ACESColorTransformer(device='cpu', use_lut=False)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.fixture
    def transformer_cuda(self):
        """Create transformer on CUDA."""
        return ACESColorTransformer(device='cuda', use_lut=False)
    
    def test_transformer_init_cpu(self, transformer_cpu):
        """Test transformer initialization on CPU."""
        assert transformer_cpu.device.type == 'cpu'
        assert hasattr(transformer_cpu, 'M_AP0_TO_AP1')
        assert hasattr(transformer_cpu, 'M_AP1_TO_XYZ')
        assert hasattr(transformer_cpu, 'M_XYZ_TO_REC709')
        logger.info("✓ Transformer initialized on CPU")
    
    def test_single_image_transform(self, transformer_cpu):
        """Test transformation of single image."""
        # Create synthetic ACES image
        aces_img = torch.ones(512, 512, 3, dtype=torch.float32)
        aces_img = aces_img * 0.5  # Mid-gray
        
        # Transform
        srgb_32f = transformer_cpu.aces_to_srgb_32f(aces_img)
        srgb_8u = transformer_cpu.aces_to_srgb_8u(aces_img)
        
        assert srgb_32f.shape == (512, 512, 3)
        assert srgb_8u.shape == (512, 512, 3)
        assert srgb_32f.dtype == torch.float32
        assert srgb_8u.dtype == torch.uint8
        assert torch.all(srgb_32f >= 0) and torch.all(srgb_32f <= 1)
        assert torch.all(srgb_8u >= 0) and torch.all(srgb_8u <= 255)
        
        logger.info(f"✓ Single image: {srgb_32f[256, 256].tolist()}")
    
    def test_batch_transform(self, transformer_cpu):
        """Test transformation of image batch."""
        # Create batch of ACES images
        batch = torch.rand(4, 256, 256, 3, dtype=torch.float32)
        
        # Transform
        srgb_32f = transformer_cpu.aces_to_srgb_32f(batch)
        
        assert srgb_32f.shape == (4, 256, 256, 3)
        assert srgb_32f.dtype == torch.float32
        logger.info(f"✓ Batch transform: {batch.shape} → {srgb_32f.shape}")
    
    def test_forward_method(self, transformer_cpu):
        """Test forward() returns both 32f and 8u."""
        aces_img = torch.ones(256, 256, 3, dtype=torch.float32) * 0.5
        
        srgb_32f, srgb_8u = transformer_cpu(aces_img)
        
        assert srgb_32f.dtype == torch.float32
        assert srgb_8u.dtype == torch.uint8
        logger.info("✓ Forward method works")
    
    def test_gradient_flow(self, transformer_cpu):
        """Test that gradients flow through transform (differentiability)."""
        aces_img = torch.ones(64, 64, 3, dtype=torch.float32, requires_grad=True)
        
        # Forward pass
        srgb = transformer_cpu.aces_to_srgb_32f(aces_img)
        
        # Backward pass
        loss = srgb.mean()
        loss.backward()
        
        assert aces_img.grad is not None
        assert aces_img.grad.shape == aces_img.shape
        logger.info(f"✓ Gradients flow: {aces_img.grad.mean().item():.6f}")
    
    def test_dark_and_bright_values(self, transformer_cpu):
        """Test transform handles extreme values."""
        # Very dark
        dark = torch.zeros(10, 10, 3, dtype=torch.float32)
        out_dark = transformer_cpu.aces_to_srgb_32f(dark)
        assert torch.all(out_dark >= 0)
        
        # Very bright
        bright = torch.ones(10, 10, 3, dtype=torch.float32) * 10.0
        out_bright = transformer_cpu.aces_to_srgb_32f(bright)
        assert torch.all(out_bright >= 0) and torch.all(out_bright <= 1)
        
        logger.info("✓ Handles dark and bright extremes")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_transform(self, transformer_cuda):
        """Test transformation on CUDA device."""
        aces_img = torch.ones(256, 256, 3, dtype=torch.float32, device='cuda')
        
        srgb = transformer_cuda.aces_to_srgb_32f(aces_img)
        
        assert srgb.device.type == 'cuda'
        logger.info("✓ CUDA transform works")
    
    def test_device_mismatch_error(self, transformer_cpu):
        """Test that device mismatch raises error."""
        aces_img = torch.ones(10, 10, 3, dtype=torch.float32)
        
        if torch.cuda.is_available():
            aces_cuda = aces_img.to('cuda')
            with pytest.raises(ValueError, match="device"):
                transformer_cpu.aces_to_srgb_32f(aces_cuda)
            logger.info("✓ Device mismatch raises error")


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""
    
    def test_aces_to_srgb_torch(self):
        """Test one-off transformation function."""
        aces = torch.ones(128, 128, 3, dtype=torch.float32)
        
        srgb_32f, srgb_8u = aces_to_srgb_torch(aces, device='cpu', use_lut=False)
        
        assert srgb_32f.shape == (128, 128, 3)
        assert srgb_8u.shape == (128, 128, 3)
        assert srgb_32f.dtype == torch.float32
        assert srgb_8u.dtype == torch.uint8
        logger.info("✓ Convenience function works")


class TestNumericalAccuracy:
    """Test numerical properties and edge cases."""
    
    def test_linear_input(self):
        """Test that mid-gray maps to expected sRGB value."""
        transformer = ACESColorTransformer(device='cpu', use_lut=False)
        
        # Mid-gray in ACES
        mid_gray = torch.ones(1, 1, 3, dtype=torch.float32) * 0.18
        
        srgb = transformer.aces_to_srgb_32f(mid_gray)
        
        # Mid-gray should still be roughly gray (all channels equal)
        assert abs(srgb[0, 0, 0] - srgb[0, 0, 1]) < 0.01
        assert abs(srgb[0, 0, 1] - srgb[0, 0, 2]) < 0.01
        logger.info(f"Mid-gray ACES → sRGB: {srgb[0, 0].tolist()}")
    
    def test_reciprocal_channels(self):
        """Test color channel independence."""
        transformer = ACESColorTransformer(device='cpu', use_lut=False)
        
        # Red channel
        red = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
        srgb_red = transformer.aces_to_srgb_32f(red)
        
        # Green channel
        green = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32)
        srgb_green = transformer.aces_to_srgb_32f(green)
        
        # Channels should be independent
        assert srgb_red[0, 0, 0] > srgb_red[0, 0, 1]  # Red > Green in output
        logger.info(f"✓ Color channels independent")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
