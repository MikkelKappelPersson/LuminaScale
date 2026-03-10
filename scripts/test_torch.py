#!/usr/bin/env python3
"""Simple PyTorch test script to verify installation."""

import torch

print("PyTorch Test Script")
print("=" * 50)

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

# Create simple tensors
print("\nTensor operations:")
x = torch.tensor([1.0, 2.0, 3.0])
print(f"CPU tensor: {x}")

if cuda_available:
    x_gpu = x.cuda()
    print(f"GPU tensor: {x_gpu}")
    y_gpu = x_gpu * 2
    print(f"GPU tensor * 2: {y_gpu}")
    print(f"Result on CPU: {y_gpu.cpu()}")

print("\n" + "=" * 50)
print("✓ PyTorch installation successful!")
