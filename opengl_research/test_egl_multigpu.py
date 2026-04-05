#!/usr/bin/env python3
"""Test EGL multi-GPU support on HPC compute nodes."""

import ctypes
import os
import sys
from ctypes import c_void_p

try:
    from OpenGL.EGL import (
        eglGetDisplay,
        eglInitialize,
        eglGetProcAddress,
        EGL_DEFAULT_DISPLAY,
        EGL_PLATFORM_DEVICE_EXT,
    )
    print("[✓] OpenGL.EGL imports OK\n")
except ImportError as e:
    print(f"[✗] OpenGL import failed: {e}")
    print("    Install: pip install PyOpenGL")
    sys.exit(1)

print("=" * 60)
print("GPU OpenGL Multi-GPU Diagnostic (EGL)")
print("=" * 60)

# ============================================================================
# 1. CHECK CUDA SETUP
# ============================================================================
print("\n1. CUDA GPU Setup:")
print("-" * 60)
os.system("nvidia-smi --query-gpu=index,name --format=csv,noheader")

cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
local_rank = os.environ.get("LOCAL_RANK", "unset")
print(f"\nCUDA_VISIBLE_DEVICES: {cuda_visible or '(all GPUs)'}")
print(f"LOCAL_RANK (from DDP): {local_rank}")

# ============================================================================
# 2. QUERY EGL DEVICES
# ============================================================================
print("\n2. EGL Device Discovery:")
print("-" * 60)
try:
    eglQueryDevicesEXT = eglGetProcAddress("eglQueryDevicesEXT")
    if not eglQueryDevicesEXT:
        print("[✗] eglQueryDevicesEXT not available")
        print("    NVIDIA Driver too old (need >= 358 for multi-GPU EGL)")
        print("    Contact HPC admin to update driver")
        sys.exit(1)
    
    MAX_DEVICES = 4
    egl_devices = (c_void_p * MAX_DEVICES)()
    num_devices = ctypes.c_int()
    
    result = eglQueryDevicesEXT(MAX_DEVICES, egl_devices, ctypes.byref(num_devices))
    
    if result:
        print(f"[✓] EGL detected {num_devices.value} GPU devices")
    else:
        print(f"[✗] eglQueryDevicesEXT returned False")
        sys.exit(1)
except Exception as e:
    print(f"[✗] Device query failed: {e}")
    sys.exit(1)

# ============================================================================
# 3. TEST EGL INITIALIZATION ON EACH DEVICE
# ============================================================================
print("\n3. EGL Initialization per Device:")
print("-" * 60)

all_ok = True
for device_idx in range(num_devices.value):
    try:
        eglGetPlatformDisplayEXT = eglGetProcAddress("eglGetPlatformDisplayEXT")
        if not eglGetPlatformDisplayEXT:
            print(f"Device {device_idx}: [✗] eglGetPlatformDisplayEXT unavailable")
            all_ok = False
            continue
        
        display = eglGetPlatformDisplayEXT(
            EGL_PLATFORM_DEVICE_EXT,
            egl_devices[device_idx],
            0
        )
        
        if not display or display == -1:
            print(f"Device {device_idx}: [✗] eglGetPlatformDisplayEXT returned invalid handle")
            all_ok = False
            continue
        
        major, minor = ctypes.c_int(), ctypes.c_int()
        result = eglInitialize(display, ctypes.byref(major), ctypes.byref(minor))
        
        if result:
            print(f"Device {device_idx}: [✓] EGL {major.value}.{minor.value} initialized")
        else:
            print(f"Device {device_idx}: [✗] eglInitialize failed (EGL_NOT_INITIALIZED)")
            all_ok = False
    except Exception as e:
        print(f"Device {device_idx}: [✗] Exception: {e}")
        all_ok = False

# ============================================================================
# 4. SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)

if all_ok and num_devices.value >= 1:
    print("\n✓ SUCCESS: GPU OpenGL (EGL) is supported on this HPC node")
    print(f"  - {num_devices.value} GPU device(s) detected")
    print("  - Ready for multi-GPU training with DDP + GPU OCIO transforms")
    print("\nNext steps:")
    print("  1. Implement _initialize_egl_multigpu() in gpu_torch_processor.py")
    print("  2. Add get_local_gpu_id() to dataset_pair_generator.py")
    print("  3. Test single-GPU training first")
    print("  4. Then test multi-GPU with DDP")
    sys.exit(0)
else:
    print("\n✗ PROBLEM: GPU OpenGL (EGL) not fully supported")
    
    if num_devices.value == 0:
        print("  - No EGL devices detected (GPU driver issue)")
    else:
        print(f"  - Some devices failed initialization")
    
    print("\nTroubleshooting:")
    print("  1. Check NVIDIA driver version:")
    print("     nvidia-smi --query-gpu=driver_version --format=csv")
    print("  2. Required: driver >= 358 (for multi-GPU EGL support)")
    print("  3. If driver is old, contact HPC admin to update")
    print("  4. For EGL >= 358:")
    print("     - NVIDIA 355–357: EGL ES only (not full OpenGL)")
    print("     - NVIDIA 358+: Desktop OpenGL + multi-GPU")
    print("     - NVIDIA 500+: Recommended")
    
    sys.exit(1)
