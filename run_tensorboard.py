#!/usr/bin/env python
"""Helper script to launch TensorBoard with setuptools pre-loaded and smart defaults."""

import sys
import setuptools  # Pre-load setuptools to ensure pkg_resources is available
from pathlib import Path

# Import tensorboard after setuptools is loaded
from tensorboard.main import run_main

if __name__ == "__main__":
    # Smart defaults: find tensorboard directories and use reload_multifile
    # Check for common tensorboard log locations
    possible_dirs = [
        Path("outputs"),  # Hydra default
        Path("dataset/temp/test_run/tensorboard"),  # Test runs
        Path("checkpoints/tensorboard"),  # Local checkpoints
    ]
    
    log_dir = None
    for path in possible_dirs:
        if path.exists():
            log_dir = str(path.resolve())
            break
    
    # Build tensorboard args
    if log_dir:
        sys.argv = ["tensorboard", f"--logdir={log_dir}", "--reload_multifile=true"]
        print(f"TensorBoard: monitoring {log_dir}")
    else:
        sys.argv = ["tensorboard"]
        print("TensorBoard: no log directory found, using current directory")
    
    run_main()
