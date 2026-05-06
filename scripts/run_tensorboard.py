from __future__ import annotations

import argparse
import os
import sys

from tensorboard.main import run_main


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for TensorBoard launcher."""
    parser = argparse.ArgumentParser(description="Run TensorBoard with LuminaScale defaults")
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.environ.get("TB_LOGDIR", "outputs/training"),
        help="TensorBoard log directory (default: TB_LOGDIR or outputs/training)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=os.environ.get("TB_PORT", "6008"),
        help="TensorBoard port (default: TB_PORT or 6008)",
    )
    parser.add_argument(
        "--bind-all",
        action="store_true",
        default=True,
        help="Bind TensorBoard to all interfaces (default: enabled)",
    )
    parser.add_argument(
        "--load-fast",
        type=str,
        default="false",
        help="TensorBoard load_fast value (default: false)",
    )
    return parser.parse_args()


def main() -> None:
    """Run TensorBoard with parsed options."""
    args = parse_args()

    argv = [
        "tensorboard",
        f"--logdir={args.logdir}",
        f"--port={args.port}",
        f"--load_fast={args.load_fast}",
    ]
    if args.bind_all:
        argv.append("--bind_all")

    sys.argv = argv
    run_main()


if __name__ == "__main__":
    main()
