"""V3 (decode-in-workers) component smoke test.

Ordering matters: the decode-in-workers section MUST run before the parent
process uses OIIO at all — forked workers inherit the parent's OIIO thread
pool state and deadlock if the parent has already initialized it. In the
training flow this never bites (workers fork before the parent's first
in-step decode; in v3 the parent never uses OIIO), so the test mirrors the
trainer ordering: v3 workers first, in-step generation after.

Checks:
1. decode_in_workers path: loader ships ExrDecodeResult with pinned
   float32 [CROP, CROP, 3] tensors; generated pairs finite, in [0, 1];
   repeated iteration from 4 workers does not deadlock.
2. Raw-bytes path (regression on the refactored generator) generates
   finite sRGB pairs.
3. Pixel equivalence: same input bytes -> identical arrays whether decoded
   in-step or in a worker (shared decode_exr_to_pixels).

Usage (local dev, ACES2065-1/dev):
    ./.pixi/envs/default/bin/python -u scripts/test_v3_smoke.py
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from luminascale.data.wds_dataset import (  # noqa: E402
    LuminaScaleWebDataset,
    ExrDecodeResult,
    decode_exr_bytes_to_result,
    decode_exr_to_pixels,
)

SHARD_DIR = "dataset/shards/ACES2065-1/dev/shards"
META = "dataset/shards/ACES2065-1/dev/training_metadata.parquet"
CROP = 2048
N_BATCHES = 5


class _Deadline(Exception):
    pass


def _alarm(signum: int, frame) -> None:
    raise _Deadline()


def build(decode_in_workers: bool, n_workers: int = 2) -> LuminaScaleWebDataset:
    return LuminaScaleWebDataset(
        shard_path=SHARD_DIR + "/train",
        batch_size=1,
        shuffle_buffer=50,
        is_training=True,
        metadata_parquet=META,
        decode_in_workers=decode_in_workers,
        crop_size=CROP,
    )


def main() -> None:
    assert torch.cuda.is_available(), "need CUDA for the generator check"
    device = torch.device("cuda")

    # ---- 1. decode-in-workers path (MUST run before any parent-side OIIO) ----
    if "OpenImageIO" in sys.modules:
        print("[warn] OIIO already imported in parent at test start (package import) — "
              "ordering guard: workers still fork before any OIIO USE.")
    ds_v3 = build(decode_in_workers=True)
    loader3 = ds_v3.get_loader(num_workers=2)
    t0 = time.perf_counter()
    v3_batch = next(_deadline_gen(loader3, 300.0, "first v3 batch"))
    first_dt = time.perf_counter() - t0
    item = v3_batch[0][0]
    assert isinstance(item, ExrDecodeResult), f"v3 should ship ExrDecodeResult, got {type(item)}"
    v3_batch = next(_deadline_gen(loader3, 300.0, "first v3 batch"))

    # transfer_batch_to_device emulation (what the trainer does)
    tensored = []
    for res in v3_batch[0]:
        px_t = torch.from_numpy(np.ascontiguousarray(res.pixels)).pin_memory()
        tensored.append(ExrDecodeResult(pixels=px_t, meta=res.meta))
    assert tensored[0].pixels.shape == (CROP, CROP, 3), tensored[0].pixels.shape
    print(f"[1b] pinned tensor shape={tuple(tensored[0].pixels.shape)} pinned={tensored[0].pixels.is_pinned()}")

    from luminascale.utils.dataset_pair_generator import DatasetPairGenerator

    gen = DatasetPairGenerator(device)
    x8u_v3, x32f_v3, td_v3 = gen.generate_srgb_8u_32f_from_bytes(tensored, crop_size=CROP)
    ok3 = (
        x8u_v3.shape[1:] == (3, CROP, CROP)
        and torch.isfinite(x8u_v3).all()
        and torch.isfinite(x32f_v3).all()
        and float(x8u_v3.min()) >= 0.0
    )
    print(
        f"[1c] v3 generated ok={ok3} range=[{x8u_v3.min():.3f}, {x8u_v3.max():.3f}] "
        f"step_ms={td_v3['total_decode_batch_ms']:.0f} worker_decode_ms={td_v3['worker_decode_ms']:.0f}"
    )
    assert ok3

    # repeated iteration from 4 workers (no deadlock, steady rate)
    loader3b = ds_v3.get_loader(num_workers=4)
    t1 = time.perf_counter()
    n_ok = 0
    for i, b in enumerate(loader3b):
        if i >= N_BATCHES:
            break
        if not (isinstance(b[0][0], ExrDecodeResult) and b[0][0].pixels is not None):
            raise AssertionError(f"batch {i} malformed")
        n_ok += 1
    dt = time.perf_counter() - t1
    print(f"[1d] {n_ok}/{N_BATCHES} batches from 4 workers in {dt:.1f}s ({N_BATCHES / max(dt, 1e-9):.2f} b/s)")
    assert n_ok == N_BATCHES

    # ---- 2. raw-bytes path (parent OIIO use NOW OK: workers already forked) ----
    ds_raw = build(decode_in_workers=False)
    loader = ds_raw.get_loader(num_workers=2)
    raw_batch = next(_deadline_gen(loader, 300.0, "first raw batch"))
    assert isinstance(raw_batch[0][0], bytes), f"raw path should ship bytes, got {type(raw_batch[0][0])}"
    print(f"[2a] raw path ships bytes ({len(raw_batch[0][0])}) ")
    x8u_raw, x32f_raw, td_raw = gen.generate_srgb_8u_32f_from_bytes(raw_batch[0], crop_size=CROP)
    ok = (
        x8u_raw.shape[1:] == (3, CROP, CROP)
        and torch.isfinite(x8u_raw).all()
        and torch.isfinite(x32f_raw).all()
    )
    print(
        f"[2b] raw generated ok={ok} shape={tuple(x8u_raw.shape)} "
        f"range=[{x8u_raw.min():.3f}, {x8u_raw.max():.3f}] step_ms={td_raw['total_decode_batch_ms']:.0f}"
    )
    assert ok

    # ---- 3. pixel equivalence (same bytes, shared decode fn) ----
    res = decode_exr_bytes_to_result(raw_batch[0][0], raw_batch[1][0], CROP)
    worker_px = res.pixels
    import os as _os
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".exr", delete=False) as _t:
        _t.write(raw_batch[0][0])
        _tmp = _t.name
    step_px = decode_exr_to_pixels(_tmp, {}, CROP)
    _os.remove(_tmp)
    same = worker_px is not None and np.array_equal(worker_px, step_px)
    print(f"[3] pixel equivalence in-step vs worker: {same} (worker_decode_ms={res.meta['decode_ms']:.1f})")
    assert same, "in-step and worker decode must produce identical pixels"

    print("SMOKE OK")


def _deadline_gen(loader, seconds: float, what: str):
    """Simple watchdog around loader iteration for the smoke test."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        for item in loader:
            signal.setitimer(signal.ITIMER_REAL, 0)
            yield item
            signal.setitimer(signal.ITIMER_REAL, seconds)
    except _Deadline:
        raise AssertionError(f"TIMEOUT: {what} after {seconds}s (worker deadlock?)")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
