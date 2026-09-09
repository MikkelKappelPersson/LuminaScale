"""Recompute per-cell throughput stats from run artifacts (TB gaps + profiler).

Prints one line per diag_baseline_* output dir: label, node, per-epoch
mean/median s/batch (first 5 steps dropped, epoch-boundary excluded) and
in-step run_training_batch mean over calls.
"""
import glob, os, statistics, sys
ROOT = os.environ["HOME"] + "/projects/LuminaScale"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for out in sorted(glob.glob(ROOT + "/outputs/diag_baseline_*")):
    name = os.path.basename(out)
    cands = glob.glob(out + "/**/events.out.tfevents.*", recursive=True)
    if not cands:
        print(f"{name}: NO TB EVENTS"); continue
    ea = EventAccumulator(max(cands, key=os.path.getmtime)); ea.Reload()
    tags = [t for t in ea.Tags()["scalars"] if "train" in t.lower() and "loss" in t.lower()]
    if not tags:
        print(f"{name}: NO SCALARS"); continue
    evs = ea.Scalars(tags[0])
    gaps = [(b.step, b.wall_time - a.wall_time) for a, b in zip(evs, evs[1:])]
    parts = [name.replace("diag_baseline_", "")]
    for lo, hi, lab in [(0, 39, "ep1"), (41, 79, "ep2")]:
        g = [d for s, d in gaps if lo < s <= hi][5:]
        if g:
            parts.append(f"{lab}: mean {sum(g)/len(g):.3f} med {statistics.median(g):.3f} (n={len(g)})")
    profs = sorted(glob.glob(out + "/**/fit-training_profile*", recursive=True), key=os.path.getmtime)
    if profs:
        for line in open(profs[-1]):
            if "run_training_batch" in line:
                p = [x.strip() for x in line.split("|") if x.strip()]
                parts.append(f"in-step {p[2]}s/{p[3]}calls")
                break
    print(" | ".join(parts), flush=True)
