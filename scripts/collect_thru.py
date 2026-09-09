"""Recompute per-run throughput stats from run artifacts, classified by config.

Walks every timestamped run dir under outputs/diag_baseline_*, reads the
saved Hydra config (num_workers, decode_in_workers) and reports per-epoch
TB-gap stats + in-step profiler time. Directory labels may be wrong (early
harness labelled before-runs 'v3'); the saved config is authoritative.
"""
import glob, os, statistics
import yaml
ROOT = os.environ["HOME"] + "/projects/LuminaScale"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

rows = []
for out in sorted(glob.glob(ROOT + "/outputs/diag_baseline_*/*")):
    if not os.path.isdir(out):
        continue
    cfgf = glob.glob(out + "/config.yaml") + glob.glob(out + "/.hydra/config.yaml")
    if not cfgf:
        continue
    try:
        cfg = yaml.safe_load(open(cfgf[0]))
    except Exception:
        continue
    nw = cfg.get("num_workers"); v3 = cfg.get("decode_in_workers", False)
    cands = glob.glob(out + "/**/events.out.tfevents.*", recursive=True)
    if not cands:
        continue
    ea = EventAccumulator(max(cands, key=os.path.getmtime)); ea.Reload()
    tags = [t for t in ea.Tags()["scalars"] if "train" in t.lower() and "loss" in t.lower()]
    if not tags:
        continue
    evs = ea.Scalars(tags[0])
    gaps = [(b.step, b.wall_time - a.wall_time) for a, b in zip(evs, evs[1:])]
    line = f"nw={nw} v3={v3} dir={os.path.basename(os.path.dirname(out))[:40]}"
    for lo, hi, lab in [(0, 39, "ep1"), (41, 79, "ep2")]:
        g = [d for s, d in gaps if lo < s <= hi][5:]
        if g:
            line += f" | {lab} {sum(g)/len(g):.3f}/{statistics.median(g):.3f}"
    profs = sorted(glob.glob(out + "/**/fit-training_profile*", recursive=True), key=os.path.getmtime)
    if profs:
        for l in open(profs[-1]):
            if "run_training_batch" in l:
                p = [x.strip() for x in l.split("|") if x.strip()]
                line += f" | in-step {float(p[3])/int(p[2]):.3f} ({p[2]} calls)"
                break
    print(line, flush=True)
