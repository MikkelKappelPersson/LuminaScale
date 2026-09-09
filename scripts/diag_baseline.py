"""2-epoch dev throughput benchmark (before/after harness).

Controls (env vars, set by the .sbatch submitter):
  NW   num_workers override (default 1 = config default)
  V3   set to "1" for the decode-in-workers build (changes only the
       output-dir label so before/after cells stay comparable)

All other config is identical to the ratified baseline protocol
(dev shards, epochs=2, per-job output_dir).
"""
import sys, time, os, glob, socket, subprocess, statistics
H = os.environ["HOME"]
ROOT = H + "/projects/LuminaScale"
sys.path.insert(0, ROOT + "/src")
NODE = socket.gethostname()
NW = os.environ.get("NW", "1")
LABEL = os.environ.get("V3") and "v3" or "before"
gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
print(f"node: {NODE} gpu: {gpu} nw: {NW} label: {LABEL}", flush=True)

out_dir = ROOT + f"/outputs/diag_baseline_{LABEL}_nw{NW}_" + NODE
env = dict(os.environ)
env.pop("MLFLOW_TRACKING_URI", None)
cmd = [sys.executable, "-u", "scripts/train_dequant_net.py", "--config-name=dequant_dev",
       "epochs=2",
       f"num_workers={NW}",
       "output_dir=" + out_dir,
       "shard_path=dataset/shards/ACEScct/dev/shards/train",
       "val_shard_path=dataset/shards/ACEScct/dev/shards/val",
       "metadata_parquet=dataset/shards/ACEScct/dev/training_metadata.parquet"]
t0 = time.time()
r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
print("wall", round(time.time()-t0), "s rc=", r.returncode, flush=True)
if r.returncode != 0:
    print("STDOUT tail:", r.stdout[-1500:], flush=True)
    print("STDERR tail:", r.stderr[-1500:], flush=True)
    sys.exit(0)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
cands = [f for f in glob.glob(out_dir + "/**/events.out.tfevents.*", recursive=True) if os.path.getmtime(f) > t0]
ea = EventAccumulator(max(cands, key=os.path.getmtime)); ea.Reload()
tags = [t for t in ea.Tags()["scalars"] if "train" in t.lower() and "loss" in t.lower()]
evs = ea.Scalars(tags[0])
gaps = [(b.step, b.wall_time - a.wall_time) for a, b in zip(evs, evs[1:])]
print("total steps:", len(evs), flush=True)
for lo, hi, label in [(0, 39, "epoch1-cold"), (41, 79, "epoch2-warm")]:
    g = [d for s, d in gaps if lo < s <= hi][5:]
    if g:
        print(f"{label}: n={len(g)} mean {sum(g)/len(g):.3f} median {statistics.median(g):.3f} s/batch; min {min(g):.2f} max {max(g):.2f}", flush=True)
profs = sorted(glob.glob(out_dir + "/**/fit-training_profile*", recursive=True), key=os.path.getmtime)
for line in open(profs[-1]):
    if "run_training_batch" in line:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        print("in-step run_training_batch mean:", parts[2], "s over", parts[3], "calls", flush=True)
        break
