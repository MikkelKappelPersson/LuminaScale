"""Multi-GPU (DDP) throughput scaling cell — companion to diag_baseline.py.

Protocol (see med10-journal/experiments/multi-gpu-ddp-throughput.md):
  1/2/4x L40S, decode_in_workers=True (V3 baseline), NW=4, 2-epoch dev.
  Metric: wall s/batch per rank + samples/s (= NGPU / mean gap, batch_size=1).

Controls (env vars, set by the .sbatch submitter):
  NGPU  number of GPUs allocated by sbatch (default 1)
  NW    num_workers per rank (default 4)

Steps/epoch = 40 // NGPU on the dev set (batch_size=1), so the TB-gap epoch
windows are computed from NGPU instead of hardcoded.
"""
import sys, time, os, glob, socket, subprocess, statistics
H = os.environ["HOME"]
ROOT = H + "/projects/LuminaScale"
sys.path.insert(0, ROOT + "/src")
NODE = socket.gethostname()
NGPU = int(os.environ.get("NGPU", "1"))
NW = os.environ.get("NW", "4")
gpu_names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip().splitlines()
print(f"node: {NODE} ngpu: {NGPU} gpu: {gpu_names} nw/rank: {NW}", flush=True)
subprocess.run(["nvidia-smi", "topo", "-m"])  # record topology once per cell

SEP = 40 // NGPU  # steps per epoch on the 40-image dev set, batch_size=1
out_dir = ROOT + f"/outputs/diag_ddp_g{NGPU}_nw{NW}_" + NODE
env = dict(os.environ)
env.pop("MLFLOW_TRACKING_URI", None)
# Rank 0 (this process) builds its datasets BEFORE Lightning initialises the
# process group, so seed WORLD_SIZE the way Lightning seeds its rank-1+
# subprocesses — otherwise rank0's pipeline is built thinking world=1
# (no rank filter, single_node_only splitter) and dies on first iteration.
# NB: do NOT set LOCAL_RANK here — LightningEnvironment treats a present
# LOCAL_RANK as "processes managed externally" and will not spawn ranks 1+.
env["WORLD_SIZE"] = str(NGPU)
# NB: pass as hydra-quoted string ('0,1') — bare commas are ambiguous to hydra
# and a bare '0' coerces to int 0, which Lightning rejects. For NGPU=1 keep
# the config default (auto); Lightning rejects a plain '0' string as well.
device_ids = ",".join(str(i) for i in range(NGPU))
cmd = [sys.executable, "-u", "scripts/train_dequant_net.py", "--config-name=dequant_dev",
       "epochs=2",
       f"num_workers={NW}",
       *( [f"devices='{device_ids}'"] if NGPU > 1 else [] ),
       f"output_dir={out_dir}",
       "shard_path=dataset/shards/ACEScct/dev/shards/train",
       "val_shard_path=dataset/shards/ACEScct/dev/shards/val",
       "metadata_parquet=dataset/shards/ACEScct/dev/training_metadata.parquet"]
# V3 mode (decode_in_workers=True) is the default here — the throughput
# experiment record compares against the V3 L40S baseline (0.43–0.46 s/batch).
# V3=0 falls back to the in-step-decode path (config default).
if os.environ.get("V3", "1") == "1":
    cmd.append("decode_in_workers=True")
t0 = time.time()
logf = open(out_dir + ".run.log", "w")
# Stream training output live (capture_output blinded us to hangs); faulthandler
# lets us kill -ABRT a hung rank and read its Python stack from the stream.
env["PYTHONFAULTHANDLER"] = "1"
env.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
env.setdefault("NCCL_DEBUG", "WARN")
rank0 = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in rank0.stdout:
    logf.write(line); logf.flush()
    if any(k in line for k in ("Epoch 0", "Epoch 1", "it/s", "Error", "Traceback")):
        print(line.rstrip(), flush=True)
r = type("R", (), {"returncode": rank0.wait(), "stdout": "", "stderr": ""})()
logf.close()
print("wall", round(time.time() - t0), "s rc=", r.returncode, flush=True)
if r.returncode != 0:
    print(open(out_dir + ".run.log").read()[-2000:], flush=True)
    sys.exit(0)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
cands = [f for f in glob.glob(out_dir + "/**/events.out.tfevents.*", recursive=True)
         if os.path.getmtime(f) > t0]
ea = EventAccumulator(max(cands, key=os.path.getmtime)); ea.Reload()
tags = [t for t in ea.Tags()["scalars"] if "train" in t.lower() and "loss" in t.lower()]
evs = ea.Scalars(tags[0])
gaps = [(b.step, b.wall_time - a.wall_time) for a, b in zip(evs, evs[1:])]
print(f"total steps: {len(evs)} (steps/epoch = {SEP})", flush=True)
# epoch1: 0 < step < SEP ; epoch2: SEP < step < 2*SEP ; first 5 gaps of each dropped
for lo, hi, label in [(0, SEP, "epoch1-cold"), (SEP, 2 * SEP, "epoch2-warm")]:
    g = [d for s, d in gaps if lo < s < hi][5:]
    if g:
        mean = sum(g) / len(g)
        print(f"{label}: n={len(g)} mean {mean:.3f} median {statistics.median(g):.3f} "
              f"s/batch/rank; min {min(g):.2f} max {max(g):.2f} | "
              f"samples/s {NGPU / mean:.2f}", flush=True)
profs = sorted(glob.glob(out_dir + "/**/fit-training_profile*", recursive=True),
               key=os.path.getmtime)
for line in open(profs[-1]):
    if "run_training_batch" in line:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        print("in-step run_training_batch mean:", parts[2], "s over", parts[3],
              "calls", flush=True)
        break
