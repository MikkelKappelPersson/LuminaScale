import sys, time, os, glob, socket, subprocess
H = os.environ["HOME"]
ROOT = H + "/projects/LuminaScale"
sys.path.insert(0, ROOT + "/src")
GPU = socket.gethostname()
print("node:", socket.gethostname(), "gpu:", subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(), flush=True)

def tb_gaps(out_dir, t_start):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    cands = [f for f in glob.glob(out_dir + "/**/events.out.tfevents.*", recursive=True) if os.path.getmtime(f) > t_start]
    if not cands:
        return None
    ea = EventAccumulator(max(cands, key=os.path.getmtime)); ea.Reload()
    tags = [t for t in ea.Tags()["scalars"] if "train" in t.lower() and "loss" in t.lower()]
    if not tags:
        return None
    evs = ea.Scalars(tags[0])
    gaps = [b.wall_time - a.wall_time for a, b in zip(evs, evs[1:])]
    if len(gaps) < 10:
        return {"n": len(evs), "mean": 0.0, "note": "only " + str(len(evs)) + " steps", "first5": gaps[:5], "last5": gaps[-5:]}
    steady = gaps[5:]
    return {"n": len(evs), "mean": sum(steady)/len(steady), "first5": [round(g,2) for g in gaps[:5]], "last5": [round(g,2) for g in gaps[-5:]]}

def run(tag, num_workers):
    out_dir = ROOT + "/outputs/diag_" + GPU
    env = dict(os.environ)
    env.pop("MLFLOW_TRACKING_URI", None)
    cmd = [sys.executable, "-u", "scripts/train_dequant_net.py", "--config-name=dequant_dev",
           "epochs=1", "num_workers=" + str(num_workers),
           "output_dir=" + out_dir,
           "shard_path=dataset/shards/ACEScct/dev/shards/train",
           "val_shard_path=dataset/shards/ACEScct/dev/shards/val",
           "metadata_parquet=dataset/shards/ACEScct/dev/training_metadata.parquet"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    print("[" + tag + "] wall", round(wall), "s rc=", r.returncode, flush=True)
    if r.returncode != 0:
        print("STDOUT tail:", r.stdout[-1500:], flush=True)
        print("STDERR tail:", r.stderr[-1500:], flush=True)
        return
    g = tb_gaps(out_dir, t0)
    if g is None:
        print("[" + tag + "] no TB events found in", out_dir, flush=True)
    else:
        print("[" + tag + "] steady", round(g["mean"], 2), "s/batch over", g["n"], "steps; first5", g["first5"], "last5", g["last5"], g.get("note",""), flush=True)

run("nw=1-cold", 1)
run("nw=4-warm", 4)
