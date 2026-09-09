import sys, time, os, glob, socket
H = os.environ["HOME"]
sys.path.insert(0, H + "/projects/LuminaScale/src")
print("node:", socket.gethostname(), flush=True)

# raw CephFS reference on a fresh shard (cold)
t0 = time.time()
n = 0
with open(H + "/projects/LuminaScale/dataset/shards/ACEScct/full/shards/train/train-000039.tar", "rb") as f:
    while True:
        b = f.read(1024*1024)
        if not b: break
        n += len(b)
dt = time.time() - t0
print(f"raw cold seq read: {n/1e6:.0f} MB in {dt:.1f}s = {n/1e6/dt:.0f} MB/s", flush=True)

import webdataset as wds
from luminascale.data.wds_dataset import decode_exr_and_json, collate_wds_batch

def run(tag, num_workers, pat):
    shards = sorted(glob.glob(H + "/projects/LuminaScale/dataset/shards/ACEScct/full/shards/train/" + pat))
    ds = wds.WebDataset(shards, resampled=False, empty_check=False, shardshuffle=100)
    ds = ds.select(wds.shardlists.split_by_worker)
    ds = ds.shuffle(50)
    ds = ds.map(decode_exr_and_json).batched(1).map(collate_wds_batch)
    t0 = time.time()
    it = iter(wds.WebLoader(ds, batch_size=None, num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers>0, prefetch_factor=2 if num_workers>0 else None))
    times = []
    for i in range(40):
        t1 = time.time()
        b = next(it)
        times.append(time.time() - t1)
    total = time.time() - t0
    excl = total - times[0]
    print(f"{tag}: total {total:.1f}s (first batch {times[0]:.1f}s incl. buffer fill + worker start), steady mean {excl/39:.2f}s/batch", flush=True)
    print(f"   per-batch: {[round(t,2) for t in times[:8]]} ... {[round(t,2) for t in times[-4:]]}", flush=True)

run("nw=1", 1, "train-00004[0-5].tar")
run("nw=4", 4, "train-00004[6-9].tar")
run("nw=8", 8, "train-00005[0-3].tar")
