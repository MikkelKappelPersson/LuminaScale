# dataset/ — machine data layout

Type-based layout for training data. This folder is gitignored **except
this README**; the data itself lives per machine as described below.
**eos-ext is the single source of truth** — working slices are propagated
outward to the other machines.

## Layout

```
dataset/
├── raw/      Immutable raw source datasets (MIT-Adobe 5K, PPR10K).
│             Source of truth for every derivation. Never modified.
├── exr/      ACES2065-1 EXR conversions derived from raw/ (the bake
│             stage's input; also holds the inference reference image).
└── shards/   Baked webdataset shards per colour space:
              shards/<COLORSPACE>/{dev,full}/{shards/{train,val,test},
              training_metadata.parquet}. Disposable — reproducible from
              exr/ via scripts/bake_wds_local.sh (override its paths for
              this layout).
```

Training code consumes shards via explicit config overrides
(`shard_path`, `metadata_parquet`) pointing into `shards/...` — see
`src/luminascale/data/wds_dataset.py` (directory mode globs `*.tar`).

## What lives on each machine (2026-09-06)

| Machine | Contents |
|---|---|
| **eos-ext** (source of truth) | everything: `raw/` (MIT-Adobe_5K 46 G, PPR10K 314 G), `exr/ACES2065-1/` (1.3 T), `shards/{ACES2065-1,ACEScct}` (268 G + 186 G) |
| **HPC ai-fe02** (training) | `shards/ACEScct/{dev,full}` (186 G) + `exr/ACES2065-1/` reference image for inference |
| **Desktop** (smoke tests) | `shards/ACES2065-1/dev` (1.3 G) |

## Conventions

- Colour-space variants: `ACES2065-1` (linear) and `ACEScct` (log). The
  training variant is a design decision (see med10-journal OQ3, grade
  space); HPC training currently uses ACEScct.
- `dev` = single-shard smoke sets for pipeline checks; `full` = complete
  80/10/10 bake (max 3 GB per shard).
- New variants are baked on eos-ext, then propagated outward.
- When moving data between machines, preserve this layout exactly.
