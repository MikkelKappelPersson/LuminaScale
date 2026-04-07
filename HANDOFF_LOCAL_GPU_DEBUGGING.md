# Handoff: Local GPU Debugging for WebDataset Pipeline

**Date**: April 7, 2026  
**Branch**: `WebDataset`  
**Current Status**: Freeze identified in Lightning training loop; collate function solution implemented but not yet validated locally

---

## Problem Summary

The training pipeline freezes after Lightning initializes but before `training_step()` is ever called. The freeze occurs during batch iteration, after batches are loaded and decoded.

### Root Cause Identified
WebDataset's `.batched()` function returns raw Python lists of `(exr_bytes, metadata)` tuples. Lightning's trainer expects batches to be properly formatted and doesn't know how to collate these. This causes the data loader to hang indefinitely waiting for proper batch formatting.

### Evidence
- ✅ Batches ARE being decoded (we see `[DECODE]` messages)
- ✅ Batches ARE being created 
- ❌ `[TRAINING_STEP]` is NEVER printed
- ❌ Process hangs after `on_train_epoch_start` before any training

---

## Solution Implemented

### New Collate Function
Added `collate_wds_batch()` in `src/luminascale/data/wds_dataset.py`:
- Converts WebDataset batch format (list of tuples) → `(exr_bytes_list, metadata_list)` tuple
- Separates EXR bytes and metadata for GPU processing
- Added debug logging to track collation

### Modified Files
1. **`src/luminascale/data/wds_dataset.py`**
   - Added `collate_wds_batch()` function
   - Updated `decode_exr_and_json()` docstring to specify return type
   - Added collate function mapping in `LuminaScaleWebDataset.__init__()`
   - Added extensive debug logging

2. **`src/luminascale/training/dequantization_trainer.py`**
   - Enhanced `LuminaScaleModule.training_step()` with batch type detection and debug prints
   - Added device checking in `_process_batch()`
   - Enhanced debug output for batch processing

3. **`scripts/train_dequantization_net_wds.py`**
   - Added `DebugCallback` class to trace training lifecycle
   - Added extensive print statements for initialization flow
   - Integrated callback into trainer

---

## Next Steps for Local GPU Debugging

### 1. Test the Collate Function Fix
```bash
# On local GPU with datasets available
python scripts/train_dequantization_net_wds.py \
  --config-name=wds \
  'shard_path="dataset/temp/shards/train/train-{000000..000001}.tar"' \
  epochs=1 \
  batch_size=2 \
  num_workers=1
```

**Expected Output Flow:**
```
[COLLATE] Collating batch with N samples
[COLLATE] ✓ Batch ready: N samples, returning as (list, list)
[TRAINING_STEP] ========== BATCH 0 RECEIVED ==========
[TRAINING_STEP] Processing WebDataset raw byte batch...
[TRAINING_STEP] GPU Processing took XXXms
[TRAINING_STEP] About to call model forward pass...
```

### 2. If Training Step is Still Not Called
Check these in order:
1. Is the collate function being invoked? (Look for `[COLLATE]` messages)
2. What is the actual batch format after collation? (Add more debug prints)
3. Is Lightning's WebLoader properly iterating? (Check WebLoader logs)

### 3. If GPU Processing Fails
The `generate_batch_from_bytes()` function will need adjustment for:
- OIIO EXR decoding on local GPU (may differ from HPC)
- Device mismatch issues (already added device checking in `_process_batch()`)
- Memory constraints (batch_size may need adjustment)

---

## Key Code Locations

### Data Loading
- Decoder: [`src/luminascale/data/wds_dataset.py`](src/luminascale/data/wds_dataset.py#L16) - `decode_exr_and_json()`
- Collate: [`src/luminascale/data/wds_dataset.py`](src/luminascale/data/wds_dataset.py#L49) - `collate_wds_batch()`
- WebDataset setup: [`src/luminascale/data/wds_dataset.py`](src/luminascale/data/wds_dataset.py#L77) - `LuminaScaleWebDataset.__init__()`

### Training
- Training step: [`src/luminascale/training/dequantization_trainer.py`](src/luminascale/training/dequantization_trainer.py#L332) - `LuminaScaleModule.training_step()`
- Batch processor: [`src/luminascale/training/dequantization_trainer.py`](src/luminascale/training/dequantization_trainer.py#L567) - `DequantizationTrainer._process_batch()`
- GPU pipeline: [`src/luminascale/utils/dataset_pair_generator.py`](src/luminascale/utils/dataset_pair_generator.py#L47) - `generate_batch_from_bytes()`

### Entry Point & Debugging
- Main script: [`scripts/train_dequantization_net_wds.py`](scripts/train_dequantization_net_wds.py)
- Config: [`configs/wds.yaml`](configs/wds.yaml)

---

## Debug Logging Quick Reference

### Frontend (visible in terminal)
- `[MAIN]` - Script initialization
- `[WDS.__init__]` - WebDataset setup
- `[WDS.get_loader]` - WebLoader creation
- `[DECODE]` - Sample decoding
- `[COLLATE]` - Batch collation
- `[DEBUG_CALLBACK]` - Lightning lifecycle
- `[TRAINING_STEP]` - Training loop entry
- `[PROCESS_BATCH]` - GPU batch processing

### Backend (check logs when needed)
- Logger messages with timestamps: `[2026-04-07 HH:MM:SS][module][LEVEL]`

---

## Testing Checklist

- [ ] Collate function runs (see `[COLLATE]` messages)
- [ ] Training step is called (see `[TRAINING_STEP]` messages)
- [ ] GPU batch processing starts (see `[PROCESS_BATCH]` messages)
- [ ] OIIO EXR decoding succeeds on local GPU
- [ ] Loss is computed and backward pass runs
- [ ] Model weights update
- [ ] First epoch completes

---

## Important Notes

1. **Device Management**: The code now checks and updates device in `_process_batch()` to handle Lightning DDP correctly
2. **Batch Format**: Must be `(exr_bytes_list, metadata_list)` tuple for the training loop
3. **GPU Memory**: Large EXR files (157MB+) may cause OOM on some GPUs; adjust batch_size if needed
4. **Timeouts**: Previous HPC runs used 60-second timeout; local runs may be faster but could hang indefinitely if there's a deadlock

---

## Configuration

**Active Config**: `configs/wds.yaml`

Key settings to adjust for local testing:
```yaml
batch_size: 2          # Start small for GPU testing
num_workers: 1         # Single worker on local GPU
epochs: 1              # Just one epoch for quick testing
shard_path: "dataset/temp/shards/train/train-{000000..000001}.tar"  # Use 2 shards for fast iteration
```

---

## Success Criteria

✅ **Local GPU Debug Complete When:**
1. `[TRAINING_STEP]` messages appear in terminal
2. GPU batch processing completes without errors
3. Model forward pass runs
4. Loss is logged
5. First epoch completes successfully

Then re-test on HPC with larger dataset and full shard range.
