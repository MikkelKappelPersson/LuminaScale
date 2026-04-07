# Local GPU Testing with Pixi

## Setup

### 1. Install Pixi (if not already installed)
```bash
curl -fsSL https://pixi.sh/install.sh | bash
# Add pixi to PATH
export PATH="$HOME/.pixi/bin:$PATH"
```

### 2. Create Pixi Environment
```bash
cd /path/to/LuminaScale
pixi install
```

This will:
- Install all conda dependencies (PyTorch GPU, Lightning, Hydra, etc.)
- Install PyPI dependencies including `webdataset`, `pyarrow`, `fastparquet`
- Set up CUDA 12.8 environment

### 3. Activate Pixi Environment
```bash
pixi shell
```

Or run commands directly with:
```bash
pixi run python scripts/train_dequantization_net_wds.py ...
```

---

## Test WebDataset Training Locally

### Quick Test (2 shards, 1 batch)
```bash
cd /path/to/LuminaScale

# Option A: Using pixi shell
pixi shell
python scripts/train_dequantization_net_wds.py \
  --config-name=wds \
  'shard_path="dataset/temp/shards/train/train-{000000..000001}.tar"' \
  epochs=1 \
  batch_size=2 \
  num_workers=1

# Option B: Direct pixi run
pixi run python scripts/train_dequantization_net_wds.py \
  --config-name=wds \
  'shard_path="dataset/temp/shards/train/train-{000000..000001}.tar"' \
  epochs=1 \
  batch_size=2 \
  num_workers=1
```

### Expected Output Sequence

1. **Initialization**
   ```
   [MAIN] Starting training initialization...
   [WDS.__init__] Starting initialization...
   [WDS.__init__] ✓ WebDataset initialization complete!
   [MAIN] ✓ WebLoader created
   ```

2. **Data Loading** (you'll see these repeating)
   ```
   [DECODE] Processing sample: 1004_0
   [DECODE] EXR data size: 157170866 bytes
   [DECODE] ✓ Decoded in X.XXms
   ```

3. **Collation**
   ```
   [COLLATE] Collating batch with 2 samples
   [COLLATE]   Sample 0: XXXXXXX bytes
   [COLLATE]   Sample 1: XXXXXXX bytes
   [COLLATE] ✓ Batch ready: 2 samples, returning as (list, list)
   ```

4. **Training Loop** ⭐ **THIS IS THE KEY TEST**
   ```
   [DEBUG_CALLBACK] on_train_start called
   [DEBUG_CALLBACK] on_train_epoch_start called (epoch=0)
   [TRAINING_STEP] ========== BATCH 0 RECEIVED ==========
   [TRAINING_STEP] Processing WebDataset raw byte batch...
   [TRAINING_STEP] GPU Processing took XXXms
   [TRAINING_STEP] About to call model forward pass...
   ```

   ✅ **SUCCESS** if you see `[TRAINING_STEP]` and `GPU Processing` messages
   ❌ **FAILURE** if process hangs/freezes before these messages appear

---

## Troubleshooting on Local GPU

### Issue: CUDA/GPU not Found
```bash
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

If returns `False`:
- Check local GPU drivers: `nvidia-smi`
- Verify CUDA 12.8 compatibility with your GPU
- Try updating GPU drivers

### Issue: OIIO Import Fails
```bash
pixi run python -c "import OpenImageIO; print(OpenImageIO.__version__)"
```

Expected: version 3.1.11 or higher

If fails:
- Check pixi environment: `pixi env info`
- Reinstall: `pixi install --force-reinstall`

### Issue: OutOfMemory (OOM) During EXR Decoding
- Reduce batch_size: `batch_size=1`
- Reduce num_workers: `num_workers=0`
- Use smaller shards (only test-{000000..000000}.tar single shard)

### Issue: Training Loop Hangs (Not Reaching [TRAINING_STEP])
This is the main bug we're testing for. If this happens:
1. Check for `[COLLATE]` messages - if missing, collate function not called
2. Check for any error messages after `Epoch 0: |` line
3. Try with `num_workers=0` to eliminate worker process issues
4. Add more debug output by checking terminal continuously

### Enable Extra Debug Output
If needed, modify `src/luminascale/training/dequantization_trainer.py`:
```python
# In training_step(), add before batch processing:
print(f"[TRAINING_STEP] Full batch structure:")
print(f"  Type: {type(batch)}")
print(f"  Length: {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")
if isinstance(batch, (list, tuple)) and len(batch) > 0:
    print(f"  Element 0 type: {type(batch[0])}")
```

---

## Next: Testing Workflow

### Phase 1: Verify Data Loading ✓
- [ ] `pixi shell` succeeds
- [ ] `pixi install` completes without errors
- [ ] Run test and see `[DECODE]` and `[COLLATE]` messages

### Phase 2: Verify Training Loop (MAIN TEST)
- [ ] See `[TRAINING_STEP]` messages
- [ ] Model forward pass runs
- [ ] Loss is computed and logged
- [ ] First epoch completes

### Phase 3: If Phase 2 Still Hangs
- Need to add more debug to Lightning's DataLoader iteration
- Check if WebLoader is properly yielding batches
- May need to investigate webdataset/Lightning integration

---

## Important Notes

- **Batch Format**: Must be exactly `(exr_bytes_list, metadata_list)` tuple after collation
- **Device**: Code auto-detects GPU, but verify with `nvidia-smi` before running
- **Timeout**: No timeout on local runs, so hangs will be indefinite (use Ctrl+C)
- **Logs**: Training logs saved to `outputs/` directory with timestamps

---

## File Locations Reference

- Entry point: `scripts/train_dequantization_net_wds.py`
- Config: `configs/wds.yaml`
- Data loader: `src/luminascale/data/wds_dataset.py`
- Training: `src/luminascale/training/dequantization_trainer.py`
- GPU pipeline: `src/luminascale/utils/dataset_pair_generator.py`
