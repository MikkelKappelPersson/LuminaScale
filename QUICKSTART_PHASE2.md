# Quick Start for New Agent - LuminaScale PyTorch ACES Phase 2

**TL;DR**: Phase 1 (implementation) done. You need to test it, benchmark it, integrate it.

---

## What to Do

### Day 1: Validate Implementation (4 hours)

```bash
# Clone
git clone https://github.com/MikkelKappelPersson/LuminaScale.git
cd LuminaScale

# Setup
pixi install

# Run tests
pixi run pytest tests/test_pytorch_aces_transformer.py -v

# Quick manual test
pixi run python << 'EOF'
from src.luminascale.utils.pytorch_aces_transformer import ACESColorTransformer
import torch

transformer = ACESColorTransformer(device='cuda', use_lut=False)
aces = torch.ones(256, 256, 3, device='cuda')
srgb = transformer.aces_to_srgb_32f(aces)
print(f"✅ Works! Output: {srgb.shape}")
EOF
```

**Success**: All 30+ tests pass, manual test works

### Day 2-3: Benchmark & Integrate (8 hours)

**Create** `scripts/benchmark_pytorch_vs_ocio.py`:
- Compare PyTorch output vs OCIO reference
- Target: PSNR > 28dB (analytical mode)
- Measure speed: expect 3-5× faster

**Integrate** (3 files) - Switch from OCIO GPU to PyTorch:
1. `src/luminascale/utils/io.py` - Add flag to use PyTorch
2. `src/luminascale/utils/dataset_pair_generator.py` - Replace ocio_processor
3. `scripts/generate_on_the_fly_dataset.py` - Same replacement

**Test** integration on real data pipeline

### Day 4: Performance Profile (2 hours)

```bash
# Create benchmark script
pixi run python scripts/profile_color_transforms.py
```

Expected: 2-2.5ms vs 8-11ms (OCIO) = 3-5× speedup

---

## Key Files to Know

**New Code** (Phase 1 - Already Done):
- `src/luminascale/utils/pytorch_aces_transformer.py` (860 lines) - THE CORE
- `tests/test_pytorch_aces_transformer.py` (450 lines) - Tests

**To Update** (Phase 2 - Your Job):
- `src/luminascale/utils/io.py` - aces_to_display_gpu()
- `src/luminascale/utils/dataset_pair_generator.py` - replace ocio_processor
- `scripts/generate_on_the_fly_dataset.py` - same

**Reference Docs**:
- `PHASE_1_COMPLETION_REPORT.md` - What was built
- `ACES_IMPLEMENTATION_MATHEMATICS.md` - Math
- `PHASE_2_HANDOFF_DOCUMENT.md` - Detailed guide (read if stuck)

---

## Core Transformation

```python
# This is what you're replacing everywhere:

# OLD (OpenGL-based, 8-11ms, HPC problems):
processor = GPUTorchProcessor(headless=True)
srgb = processor.apply_ocio_torch(aces_tensor)

# NEW (PyTorch-native, 2-2.5ms, pure CUDA):
transformer = ACESColorTransformer(device=aces_tensor.device, use_lut=False)
srgb = transformer.aces_to_srgb_32f(aces_tensor)
```

---

## Success Metrics

Phase 2 is done when:
- ✅ Tests pass
- ✅ PSNR > 28 dB (accuracy OK in analytical mode)
- ✅ 3-5× speedup measured
- ✅ Integration doesn't break existing code
- ✅ Data pipeline still works with new transforms

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Tests fail | Check `torch.cuda.is_available()` |
| LUT extraction fails | Use `use_lut=False` (analytical mode) |
| Low accuracy | Expected with analytical tone mapping (~1-3% error) |
| Integration breaks | Add fallback flag: `use_pytorch_aces: true` in config |
| GPU memory issues | Reduce batch size or use CPU mode |

---

## Questions?

See `PHASE_2_HANDOFF_DOCUMENT.md` for detailed guide.

Key sections:
- "Environment Setup for Phase 2"
- "Phase 2 Detailed Tasks"
- "Troubleshooting"
- "Integration Testing"

---

## One More Thing

When you're done with Phase 2:
1. Run full test suite one more time
2. Update `PHASE_1_COMPLETION_REPORT.md` with actual benchmark results
3. Create PR or commit summary
4. Ready for Phase 3 (production deployment)

---

**Start with**: Reading PHASE_2_HANDOFF_DOCUMENT.md  
**Then**: Run the test suite (Day 1)  
**Then**: Create benchmark (Day 2-3)  
**Then**: Integrate (Day 3-4)  

Let's go! 🚀
