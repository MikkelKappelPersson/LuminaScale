# Phase 2 Code Integration Reference

**Purpose**: Exact code snippets showing what needs to change in Phase 2

Copy-paste ready with context.

---

## 1. `src/luminascale/utils/io.py`

**Function to Update**: `aces_to_display_gpu()`

### Before (Current - OCIO GPU)

```python
def aces_to_display_gpu(
    aces_tensor: torch.Tensor,
    input_cs: str = "ACES2065-1",
    display: str = "sRGB - Display",
    view: str = "ACES 2.0 - SDR 100 nits (Rec.709)",
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-accelerated ACES-to-display color transform using OCIO."""
    
    from .gpu_torch_processor import GPUTorchProcessor

    if not aces_tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    processor = GPUTorchProcessor(headless=True)
    srgb_32bit, srgb_8bit = processor.apply_ocio_torch(
        aces_tensor,
        input_cs=input_cs,
        display=display,
        view=view,
    )
    processor.cleanup()
    return srgb_32bit, srgb_8bit
```

### After (PyTorch-native with fallback)

```python
def aces_to_display_gpu(
    aces_tensor: torch.Tensor,
    input_cs: str = "ACES2065-1",
    display: str = "sRGB - Display",
    view: str = "ACES 2.0 - SDR 100 nits (Rec.709)",
    use_pytorch: bool = True,  # NEW PARAMETER
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-accelerated ACES-to-display color transform.
    
    Now supports both PyTorch-native and OCIO backends.
    PyTorch is faster (2-3ms) but may have slight accuracy differences.
    OCIO is reference implementation but requires OpenGL/EGL.
    
    Args:
        aces_tensor: ACES2065-1 tensor on CUDA
        use_pytorch: Use PyTorch native transform (default: True)
        ...other args same...
    
    Returns:
        (srgb_32bit, srgb_8bit): Same as before
    """
    
    if use_pytorch:
        # NEW: PyTorch-native implementation
        from .pytorch_aces_transformer import ACESColorTransformer
        
        if not aces_tensor.is_cuda:
            # Can work on CPU too, but slower
            pass
        
        transformer = ACESColorTransformer(
            device=aces_tensor.device,
            use_lut=False  # Analytical tone mapping for now
        )
        
        srgb_32bit = transformer.aces_to_srgb_32f(aces_tensor)
        srgb_8bit = transformer.aces_to_srgb_8u(aces_tensor)
        
    else:
        # FALLBACK: Original OCIO implementation
        from .gpu_torch_processor import GPUTorchProcessor

        if not aces_tensor.is_cuda:
            raise ValueError("Input tensor must be on CUDA device")

        processor = GPUTorchProcessor(headless=True)
        srgb_32bit, srgb_8bit = processor.apply_ocio_torch(
            aces_tensor,
            input_cs=input_cs,
            display=display,
            view=view,
        )
        processor.cleanup()
    
    return srgb_32bit, srgb_8bit
```

**Testing**:
```python
# Test both modes work
aces = torch.randn(256, 256, 3, device='cuda')

# PyTorch (new)
srgb_pytorch = aces_to_display_gpu(aces, use_pytorch=True)

# OCIO (fallback)
srgb_ocio = aces_to_display_gpu(aces, use_pytorch=False)
```

---

## 2. `src/luminascale/utils/dataset_pair_generator.py`

**Class to Update**: `DatasetPairGenerator`

### In `__init__()` method

**Before**:
```python
def __init__(
    self,
    lmdb_env: lmdb.Environment,
    device: torch.device,
    keys_cache: list[str] | None = None,
) -> None:
    """Initialize pair generator."""
    
    self.lmdb_env = lmdb_env
    self.device = device
    self._load_keys()
    
    # Old: OCIO processor
    self.ocio_processor = GPUTorchProcessor(headless=True)
    self.cdl_processor = CDLProcessor(device)
```

**After**:
```python
def __init__(
    self,
    lmdb_env: lmdb.Environment,
    device: torch.device,
    keys_cache: list[str] | None = None,
) -> None:
    """Initialize pair generator."""
    
    self.lmdb_env = lmdb_env
    self.device = device
    self._load_keys()
    
    # New: PyTorch ACES transformer
    from .pytorch_aces_transformer import ACESColorTransformer
    self.pytorch_transformer = ACESColorTransformer(
        device=device, 
        use_lut=False
    )
    self.cdl_processor = CDLProcessor(device)
```

### In `load_aces_and_transform()` method

**Before**:
```python
def load_aces_and_transform(
    self, key: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ACES from LMDB and return both sRGB transforms."""
    
    aces_tensor = self._load_aces_from_lmdb(key).to(self.device, non_blocking=True)
    srgb_32f, srgb_8u = self.ocio_processor.apply_ocio_torch(aces_tensor)
    return srgb_32f, srgb_8u
```

**After**:
```python
def load_aces_and_transform(
    self, key: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ACES from LMDB and return both sRGB transforms."""
    
    aces_tensor = self._load_aces_from_lmdb(key).to(self.device, non_blocking=True)
    srgb_32f, srgb_8u = self.pytorch_transformer(aces_tensor)
    return srgb_32f, srgb_8u
```

### In `load_aces_apply_cdl_and_transform()` method

**Before**:
```python
def load_aces_apply_cdl_and_transform(
    self, key: str, cdl_params: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ACES, apply CDL, and transform to sRGB."""
    
    # ... existing code ...
    
    srgb_32f, srgb_8u = self.ocio_processor.apply_ocio_torch(graded_aces)
```

**After**:
```python
def load_aces_apply_cdl_and_transform(
    self, key: str, cdl_params: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ACES, apply CDL, and transform to sRGB."""
    
    # ... existing code ...
    
    srgb_32f, srgb_8u = self.pytorch_transformer(graded_aces)
```

**Testing**:
```python
# Verify loads still work
import lmdb
env = lmdb.open("dataset/training_data.lmdb", readonly=True)
gen = DatasetPairGenerator(env, device='cuda')
srgb_32f, srgb_8u = gen.load_aces_and_transform(gen._load_keys()[0])
print(f"✅ Works: {srgb_32f.shape}")
```

---

## 3. `scripts/generate_on_the_fly_dataset.py`

**Class to Update**: `OnTheFlyACESDataset`

### In `__init__()` method

**Before**:
```python
class OnTheFlyACESDataset:
    def __init__(self, ...):
        # ... other init code ...
        
        self.ocio_processor = GPUTorchProcessor(headless=True)
```

**After**:
```python
class OnTheFlyACESDataset:
    def __init__(self, ...):
        # ... other init code ...
        
        from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer
        self.pytorch_transformer = ACESColorTransformer(
            device=device,
            use_lut=False
        )
```

### In `iter_batches()` method

**Before**:
```python
def iter_batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Generate batches of (ACES, sRGB) pairs."""
    
    # ... batch collection code ...
    
    srgb_32f, srgb_8u = self.ocio_processor.apply_ocio_torch(graded_aces)
```

**After**:
```python
def iter_batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Generate batches of (ACES, sRGB) pairs."""
    
    # ... batch collection code ...
    
    srgb_32f, srgb_8u = self.pytorch_transformer(graded_aces)
```

---

## 4. Optional: Add Config Flag (for easy switching)

**In** `configs/default.yaml`:

```yaml
# ... existing config ...

# Color transform backend
# use_pytorch_aces: true  -> PyTorch native (fast, 2-3ms)
# use_pytorch_aces: false -> OCIO reference (slow, 8-11ms)
use_pytorch_aces: true
```

**Then in code**:
```python
config = load_config("configs/default.yaml")

aces_srgb = aces_to_display_gpu(
    aces_tensor,
    use_pytorch=config.get("use_pytorch_aces", True)
)
```

---

## Summary of Changes

| File | Change | Complexity |
|------|--------|------------|
| io.py | Add `use_pytorch` flag + conditional | Medium |
| dataset_pair_generator.py | Replace ocio_processor with pytorch_transformer | Easy |
| generate_on_the_fly_dataset.py | Replace ocio_processor with pytorch_transformer | Easy |
| configs/default.yaml | Add `use_pytorch_aces` flag (optional) | Easy |

**Total lines to change**: ~20 lines across 3 files + optional config

---

## Testing Each Change

### Test 1: io.py
```python
from src.luminascale.utils.io import aces_to_display_gpu
import torch

aces = torch.randn(256, 256, 3, device='cuda')

# Test PyTorch path
srgb_pytorch = aces_to_display_gpu(aces, use_pytorch=True)
print(f"✅ PyTorch: {srgb_pytorch[0].shape}")

# Test OCIO path (if available)
try:
    srgb_ocio = aces_to_display_gpu(aces, use_pytorch=False)
    print(f"✅ OCIO: {srgb_ocio[0].shape}")
except Exception as e:
    print(f"⚠️ OCIO not available: {e}")
```

### Test 2: dataset_pair_generator.py
```python
from src.luminascale.utils.dataset_pair_generator import DatasetPairGenerator
import lmdb

env = lmdb.open("dataset/training_data.lmdb", readonly=True)
gen = DatasetPairGenerator(env, device='cuda')

key = gen._load_keys()[0]
srgb_32f, srgb_8u = gen.load_aces_and_transform(key)
print(f"✅ load_aces_and_transform: {srgb_32f.shape}")

# Test with CDL
params = {"slope": [1.0, 1.0, 1.0], "offset": [0.0, 0.0, 0.0]}
srgb_32f, srgb_8u = gen.load_aces_apply_cdl_and_transform(key, params)
print(f"✅ load_aces_apply_cdl_and_transform: {srgb_32f.shape}")
```

### Test 3: generate_on_the_fly_dataset.py
```python
from scripts.generate_on_the_fly_dataset import OnTheFlyACESDataset

dataset = OnTheFlyACESDataset(num_images=100, batch_size=4, device='cuda')

for aces_batch, srgb_batch in dataset.iter_batches():
    print(f"✅ Batch: ACES {aces_batch.shape} → sRGB {srgb_batch.shape}")
    break  # Just test first batch
```

---

## Rollback Plan (if needed)

If PyTorch transformer doesn't work:
1. Revert changes (git checkout)
2. Use `use_pytorch=False` flag to fallback to OCIO
3. File an issue with error details
4. Run original OCIO pipeline while debugging

All changes are isolated and can be reverted independently.

---

## That's It!

Those are all the changes needed for Phase 2. Each is straightforward.

Follow this checklist:
- [ ] Update io.py
- [ ] Update dataset_pair_generator.py
- [ ] Update generate_on_the_fly_dataset.py
- [ ] Test each change
- [ ] Run full test suite
- [ ] Create benchmark
- [ ] Document results

See you in Phase 3! 🚀
