"""Integration guide for improved loss functions and learning rate scheduling.

This document shows how to integrate the new loss functions and schedulers
into the existing dequantization trainer to improve performance.
"""

# STEP 1: In dequantization_trainer.py, update imports:
# ======================================================

from luminascale.training.losses import CombinedDequantizationLoss
from luminascale.training.schedulers import CosineAnnealingWarmupScheduler


# STEP 2: In LuminaScaleModule.__init__(), add after self.automatic_optimization = False:
# =======================================================================================

# Initialize combined loss function
self.loss_fn = CombinedDequantizationLoss(
    use_l2=True,
    use_tv=True,                    # Add Total Variation (smoothness)
    use_perceptual=False,           # Set to True if you have VGG available
    use_banding_aware=True,         # Specifically target banding
    tv_weight=0.1,                  # Adjust based on smoothness preference
    perceptual_weight=0.1,
    banding_weight=0.2,             # Increase to emphasize banding removal
)


# STEP 3: Replace masked_l2_loss call in _train_on_image():
# ===========================================================

# OLD CODE (around line 248):
#   crop_loss = masked_l2_loss(y_hat, y_crop, mask)

# NEW CODE:
loss_components = self.loss_fn(y_hat, y_crop, mask)
crop_loss = loss_components["total"]

# Optionally log individual components:
if batch_idx == 0 and patch_idx == 0:
    logger.info(f"[DEBUG] L2: {loss_components['l2']:.4e}, "
                f"TV: {loss_components['tv']:.4e}, "
                f"Banding: {loss_components['banding']:.4e}")


# STEP 4: Update configure_optimizers() to add learning rate scheduling:
# ======================================================================

def configure_optimizers(self):
    """Configure optimizer with learning rate scheduling."""
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    
    # Add cosine annealing scheduler for better convergence
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_epochs=2,          # Warm up for first 2 epochs
        total_epochs=self.trainer.max_epochs if hasattr(self, 'trainer') else 50,
    )
    
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",  # Update LR every epoch
            "frequency": 1,
        }
    }


# STEP 5 (OPTIONAL): Adjust training config (configs/default.yaml):
# ===================================================================

changes:
  learning_rate: 1e-3          # Can be higher now with scheduling
  epochs: 50                    # Can train longer with scheduling
  # Add new config parameters:
  tv_loss_weight: 0.1          # Total variation weight
  banding_loss_weight: 0.2     # Banding-aware loss weight
  use_perceptual_loss: false   # Set to true for slower but better quality


# STEP 6: Expected improvements:
# ================================

Results you should see:
1. ✓ Loss won't plateau as aggressively (continues decreasing)
2. ✓ Output images less quantized/banded (smoother gradients)
3. ✓ Training may be slower (extra losses) but better quality
4. ✓ Learning rate scheduling helps escape early plateaus

Tuning tips:
- Increase `banding_weight` if you still see banding
- Increase `tv_weight` if output is too smooth/loses detail
- Use `use_perceptual=True` for highest quality (requires VGG downloads)
- Monitor TensorBoard to see each loss component's contribution


# MINIMAL INTEGRATION (quickest fix):
# ====================================

If you want to integrate quickly without full refactoring:

1. Add to imports:
   from luminascale.training.losses import TotalVariationLoss
   
2. In LuminaScaleModule.__init__():
   self.tv_loss = TotalVariationLoss(weight=0.1)
   
3. In _train_on_image() after computing crop_loss:
   tv_penalty = self.tv_loss(y_hat)
   crop_loss = crop_loss + tv_penalty
   
4. Add scheduler to configure_optimizers() - see STEP 4

This minimal approach adds smoothness penalty without restructuring existing code.
"""

# Example: Minimum changes to dequantization_trainer.py
# =======================================================
"""
In dequantization_trainer.py:

1. Add import at top:
   from luminascale.training.losses import TotalVariationLoss
   from luminascale.training.schedulers import CosineAnnealingWarmupScheduler

2. In LuminaScaleModule.__init__():
   self.tv_loss = TotalVariationLoss(weight=0.1)

3. In _train_on_image() method, replace line 248:
   OLD: crop_loss = masked_l2_loss(y_hat, y_crop, mask)
   NEW: l2_loss = masked_l2_loss(y_hat, y_crop, mask)
        tv_penalty = self.tv_loss(y_hat)
        crop_loss = l2_loss + tv_penalty

4. In configure_optimizers():
   OLD: return optim.Adam(self.parameters(), lr=self.learning_rate)
   NEW: optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer, warmup_epochs=2, total_epochs=50
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }
"""
