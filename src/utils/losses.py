"""
src/utils/losses.py
────────────────────
Combined Cross-Entropy + Dice loss for semantic segmentation.

Why both?
  • Cross-Entropy  → correct per-pixel probability calibration
  • Dice           → combats class imbalance; small objects (Rocks, Flowers)
                     are not drowned out by the dominant Landscape/Sky classes
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dice Loss ─────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    For each class c:
        dice_c = (2 · ΣpŷI) / (Σp + Σŷ + ε)

    Final loss = 1 − mean(dice_c)   over non-ignored classes.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.smooth        = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [B, C, H, W]  raw (un-softmaxed) predictions
            targets : [B, H, W]     integer class labels
        Returns:
            scalar loss
        """
        # Build valid-pixel mask
        valid = (targets != self.ignore_index)           # [B, H, W]

        # Temporarily replace ignore pixels with 0 so one-hot encoding works
        targets_clean = targets.clone()
        targets_clean[~valid] = 0

        probs = F.softmax(logits, dim=1)                 # [B, C, H, W]
        C = self.num_classes

        # One-hot targets → [B, C, H, W]
        one_hot = F.one_hot(targets_clean, C).permute(0, 3, 1, 2).float()

        # Zero out ignored pixels in both prediction and target
        valid_f = valid.unsqueeze(1).float()             # [B, 1, H, W]
        probs   = probs   * valid_f
        one_hot = one_hot * valid_f

        # Per-class Dice
        dims = (0, 2, 3)   # reduce over batch, H, W
        intersection = (probs * one_hot).sum(dim=dims)   # [C]
        cardinality  = probs.sum(dim=dims) + one_hot.sum(dim=dims)  # [C]

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


# ── Combined Loss ─────────────────────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    """
    loss = α · CrossEntropy  +  β · Dice

    α and β are read from config["training"]["loss"].
    Optionally accepts per-class weights to further boost rare classes.
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        assert abs(ce_weight + dice_weight - 1.0) < 1e-4, \
            "ce_weight + dice_weight should sum to 1.0"

        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Returns a dict with individual + combined losses
        (useful for TensorBoard logging).
        """
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total = self.ce_weight * ce + self.dice_weight * dice
        return {"loss": total, "ce_loss": ce, "dice_loss": dice}


# ── Factory ───────────────────────────────────────────────────────────────────

def build_loss(cfg: dict, device: torch.device) -> SegmentationLoss:
    loss_cfg = cfg["training"]["loss"]
    class_weights = None

    if loss_cfg.get("class_weights") == "balanced":
        # Placeholder — fill in real pixel counts after dataset analysis
        # e.g. weights = total_pixels / (num_classes * per_class_count)
        print("[Loss] class_weights='balanced' detected — "
              "run scripts/compute_class_weights.py to generate weights.")

    return SegmentationLoss(
        num_classes   = cfg["data"]["num_classes"],
        ce_weight     = loss_cfg["cross_entropy_weight"],
        dice_weight   = loss_cfg["dice_weight"],
        class_weights = class_weights,
        ignore_index  = cfg["data"].get("ignore_index", 255),
    ).to(device)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, C, H, W = 2, 9, 128, 128
    logits  = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    targets[0, :10, :10] = 255   # simulate ignored pixels

    criterion = SegmentationLoss(num_classes=C)
    losses = criterion(logits, targets)
    for k, v in losses.items():
        print(f"  {k:12s} = {v.item():.4f}")
