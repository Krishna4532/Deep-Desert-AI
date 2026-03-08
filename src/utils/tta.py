"""
src/utils/tta.py
─────────────────
Test-Time Augmentation (Solution C).

Strategy: for each test image we run the model on:
  1. Original image
  2. Horizontally-flipped image  → un-flip the prediction → average

Averaging probability maps (not logits) gives a smoother,
more robust segmentation — especially at the edges of bushes
and complex vegetation where a single forward pass tends to be noisy.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAWrapper(nn.Module):
    """
    Wraps any segmentation model with Test-Time Augmentation.

    Supported augmentations (controlled by `flips` argument):
        "horizontal"  — left-right flip
        "vertical"    — top-bottom flip  (less common for ground robots)

    Usage:
        model = DeepLabV3Plus(...)
        tta_model = TTAWrapper(model, flips=["horizontal"])

        with torch.no_grad():
            logits = tta_model(image_batch)   # same interface as the raw model
    """

    def __init__(
        self,
        model: nn.Module,
        flips: List[str] = ("horizontal",),
    ):
        super().__init__()
        self.model = model
        self.flips = list(flips)

        # Validate
        valid = {"horizontal", "vertical"}
        for f in self.flips:
            if f not in valid:
                raise ValueError(f"Unknown TTA flip: '{f}'. Choose from {valid}")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-1])   # flip width

    @staticmethod
    def _vflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-2])   # flip height

    def _augment(self, x: torch.Tensor, flip: str) -> torch.Tensor:
        if flip == "horizontal":
            return self._hflip(x)
        elif flip == "vertical":
            return self._vflip(x)
        raise ValueError(flip)

    def _deaugment(self, probs: torch.Tensor, flip: str) -> torch.Tensor:
        """Un-flip prediction so it aligns with the original orientation."""
        return self._augment(probs, flip)   # flipping is its own inverse

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns averaged logit-equivalent scores [B, C, H, W].
        Averaging is done in probability space then converted back so
        downstream code can still apply argmax directly.
        """
        # Original prediction
        logits = self.model(x)
        probs  = F.softmax(logits, dim=1)   # [B, C, H, W]

        prob_sum = probs
        n        = 1

        for flip in self.flips:
            x_aug         = self._augment(x, flip)
            logits_aug    = self.model(x_aug)
            probs_aug     = F.softmax(logits_aug, dim=1)
            probs_deaug   = self._deaugment(probs_aug, flip)
            prob_sum = prob_sum + probs_deaug
            n        += 1

        avg_probs = prob_sum / n   # [B, C, H, W]

        # Return log-probs so the caller can use argmax or NLL loss as usual
        # (soft-argmax is equivalent to argmax on avg_probs)
        return torch.log(avg_probs.clamp(min=1e-8))


def build_tta(model: nn.Module, cfg: dict) -> nn.Module:
    """
    Returns model wrapped with TTA if enabled in config,
    otherwise returns the raw model.
    """
    eval_cfg = cfg.get("evaluation", {})
    if eval_cfg.get("tta_enabled", False):
        flips = eval_cfg.get("tta_flips", ["horizontal"])
        print(f"[TTA] Enabled — flips: {flips}")
        return TTAWrapper(model, flips=flips)
    return model


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 9, 1)
        def forward(self, x):
            return self.conv(x)

    model = DummyModel()
    tta   = TTAWrapper(model, flips=["horizontal"])

    dummy = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = tta(dummy)
    print(f"TTA output shape: {out.shape}")   # → [2, 9, 128, 128]
