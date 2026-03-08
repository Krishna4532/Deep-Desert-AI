"""
src/utils/metrics.py
─────────────────────
Segmentation metrics: mIoU, per-class IoU, pixel accuracy.

Uses a confusion matrix accumulated over the full dataset
for numerically stable results.
"""

from typing import Dict, List, Optional
import numpy as np
import torch


class SegmentationMetrics:
    """
    Streaming confusion-matrix-based metrics accumulator.

    Usage:
        metrics = SegmentationMetrics(num_classes=9, ignore_index=255)
        for preds, targets in dataloader:
            metrics.update(preds, targets)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    # ── state ─────────────────────────────────────────────────────────────────

    def reset(self):
        self._conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds   : [B, H, W] or [B, C, H, W] (argmax is applied if C>1)
            targets : [B, H, W]  long tensor with class IDs
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)

        preds_np   = preds.cpu().numpy().astype(np.int64).flatten()
        targets_np = targets.cpu().numpy().astype(np.int64).flatten()

        # Remove ignored pixels
        valid = targets_np != self.ignore_index
        preds_np   = preds_np[valid]
        targets_np = targets_np[valid]

        # Accumulate into confusion matrix
        np.add.at(
            self._conf,
            (targets_np, preds_np),
            1,
        )

    # ── compute ───────────────────────────────────────────────────────────────

    def compute(self) -> Dict[str, float]:
        conf = self._conf.astype(np.float64)

        tp  = np.diag(conf)                        # [C]
        fp  = conf.sum(axis=0) - tp                # [C]  false positives
        fn  = conf.sum(axis=1) - tp                # [C]  false negatives

        # IoU per class (only classes that appear in GT)
        denom = tp + fp + fn
        iou_per_class = np.where(denom > 0, tp / denom, np.nan)

        # mIoU — mean over classes that actually appear
        valid_mask = ~np.isnan(iou_per_class)
        miou = float(np.nanmean(iou_per_class)) if valid_mask.any() else 0.0

        # Pixel accuracy
        total_correct = tp.sum()
        total_pixels  = conf.sum()
        pixel_acc = float(total_correct / total_pixels) if total_pixels > 0 else 0.0

        # Mean class accuracy
        per_class_acc = np.where(
            conf.sum(axis=1) > 0,
            tp / conf.sum(axis=1),
            np.nan,
        )
        mean_class_acc = float(np.nanmean(per_class_acc))

        return {
            "miou":             miou,
            "pixel_accuracy":   pixel_acc,
            "mean_class_acc":   mean_class_acc,
            "per_class_iou":    iou_per_class.tolist(),   # length = num_classes
            "per_class_acc":    per_class_acc.tolist(),
        }

    def compute_and_reset(self) -> Dict[str, float]:
        results = self.compute()
        self.reset()
        return results

    # ── pretty print ──────────────────────────────────────────────────────────

    def summary(
        self,
        class_names: Optional[List[str]] = None,
        results: Optional[Dict] = None,
    ) -> str:
        if results is None:
            results = self.compute()
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]

        lines = [
            "─" * 50,
            f"  mIoU            : {results['miou']          * 100:.2f} %",
            f"  Pixel Accuracy  : {results['pixel_accuracy']* 100:.2f} %",
            f"  Mean Class Acc  : {results['mean_class_acc']* 100:.2f} %",
            "─" * 50,
            f"  {'Class':<20s}  {'IoU':>8s}  {'Acc':>8s}",
            "─" * 50,
        ]
        for i, name in enumerate(class_names):
            iou = results["per_class_iou"][i]
            acc = results["per_class_acc"][i]
            iou_s = f"{iou*100:>7.2f}%" if not np.isnan(iou) else "    N/A"
            acc_s = f"{acc*100:>7.2f}%" if not np.isnan(acc) else "    N/A"
            lines.append(f"  {name:<20s}  {iou_s}  {acc_s}")
        lines.append("─" * 50)
        return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.dataset.offroad_dataset import CLASS_NAMES
    m = SegmentationMetrics(num_classes=9)

    # Simulate near-perfect predictions
    targets = torch.randint(0, 9, (4, 512, 512))
    preds   = targets.clone()
    preds[0, :50, :50] = (preds[0, :50, :50] + 1) % 9  # add some errors

    m.update(preds, targets)
    results = m.compute()
    print(m.summary(class_names=CLASS_NAMES, results=results))
