"""
eval.py
────────
Evaluate a trained checkpoint on the test set.
Supports Test-Time Augmentation (TTA) via --tta flag.

Usage:
    python eval.py --checkpoint outputs/checkpoints/best.pth
    python eval.py --checkpoint outputs/checkpoints/best.pth --tta
    python eval.py --checkpoint outputs/checkpoints/best.pth --split val
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset.offroad_dataset import build_dataloaders, CLASS_NAMES
from src.models.deeplabv3plus    import build_model
from src.utils.metrics           import SegmentationMetrics
from src.utils.tta               import build_tta
from src.utils.visualize         import save_prediction_grid


@torch.no_grad()
def evaluate(model, loader, device, cfg, save_dir: str = None):
    model.eval()
    metrics = SegmentationMetrics(
        num_classes  = cfg["data"]["num_classes"],
        ignore_index = cfg["data"].get("ignore_index", 255),
    )

    vis_saved = 0
    max_vis   = 5   # save up to 5 visualisation grids

    for batch_idx, (images, masks, meta) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits = model(images)
        preds  = logits.argmax(dim=1)
        metrics.update(preds, masks)

        # Save visualisations
        if save_dir and vis_saved < max_vis:
            vis_path = Path(save_dir) / f"batch_{batch_idx:04d}.png"
            save_prediction_grid(
                images.cpu(), masks.cpu(), preds.cpu(),
                save_path=str(vis_path), max_samples=4,
            )
            vis_saved += 1

        if batch_idx % 10 == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]", end="\r")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--tta",        action="store_true",
                        help="Enable Test-Time Augmentation")
    parser.add_argument("--save_vis",   action="store_true",
                        help="Save prediction visualisations to outputs/")
    args = parser.parse_args()

    # Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override TTA setting from CLI flag
    if args.tta:
        cfg["evaluation"]["tta_enabled"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[Checkpoint] Loaded  (trained epoch={ckpt.get('epoch','?')}  "
          f"best_mIoU={ckpt.get('best_miou',0):.4f})")

    # Optionally wrap with TTA
    model = build_tta(model, cfg)

    # Dataloader for chosen split only
    # We re-use build_dataloaders but only need the requested split
    loaders = build_dataloaders(cfg)
    loader  = loaders[args.split]
    print(f"[Eval] Split: {args.split}  ({len(loader.dataset)} samples)")

    # Evaluate
    tta_tag  = "_tta" if cfg["evaluation"].get("tta_enabled") else ""
    save_dir = None
    if args.save_vis:
        save_dir = Path(cfg["output"]["vis_dir"]) / f"{args.split}{tta_tag}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Eval] Saving visualisations → {save_dir}")

    metrics = evaluate(model, loader, device, cfg, save_dir=str(save_dir) if save_dir else None)
    results = metrics.compute()

    print("\n" + "="*60)
    print(f"  Results on [{args.split}]{tta_tag}")
    print("="*60)
    print(metrics.summary(class_names=CLASS_NAMES, results=results))

    # Highlight the engineering insight (Dry Grass vs Ground Clutter)
    dry_grass_iou     = results["per_class_iou"][3]
    ground_clutter_iou= results["per_class_iou"][8]
    print("\n[Engineering Insight] Known failure case:")
    print(f"  Dry Grass     IoU: {dry_grass_iou*100:.2f}%")
    print(f"  Ground Clutter IoU: {ground_clutter_iou*100:.2f}%")
    if dry_grass_iou < 0.6 or ground_clutter_iou < 0.6:
        print("  → Visual similarity between these classes reduces IoU. "
              "Consider adding more augmentation or class-weighted loss.")


if __name__ == "__main__":
    main()
