"""
predict.py
───────────
Run inference on a single image (or folder of images).

Usage:
    python predict.py --checkpoint outputs/checkpoints/best.pth \
                      --input path/to/image.png

    python predict.py --checkpoint outputs/checkpoints/best.pth \
                      --input path/to/folder/ \
                      --output outputs/predictions/ \
                      --tta
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.deeplabv3plus import build_model
from src.utils.tta            import build_tta
from src.utils.visualize      import mask_to_rgb, overlay_mask, make_legend, CLASS_NAMES


def build_inference_transform(cfg: dict) -> A.Compose:
    aug  = cfg["augmentation"]["val"]
    norm = aug["normalize"]
    return A.Compose([
        A.Resize(*aug["resize"]),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])


@torch.no_grad()
def predict_single(
    model,
    image_path: str,
    transform: A.Compose,
    device: torch.device,
    original_size: bool = True,
) -> dict:
    """
    Run model on one image.

    Returns:
        {
          "pred_mask":  np.ndarray  (H, W)  class IDs
          "seg_rgb":    np.ndarray  (H, W, 3) coloured mask
          "overlay":    np.ndarray  (H, W, 3) blended image+mask
          "confidence": np.ndarray  (H, W)  max softmax score
        }
    """
    orig = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = orig.shape[:2]

    augmented = transform(image=orig)
    tensor    = augmented["image"].unsqueeze(0).to(device)   # [1,3,H,W]

    logits    = model(tensor)                   # [1, C, H, W]
    probs     = F.softmax(logits, dim=1)        # [1, C, H, W]
    pred      = probs.argmax(dim=1).squeeze(0)  # [H, W]
    conf      = probs.max(dim=1).values.squeeze(0)  # [H, W]

    pred_np = pred.cpu().numpy().astype(np.uint8)
    conf_np = conf.cpu().numpy()

    # Upsample back to original resolution if requested
    if original_size:
        pred_pil = Image.fromarray(pred_np).resize(
            (orig_w, orig_h), Image.NEAREST
        )
        pred_np = np.array(pred_pil)
        conf_pil = Image.fromarray((conf_np * 255).astype(np.uint8)).resize(
            (orig_w, orig_h), Image.BILINEAR
        )
        conf_np = np.array(conf_pil).astype(np.float32) / 255.0

    seg_rgb = mask_to_rgb(pred_np)
    blend   = overlay_mask(orig if original_size else
                           np.array(Image.fromarray(orig).resize(
                               (pred_np.shape[1], pred_np.shape[0])
                           )),
                           pred_np, alpha=0.5)

    return {
        "pred_mask":  pred_np,
        "seg_rgb":    seg_rgb,
        "overlay":    blend,
        "confidence": conf_np,
    }


def save_results(result: dict, save_dir: str, stem: str):
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    Image.fromarray(result["seg_rgb"]).save(out / f"{stem}_seg.png")
    Image.fromarray(result["overlay"]).save(out / f"{stem}_overlay.png")

    # Confidence heat-map (brighter = more confident)
    conf_img = (result["confidence"] * 255).astype(np.uint8)
    Image.fromarray(conf_img, mode="L").save(out / f"{stem}_confidence.png")

    print(f"  Saved: {out / stem}_*.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input",      required=True,
                        help="Single image path or directory of images")
    parser.add_argument("--output",     default="outputs/predictions",
                        help="Directory to save results")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--tta",        action="store_true")
    parser.add_argument("--no_resize",  action="store_true",
                        help="Keep predictions at model output size instead of "
                             "resizing back to original resolution")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.tta:
        cfg["evaluation"]["tta_enabled"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = build_tta(model, cfg)
    model.eval()

    transform = build_inference_transform(cfg)

    # Collect image paths
    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        )
    else:
        image_paths = [input_path]

    print(f"[Predict] Running on {len(image_paths)} image(s)...")

    for img_path in image_paths:
        print(f"  {img_path.name}")
        result = predict_single(
            model, str(img_path), transform, device,
            original_size=not args.no_resize,
        )
        save_results(result, args.output, img_path.stem)

    # Save legend alongside predictions
    legend = make_legend()
    legend_path = Path(args.output) / "class_legend.png"
    Image.fromarray(legend).save(legend_path)
    print(f"\n[Done] Results saved to: {args.output}")
    print(f"       Class legend  : {legend_path}")


if __name__ == "__main__":
    main()
