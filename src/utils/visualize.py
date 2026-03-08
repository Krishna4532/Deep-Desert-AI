"""
src/utils/visualize.py
───────────────────────
Utilities for turning class-ID masks into colourful overlays
and saving prediction grids for TensorBoard / disk.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


# ── Colour palette ────────────────────────────────────────────────────────────
CLASS_INFO = [
    {"name": "Sky",           "color": (70,  130, 180)},
    {"name": "Landscape",     "color": (124, 179,  66)},
    {"name": "Rocks",         "color": (112, 128, 144)},
    {"name": "Dry Grass",     "color": (210, 180, 140)},
    {"name": "Trees",         "color": ( 34,  85,  34)},
    {"name": "Lush Bushes",   "color": (  0, 168, 107)},
    {"name": "Logs",          "color": (101,  67,  33)},
    {"name": "Flowers",       "color": (255, 105, 180)},
    {"name": "Ground Clutter","color": (169, 169, 169)},
]

# Fast lookup table: index → RGB
_PALETTE = np.array([c["color"] for c in CLASS_INFO], dtype=np.uint8)  # [C, 3]
CLASS_NAMES = [c["name"] for c in CLASS_INFO]


# ── Core colouring ────────────────────────────────────────────────────────────

def mask_to_rgb(mask: np.ndarray, ignore_color: Tuple[int,int,int] = (0,0,0)) -> np.ndarray:
    """
    Convert (H,W) class-ID mask → (H,W,3) RGB image.
    Pixels with value 255 (ignore) are painted `ignore_color`.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    valid = mask < len(CLASS_INFO)
    rgb[valid] = _PALETTE[mask[valid]]

    # Ignored pixels
    ignore = mask == 255
    rgb[ignore] = np.array(ignore_color, dtype=np.uint8)
    return rgb


def overlay_mask(
    image: np.ndarray,    # H W 3  uint8  [0,255]
    mask:  np.ndarray,    # H W    int    class IDs
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend original RGB image with coloured segmentation mask."""
    seg_rgb = mask_to_rgb(mask)
    blended = (
        (1 - alpha) * image.astype(np.float32) +
        alpha       * seg_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return blended


# ── Grid visualisation ────────────────────────────────────────────────────────

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 HWC numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1, 2, 0).numpy()  # CHW → HWC
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


def make_prediction_grid(
    images:  torch.Tensor,    # [B, 3, H, W]  normalised
    gt_masks:torch.Tensor,    # [B, H, W]     class IDs
    pr_masks:torch.Tensor,    # [B, H, W]     class IDs  (argmax already applied)
    max_samples: int = 4,
    cell_pad: int = 4,
) -> np.ndarray:
    """
    Build a side-by-side grid:
        [ Input | Ground-Truth overlay | Prediction overlay ]
    for up to `max_samples` images.

    Returns a single (H, W, 3) uint8 numpy array.
    """
    B = min(images.shape[0], max_samples)
    rows = []

    for i in range(B):
        img_np = _denormalize(images[i])
        gt_np  = gt_masks[i].cpu().numpy()
        pr_np  = pr_masks[i].cpu().numpy()

        gt_overlay = overlay_mask(img_np, gt_np)
        pr_overlay = overlay_mask(img_np, pr_np)

        # Pad cells
        p = cell_pad
        def pad(x):
            return np.pad(x, ((p,p),(p,p),(0,0)), constant_values=255)

        row = np.concatenate([pad(img_np), pad(gt_overlay), pad(pr_overlay)], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    return grid


def save_prediction_grid(
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pr_masks: torch.Tensor,
    save_path: str,
    max_samples: int = 4,
):
    """Save prediction grid PNG to disk."""
    grid = make_prediction_grid(images, gt_masks, pr_masks, max_samples=max_samples)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(save_path)
    print(f"[Visualize] Saved → {save_path}")


# ── Legend ────────────────────────────────────────────────────────────────────

def make_legend(cell_size: int = 30, font_size: int = 16) -> np.ndarray:
    """Generate a class-colour legend as a numpy image."""
    n   = len(CLASS_INFO)
    w   = 250
    h   = n * cell_size + 10
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for i, info in enumerate(CLASS_INFO):
        y0 = 5 + i * cell_size
        y1 = y0 + cell_size - 4
        draw.rectangle([5, y0, 5 + cell_size - 4, y1], fill=info["color"])
        draw.text((5 + cell_size + 4, y0 + 2), info["name"], fill=(0,0,0), font=font)

    return np.array(img)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test mask colouring
    mask = np.random.randint(0, 9, (128, 128), dtype=np.uint8)
    rgb  = mask_to_rgb(mask)
    Image.fromarray(rgb).save("/tmp/test_mask.png")
    print("Saved test mask to /tmp/test_mask.png")

    # Test legend
    legend = make_legend()
    Image.fromarray(legend).save("/tmp/test_legend.png")
    print("Saved legend to /tmp/test_legend.png")
