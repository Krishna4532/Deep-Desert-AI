"""
scripts/generate_synthetic_data.py
────────────────────────────────────
Generates fake desert scene RGB images + matching colour-coded
segmentation masks so you can test the full pipeline without a simulator.

Produces N frames in data/raw/ with the naming convention:
    FrameXXXX_rgb.png   ← procedurally-painted desert scene
    FrameXXXX_seg.png   ← matching RGB colour-coded mask

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --n 200 --size 640 480
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── Class colours (must match config.yaml) ───────────────────────────────────
CLASSES = {
    "Sky":            (70,  130, 180),
    "Landscape":      (124, 179,  66),
    "Rocks":          (112, 128, 144),
    "Dry Grass":      (210, 180, 140),
    "Trees":          ( 34,  85,  34),
    "Lush Bushes":    (  0, 168, 107),
    "Logs":           (101,  67,  33),
    "Flowers":        (255, 105, 180),
    "Ground Clutter": (169, 169, 169),
}


def make_frame(width: int, height: int, seed: int):
    """Generate one (rgb_image, seg_mask) pair as PIL Images."""
    rng = random.Random(seed)
    np.random.seed(seed)

    rgb  = Image.new("RGB", (width, height))
    seg  = Image.new("RGB", (width, height))
    rd   = ImageDraw.Draw(rgb)
    sd   = ImageDraw.Draw(seg)

    # ── Sky (top ~35-45% of image) ───────────────────────────────────────────
    horizon = int(height * rng.uniform(0.35, 0.45))
    sky_top    = tuple(np.clip(np.array(CLASSES["Sky"])    + rng.randint(-20, 20), 0, 255).tolist())
    ground_col = tuple(np.clip(np.array(CLASSES["Dry Grass"]) + rng.randint(-15, 15), 0, 255).tolist())

    rd.rectangle([0, 0, width, horizon], fill=sky_top)
    sd.rectangle([0, 0, width, horizon], fill=CLASSES["Sky"])

    # ── Ground (below horizon) ───────────────────────────────────────────────
    rd.rectangle([0, horizon, width, height], fill=ground_col)
    sd.rectangle([0, horizon, width, height], fill=CLASSES["Dry Grass"])

    # ── Landscape patches ────────────────────────────────────────────────────
    for _ in range(rng.randint(2, 5)):
        x0 = rng.randint(0, width)
        y0 = rng.randint(horizon, height)
        x1 = x0 + rng.randint(60, 200)
        y1 = y0 + rng.randint(20, 80)
        c  = tuple(np.clip(np.array(CLASSES["Landscape"]) + rng.randint(-20, 20), 0, 255).tolist())
        rd.ellipse([x0, y0, x1, y1], fill=c)
        sd.ellipse([x0, y0, x1, y1], fill=CLASSES["Landscape"])

    # ── Trees (tall, near horizon) ───────────────────────────────────────────
    for _ in range(rng.randint(1, 4)):
        tx = rng.randint(0, width)
        ty = horizon - rng.randint(60, 140)
        tw = rng.randint(25, 55)
        th = rng.randint(80, 160)
        trunk_c = tuple(np.clip(np.array(CLASSES["Logs"]) + rng.randint(-10, 10), 0, 255).tolist())
        # trunk
        rd.rectangle([tx-5, ty+th//2, tx+5, ty+th], fill=trunk_c)
        sd.rectangle([tx-5, ty+th//2, tx+5, ty+th], fill=CLASSES["Logs"])
        # canopy
        tree_c = tuple(np.clip(np.array(CLASSES["Trees"]) + rng.randint(-15, 15), 0, 255).tolist())
        rd.ellipse([tx-tw, ty, tx+tw, ty+th//2+20], fill=tree_c)
        sd.ellipse([tx-tw, ty, tx+tw, ty+th//2+20], fill=CLASSES["Trees"])

    # ── Lush Bushes ──────────────────────────────────────────────────────────
    for _ in range(rng.randint(2, 6)):
        bx = rng.randint(0, width)
        by = rng.randint(horizon, height - 30)
        bw = rng.randint(30, 90)
        bh = rng.randint(20, 55)
        bc = tuple(np.clip(np.array(CLASSES["Lush Bushes"]) + rng.randint(-20, 20), 0, 255).tolist())
        rd.ellipse([bx, by, bx+bw, by+bh], fill=bc)
        sd.ellipse([bx, by, bx+bw, by+bh], fill=CLASSES["Lush Bushes"])

    # ── Rocks ────────────────────────────────────────────────────────────────
    for _ in range(rng.randint(3, 10)):
        rx = rng.randint(0, width)
        ry = rng.randint(horizon + 10, height - 10)
        rs = rng.randint(10, 45)
        rc = tuple(np.clip(np.array(CLASSES["Rocks"]) + rng.randint(-25, 25), 0, 255).tolist())
        rd.ellipse([rx, ry, rx+rs, ry+rs//2], fill=rc)
        sd.ellipse([rx, ry, rx+rs, ry+rs//2], fill=CLASSES["Rocks"])

    # ── Logs ─────────────────────────────────────────────────────────────────
    for _ in range(rng.randint(0, 3)):
        lx = rng.randint(0, width - 80)
        ly = rng.randint(horizon + 20, height - 20)
        lc = tuple(np.clip(np.array(CLASSES["Logs"]) + rng.randint(-15, 15), 0, 255).tolist())
        rd.rectangle([lx, ly, lx + rng.randint(50, 120), ly + rng.randint(8, 18)], fill=lc)
        sd.rectangle([lx, ly, lx + rng.randint(50, 120), ly + rng.randint(8, 18)], fill=CLASSES["Logs"])

    # ── Flowers (small dots) ─────────────────────────────────────────────────
    for _ in range(rng.randint(0, 12)):
        fx = rng.randint(0, width)
        fy = rng.randint(horizon + 10, height)
        fs = rng.randint(4, 12)
        fc = tuple(np.clip(np.array(CLASSES["Flowers"]) + rng.randint(-30, 30), 0, 255).tolist())
        rd.ellipse([fx, fy, fx+fs, fy+fs], fill=fc)
        sd.ellipse([fx, fy, fx+fs, fy+fs], fill=CLASSES["Flowers"])

    # ── Ground Clutter (scattered small patches) ─────────────────────────────
    for _ in range(rng.randint(5, 15)):
        cx = rng.randint(0, width)
        cy = rng.randint(horizon + 5, height)
        cs = rng.randint(5, 20)
        cc = tuple(np.clip(np.array(CLASSES["Ground Clutter"]) + rng.randint(-20, 20), 0, 255).tolist())
        rd.rectangle([cx, cy, cx+cs, cy+cs//2], fill=cc)
        sd.rectangle([cx, cy, cx+cs, cy+cs//2], fill=CLASSES["Ground Clutter"])

    # Slight blur on RGB to look more natural (not on seg mask!)
    rgb = rgb.filter(ImageFilter.GaussianBlur(radius=1))

    return rgb, seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int,   default=100,
                        help="Number of frames to generate (default: 100)")
    parser.add_argument("--size",    type=int,   nargs=2, default=[640, 480],
                        metavar=("W", "H"),
                        help="Image size in pixels (default: 640 480)")
    parser.add_argument("--out_dir", type=str,   default="data/raw")
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    W, H = args.size
    print(f"Generating {args.n} synthetic desert frames ({W}×{H}) → {out}/")

    for i in range(args.n):
        frame_id = f"Frame{i+1:04d}"
        rgb, seg = make_frame(W, H, seed=args.seed + i)
        rgb.save(out / f"{frame_id}_rgb.png")
        seg.save(out / f"{frame_id}_seg.png")

        if (i + 1) % 10 == 0 or (i + 1) == args.n:
            print(f"  [{i+1:>4d}/{args.n}] done")

    print(f"\nDone! {args.n} frame pairs saved to {out}/")
    print("Next step:  python scripts\\prepare_data.py")


if __name__ == "__main__":
    main()