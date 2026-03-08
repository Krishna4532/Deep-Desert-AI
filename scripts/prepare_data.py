"""
scripts/prepare_data.py
────────────────────────
Organise your raw synthetic dataset into the expected directory structure
and create train / val / test split files.

Raw dataset layout (what your simulator exports):
    data/raw/
        FrameXXXX_rgb.png        ← RGB camera frames
        FrameXXXX_seg.png        ← Segmentation masks (RGB colour-coded)
        ...

After running this script:
    data/processed/
        images/   XXXX.png
        masks/    XXXX.png   (single-channel class IDs  0-8)
    data/splits/
        train.txt
        val.txt
        test.txt

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --raw_dir data/raw --split_ratio 0.7 0.15 0.15
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Class colour → ID mapping ─────────────────────────────────────────────────
# Must match config.yaml and offroad_dataset.py
CLASS_COLORS = {
    (70,  130, 180): 0,   # Sky
    (124, 179,  66): 1,   # Landscape
    (112, 128, 144): 2,   # Rocks
    (210, 180, 140): 3,   # Dry Grass
    ( 34,  85,  34): 4,   # Trees
    (  0, 168, 107): 5,   # Lush Bushes
    (101,  67,  33): 6,   # Logs
    (255, 105, 180): 7,   # Flowers
    (169, 169, 169): 8,   # Ground Clutter
}


def rgb_mask_to_class_ids(rgb_mask_path: Path) -> np.ndarray:
    """Convert a colour-coded RGB mask PNG to a single-channel class-ID mask."""
    rgb = np.array(Image.open(rgb_mask_path).convert("RGB"))
    h, w, _ = rgb.shape
    class_mask = np.full((h, w), fill_value=255, dtype=np.uint8)

    for color, class_id in CLASS_COLORS.items():
        match = (
            (rgb[:, :, 0] == color[0]) &
            (rgb[:, :, 1] == color[1]) &
            (rgb[:, :, 2] == color[2])
        )
        class_mask[match] = class_id

    unmatched = (class_mask == 255).sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} pixels not matched in {rgb_mask_path.name}")
    return class_mask


def prepare(
    raw_dir: str  = "data/raw",
    out_dir: str  = "data/processed",
    split_dir: str = "data/splits",
    split_ratio: tuple = (0.70, 0.15, 0.15),
    seed: int = 42,
    rgb_suffix: str  = "_rgb.png",
    seg_suffix: str  = "_seg.png",
):
    raw  = Path(raw_dir)
    out  = Path(out_dir)
    sdir = Path(split_dir)

    img_out  = out / "images"
    mask_out = out / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)

    # Discover frame IDs
    rgb_files = sorted(raw.glob(f"*{rgb_suffix}"))
    if not rgb_files:
        print(f"[prepare_data] No files matching *{rgb_suffix} in {raw_dir}")
        print("  Create some synthetic frames first, then re-run.")
        return

    frame_ids = []
    for rgb_f in tqdm(rgb_files, desc="Converting masks"):
        # Derive frame ID (strip suffix)
        stem = rgb_f.name.replace(rgb_suffix, "")
        seg_f = raw / f"{stem}{seg_suffix}"

        if not seg_f.exists():
            print(f"  SKIP: no seg mask for {rgb_f.name}")
            continue

        # Copy RGB image
        shutil.copy(rgb_f, img_out / f"{stem}.png")

        # Convert colour mask → class ID mask
        class_mask = rgb_mask_to_class_ids(seg_f)
        Image.fromarray(class_mask).save(mask_out / f"{stem}.png")

        frame_ids.append(stem)

    print(f"\n[prepare_data] Processed {len(frame_ids)} frames")

    # Split
    random.seed(seed)
    random.shuffle(frame_ids)
    n = len(frame_ids)
    n_train = int(n * split_ratio[0])
    n_val   = int(n * split_ratio[1])

    splits = {
        "train": frame_ids[:n_train],
        "val":   frame_ids[n_train:n_train + n_val],
        "test":  frame_ids[n_train + n_val:],
    }

    for name, ids in splits.items():
        path = sdir / f"{name}.txt"
        path.write_text("\n".join(ids) + "\n")
        print(f"  {name:5s}: {len(ids):>5d} samples  → {path}")

    print("\n[prepare_data] Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",     default="data/raw")
    parser.add_argument("--out_dir",     default="data/processed")
    parser.add_argument("--split_dir",   default="data/splits")
    parser.add_argument("--split_ratio", nargs=3, type=float,
                        default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb_suffix",  default="_rgb.png")
    parser.add_argument("--seg_suffix",  default="_seg.png")
    args = parser.parse_args()

    prepare(
        raw_dir    =args.raw_dir,
        out_dir    =args.out_dir,
        split_dir  =args.split_dir,
        split_ratio=tuple(args.split_ratio),
        seed       =args.seed,
        rgb_suffix =args.rgb_suffix,
        seg_suffix =args.seg_suffix,
    )
