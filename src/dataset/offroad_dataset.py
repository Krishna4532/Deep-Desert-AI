"""
src/dataset/offroad_dataset.py
──────────────────────────────
PyTorch Dataset for the Offroad Autonomy segmentation task.

Expected directory layout (after running scripts/prepare_data.py):
    data/processed/
        images/   <frame_id>.png
        masks/    <frame_id>.png   (single-channel, class IDs 0-8)
    data/splits/
        train.txt
        val.txt
        test.txt
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


# ── Colour palette (matches config.yaml) ─────────────────────────────────────
CLASS_INFO: List[Dict] = [
    {"name": "Sky",           "id": 0, "color": (70,  130, 180)},
    {"name": "Landscape",     "id": 1, "color": (124, 179,  66)},
    {"name": "Rocks",         "id": 2, "color": (112, 128, 144)},
    {"name": "Dry Grass",     "id": 3, "color": (210, 180, 140)},
    {"name": "Trees",         "id": 4, "color": ( 34,  85,  34)},
    {"name": "Lush Bushes",   "id": 5, "color": (  0, 168, 107)},
    {"name": "Logs",          "id": 6, "color": (101,  67,  33)},
    {"name": "Flowers",       "id": 7, "color": (255, 105, 180)},
    {"name": "Ground Clutter","id": 8, "color": (169, 169, 169)},
]
NUM_CLASSES  = len(CLASS_INFO)
CLASS_NAMES  = [c["name"]  for c in CLASS_INFO]
CLASS_COLORS = [c["color"] for c in CLASS_INFO]   # list of (R,G,B)

# Pre-build colour → class-id lookup for fast mask decoding
_COLOR_TO_ID: Dict[Tuple[int,int,int], int] = {
    tuple(c["color"]): c["id"] for c in CLASS_INFO
}


# ── Augmentation builders ─────────────────────────────────────────────────────

def build_train_transforms(cfg: dict) -> A.Compose:
    aug = cfg["augmentation"]["train"]
    norm = aug["normalize"]
    return A.Compose([
        A.Resize(*aug["resize"]),
        A.RandomCrop(*aug["random_crop"]),
        A.HorizontalFlip(p=0.5 if aug["horizontal_flip"] else 0.0),
        A.VerticalFlip  (p=0.5 if aug["vertical_flip"]   else 0.0),
        A.ColorJitter(
            brightness=aug["color_jitter"]["brightness"],
            contrast  =aug["color_jitter"]["contrast"],
            saturation=aug["color_jitter"]["saturation"],
            hue       =aug["color_jitter"]["hue"],
            p=0.8,
        ),
        A.Rotate(limit=aug["random_rotation"], p=0.5),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])


def build_val_transforms(cfg: dict) -> A.Compose:
    aug = cfg["augmentation"]["val"]
    norm = aug["normalize"]
    return A.Compose([
        A.Resize(*aug["resize"]),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class OffroadDataset(Dataset):
    """
    Returns:
        image  : FloatTensor [3, H, W], normalised
        mask   : LongTensor  [H, W],    class IDs (0-8), 255 = ignore
        meta   : dict with frame_id, image_path, mask_path
    """

    def __init__(
        self,
        cfg: dict,
        split: str = "train",           # "train" | "val" | "test"
        transform: Optional[A.Compose] = None,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.cfg       = cfg
        self.split     = split
        self.transform = transform
        self.ignore_index = cfg["data"].get("ignore_index", 255)

        data_root   = Path(cfg["data"]["root"])
        splits_dir  = Path(cfg["data"]["splits_dir"])
        self.img_dir  = data_root / cfg["data"]["image_dir"]
        self.mask_dir = data_root / cfg["data"]["mask_dir"]
        self.img_ext  = cfg["data"]["image_ext"]
        self.mask_ext = cfg["data"]["mask_ext"]

        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Run  python scripts/prepare_data.py  first."
            )
        self.frame_ids = [
            line.strip() for line in split_file.read_text().splitlines()
            if line.strip()
        ]
        print(f"[OffroadDataset] {split:5s} → {len(self.frame_ids):>5d} samples")

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_rgb_mask(rgb_mask: np.ndarray) -> np.ndarray:
        """Convert an (H,W,3) RGB segmentation mask → (H,W) class-ID mask."""
        h, w, _ = rgb_mask.shape
        class_mask = np.full((h, w), fill_value=255, dtype=np.uint8)
        for color, class_id in _COLOR_TO_ID.items():
            match = (
                (rgb_mask[:, :, 0] == color[0]) &
                (rgb_mask[:, :, 1] == color[1]) &
                (rgb_mask[:, :, 2] == color[2])
            )
            class_mask[match] = class_id
        return class_mask

    # ── core ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        frame_id = self.frame_ids[idx]

        img_path  = self.img_dir  / f"{frame_id}{self.img_ext}"
        mask_path = self.mask_dir / f"{frame_id}{self.mask_ext}"

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # Load mask — supports both single-channel (grayscale class IDs)
        # and 3-channel (RGB colour-coded) masks
        raw_mask = np.array(Image.open(mask_path))
        if raw_mask.ndim == 3:
            mask = self._decode_rgb_mask(raw_mask)
        else:
            mask = raw_mask.astype(np.uint8)

        # Albumentations expects mask as uint8
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # FloatTensor [3,H,W]
            mask  = augmented["mask"]    # Tensor [H,W]

        mask = mask.long()

        meta = {
            "frame_id":   frame_id,
            "image_path": str(img_path),
            "mask_path":  str(mask_path),
        }
        return image, mask, meta


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Dict[str, DataLoader]:
    """Create train / val / test DataLoaders from config."""
    train_tf = build_train_transforms(cfg)
    val_tf   = build_val_transforms(cfg)

    datasets = {
        "train": OffroadDataset(cfg, split="train", transform=train_tf),
        "val":   OffroadDataset(cfg, split="val",   transform=val_tf),
        "test":  OffroadDataset(cfg, split="test",  transform=val_tf),
    }

    loaders = {}
    for split, ds in datasets.items():
        is_train = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size =cfg["training"]["batch_size"] if is_train
                        else cfg["evaluation"]["batch_size"],
            shuffle    =is_train,
            num_workers=cfg["data"]["num_workers"],
            pin_memory =cfg["data"]["pin_memory"],
            drop_last  =is_train,
        )
    return loaders


# ── Quick sanity-check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ds = OffroadDataset(cfg, split="train", transform=build_val_transforms(cfg))
    img, mask, meta = ds[0]
    print(f"Image shape : {img.shape}  dtype={img.dtype}")
    print(f"Mask  shape : {mask.shape} dtype={mask.dtype}")
    print(f"Unique IDs  : {mask.unique().tolist()}")
    print(f"Frame ID    : {meta['frame_id']}")
