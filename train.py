"""
train.py
─────────
Main training script for Offroad Autonomy Semantic Segmentation.

Usage:
    python train.py                           # uses configs/config.yaml
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume outputs/checkpoints/last.pth
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset.offroad_dataset import build_dataloaders, CLASS_NAMES
from src.models.deeplabv3plus    import build_model
from src.utils.losses            import build_loss
from src.utils.metrics           import SegmentationMetrics
from src.utils.visualize         import save_prediction_grid


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Optimiser & Scheduler ─────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg["training"]["optimizer"]
    lr      = opt_cfg["lr"]
    wd      = opt_cfg["weight_decay"]
    mult    = opt_cfg["backbone_lr_multiplier"]

    param_groups = model.parameter_groups(base_lr=lr, backbone_multiplier=mult)
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sched_cfg    = cfg["training"]["scheduler"]
    total_epochs = cfg["training"]["epochs"]
    warmup       = sched_cfg["warmup_epochs"]
    min_lr       = sched_cfg["min_lr"]

    warmup_steps = warmup * steps_per_epoch
    total_steps  = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1 + np.cos(np.pi * progress))
        # Scale so LR doesn't drop below min_lr
        base_lr = cfg["training"]["optimizer"]["lr"]
        return max(cosine, min_lr / base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"[Checkpoint] Resumed from epoch {ckpt.get('epoch', '?')}  "
          f"(best mIoU={ckpt.get('best_miou', 0):.4f})")
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)


# ── One epoch of training ─────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scheduler,
    criterion, scaler, device, cfg, epoch, writer,
):
    model.train()
    accum_steps = cfg["training"]["accumulate_grad_batches"]
    log_every   = cfg["output"]["log_every_n_steps"]
    use_amp     = cfg["training"]["mixed_precision"]
    clip_val    = cfg["training"]["gradient_clip"]

    running = {"loss": 0.0, "ce_loss": 0.0, "dice_loss": 0.0}
    optimizer.zero_grad()

    for step, (images, masks, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            losses = criterion(logits, masks)
            loss   = losses["loss"] / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        for k in running:
            running[k] += losses[k].item()

        global_step = epoch * len(loader) + step
        if step % log_every == 0:
            lr = optimizer.param_groups[-1]["lr"]
            avg_loss = running["loss"] / (step + 1)
            print(f"  Epoch {epoch:03d} [{step+1:04d}/{len(loader):04d}]  "
                  f"loss={avg_loss:.4f}  lr={lr:.2e}")
            if writer:
                writer.add_scalar("train/loss",      losses["loss"].item(),    global_step)
                writer.add_scalar("train/ce_loss",   losses["ce_loss"].item(), global_step)
                writer.add_scalar("train/dice_loss", losses["dice_loss"].item(),global_step)
                writer.add_scalar("train/lr",        lr,                       global_step)

    n = len(loader)
    return {k: v / n for k, v in running.items()}


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, cfg, epoch, writer):
    model.eval()
    metrics   = SegmentationMetrics(
        num_classes  = cfg["data"]["num_classes"],
        ignore_index = cfg["data"].get("ignore_index", 255),
    )
    val_loss  = 0.0
    vis_batch = None   # save first batch for visualisation

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits = model(images)
        losses = criterion(logits, masks)
        val_loss += losses["loss"].item()

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

        if vis_batch is None:
            vis_batch = (images.cpu(), masks.cpu(), preds.cpu())

    results = metrics.compute()
    avg_loss = val_loss / len(loader)

    print(metrics.summary(class_names=CLASS_NAMES, results=results))
    print(f"  Val loss: {avg_loss:.4f}")

    if writer:
        writer.add_scalar("val/loss",           avg_loss,             epoch)
        writer.add_scalar("val/miou",           results["miou"],      epoch)
        writer.add_scalar("val/pixel_accuracy", results["pixel_accuracy"], epoch)
        for i, name in enumerate(CLASS_NAMES):
            iou = results["per_class_iou"][i]
            if not np.isnan(iou):
                writer.add_scalar(f"val/iou_{name}", iou, epoch)

    # Save visualisation
    vis_every = cfg["output"]["vis_every_n_epochs"]
    if vis_batch and (epoch % vis_every == 0 or epoch == 0):
        vis_path = Path(cfg["output"]["vis_dir"]) / f"epoch_{epoch:03d}.png"
        save_prediction_grid(*vis_batch, save_path=str(vis_path), max_samples=4)

    return results, avg_loss


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # Data
    loaders = build_dataloaders(cfg)

    # Model
    model = build_model(cfg).to(device)
    print(f"[Model] DeepLabV3+ with ResNet-50 backbone")

    # Loss, optimiser, scheduler
    criterion = build_loss(cfg, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(loaders["train"]))
    scaler    = GradScaler(enabled=cfg["training"]["mixed_precision"])

    # Resume
    start_epoch = 0
    best_miou   = 0.0
    if args.resume:
        start_epoch, best_miou = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    # TensorBoard
    log_dir = Path(cfg["output"]["log_dir"]) / cfg["project"]["experiment"]
    writer  = SummaryWriter(log_dir=str(log_dir))
    print(f"[TensorBoard] tensorboard --logdir {log_dir}")

    # ── Training loop ─────────────────────────────────────────────────────────
    total_epochs = cfg["training"]["epochs"]
    ckpt_dir     = Path(cfg["output"]["checkpoint_dir"])

    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch+1} / {total_epochs}")
        print(f"{'='*60}")

        # Train
        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, scheduler,
            criterion, scaler, device, cfg, epoch, writer,
        )

        # Validate
        val_results, val_loss = validate(
            model, loaders["val"], criterion, device, cfg, epoch, writer,
        )

        miou = val_results["miou"]
        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            print(f"  ★  New best mIoU: {best_miou:.4f}")

        # Save checkpoint
        state = {
            "epoch":           epoch + 1,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou":       best_miou,
            "config":          cfg,
        }
        save_checkpoint(state, str(ckpt_dir / "last.pth"))
        if is_best:
            save_checkpoint(state, str(ckpt_dir / "best.pth"))

    writer.close()
    print(f"\n[Done] Best validation mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
