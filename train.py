"""
Training loop with BCE + 0.5 * (1 - soft IoU), Adam (1e-3), early stopping,
and optional checkpoint resume (model + optimizer + epoch).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from checkpoint_io import torch_load_ckpt
from data_loader import build_dataloaders
from sod_model import build_model


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean soft IoU across batch; returns scalar IoU in [0,1]."""
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    iou_weight: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pos_weight is None:
        bce = torch.nn.functional.binary_cross_entropy(pred, target)
    else:
        # binary_cross_entropy has no pos_weight arg; emulate class weighting:
        # positive pixels get `pos_weight`, background pixels get weight 1.
        w = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
        bce = torch.nn.functional.binary_cross_entropy(pred, target, weight=w)
    iou = soft_iou_loss(pred, target)
    return bce + iou_weight * (1.0 - iou)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_weight: float,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        pred = model(x)
        loss = combined_loss(pred, y, iou_weight=iou_weight, pos_weight=pos_weight)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_iou += soft_iou_loss(pred, y).item() * bs
        n += bs
    return total_loss / max(n, 1), total_iou / max(n, 1)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    best_val_loss: float,
    meta: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "meta": meta,
        },
        path,
    )
    print(f"[checkpoint] saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Adam],
    device: torch.device,
) -> Tuple[int, float]:
    ckpt = torch_load_ckpt(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = int(ckpt.get("epoch", 0))
    best = float(ckpt.get("best_val_loss", float("inf")))
    print(f"[checkpoint] resumed from {path} (epoch={epoch}, best_val_loss={best:.6f})")
    return epoch, best


def train() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--images", type=str, default="images")
    ap.add_argument("--masks", type=str, default="masks")
    ap.add_argument("--image_size", type=int, default=128, choices=(128, 224))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--iou_loss_weight", type=float, default=0.5)
    ap.add_argument(
        "--bce_pos_weight",
        type=float,
        default=15.0,
        help="Positive-class weight for BCE (salient pixels are rare). Set 0 to disable. "
        "~10–20 often helps IoU on ECSSD.",
    )
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Stop if val loss does not improve for this many epochs. Use 0 to disable (train for full --epochs).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=("baseline", "improved", "unet"),
        help="Use 'unet' for best ECSSD metrics (skip U-Net, still no pretrained weights).",
    )
    ap.add_argument("--run_name", type=str, default="run1")
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    ap.add_argument("--save_every_epoch", action="store_true", help="Save full checkpoint after each epoch")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, _, _ = build_dataloaders(
        args.data_root,
        args.images,
        args.masks,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(args.variant, in_ch=3, base=32).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    pw: Optional[torch.Tensor] = None
    if args.bce_pos_weight > 0:
        pw = torch.tensor(args.bce_pos_weight, device=device, dtype=torch.float32)
        print("BCE positive weight:", args.bce_pos_weight)

    start_epoch = 0
    best_val = float("inf")
    run_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.resume:
        start_epoch, best_val = load_checkpoint(args.resume, model, optimizer, device)

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "epochs_ran": [],
        "config": vars(args),
    }
    no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = combined_loss(
                pred, y, iou_weight=args.iou_loss_weight, pos_weight=pw
            )
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            running += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=f"{running/max(seen,1):.4f}")

        train_loss = running / max(seen, 1)
        val_loss, val_iou = validate_one_epoch(
            model, val_loader, device, args.iou_loss_weight, pos_weight=pw
        )
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["epochs_ran"].append(epoch + 1)

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_soft_iou={val_iou:.4f} lr={lr_now:.2e} ({dt:.1f}s)"
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            best_path = os.path.join(run_dir, "best.pt")
            save_checkpoint(best_path, model, optimizer, epoch + 1, best_val, {"variant": args.variant})
        else:
            no_improve += 1

        if args.save_every_epoch:
            ep_path = os.path.join(run_dir, f"epoch_{epoch+1:04d}.pt")
            save_checkpoint(ep_path, model, optimizer, epoch + 1, best_val, {"variant": args.variant})

        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"Early stopping (no val improvement for {args.early_stop_patience} epochs).")
            break

    print("Training finished. Best val_loss:", best_val)


if __name__ == "__main__":
    train()
