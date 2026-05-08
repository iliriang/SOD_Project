"""
Evaluation metrics (IoU, precision, recall, F1, MAE) and result visualizations.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from checkpoint_io import torch_load_ckpt
from data_loader import SodDataset, list_image_mask_pairs, train_val_test_indices
from sod_model import build_model


def binary_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    thr: float = 0.5,
    *,
    gt_thr: float = 0.5,
) -> Dict[str, float]:
    """Binarize predictions at ``thr`` and ground truth at ``gt_thr`` (keep gt_thr=0.5 when sweeping ``thr``)."""
    p = pred >= thr
    g = gt >= gt_thr
    tp = np.logical_and(p, g).sum(dtype=np.float64)
    fp = np.logical_and(p, np.logical_not(g)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(p), g).sum(dtype=np.float64)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    inter = tp
    union = np.logical_or(p, g).sum(dtype=np.float64) + 1e-8
    iou = inter / union

    mae = float(np.mean(np.abs(pred.astype(np.float64) - gt.astype(np.float64))))
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mae": mae,
    }


def macro_aggregate_at_threshold(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    thr: float,
    gt_thr: float = 0.5,
) -> Dict[str, float]:
    """Mean per-image IoU/P/R/F1/MAE (same pooling as evaluate_checkpoint)."""
    keys = ["iou", "precision", "recall", "f1", "mae"]
    sums = {k: 0.0 for k in keys}
    n = len(pairs)
    if not n:
        return {k: 0.0 for k in keys}
    for pred, gt in pairs:
        m = binary_metrics(pred, gt, thr=thr, gt_thr=gt_thr)
        for k in keys:
            sums[k] += m[k]
    return {k: sums[k] / n for k in keys}


def sweep_thresholds_macro(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    thresholds: np.ndarray,
    gt_thr: float = 0.5,
) -> List[Dict[str, float]]:
    """Macro-averaged metrics at each prediction threshold."""
    rows: List[Dict[str, float]] = []
    for thr in thresholds:
        thr_f = float(thr)
        m = macro_aggregate_at_threshold(pairs, thr_f, gt_thr=gt_thr)
        rows.append({"threshold": thr_f, **m})
    return rows


def soft_iou_numpy(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(pred.astype(np.float64), 0.0, 1.0)
    g = np.clip(gt.astype(np.float64), 0.0, 1.0)
    inter = float((p * g).sum())
    union = float(p.sum() + g.sum() - inter)
    return (inter + eps) / (union + eps)


def pick_threshold(rows: List[Dict[str, float]], target_precision: float) -> Optional[Dict[str, float]]:
    """Row whose precision is closest to ``target_precision``."""
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(r["precision"] - target_precision))
    return best


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: str,
    data_root: str,
    images_sub: str,
    masks_sub: str,
    image_size: int,
    batch_size: int,
    variant: str,
    num_workers: int,
    seed: int,
    device: torch.device,
    max_batches: int = 0,
    bin_threshold: float = 0.5,
    gt_threshold: float = 0.5,
    pred_gt_store: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[Dict[str, float], List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]]:
    image_dir = os.path.join(data_root, images_sub)
    mask_dir = os.path.join(data_root, masks_sub)
    pairs = list_image_mask_pairs(image_dir, mask_dir)
    train_idx, val_idx, test_idx = train_val_test_indices(len(pairs), seed)
    test_pairs = [pairs[i] for i in test_idx.tolist()]

    ds = SodDataset(test_pairs, image_size=image_size, augment=False, seed=seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = build_model(variant, in_ch=3, base=32).to(device)
    ckpt = torch_load_ckpt(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sums = {k: 0.0 for k in ["iou", "precision", "recall", "f1", "mae", "soft_iou"]}
    count = 0
    viz_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

    for bi, batch in enumerate(tqdm(loader, desc="Test")):
        if max_batches and bi >= max_batches:
            break
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        pred = model(x)
        pb = pred.cpu().numpy()
        gb = y.cpu().numpy()
        paths = batch["image_path"] if "image_path" in batch else [""] * x.size(0)

        for i in range(pb.shape[0]):
            p1 = pb[i, 0]
            g1 = gb[i, 0]
            m = binary_metrics(p1, g1, thr=bin_threshold, gt_thr=gt_threshold)
            for k in ["iou", "precision", "recall", "f1", "mae"]:
                sums[k] += m[k]
            sums["soft_iou"] += soft_iou_numpy(p1, g1)
            count += 1
            if pred_gt_store is not None:
                pred_gt_store.append((p1.copy(), g1.copy()))

        # store first batch for grids
        if not viz_samples and bi == 0:
            for i in range(min(4, pb.shape[0])):
                inp = x[i].cpu().numpy().transpose(1, 2, 0)
                inp = np.clip(inp, 0, 1)
                pv = paths[i] if isinstance(paths, (list, tuple)) and i < len(paths) else ""
                viz_samples.append((inp, gb[i, 0], pb[i, 0], pv))

    agg = {k: sums[k] / max(count, 1) for k in sums}
    return agg, viz_samples


def plot_panel(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    out_path: str,
    title: str,
) -> None:
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for row, (img, gt, pr, _) in enumerate(samples):
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Input")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground truth")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(pr, cmap="inferno", vmin=0, vmax=1)
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")
        overlay = img.copy()
        heat = plt.cm.hot(pr)[:, :, :3]
        alpha = np.expand_dims(pr, -1)
        blended = overlay * (1 - 0.45 * alpha) + heat * (0.45 * alpha)
        blended = np.clip(blended, 0, 1)
        axes[row, 3].imshow(blended)
        axes[row, 3].set_title("Overlay")
        axes[row, 3].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved visualization:", out_path)


def benchmark_inference_ms(
    ckpt_path: str,
    variant: str,
    image_size: int,
    device: torch.device,
    n_runs: int = 21,
    warmup: int = 5,
) -> float:
    model = build_model(variant, in_ch=3, base=32).to(device)
    ckpt = torch_load_ckpt(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--images", type=str, default="images")
    ap.add_argument("--masks", type=str, default="masks")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--variant", type=str, default="baseline", choices=("baseline", "improved", "unet")
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarize predictions at this value for IoU / P / R / F1. "
        "If outputs sit low (~0.3–0.4), try lowering (e.g. 0.35). "
        "If precision is low (too much predicted foreground), raise (e.g. 0.6–0.72).",
    )
    ap.add_argument(
        "--gt_threshold",
        type=float,
        default=0.5,
        help="Binarize ground-truth masks at this value (keep 0.5 for ECSSD PNG masks).",
    )
    ap.add_argument(
        "--threshold_sweep",
        action="store_true",
        help="After inference, sweep prediction thresholds on the stored test preds and "
        "report best F1 and threshold closest to --target_precision (writes threshold_sweep.json).",
    )
    ap.add_argument("--threshold_sweep_from", type=float, default=0.35)
    ap.add_argument("--threshold_sweep_to", type=float, default=0.85)
    ap.add_argument("--threshold_sweep_steps", type=int, default=51)
    ap.add_argument(
        "--target_precision",
        type=float,
        default=0.7,
        help="Used with --threshold_sweep to print the row whose precision is closest to this.",
    )
    ap.add_argument("--max_batches", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    pred_gt_store: Optional[List[Tuple[np.ndarray, np.ndarray]]] = (
        [] if args.threshold_sweep else None
    )

    metrics, samples = evaluate_checkpoint(
        args.checkpoint,
        args.data_root,
        args.images,
        args.masks,
        args.image_size,
        args.batch_size,
        args.variant,
        args.num_workers,
        args.seed,
        device,
        max_batches=args.max_batches,
        bin_threshold=args.threshold,
        gt_threshold=args.gt_threshold,
        pred_gt_store=pred_gt_store,
    )

    med_ms = benchmark_inference_ms(
        args.checkpoint, args.variant, args.image_size, device
    )
    metrics_out = dict(metrics)
    metrics_out["median_infer_ms_per_image"] = med_ms
    metrics_out["binary_threshold_used"] = args.threshold
    metrics_out["gt_threshold_used"] = args.gt_threshold

    sweep_path = os.path.join(args.out_dir, "threshold_sweep.json")
    if args.threshold_sweep and pred_gt_store:
        thresholds = np.linspace(
            args.threshold_sweep_from,
            args.threshold_sweep_to,
            max(2, args.threshold_sweep_steps),
        ).astype(np.float64)
        rows = sweep_thresholds_macro(pred_gt_store, thresholds, gt_thr=args.gt_threshold)
        with open(sweep_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        best_f1 = max(rows, key=lambda r: r["f1"])
        near_p = pick_threshold(rows, args.target_precision)
        print("")
        print("=== threshold_sweep.json (prediction threshold vs macro P/R/F1/IoU) ===")
        print(
            f"Best F1: threshold={best_f1['threshold']:.4f} "
            f"precision={best_f1['precision']:.4f} recall={best_f1['recall']:.4f} "
            f"f1={best_f1['f1']:.4f} iou={best_f1['iou']:.4f}"
        )
        if near_p is not None:
            print(
                f"Closest to precision={args.target_precision:.2f}: threshold={near_p['threshold']:.4f} "
                f"precision={near_p['precision']:.4f} recall={near_p['recall']:.4f} "
                f"f1={near_p['f1']:.4f} iou={near_p['iou']:.4f}"
            )
            metrics_out["suggested_threshold_for_target_precision"] = near_p["threshold"]
        metrics_out["suggested_threshold_for_best_macro_f1"] = best_f1["threshold"]
        print("Wrote", sweep_path)

    path_json = os.path.join(args.out_dir, "metrics.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print("Metrics:", json.dumps(metrics_out, indent=2))

    plot_panel(
        samples,
        os.path.join(args.out_dir, "comparison_panel.png"),
        title=f"SOD qualitative results ({args.variant})",
    )


if __name__ == "__main__":
    main()
