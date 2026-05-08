#!/usr/bin/env python3
"""
After training ECSSD baseline and improved models, summarize metrics for the report table.

Example:
  python evaluate.py --checkpoint checkpoints/ecssd_baseline/best.pt --data_root data/ecssd \\
      --images images --masks ground_truth_mask --variant baseline --out_dir results/ecssd_eval_baseline
  python evaluate.py --checkpoint checkpoints/ecssd_improved/best.pt --data_root data/ecssd \\
      --images images --masks ground_truth_mask --variant improved --out_dir results/ecssd_eval_improved
  python compare_experiments.py results/ecssd_eval_baseline/metrics.json results/ecssd_eval_improved/metrics.json \\
      --labels ecssd_baseline ecssd_improved
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "metrics",
        nargs="+",
        type=str,
        help="One or more metrics.json files (e.g. baseline improved unet).",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        type=str,
        default=[],
        help="Column labels; if omitted, uses run_1, run_2, …",
    )
    ap.add_argument("--out", type=str, default="", help="Write comparison JSON here")
    args = ap.parse_args()

    paths = [Path(p) for p in args.metrics]
    rows = [load_json(p) for p in paths]

    if args.labels:
        if len(args.labels) != len(paths):
            ap.error(f"Need one --labels entry per metrics file ({len(paths)} files).")
        labels = [lb.replace("|", "") for lb in args.labels]
    else:
        labels = [f"run_{i + 1}" for i in range(len(paths))]

    keys = ["iou", "precision", "recall", "f1", "mae", "median_infer_ms_per_image"]

    header = "| Metric | " + " | ".join(labels) + " |\n"
    sep = "| --- | " + " | ".join(["---"] * len(labels)) + " |\n"

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, int):
            return f"{v}"
        return str(v)

    md = header + sep
    for k in keys:
        md += "| {} | {} |\n".format(k, " | ".join(fmt(r.get(k)) for r in rows))

    print(md)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {lb: r for lb, r in zip(labels, rows)}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("Wrote", out_path)


if __name__ == "__main__":
    main()
