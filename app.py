"""Gradio demo: upload an image -> saliency mask, overlay, and inference time."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from checkpoint_io import torch_load_ckpt
from sod_model import build_model


def _mask_to_display_rgb(mask_01: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Build an RGB image Gradio can show reliably. Natural images with a weak / flat
    model output look like solid gray as raw uint8; we min–max stretch for display
    only and add stats to the caption.
    """
    m = np.clip(mask_01.astype(np.float64), 0.0, 1.0)
    mn, mx = float(m.min()), float(m.max())
    mean = float(m.mean())
    span = mx - mn
    stats = f"mask min={mn:.3f} max={mx:.3f} mean={mean:.3f} (span={span:.4f})"
    if span < 1e-8:
        m = np.clip(m, 0.0, 1.0)
        gray = np.full((*m.shape, 3), 128, dtype=np.uint8)
        return gray, stats + " | (constant mask)"

    norm = (m - mn) / max(span, 1e-8)
    rgb = plt.cm.inferno(norm)[..., :3]
    vis = (np.clip(rgb * 255.0, 0, 255)).astype(np.uint8)
    if span < 0.08:
        stats = (
            stats
            + " | WARNING: map is almost flat — launch with the right ECSSD checkpoint, e.g. "
            "python app.py --checkpoint checkpoints/ecssd_baseline/best.pt --variant baseline --image_size 128"
        )
    return vis, stats


def _load_rgb(path_or_pil):
    if isinstance(path_or_pil, np.ndarray):
        arr = path_or_pil
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        img = Image.fromarray(arr.astype("uint8"))
    else:
        img = Image.open(path_or_pil).convert("RGB")
    return img


def _make_predict_fn(ckpt_path: str, variant: str, image_size: int):
    """Load weights once; each upload only runs the forward pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(variant, in_ch=3, base=32).to(device)
    ckpt = torch_load_ckpt(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    @torch.no_grad()
    def predict(input_image):
        if input_image is None:
            return None, None, "Upload an image first."
        pil = _load_rgb(input_image).convert("RGB")
        w, h = pil.size
        arr = np.asarray(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        resized = F.interpolate(
            tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
        )

        x = resized.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mask_s = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        mask_np = F.interpolate(mask_s, size=(h, w), mode="bilinear", align_corners=False)
        mask_np = mask_np[0, 0].clamp(0, 1).cpu().numpy()

        heat = plt.cm.inferno(mask_np)[..., :3].astype(np.float32)
        alpha = 0.45 * np.clip(mask_np[..., None], 0.0, 1.0)
        overlay = arr * (1.0 - alpha) + heat * alpha
        overlay = np.clip(overlay, 0, 1)

        mask_rgb, stats = _mask_to_display_rgb(mask_np)

        caption = (
            f"{stats}. | Inference ({image_size}px forward): {infer_ms:.2f} ms on {device}"
        )

        return mask_rgb, (overlay * 255).astype(np.uint8), caption

    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ecssd_baseline/best.pt",
        help="Path to best.pt from train.py (ECSSD: checkpoints/ecssd_baseline/best.pt)",
    )
    ap.add_argument(
        "--variant", type=str, default="baseline", choices=("baseline", "improved", "unet")
    )
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Create a Gradio share link.")
    args = ap.parse_args()

    if not Path(args.checkpoint).is_file():
        print(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train ECSSD first, then e.g.:\n"
            "  python app.py --checkpoint checkpoints/ecssd_baseline/best.pt --variant baseline\n"
            "or: python app.py --checkpoint checkpoints/ecssd_improved/best.pt --variant improved"
        )
        sys.exit(1)

    predict_fn = _make_predict_fn(args.checkpoint, args.variant, args.image_size)

    iface = gr.Interface(
        fn=predict_fn,
        inputs=[gr.Image(type="numpy", label="Upload image")],
        outputs=[
            gr.Image(label="Predicted saliency (inferno, min–max stretched for display)"),
            gr.Image(label="Overlay on input resolution"),
            gr.Text(label="Stats & timing"),
        ],
        title="Salient Object Detection",
        description="Upload an RGB image to get a segmentation-style saliency mask.",
    )
    iface.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
