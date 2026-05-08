"""
Dataset loading, preprocessing, and augmentation for salient object detection.

Expected directory layout (pair images with masks by matching stem):

    data_root/
        images/   (or im/, DUTS-TR-Image/, etc.)
        masks/    (or gt/, DUTS-TR-Mask/, etc.)

Masks may be binary or soft; they are loaded as single-channel float in [0, 1].
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset, Subset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_mask_pairs(
    image_dir: str,
    mask_dir: str,
) -> List[Tuple[str, str]]:
    """Match image files to mask files by basename (without extension)."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    stem_to_mask: dict[str, str] = {}
    for name in os.listdir(mask_dir):
        path = os.path.join(mask_dir, name)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(name)
        if ext.lower() not in IMG_EXTS:
            continue
        stem_to_mask[base] = path

    pairs: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(image_dir)):
        path = os.path.join(image_dir, name)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(name)
        if ext.lower() not in IMG_EXTS:
            continue
        if base not in stem_to_mask:
            continue
        pairs.append((path, stem_to_mask[base]))

    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found under {image_dir!r} and {mask_dir!r}. "
            "Ensure basenames match (e.g. im1.jpg <-> im1.png)."
        )
    return pairs


def train_val_test_indices(
    n: int,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _read_mask(path: str) -> np.ndarray:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    m = gray.astype(np.float32) / 255.0
    m = np.clip(m, 0.0, 1.0)
    return m


def _resize_pair(
    img: np.ndarray,
    mask: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r


def _random_crop_pair(
    img: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[0], img.shape[1]
    if h < crop_size or w < crop_size:
        scale = (crop_size / min(h, w)) + 0.01
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[0], img.shape[1]
    top = int(rng.integers(0, h - crop_size + 1))
    left = int(rng.integers(0, w - crop_size + 1))
    img_c = img[top : top + crop_size, left : left + crop_size]
    mask_c = mask[top : top + crop_size, left : left + crop_size]
    return img_c, mask_c


def _brightness_mult(img: np.ndarray, factor: float) -> np.ndarray:
    out = img * factor
    return np.clip(out, 0.0, 1.0)


class SodDataset(Dataset):
    """Paired RGB image + single-channel saliency mask."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        image_size: int,
        augment: bool,
        seed: int = 42,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ip, mp = self.pairs[index]
        img = _read_rgb(ip)
        mask = _read_mask(mp)

        if self.augment:
            img, mask = _random_crop_pair(
                img, mask, max(self.image_size, int(self.image_size * 1.1)), self._rng
            )
            img, mask = _resize_pair(img, mask, self.image_size)

            if self._rng.random() < 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

            bf = float(self._rng.uniform(0.75, 1.25))
            img = _brightness_mult(img, bf)
        else:
            img, mask = _resize_pair(img, mask, self.image_size)

        # NCHW
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        y = torch.from_numpy(mask).unsqueeze(0).contiguous()
        return {"image": x, "mask": y, "image_path": ip}


def build_dataloaders(
    data_root: str,
    image_subdir: str,
    mask_subdir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[Tuple[str, str]]]:
    image_dir = os.path.join(data_root, image_subdir)
    mask_dir = os.path.join(data_root, mask_subdir)
    pairs = list_image_mask_pairs(image_dir, mask_dir)

    train_idx, val_idx, test_idx = train_val_test_indices(
        len(pairs), seed, train_ratio, val_ratio
    )

    full = SodDataset(pairs, image_size=image_size, augment=False, seed=seed)
    train_ds = SodDataset(pairs, image_size=image_size, augment=True, seed=seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_ds, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(full, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        Subset(full, test_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader, pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test SOD data loading.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--images", type=str, default="images")
    p.add_argument("--masks", type=str, default="masks")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    tl, vl, xl, pairs = build_dataloaders(
        args.data_root,
        args.images,
        args.masks,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
    )
    b = next(iter(tl))
    print("Pairs:", len(pairs))
    print("Train batches:", len(tl))
    print("Batch image shape:", tuple(b["image"].shape))


if __name__ == "__main__":
    main()
