from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DefectPatch:
    img_path: Path
    mask_path: Path


def _load_rgb(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def _load_mask(path: Path, size: int) -> np.ndarray:
    m = Image.open(path).convert("L").resize((size, size), resample=Image.NEAREST)
    arr = (np.asarray(m).astype(np.float32) / 255.0)  # [0,1]
    return (arr > 0.5).astype(np.float32)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _alpha_blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    # dst/src: [H,W,3], alpha: [H,W,1]
    return dst * (1.0 - alpha) + src * alpha


def synthesize_one(
    normal_img_path: Path,
    patch_bank: list[DefectPatch],
    image_size: int,
    rng: random.Random,
    mode: str = "alpha",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (synthetic_rgb_float01, used_mask_float01) in full image coordinates.
    """
    base = _load_rgb(normal_img_path, image_size)
    dp = rng.choice(patch_bank)
    defect_img = _load_rgb(dp.img_path, image_size)
    defect_mask = _load_mask(dp.mask_path, image_size)

    bb = _bbox_from_mask(defect_mask)
    if bb is None:
        # fallback: no-op
        return base, np.zeros((image_size, image_size), dtype=np.float32)

    x0, y0, x1, y1 = bb
    patch = defect_img[y0:y1, x0:x1, :]
    pmask = defect_mask[y0:y1, x0:x1]

    # moderate geometric jitter
    scale = rng.uniform(0.7, 1.3)
    ph, pw = patch.shape[0], patch.shape[1]
    nh, nw = max(4, int(ph * scale)), max(4, int(pw * scale))

    patch_img = Image.fromarray((patch * 255).astype(np.uint8))
    patch_img = patch_img.resize((nw, nh), resample=Image.BILINEAR)
    patch = np.asarray(patch_img).astype(np.float32) / 255.0

    mask_img = Image.fromarray((pmask * 255).astype(np.uint8))
    mask_img = mask_img.resize((nw, nh), resample=Image.NEAREST)
    pmask = (np.asarray(mask_img).astype(np.float32) / 255.0) > 0.5
    pmask = pmask.astype(np.float32)

    # random placement on base (ensure within bounds)
    max_x = max(0, image_size - nw)
    max_y = max(0, image_size - nh)
    ox = rng.randint(0, max_x) if max_x > 0 else 0
    oy = rng.randint(0, max_y) if max_y > 0 else 0

    out = base.copy()
    used = np.zeros((image_size, image_size), dtype=np.float32)

    alpha = pmask[..., None]
    roi = out[oy : oy + nh, ox : ox + nw, :]

    if mode in ("alpha", "poisson_like"):
        # poisson_like is approximated by blurred alpha (cheap, no opencv dependency)
        if mode == "poisson_like":
            # robust blur using PIL (avoids manual indexing edge cases)
            from PIL import ImageFilter

            a_img = Image.fromarray((alpha[..., 0] * 255).astype(np.uint8))
            a_img = a_img.filter(ImageFilter.GaussianBlur(radius=2))
            blurred = np.asarray(a_img).astype(np.float32) / 255.0
            alpha = blurred[..., None]

        blended = _alpha_blend(roi, patch, alpha)
        out[oy : oy + nh, ox : ox + nw, :] = blended
        used[oy : oy + nh, ox : ox + nw] = np.maximum(used[oy : oy + nh, ox : ox + nw], alpha[..., 0])

    return out, used


def build_patch_bank(real_defect_samples: list["Sample"]) -> list[DefectPatch]:
    from rdi.datasets.torch_dataset import Sample

    bank: list[DefectPatch] = []
    for s in real_defect_samples:
        if not isinstance(s, Sample):
            continue
        if s.mask_path is None:
            continue
        bank.append(DefectPatch(img_path=s.image_path, mask_path=s.mask_path))
    return bank


