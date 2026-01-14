from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision

from rdi.metrics import auroc_score
from rdi.torch_utils import imagenet_normalize, load_image_tensor, seed_everything
from rdi.utils import ensure_dir, save_json


@dataclass(frozen=True)
class PaDiMResult:
    auroc: float


class _ResNet18Features(torch.nn.Module):
    """
    Feature extractor returning a spatial feature map.
    We use layer3 output (C=256, H=W=16 for 256x256 input).
    """

    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # [B,C,H,W]


def _build_batch(paths: list[Path], image_size: int, device: torch.device) -> torch.Tensor:
    norm = imagenet_normalize()
    xs = []
    for p in paths:
        x = load_image_tensor(p, image_size)
        x = norm(x)
        xs.append(x)
    return torch.stack(xs, dim=0).to(device)


def run_padim(
    train_normal_paths: list[Path],
    test_paths: list[tuple[Path, int]],  # (path,label)
    out_dir: Path,
    image_size: int = 256,
    batch_size: int = 16,
    seed: int = 42,
    device: str = "mps",
    reg_eps: float = 1e-2,
) -> PaDiMResult:
    """
    Lightweight PaDiM-style baseline:
    - Extract spatial features (ResNet18 layer3).
    - Fit per-location Gaussian (mean, covariance) on normal train set.
    - Score each image by max Mahalanobis distance across locations.
    """
    seed_everything(seed)
    ensure_dir(out_dir)

    dev = torch.device(device if device == "mps" and torch.backends.mps.is_available() else "cpu")
    feat = _ResNet18Features().to(dev).eval()

    # Fit gaussian per spatial position
    with torch.no_grad():
        feats_all = []
        for i in range(0, len(train_normal_paths), batch_size):
            batch_paths = train_normal_paths[i : i + batch_size]
            xb = _build_batch(batch_paths, image_size, dev)
            fb = feat(xb).detach().cpu().numpy()  # [B,C,H,W]
            feats_all.append(fb)
        F = np.concatenate(feats_all, axis=0)  # [N,C,H,W]

    N, C, H, W = F.shape
    X = F.transpose(0, 2, 3, 1).reshape(N, H * W, C)  # [N,HW,C]

    mu = X.mean(axis=0)  # [HW,C]
    cov = np.zeros((H * W, C, C), dtype=np.float32)
    for j in range(H * W):
        xj = X[:, j, :]  # [N,C]
        cj = np.cov(xj, rowvar=False)
        cov[j] = cj.astype(np.float32) + reg_eps * np.eye(C, dtype=np.float32)

    inv_cov = np.linalg.inv(cov)  # [HW,C,C]

    # Score test images
    ys: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(test_paths), batch_size):
            batch = test_paths[i : i + batch_size]
            paths = [p for p, _ in batch]
            labels = [y for _, y in batch]
            xb = _build_batch(paths, image_size, dev)
            fb = feat(xb).detach().cpu().numpy()  # [B,C,H,W]
            B = fb.shape[0]
            Xb = fb.transpose(0, 2, 3, 1).reshape(B, H * W, C)  # [B,HW,C]
            # mahalanobis per position
            diff = Xb - mu[None, :, :]  # [B,HW,C]
            # compute d^2 = diff^T inv_cov diff
            d2 = np.einsum("bhc,hcd,bhd->bh", diff, inv_cov, diff)  # [B,HW]
            s = d2.max(axis=1)  # image-level score
            ys.extend([int(v) for v in labels])
            scores.extend([float(v) for v in s])

    auroc = auroc_score(ys, scores)
    save_json(out_dir / "metrics.json", {"auroc": auroc, "method": "padim_lite"})
    print(f"[padim] AUROC={auroc:.4f} -> {out_dir/'metrics.json'}")
    return PaDiMResult(auroc=auroc)


