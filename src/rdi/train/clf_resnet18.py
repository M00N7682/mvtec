from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from rdi.datasets.torch_dataset import ImageBinaryDataset, Sample
from rdi.metrics import auroc_score
from rdi.torch_utils import imagenet_normalize, load_image_tensor, seed_everything
from rdi.utils import ensure_dir, save_json


@dataclass(frozen=True)
class TrainResult:
    auroc: float


def _build_loader(samples: list[Sample], image_size: int, batch_size: int, shuffle: bool):
    norm = imagenet_normalize()

    def load(p: Path):
        return load_image_tensor(p, image_size)

    def tf(x):
        return norm(x)

    ds = ImageBinaryDataset(samples=samples, load_image=load, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_and_eval(
    train_samples: list[Sample],
    test_samples: list[Sample],
    out_dir: Path,
    image_size: int = 256,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "mps",
) -> TrainResult:
    seed_everything(seed)
    ensure_dir(out_dir)

    dev = torch.device(device if device == "mps" and torch.backends.mps.is_available() else "cpu")

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = _build_loader(train_samples, image_size, batch_size, shuffle=True)
    test_loader = _build_loader(test_samples, image_size, batch_size, shuffle=False)

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        print(f"[clf] epoch {ep+1}/{epochs} loss={ep_loss/max(1,n):.4f}")

    # eval: AUROC from p(defect)
    model.eval()
    ys: list[int] = []
    ps: list[float] = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(dev)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            ps.extend(prob.tolist())
            ys.extend([int(v) for v in y.numpy().tolist()])

    auroc = auroc_score(ys, ps)
    save_json(out_dir / "metrics.json", {"auroc": auroc})
    print(f"[clf] AUROC={auroc:.4f} -> {out_dir/'metrics.json'}")
    return TrainResult(auroc=auroc)


