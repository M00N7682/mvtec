from __future__ import annotations

from pathlib import Path


def load_image_tensor(path: Path, image_size: int) -> "Tensor":
    """
    Loads RGB image -> float tensor in [0,1], shape [3,H,W] resized to image_size.
    """
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
    x = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
    return x


def imagenet_normalize() -> "Normalize":
    import torchvision.transforms as T

    return T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def seed_everything(seed: int) -> None:
    import os
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "mps"):
        # Determinism is limited on MPS; we still fix seeds.
        pass


