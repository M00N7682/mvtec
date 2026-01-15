from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rdi.datasets.torch_dataset import Sample
from rdi.models.rdi_net import RDINet
from rdi.torch_utils import seed_everything
from rdi.utils import ensure_dir, save_json


@dataclass(frozen=True)
class RDINetConfig:
    image_size: int = 256
    base_channels: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    steps: int = 800
    batch_size: int = 4
    lambda_in: float = 1.0  # inside mask reconstruction
    lambda_out: float = 0.2  # outside mask identity
    lambda_tv: float = 0.05  # alpha smoothness


def _load_rgb(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size))
    return (np.asarray(img).astype(np.float32) / 255.0)


def _load_mask(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    m = Image.open(path).convert("L").resize((size, size))
    return (np.asarray(m).astype(np.float32) / 255.0) > 0.5


def _pseudo_normal(defect_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Cheap inpainting to create pseudo-normal: fill masked region with mean boundary color + blur.
    """
    x = defect_rgb.copy()
    m = mask.astype(bool)
    # boundary ring
    from PIL import Image, ImageFilter

    mm = (m.astype(np.uint8) * 255)
    dil = Image.fromarray(mm).filter(ImageFilter.MaxFilter(size=13))
    dil_m = (np.asarray(dil).astype(np.float32) / 255.0) > 0.5
    ring = dil_m & (~m)
    if ring.any():
        color = x[ring].mean(axis=0)
    else:
        color = x[~m].mean(axis=0)
    x[m] = color

    # blur only inside enlarged region
    xi = Image.fromarray((x * 255).astype(np.uint8))
    xb = xi.filter(ImageFilter.GaussianBlur(radius=2))
    out = np.asarray(xb).astype(np.float32) / 255.0
    # keep outside exactly
    out[~m] = defect_rgb[~m]
    return out


def _to_t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).permute(2, 0, 1).float()


def _to_m(m: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(m.astype(np.float32))[None, :, :]


def _tv(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


def train_rdi_net(
    real_defect_samples: list[Sample],
    out_dir: Path,
    cfg: RDINetConfig,
    seed: int = 42,
    device: str = "mps",
) -> Path:
    """
    Train RDINet on K-shot defects via pseudo-normal reconstruction.
    Saves weights to out_dir/rdi_net.pt and returns that path.
    """
    if any(s.mask_path is None for s in real_defect_samples):
        raise ValueError("RDI-Net training requires defect masks (mask_path).")

    seed_everything(seed)
    ensure_dir(out_dir)

    dev = torch.device(device if device == "mps" and torch.backends.mps.is_available() else "cpu")

    net = RDINet(in_ch=5, base=cfg.base_channels).to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rng = np.random.default_rng(seed)
    n = len(real_defect_samples)
    if n == 0:
        raise ValueError("No real defect samples (K-shot) found for training.")

    for step in range(1, cfg.steps + 1):
        idxs = rng.integers(0, n, size=cfg.batch_size)
        xs_n = []
        xs_d = []
        ms = []
        zs = []
        for i in idxs.tolist():
            s = real_defect_samples[i]
            xd = _load_rgb(s.image_path, cfg.image_size)
            m = _load_mask(s.mask_path, cfg.image_size)
            xn = _pseudo_normal(xd, m)
            z = rng.standard_normal((cfg.image_size, cfg.image_size, 1)).astype(np.float32)
            xs_d.append(_to_t(xd))
            xs_n.append(_to_t(xn))
            ms.append(_to_m(m))
            zs.append(torch.from_numpy(z).permute(2, 0, 1).float())

        x_d = torch.stack(xs_d, 0).to(dev)
        x_n = torch.stack(xs_n, 0).to(dev)
        m = torch.stack(ms, 0).to(dev)
        z = torch.stack(zs, 0).to(dev)

        out = net(x_n, m, z)
        x_hat = x_n + out.alpha * m * out.residual

        inside = (m > 0.5).float()
        outside = 1.0 - inside

        loss_in = F.l1_loss(x_hat * inside, x_d * inside)
        loss_out = F.l1_loss(x_hat * outside, x_n * outside)
        loss_tv = _tv(out.alpha)
        loss = cfg.lambda_in * loss_in + cfg.lambda_out * loss_out + cfg.lambda_tv * loss_tv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            print(
                f"[rdi] step {step}/{cfg.steps} "
                f"loss={float(loss.item()):.4f} in={float(loss_in.item()):.4f} out={float(loss_out.item()):.4f} tv={float(loss_tv.item()):.4f}"
            )

    wpath = out_dir / "rdi_net.pt"
    torch.save(net.state_dict(), wpath)
    save_json(out_dir / "train_cfg.json", cfg.__dict__)
    print(f"[rdi] saved {wpath}")
    return wpath


def synthesize_with_rdi_net(
    weights_path: Path,
    normal_paths: list[Path],
    mask_paths: list[Path],
    out_dir: Path,
    image_size: int,
    n_synth: int,
    seed: int = 42,
    device: str = "mps",
) -> list[dict]:
    """
    Generate synthetic defects by sampling a normal base + random translated mask.
    Returns synth_index items: {img, mask}.
    """
    ensure_dir(out_dir)
    synth_img_dir = ensure_dir(out_dir / "synth")
    synth_mask_dir = ensure_dir(out_dir / "masks")

    dev = torch.device(device if device == "mps" and torch.backends.mps.is_available() else "cpu")
    net = RDINet(in_ch=5, base=64).to(dev).eval()
    net.load_state_dict(torch.load(weights_path, map_location=dev, weights_only=True))

    rng = np.random.default_rng(seed)

    def load_mask(p: Path) -> np.ndarray:
        return _load_mask(p, image_size)

    def translate_mask(m: np.ndarray) -> np.ndarray:
        # random shift within +-32 pixels
        dy = int(rng.integers(-32, 33))
        dx = int(rng.integers(-32, 33))
        mm = np.roll(m, shift=(dy, dx), axis=(0, 1))
        return mm

    items = []
    for i in range(n_synth):
        npath = normal_paths[int(rng.integers(0, len(normal_paths)))]
        mpath = mask_paths[int(rng.integers(0, len(mask_paths)))]

        x = _load_rgb(npath, image_size)
        m = translate_mask(load_mask(mpath))
        z = rng.standard_normal((image_size, image_size, 1)).astype(np.float32)

        x_t = _to_t(x)[None, ...].to(dev)
        m_t = _to_m(m)[None, ...].to(dev)
        z_t = torch.from_numpy(z).permute(2, 0, 1)[None, ...].float().to(dev)

        with torch.no_grad():
            out = net(x_t, m_t, z_t)
            x_hat = x_t + out.alpha * m_t * out.residual
            x_hat = x_hat.clamp(0, 1)

        from PIL import Image

        img_out = (x_hat[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_out = (m.astype(np.uint8) * 255)

        ip = synth_img_dir / f"{i:06d}.png"
        mp = synth_mask_dir / f"{i:06d}.png"
        Image.fromarray(img_out).save(ip)
        Image.fromarray(mask_out).save(mp)
        items.append({"img": str(ip), "mask": str(mp)})

    return items


