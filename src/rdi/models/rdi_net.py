from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rdi.models.unet import UNetSmall


@dataclass(frozen=True)
class RDIOutput:
    residual: torch.Tensor  # [B,3,H,W]
    alpha: torch.Tensor  # [B,1,H,W] in [0,1]


class RDINet(nn.Module):
    """
    Residual Defect Insertion Network (minimal v0).
    Input: pseudo-normal image x_n (3ch), mask m (1ch), noise z (1ch) => 5ch.
    Output: residual r (3ch), alpha a (1ch).
    """

    def __init__(self, in_ch: int = 5, base: int = 64):
        super().__init__()
        self.backbone = UNetSmall(in_ch=in_ch, out_ch=4, base=base)

    def forward(self, x_n: torch.Tensor, m: torch.Tensor, z: torch.Tensor) -> RDIOutput:
        x = torch.cat([x_n, m, z], dim=1)
        y = self.backbone(x)
        residual = y[:, :3, :, :]
        alpha = torch.sigmoid(y[:, 3:4, :, :])
        return RDIOutput(residual=residual, alpha=alpha)


