from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Normalize:
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


