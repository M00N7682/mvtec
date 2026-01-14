from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from torch import Tensor


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int  # 0 normal, 1 defect
    mask_path: Path | None = None


class ImageBinaryDataset:
    """
    Minimal dataset wrapper that returns (image_tensor, label).
    Image loading/transforming is injected to keep dependencies light.
    """

    def __init__(
        self,
        samples: list[Sample],
        load_image: Callable[[Path], Tensor],
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        self.samples = samples
        self.load_image = load_image
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = self.load_image(s.image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(s.label)


