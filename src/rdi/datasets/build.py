from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from rdi.datasets.mvtec_split import MVTecSplit
from rdi.datasets.torch_dataset import Sample


@dataclass(frozen=True)
class BuiltDatasets:
    train_samples: list[Sample]
    test_samples: list[Sample]
    real_defect_samples: list[Sample]  # K-shot real defects (with masks if available)


def build_samples(
    split: MVTecSplit,
    train_normals_cap: int,
    seed: int,
) -> BuiltDatasets:
    rng = random.Random(seed)
    train_normals = list(split.train_normal)
    rng.shuffle(train_normals)
    train_normals = train_normals[: min(train_normals_cap, len(train_normals))]

    train_samples: list[Sample] = [Sample(p, 0, None) for p in train_normals]
    real_defect_samples: list[Sample] = [
        Sample(p, 1, split.train_defect_masks.get(p)) for p in split.train_defect
    ]
    train_samples.extend(real_defect_samples)

    test_samples: list[Sample] = [Sample(p, 0, None) for p in split.test_normal]
    test_samples.extend([Sample(p, 1, split.test_defect_masks.get(p)) for p in split.test_defect])

    return BuiltDatasets(
        train_samples=train_samples,
        test_samples=test_samples,
        real_defect_samples=real_defect_samples,
    )


