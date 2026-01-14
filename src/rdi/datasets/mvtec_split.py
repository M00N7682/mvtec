from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdi.utils import load_json


@dataclass(frozen=True)
class MVTecSplit:
    dataset: str
    category: str
    protocol: str
    seed: int
    k_defects: int
    train_normal: list[Path]
    train_defect: list[Path]
    train_defect_masks: dict[Path, Path]
    test_normal: list[Path]
    test_defect: list[Path]
    test_defect_masks: dict[Path, Path]

    @classmethod
    def from_json(cls, path: str | Path) -> "MVTecSplit":
        obj = load_json(path)

        def p(x: str) -> Path:
            return Path(x)

        train_masks = {p(k): p(v) for k, v in obj["train"]["defect_masks"].items()}
        test_masks = {p(k): p(v) for k, v in obj["test"]["defect_masks"].items()}

        return cls(
            dataset=obj["dataset"],
            category=obj["category"],
            protocol=obj["protocol"],
            seed=int(obj["seed"]),
            k_defects=int(obj["k_defects"]),
            train_normal=[p(x) for x in obj["train"]["normal"]],
            train_defect=[p(x) for x in obj["train"]["defect"]],
            train_defect_masks=train_masks,
            test_normal=[p(x) for x in obj["test"]["normal"]],
            test_defect=[p(x) for x in obj["test"]["defect"]],
            test_defect_masks=test_masks,
        )


