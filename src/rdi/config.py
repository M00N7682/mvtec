from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdi.const import DEFAULT_CATEGORIES
from rdi.utils import RunPaths, project_root


@dataclass(frozen=True)
class ExpConfig:
    # dataset
    mvtec_root: Path
    categories: tuple[str, ...] = tuple(DEFAULT_CATEGORIES)
    ks: tuple[int, ...] = (1, 5, 10)
    seed: int = 42

    # image / model
    image_size: int = 256
    device: str = "mps"  # "mps" | "cpu"

    # data sizing
    train_normals_per_category: int = 200
    synth_defects_per_category: int = 200  # 1 defect inserted per normal (subsample normals if needed)

    # training (classifier)
    clf_epochs: int = 5
    clf_batch_size: int = 32
    clf_lr: float = 3e-4
    clf_weight_decay: float = 1e-4

    # synthesis
    paste_blend: str = "alpha"  # "alpha" or "poisson_like"

    @property
    def run_paths(self) -> RunPaths:
        return RunPaths(root=project_root())

    @property
    def splits_root(self) -> Path:
        return self.run_paths.splits_dir / "mvtec_ad"

    @property
    def outputs_root(self) -> Path:
        return self.run_paths.outputs_dir


def default_config() -> ExpConfig:
    rp = RunPaths(root=project_root())
    mvtec_root = rp.processed_dir / "mvtec_ad"
    return ExpConfig(mvtec_root=mvtec_root)


