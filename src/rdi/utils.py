from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"


def project_root() -> Path:
    # src/rdi/utils.py -> src/rdi -> src -> project root
    return Path(__file__).resolve().parents[3]


def set_mps_friendly_env() -> None:
    """
    - macOS + PyTorch + (OpenMP) 환경에서 종종 발생하는 문제를 완화하기 위한 설정.
    - 필요 시 실험 스크립트에서 호출.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


