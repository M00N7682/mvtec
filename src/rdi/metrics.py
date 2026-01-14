from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    auroc: float


def auroc_score(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))


