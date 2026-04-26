"""
Abstract base class for model wrappers.

Every model module under ``src/models/`` exposes a subclass of ``ModelBase``
that implements ``fit_fold`` and ``predict``. The training driver
(``src/train.py``) loops folds and calls these methods uniformly.

Reference: PLAN_v2.md §4 (modeling roadmap).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger()


@dataclass
class FoldArtifacts:
    """In-memory record of one fold's training output."""

    fold: int
    valid_pred: np.ndarray
    test_pred: np.ndarray
    valid_auc: float
    n_iterations: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class ModelBase(ABC):
    """
    Base class for model wrappers.

    Concrete subclasses implement:

    - :attr:`name`            — short ID used in artifact filenames.
    - :attr:`matrix`          — which feature matrix this model consumes
                                (``"main"``, ``"catboost"``, or ``"nn"``).
    - :meth:`default_params`  — baseline hyperparameters from PLAN §4.1.
    - :meth:`fit_fold`        — train one fold, return predictions + AUC.
    """

    name: str = "unnamed"
    matrix: str = "main"  # one of {"main", "catboost", "nn"}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params if params is not None else self.default_params()
        self.logger = logger
        self.fold_artifacts: list[FoldArtifacts] = []

    # ─── Subclass contract ────────────────────────────────────────────────

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Return the baseline (non-tuned) hyperparameters from PLAN §4.1."""

    @abstractmethod
    def fit_fold(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        X_test: pd.DataFrame,
        *,
        fold: int,
        cat_features: list[str] | None = None,
    ) -> FoldArtifacts:
        """Train one fold; return predictions on valid & test plus AUC."""

    # ─── Concrete helpers ─────────────────────────────────────────────────

    def predictions_dir(self) -> Path:
        d = config.PREDICTIONS_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_fold_predictions(self, art: FoldArtifacts) -> None:
        """Persist per-fold predictions to ``artifacts/predictions/<name>/``."""
        d = self.predictions_dir()
        np.save(d / f"valid_fold{art.fold}.npy", art.valid_pred)
        np.save(d / f"test_fold{art.fold}.npy", art.test_pred)
        logger.debug(
            f"  {self.name} fold {art.fold}: saved valid + test predictions to {d}"
        )

    def aggregate_test_predictions(self) -> np.ndarray:
        """
        Mean of per-fold test predictions across the folds in ``self.fold_artifacts``.

        Used by ``src/ensemble.py`` and ``src/submit.py``.
        """
        if not self.fold_artifacts:
            raise RuntimeError(f"{self.name}: no fold artifacts available.")
        stacked = np.stack([a.test_pred for a in self.fold_artifacts], axis=0)
        return stacked.mean(axis=0)

    def overall_oof_auc(self, oof: np.ndarray, y: np.ndarray) -> float:
        """Compute concatenated OOF AUC (PLAN §3.2 primary metric)."""
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, oof))
