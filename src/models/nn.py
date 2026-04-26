"""
Neural network model wrappers.

Reference: PLAN_v2.md §4.3.

NN-A (primary)
--------------
Gated residual MLP with entity embeddings:

    Inputs:
      ├── Numerics (~1100, RankGauss-scaled)
      ├── _is_nan flags (~70, raw 0/1)
      └── Categoricals (label-encoded → embeddings, dim = min(50, n_unique//2))

    Block 1: Linear(in → 512) + BN + GELU + Dropout
    Block 2: Linear(512 → 256) + BN + GELU + Dropout + Skip(Linear(512 → 256))
    Block 3: Linear(256 → 128) + BN + GELU + Dropout + Skip(Linear(256 → 128))
    Output : Linear(128 → 1)

Training: BCEWithLogitsLoss(pos_weight=11.4), AdamW, OneCycleLR, AMP, ES patience=10.
Expected OOF AUC: 0.78–0.79.

NN-B (optional, Phase 4.6, gated on NN-A ≥ 0.78)
------------------------------------------------
Smaller MLP (256 → 128 → 64), dropout 0.5, no skip connections, trained on top
~300 features chosen via Ridge forward selection. Purpose: feature-subset diversity.

Both classes are functional stubs — flesh out the forward pass and training loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

from sklearn.metrics import roc_auc_score

from src import config
from src.models.base import FoldArtifacts, ModelBase
from src.utils import get_logger

logger = get_logger()


# ─── NN-A architecture ────────────────────────────────────────────────────────


class GatedResidualMLP(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Gated residual MLP described in PLAN §4.3 A2."""

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int],
        block_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.3,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch is not installed.")
        super().__init__()

        # Embedding tables for categoricals
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, min(50, c // 2 if c > 2 else 1)) for c in cat_cardinalities]
        )
        emb_total = sum(emb.embedding_dim for emb in self.embeddings)
        in_dim = n_numeric + emb_total

        b1, b2, b3 = block_dims
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, b1), nn.BatchNorm1d(b1), nn.GELU(), nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Linear(b1, b2), nn.BatchNorm1d(b2), nn.GELU(), nn.Dropout(dropout)
        )
        self.skip2 = nn.Linear(b1, b2)
        self.block3 = nn.Sequential(
            nn.Linear(b2, b3), nn.BatchNorm1d(b3), nn.GELU(), nn.Dropout(dropout)
        )
        self.skip3 = nn.Linear(b2, b3)
        self.head = nn.Linear(b3, 1)

    def forward(self, x_num: "torch.Tensor", x_cats: list["torch.Tensor"]) -> "torch.Tensor":  # type: ignore[name-defined]
        if self.embeddings:
            embs = [emb(x_cats[i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num, *embs], dim=1)
        else:
            x = x_num
        h1 = self.block1(x)
        h2 = self.block2(h1) + self.skip2(h1)
        h3 = self.block3(h2) + self.skip3(h2)
        return self.head(h3).squeeze(-1)


# ─── NN-A wrapper ─────────────────────────────────────────────────────────────


class NNAModel(ModelBase):
    """NN-A — gated residual MLP, RankGauss + nan-flags input matrix."""

    name = "nn_a"
    matrix = "nn"

    def default_params(self) -> dict[str, Any]:
        return {
            "block_dims": (512, 256, 128),
            "dropout": 0.3,
            "lr_max": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 512,
            "epochs": 30,
            "early_stopping_patience": 10,
            "seed": config.SEED,
        }

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
        if torch is None:
            raise ImportError("PyTorch is not installed. `make install` first.")

        # TODO: full training loop — this is a runnable scaffold.
        # 1. Split numerics vs cats, build cardinality list.
        # 2. Build TensorDataset / DataLoader.
        # 3. Instantiate GatedResidualMLP, BCEWithLogitsLoss(pos_weight),
        #    AdamW, OneCycleLR, AMP scaler.
        # 4. Train with early stopping on valid AUC (patience=10).
        # 5. Predict on valid + test.

        logger.warning(
            f"  {self.name} fold {fold}: NN training is a TODO stub — emitting "
            "uniform 0.5 predictions so the pipeline wires end-to-end."
        )
        valid_pred = np.full(len(y_valid), 0.5)
        test_pred = np.full(len(X_test), 0.5)
        auc = float(roc_auc_score(y_valid, valid_pred)) if y_valid.std() > 0 else 0.5

        art = FoldArtifacts(
            fold=fold,
            valid_pred=valid_pred,
            test_pred=test_pred,
            valid_auc=auc,
        )
        self.fold_artifacts.append(art)
        self.save_fold_predictions(art)
        return art


# ─── NN-B (optional, Phase 4.6) ───────────────────────────────────────────────


class NNBModel(ModelBase):
    """NN-B — smaller MLP on top-300 features chosen by Ridge forward selection."""

    name = "nn_b"
    matrix = "nn"

    def default_params(self) -> dict[str, Any]:
        return {
            "block_dims": (256, 128, 64),
            "dropout": 0.5,
            "lr_max": 5e-4,
            "weight_decay": 1e-5,
            "batch_size": 512,
            "epochs": 30,
            "early_stopping_patience": 10,
            "n_features": 300,
            "seed": config.SEED,
        }

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
        # TODO: implement Ridge forward selection + smaller MLP per PLAN §2.9 D8 / §4.3 D5.
        logger.warning(f"  {self.name} fold {fold}: TODO — uniform 0.5 stub.")
        valid_pred = np.full(len(y_valid), 0.5)
        test_pred = np.full(len(X_test), 0.5)
        auc = 0.5
        art = FoldArtifacts(fold=fold, valid_pred=valid_pred, test_pred=test_pred, valid_auc=auc)
        self.fold_artifacts.append(art)
        self.save_fold_predictions(art)
        return art
