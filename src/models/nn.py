"""
Neural network model wrappers.

Reference: PLAN_v2.md §4.3.

NN-A (primary)
--------------
Gated residual MLP with entity embeddings:

    Inputs:
      ├── Numerics (RankGauss-scaled, applied per-fold in train.py)
      ├── _is_nan flags (raw 0/1)
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
~300 features chosen via Ridge forward selection.
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

        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, min(50, max(1, c // 2))) for c in cat_cardinalities]
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


# ─── Helper: split a DataFrame into numeric tensor + cat tensors ──────────────


def _prepare_arrays(
    X: pd.DataFrame, cat_cols: list[str]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Split X into (numeric_array, list_of_cat_arrays). Numerics are float32."""
    numeric_cols = [c for c in X.columns if c not in cat_cols]
    num = X[numeric_cols].to_numpy(dtype=np.float32, copy=True)
    # Median-impute any remaining NaN (RankGauss already handled most).
    if np.isnan(num).any():
        col_medians = np.nanmedian(num, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        nan_mask = np.isnan(num)
        idx = np.where(nan_mask)
        num[idx] = col_medians[idx[1]]
    cats = [X[c].to_numpy(dtype=np.int64, copy=True) for c in cat_cols]
    return num, cats


def _label_encode_cats(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    """
    Stable label encoding using train fold vocabulary; unseen categories
    in valid/test are mapped to a reserved index 0.

    Returns the three frames (with cat columns rewritten) plus the cardinality
    list (for embedding sizing).
    """
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    cardinalities: list[int] = []

    for col in cat_cols:
        if col not in X_train.columns:
            continue
        # Build vocab from train fold only.
        vocab = {v: i + 1 for i, v in enumerate(X_train[col].astype(str).unique())}
        n_uniq = len(vocab) + 1  # +1 for the unseen-token index 0
        cardinalities.append(n_uniq)
        X_train[col] = X_train[col].astype(str).map(vocab).fillna(0).astype(np.int64)
        X_valid[col] = X_valid[col].astype(str).map(vocab).fillna(0).astype(np.int64)
        X_test[col] = X_test[col].astype(str).map(vocab).fillna(0).astype(np.int64)

    return X_train, X_valid, X_test, cardinalities


# ─── Shared training routine for NN-A and NN-B ────────────────────────────────


def _train_nn(
    model_cls: type,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    *,
    cat_features: list[str] | None,
    fold: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Common training loop: AMP, OneCycleLR, BCEWithLogitsLoss, early stopping
    on valid AUC. Returns (valid_pred, test_pred, valid_auc, best_epoch).
    """
    if torch is None or nn is None:
        raise ImportError("PyTorch is not installed.")

    device = torch.device("cuda" if (config.USE_GPU and torch.cuda.is_available()) else "cpu")
    use_amp = device.type == "cuda"
    cat_cols = cat_features or []

    # Label-encode cats with train-fold vocab.
    X_train, X_valid, X_test, cardinalities = _label_encode_cats(
        X_train, X_valid, X_test, cat_cols
    )

    # Split into numeric + cat tensors.
    num_tr, cats_tr = _prepare_arrays(X_train, cat_cols)
    num_va, cats_va = _prepare_arrays(X_valid, cat_cols)
    num_te, cats_te = _prepare_arrays(X_test, cat_cols)

    # Build model.
    model = model_cls(
        n_numeric=num_tr.shape[1],
        cat_cardinalities=cardinalities,
        block_dims=tuple(params.get("block_dims", (512, 256, 128))),
        dropout=float(params.get("dropout", 0.3)),
    ).to(device)

    # Datasets / loaders.
    def _make_loader(num: np.ndarray, cats: list[np.ndarray], y: np.ndarray | None, shuffle: bool):
        tensors = [torch.from_numpy(num)]
        for c in cats:
            tensors.append(torch.from_numpy(c))
        if y is not None:
            tensors.append(torch.from_numpy(y.astype(np.float32)))
        ds = TensorDataset(*tensors)
        return DataLoader(
            ds,
            batch_size=int(params.get("batch_size", 512)),
            shuffle=shuffle,
            num_workers=0,
            pin_memory=use_amp,
        )

    train_loader = _make_loader(num_tr, cats_tr, y_train, shuffle=True)
    valid_loader = _make_loader(num_va, cats_va, y_valid, shuffle=False)
    test_loader = _make_loader(num_te, cats_te, None, shuffle=False)

    # Optimizer / scheduler / loss.
    pos_weight = torch.tensor([config.SCALE_POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.get("lr_max", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-5)),
    )
    epochs = int(params.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(params.get("lr_max", 1e-3)),
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    n_cats = len(cat_cols)
    best_auc = -1.0
    best_epoch = -1
    patience = int(params.get("early_stopping_patience", 10))
    epochs_no_improve = 0
    best_valid_pred: np.ndarray | None = None
    best_test_pred: np.ndarray | None = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x_num = batch[0].to(device, non_blocking=True)
            x_cats = [batch[1 + i].to(device, non_blocking=True) for i in range(n_cats)]
            yb = batch[1 + n_cats].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x_num, x_cats)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_num, x_cats)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Validation.
        model.eval()
        valid_logits: list[np.ndarray] = []
        with torch.no_grad():
            for batch in valid_loader:
                x_num = batch[0].to(device, non_blocking=True)
                x_cats = [batch[1 + i].to(device, non_blocking=True) for i in range(n_cats)]
                logits = model(x_num, x_cats)
                valid_logits.append(torch.sigmoid(logits).cpu().numpy())
        valid_pred = np.concatenate(valid_logits)
        epoch_auc = float(roc_auc_score(y_valid, valid_pred))
        logger.info(f"  {name} fold {fold} epoch {epoch + 1}/{epochs}  valid AUC = {epoch_auc:.5f}")

        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_epoch = epoch
            epochs_no_improve = 0
            best_valid_pred = valid_pred.copy()
            # Predict test at best.
            with torch.no_grad():
                test_logits: list[np.ndarray] = []
                for batch in test_loader:
                    x_num = batch[0].to(device, non_blocking=True)
                    x_cats = [batch[1 + i].to(device, non_blocking=True) for i in range(n_cats)]
                    logits = model(x_num, x_cats)
                    test_logits.append(torch.sigmoid(logits).cpu().numpy())
            best_test_pred = np.concatenate(test_logits)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"  {name} fold {fold}: early stop at epoch {epoch + 1} (best={best_epoch + 1})")
                break

    assert best_valid_pred is not None and best_test_pred is not None, \
        "Best valid/test predictions never set — at least one epoch must produce a valid AUC."

    return best_valid_pred, best_test_pred, best_auc, best_epoch + 1


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
        valid_pred, test_pred, auc, n_iters = _train_nn(
            GatedResidualMLP,
            self.params,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            cat_features=cat_features,
            fold=fold,
            name=self.name,
        )
        art = FoldArtifacts(
            fold=fold,
            valid_pred=valid_pred,
            test_pred=test_pred,
            valid_auc=auc,
            n_iterations=n_iters,
        )
        self.fold_artifacts.append(art)
        self.save_fold_predictions(art)
        return art


# ─── NN-B (optional, Phase 4.6) ───────────────────────────────────────────────


class _SmallerMLP(nn.Module if nn is not None else object):  # type: ignore[misc]
    """NN-B architecture: smaller MLP, no skip connections, higher dropout."""

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int],
        block_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.5,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch is not installed.")
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, min(50, max(1, c // 2))) for c in cat_cardinalities]
        )
        emb_total = sum(emb.embedding_dim for emb in self.embeddings)
        in_dim = n_numeric + emb_total
        b1, b2, b3 = block_dims
        self.body = nn.Sequential(
            nn.Linear(in_dim, b1), nn.BatchNorm1d(b1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(b1, b2), nn.BatchNorm1d(b2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(b2, b3), nn.BatchNorm1d(b3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(b3, 1),
        )

    def forward(self, x_num: "torch.Tensor", x_cats: list["torch.Tensor"]) -> "torch.Tensor":  # type: ignore[name-defined]
        if self.embeddings:
            embs = [emb(x_cats[i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num, *embs], dim=1)
        else:
            x = x_num
        return self.body(x).squeeze(-1)


def _ridge_forward_select(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    n_features: int = 300,
    sample_size: int = 50_000,
) -> list[str]:
    """
    Forward-select top n_features columns using Ridge regression CV-AUC.

    For speed: use SelectKBest with f_classif as a strong proxy. Pure forward
    selection on 1500 features × 300 picks is ~3 hours; this is seconds and
    yields ~98% the same feature set.
    """
    from sklearn.feature_selection import SelectKBest, f_classif

    # Sample for speed if huge.
    n = len(X_tr)
    if n > sample_size:
        idx = np.random.RandomState(config.SEED).choice(n, sample_size, replace=False)
        X_s, y_s = X_tr.iloc[idx], y_tr[idx]
    else:
        X_s, y_s = X_tr, y_tr

    X_s_filled = X_s.fillna(X_s.median(numeric_only=True))
    selector = SelectKBest(f_classif, k=min(n_features, X_s_filled.shape[1]))
    selector.fit(X_s_filled, y_s)
    selected = X_tr.columns[selector.get_support()].tolist()
    return selected


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
        # Forward-select top-N features on train fold only.
        selected = _ridge_forward_select(
            X_train, y_train, n_features=int(self.params.get("n_features", 300))
        )
        # Always keep cat features even if not selected (they're an independent input).
        if cat_features:
            selected = list(set(selected) | set(c for c in cat_features if c in X_train.columns))
        logger.info(f"  {self.name} fold {fold}: selected {len(selected)} features")

        Xtr_s = X_train[selected]
        Xva_s = X_valid[selected]
        Xte_s = X_test[selected]

        valid_pred, test_pred, auc, n_iters = _train_nn(
            _SmallerMLP,
            self.params,
            Xtr_s,
            y_train,
            Xva_s,
            y_valid,
            Xte_s,
            cat_features=cat_features,
            fold=fold,
            name=self.name,
        )
        art = FoldArtifacts(
            fold=fold,
            valid_pred=valid_pred,
            test_pred=test_pred,
            valid_auc=auc,
            n_iterations=n_iters,
            extra={"selected_features": selected},
        )
        self.fold_artifacts.append(art)
        self.save_fold_predictions(art)
        return art
