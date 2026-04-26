"""
Training driver — runs 5-fold OOF training for any registered model.

Reference: PLAN_v2.md §3 (validation), §4 (modeling), §2.10 (anti-leakage).

What this file owns
-------------------
1. Load the right feature matrix for the chosen model (main / catboost / nn).
2. Loop over folds (StratifiedKFold for primary models; GroupKFold available
   for auxiliary models via ``--group``).
3. **Inside the fold loop, on a per-fold basis**:
   a. Compute OOF target encoding (smoothed) for high-cardinality cats.
   b. 🆕 D1: Build a NearestNeighbors index on `train[fold != k]`, query
      `train[fold == k]` to get TARGET_NEIGHBORS_500_MEAN. For test, build
      a full-train index and query test.
   c. 🆕 D2: Fit RankGauss (QuantileTransformer) on train fold only when
      using the NN matrix; transform train+valid inside the fold.
4. Call ``model.fit_fold(...)`` and persist artifacts.
5. Concatenate OOF predictions, compute concatenated AUC + per-fold mean ± std.

CLI
---
``python -m src.train --model lgbm --mode baseline``
``python -m src.train --model xgb  --mode tuned``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from src import config
from src.cv import load_group_folds, load_main_folds
from src.models.base import ModelBase
from src.utils import get_logger, set_seed, timer

logger = get_logger()


# ─── Model registry ───────────────────────────────────────────────────────────


def get_model(name: str, params: dict[str, Any] | None = None) -> ModelBase:
    """Return a fresh model instance by name."""
    if name == "lgbm":
        from src.models.lgbm import LGBMModel

        return LGBMModel(params)
    if name == "xgb":
        from src.models.xgb import XGBModel

        return XGBModel(params)
    if name == "catboost":
        from src.models.catboost import CatBoostModel

        return CatBoostModel(params)
    if name == "nn_a":
        from src.models.nn import NNAModel

        return NNAModel(params)
    if name == "nn_b":
        from src.models.nn import NNBModel

        return NNBModel(params)
    raise ValueError(f"Unknown model: {name!r}")


# ─── Matrix loaders ───────────────────────────────────────────────────────────


def load_feature_matrix(matrix: str) -> pl.DataFrame:
    """Load one of the three feature matrices."""
    spec = config.FEATURE_MATRICES[matrix]
    if not spec.parquet_path.exists():
        raise FileNotFoundError(
            f"Feature matrix {spec.parquet_path} missing — run `make features` first."
        )
    return pl.read_parquet(spec.parquet_path)


def split_train_test(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the assembled matrix back into train (TARGET not null) and test."""
    train = df.filter(pl.col(config.TARGET_COL).is_not_null())
    test = df.filter(pl.col(config.TARGET_COL).is_null())
    return train, test


# ─── OOF training ─────────────────────────────────────────────────────────────


def run_oof(
    model: ModelBase,
    *,
    use_group_folds: bool = False,
    cat_features: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run 5-fold OOF training for ``model``. Returns a summary dict with the
    OOF predictions and per-fold AUCs.

    Per-fold work that lives HERE (not in the feature builders), to keep
    leakage-prone steps inside the CV loop (PLAN §2.10):

    - OOF target encoding              — fold-aware
    - 🆕 D1 neighbours target mean      — fold-aware NN index
    - 🆕 D2 RankGauss                   — fit on train fold, transform valid+test fold
    """
    logger.info(config.summary())
    set_seed(config.SEED)

    # 1. Load matrix + folds
    with timer(f"load matrix '{model.matrix}'"):
        full = load_feature_matrix(model.matrix)
        train_pl, test_pl = split_train_test(full)

    folds = load_group_folds() if use_group_folds else load_main_folds()
    fold_col = "fold_group" if use_group_folds else "fold_main"

    train_pl = train_pl.join(folds, on=config.ID_COL)

    # Convert to pandas at the boundary (PLAN §A1: Polars pipeline → pandas for GBMs).
    with timer("Polars → pandas"):
        train_df = train_pl.to_pandas()
        test_df = test_pl.to_pandas()

    y = train_df[config.TARGET_COL].to_numpy()
    drop_cols = [config.ID_COL, config.TARGET_COL, fold_col]
    X = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=[config.ID_COL, config.TARGET_COL])

    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: list[float] = []

    for fold in range(config.N_FOLDS):
        valid_mask = train_df[fold_col].to_numpy() == fold
        train_mask = ~valid_mask
        logger.info(
            f"\n────── Fold {fold} ──────  "
            f"(train={train_mask.sum()}, valid={valid_mask.sum()})"
        )

        X_tr_f, X_va_f = X[train_mask].copy(), X[valid_mask].copy()
        y_tr, y_va = y[train_mask], y[valid_mask]

        # TODO: per-fold transforms go HERE
        #   1. OOF target encoding for high-cardinality cats:
        #        compute smoothed mean(TARGET) on X_tr_f, apply to X_va_f and X_test
        #        (X_test gets the encoding from the *full* train below — or fold-mean).
        #   2. 🆕 D1 neighbours target mean (for the 'main' / 'catboost' matrices):
        #        idx = NearestNeighbors(n=500).fit(X_tr_f[NEIGHBOR_COLS])
        #        nn_mean[valid_mask] = mean(y_tr[idx.kneighbors(X_va_f[NEIGHBOR_COLS])])
        #        Smoke test: assert different values per fold for the same row.
        #   3. 🆕 D2 RankGauss (only for matrix='nn'):
        #        QuantileTransformer(output_distribution='normal').fit(X_tr_f[numerics])
        #        applied to X_va_f and X_test_fold.

        with timer(f"fit fold {fold} ({model.name})"):
            art = model.fit_fold(
                X_train=X_tr_f,
                y_train=y_tr,
                X_valid=X_va_f,
                y_valid=y_va,
                X_test=X_test,
                fold=fold,
                cat_features=cat_features,
            )

        oof[valid_mask] = art.valid_pred
        fold_aucs.append(art.valid_auc)
        logger.info(f"  fold {fold} AUC = {art.valid_auc:.5f}")

    # Concatenated OOF AUC + per-fold mean / std (PLAN §3.2)
    oof_auc = float(roc_auc_score(y, oof))
    fold_mean = float(np.mean(fold_aucs))
    fold_std = float(np.std(fold_aucs))

    logger.info("")
    logger.success(f"  {model.name}: OOF AUC = {oof_auc:.5f}")
    logger.info(f"  {model.name}: per-fold = {fold_mean:.5f} ± {fold_std:.5f}")
    if fold_std > 0.005:
        logger.warning(f"  per-fold std {fold_std:.5f} > 0.005 — investigate fold imbalance")
    if abs(oof_auc - fold_mean) > 0.001:
        logger.warning(
            f"  |OOF − fold-mean| = {abs(oof_auc - fold_mean):.5f} > 0.001 — investigate"
        )

    # Persist OOF + aggregate test
    out_dir = config.PREDICTIONS_DIR / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", oof)
    np.save(out_dir / "test_mean.npy", model.aggregate_test_predictions())
    summary = {
        "model": model.name,
        "matrix": model.matrix,
        "oof_auc": oof_auc,
        "fold_aucs": fold_aucs,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "n_iterations_mean": float(np.mean([a.n_iterations for a in model.fold_artifacts])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OOF training driver.")
    p.add_argument(
        "--model",
        required=True,
        choices=["lgbm", "xgb", "catboost", "nn_a", "nn_b"],
        help="Which model wrapper to train.",
    )
    p.add_argument(
        "--mode",
        choices=["baseline", "tuned"],
        default="baseline",
        help="Use default_params() or load tuned params from artifacts/tuning_log.csv.",
    )
    p.add_argument(
        "--group",
        action="store_true",
        help="Use GroupKFold (for auxiliary monthly-table models).",
    )
    return p


def _load_tuned_params(model_name: str) -> dict[str, Any] | None:
    """
    Load tuned hyperparameters for ``model_name`` from the tuning log.

    TODO: read ``artifacts/tuning_log.csv`` (written by ``src/tune.py``) and
    return the user-confirmed-best row's params. For now, returns None
    (falls back to default_params()).
    """
    path = Path(config.ARTIFACTS_DIR / "tuning_log.csv")
    if not path.exists():
        logger.warning(f"  no tuning log at {path} — using default_params()")
        return None
    logger.warning(f"  tuned-mode loading from {path} is TODO — using default_params()")
    return None


def _main() -> None:
    args = _build_parser().parse_args()
    params = _load_tuned_params(args.model) if args.mode == "tuned" else None
    model = get_model(args.model, params=params)
    run_oof(model, use_group_folds=args.group)


if __name__ == "__main__":
    _main()
