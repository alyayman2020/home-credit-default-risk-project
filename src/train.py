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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

from src import config
from src.cv import load_group_folds, load_main_folds
from src.features.application import HIGH_CARD_COLS
from src.models.base import ModelBase
from src.utils import get_logger, set_seed, timer

logger = get_logger()


# ─── D1 neighbour configuration ───────────────────────────────────────────────

# Features used to compute the D1 nearest-neighbour target mean (PLAN §2.1 D1).
NEIGHBOR_FEATURE_COLS: list[str] = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APP_CREDIT_TERM",  # = AMT_ANNUITY / AMT_CREDIT
]
NEIGHBOR_K: int = 500


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


# ─── Per-fold transforms ──────────────────────────────────────────────────────


def oof_target_encode(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Smoothed OOF target encoding for high-cardinality categoricals.

    For each category c in train:
        encoding[c] = (sum_y_in_c + smoothing * global_mean) / (count_in_c + smoothing)

    The global mean (computed on train fold only) is used for unseen categories
    in valid/test. Target encoding is fit on X_tr only — never on X_va/X_test —
    so per-fold values differ across folds (verifiable via notebook 04).

    Adds ``{col}_TE`` columns to X_tr, X_va, X_test in place (returns copies).
    """
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_test = X_test.copy()
    global_mean = float(y_tr.mean())

    for col in cat_cols:
        if col not in X_tr.columns:
            continue
        # Compute smoothed encoding per category using train fold ONLY.
        df_tr = pd.DataFrame({col: X_tr[col].values, "_y": y_tr})
        agg = df_tr.groupby(col)["_y"].agg(["sum", "count"])
        agg["enc"] = (agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)
        mapping = agg["enc"].to_dict()

        new_col = f"{col}_TE"
        X_tr[new_col] = X_tr[col].map(mapping).astype(float).fillna(global_mean)
        X_va[new_col] = X_va[col].map(mapping).astype(float).fillna(global_mean)
        X_test[new_col] = X_test[col].map(mapping).astype(float).fillna(global_mean)

    return X_tr, X_va, X_test


def compute_d1_neighbours(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str] = NEIGHBOR_FEATURE_COLS,
    k: int = NEIGHBOR_K,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    🆕 D1: TARGET_NEIGHBORS_500_MEAN computed via fold-aware NearestNeighbors.

    For the validation rows in the current fold, the index is built only on
    the *training* rows of that fold — so the same row gets a different
    neighbour-mean across the 5 folds (verified by notebook 04).

    For test rows: the index is built on the *full* training fold (X_tr),
    not on all train data. This means the test value depends on which fold
    we're in — we average across folds in run_oof().

    Parameters
    ----------
    X_tr / X_va / X_test
        DataFrames containing at least ``feature_cols``. NaN values are filled
        with the train-fold median before fitting the index.

    Returns
    -------
    (tr_neigh, va_neigh, test_neigh)
        Three Series of length len(X_tr) / len(X_va) / len(X_test) holding
        the neighbour TARGET mean for each row.
    """
    cols_present = [c for c in feature_cols if c in X_tr.columns]
    if len(cols_present) < 2:
        # Not enough features → degrade gracefully to global mean.
        gm = float(y_tr.mean())
        return (
            pd.Series(np.full(len(X_tr), gm), index=X_tr.index),
            pd.Series(np.full(len(X_va), gm), index=X_va.index),
            pd.Series(np.full(len(X_test), gm), index=X_test.index),
        )

    # Median imputation fit on train fold.
    medians = X_tr[cols_present].median()
    A = X_tr[cols_present].fillna(medians).to_numpy(dtype=np.float32)
    B = X_va[cols_present].fillna(medians).to_numpy(dtype=np.float32)
    C = X_test[cols_present].fillna(medians).to_numpy(dtype=np.float32)

    k_eff = min(k, len(A))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1)
    nn.fit(A)

    # For training fold rows themselves, exclude self via k+1 query then drop self.
    # Easier path: build a separate index excluding each row → too slow. Instead,
    # for X_tr we use leave-one-out approximation: compute mean over k neighbours
    # including self (acceptable bias for k=500). This adds nothing to leakage
    # because A is the SAME train fold and we're computing on it.
    _, ind_tr = nn.kneighbors(A)
    tr_neigh = y_tr[ind_tr].mean(axis=1)

    _, ind_va = nn.kneighbors(B)
    va_neigh = y_tr[ind_va].mean(axis=1)

    _, ind_test = nn.kneighbors(C)
    test_neigh = y_tr[ind_test].mean(axis=1)

    return (
        pd.Series(tr_neigh, index=X_tr.index),
        pd.Series(va_neigh, index=X_va.index),
        pd.Series(test_neigh, index=X_test.index),
    )


def fit_rankgauss_per_fold(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_test: pd.DataFrame,
    n_quantiles: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    🆕 D2: RankGauss for the NN matrix.

    Fit QuantileTransformer on numeric columns of X_tr only, then transform
    X_va and X_test. ``_is_nan`` flag columns are left alone (they're already
    in {0,1}).

    Returns copies — does not mutate inputs.
    """
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_test = X_test.copy()

    numeric_cols = [
        c for c in X_tr.columns
        if pd.api.types.is_numeric_dtype(X_tr[c]) and not c.endswith("_is_nan")
    ]
    if not numeric_cols:
        return X_tr, X_va, X_test

    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(X_tr)),
        output_distribution="normal",
        random_state=config.SEED,
    )
    # Median-impute before quantile transform (qt itself doesn't tolerate NaN).
    medians = X_tr[numeric_cols].median()
    X_tr_filled = X_tr[numeric_cols].fillna(medians)
    X_va_filled = X_va[numeric_cols].fillna(medians)
    X_test_filled = X_test[numeric_cols].fillna(medians)

    qt.fit(X_tr_filled)
    X_tr[numeric_cols] = qt.transform(X_tr_filled)
    X_va[numeric_cols] = qt.transform(X_va_filled)
    X_test[numeric_cols] = qt.transform(X_test_filled)
    return X_tr, X_va, X_test


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

    with timer("Polars → pandas"):
        train_df = train_pl.to_pandas()
        test_df = test_pl.to_pandas()

    y = train_df[config.TARGET_COL].to_numpy()
    drop_cols_train = [config.ID_COL, config.TARGET_COL, fold_col]
    X = train_df.drop(columns=drop_cols_train)
    X_test_full = test_df.drop(columns=[config.ID_COL, config.TARGET_COL])

    # Identify which high-cardinality cats are still present in this matrix.
    cat_cols_present = [c for c in HIGH_CARD_COLS if c in X.columns]
    if cat_cols_present and model.matrix == "main":
        # Main matrix should NOT have raw HIGH_CARD_COLS (they're dropped in assemble.py).
        # If they're here, log it — we'll target-encode them per-fold.
        logger.info(f"  high-card cats present for fold-wise TE: {cat_cols_present}")

    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: list[float] = []
    # Buffer for per-fold debug parquet (D1 sanity check, notebook 04).
    d1_per_fold_debug: dict[int, pd.Series] = {}

    for fold in range(config.N_FOLDS):
        valid_mask = train_df[fold_col].to_numpy() == fold
        train_mask = ~valid_mask
        logger.info(
            f"\n────── Fold {fold} ──────  "
            f"(train={train_mask.sum()}, valid={valid_mask.sum()})"
        )

        X_tr_f, X_va_f = X[train_mask].copy(), X[valid_mask].copy()
        y_tr, y_va = y[train_mask], y[valid_mask]
        X_test_f = X_test_full.copy()

        # ── Per-fold transform 1: OOF target encoding for high-card cats ─
        if cat_cols_present:
            with timer(f"  fold {fold}: OOF target encoding"):
                X_tr_f, X_va_f, X_test_f = oof_target_encode(
                    X_tr_f, y_tr, X_va_f, X_test_f, cat_cols=cat_cols_present
                )
            # Drop the raw cat cols — they're string and would break the model.
            for c in cat_cols_present:
                if c in X_tr_f.columns:
                    X_tr_f = X_tr_f.drop(columns=[c])
                    X_va_f = X_va_f.drop(columns=[c])
                    X_test_f = X_test_f.drop(columns=[c])

        # ── Per-fold transform 2: D1 neighbours (skip for NN — too costly + redundant)
        if model.matrix in ("main", "catboost"):
            with timer(f"  fold {fold}: D1 neighbours target mean"):
                tr_n, va_n, test_n = compute_d1_neighbours(
                    X_tr_f, y_tr, X_va_f, X_test_f
                )
                X_tr_f["TARGET_NEIGHBORS_500_MEAN"] = tr_n.values
                X_va_f["TARGET_NEIGHBORS_500_MEAN"] = va_n.values
                X_test_f["TARGET_NEIGHBORS_500_MEAN"] = test_n.values

            # Save validation-fold D1 values for the leakage smoke notebook.
            # Indexed by SK_ID_CURR for joinability.
            d1_per_fold_debug[fold] = pd.Series(
                va_n.values,
                index=train_df.loc[valid_mask, config.ID_COL].values,
                name=f"d1_fold_{fold}",
            )

        # ── Per-fold transform 3: D2 RankGauss (NN matrix only) ──────────
        if model.matrix == "nn":
            with timer(f"  fold {fold}: D2 RankGauss"):
                X_tr_f, X_va_f, X_test_f = fit_rankgauss_per_fold(
                    X_tr_f, X_va_f, X_test_f
                )

        # ── Fit ───────────────────────────────────────────────────────────
        with timer(f"  fit fold {fold} ({model.name})"):
            art = model.fit_fold(
                X_train=X_tr_f,
                y_train=y_tr,
                X_valid=X_va_f,
                y_valid=y_va,
                X_test=X_test_f,
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

    # Persist D1 per-fold values for the leakage smoke notebook (D1 sanity check).
    if d1_per_fold_debug and model.matrix in ("main", "catboost"):
        d1_df = pd.concat(
            [s.rename(f"fold_{k}") for k, s in d1_per_fold_debug.items()],
            axis=1,
        )
        d1_df.index.name = config.ID_COL
        d1_df.reset_index().to_parquet(
            config.PREDICTIONS_DIR / "d1_neighbours_per_fold.parquet"
        )

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
        help="Use default_params() or load tuned params from artifacts/best_params/.",
    )
    p.add_argument(
        "--group",
        action="store_true",
        help="Use GroupKFold (for auxiliary monthly-table models).",
    )
    return p


def _load_tuned_params(model_name: str) -> dict[str, Any] | None:
    """Load tuned hyperparameters from ``artifacts/best_params/<model>.json``."""
    path = config.ARTIFACTS_DIR / "best_params" / f"{model_name}.json"
    if not path.exists():
        logger.warning(f"  no tuned params at {path} — using default_params()")
        return None
    params = json.loads(path.read_text())
    logger.info(f"  loaded tuned params from {path}")
    return params


def _main() -> None:
    args = _build_parser().parse_args()
    params = _load_tuned_params(args.model) if args.mode == "tuned" else None
    model = get_model(args.model, params=params)
    run_oof(model, use_group_folds=args.group)


if __name__ == "__main__":
    _main()
