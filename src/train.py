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
   b. 🆕 D1: Build a NearestNeighbors index on the 4 D1 features only
      (EXT_SOURCE_1/2/3 + APP_CREDIT_TERM) of the train fold. Query valid + test.
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

# Suffix used for nan-flag columns produced by features.assemble.to_nn_matrix.
# Must match what's written there.
NAN_FLAG_SUFFIX: str = "__isnan"


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
            f"Feature matrix {spec.parquet_path} missing — run "
            f"`uv run python -m src.features.assemble` first."
        )
    return pl.read_parquet(spec.parquet_path)


def load_catboost_cat_features() -> list[str]:
    """Read the catboost cat-feature sidecar written by assemble.to_catboost_matrix."""
    path = config.FEATURES_DIR / "cat_features.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


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

    Adds ``{col}_TE`` columns to X_tr, X_va, X_test (returns copies).
    Handles both string and pandas ``category`` dtype source columns.
    """
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_test = X_test.copy()
    global_mean = float(y_tr.mean())

    for col in cat_cols:
        if col not in X_tr.columns:
            continue
        # Cast to plain string to make ``map`` / ``groupby`` agnostic to dtype.
        # (Avoids edge cases where ``category`` dtype on test has extra/missing levels.)
        tr_vals = X_tr[col].astype("object")
        va_vals = X_va[col].astype("object")
        te_vals = X_test[col].astype("object")

        df_tr = pd.DataFrame({col: tr_vals.values, "_y": y_tr})
        agg = df_tr.groupby(col, dropna=False)["_y"].agg(["sum", "count"])
        agg["enc"] = (agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)
        mapping = agg["enc"].to_dict()

        new_col = f"{col}_TE"
        X_tr[new_col] = tr_vals.map(mapping).astype(float).fillna(global_mean).values
        X_va[new_col] = va_vals.map(mapping).astype(float).fillna(global_mean).values
        X_test[new_col] = te_vals.map(mapping).astype(float).fillna(global_mean).values

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

    Builds the index on the *training* rows of the current fold using ONLY the
    4 D1 features (EXT_SOURCE_1/2/3 + APP_CREDIT_TERM). Queries are made against
    the same 4-column subset.

    For X_va: same 4 cols, distinct rows → genuine OOF neighbour mean.
    For X_test: same 4 cols, full test set.
    For X_tr (the training rows themselves): we use the k-nearest including self
    (acceptable bias at k=500 — see PLAN §2.10).
    """
    cols_present = [c for c in feature_cols if c in X_tr.columns]
    if len(cols_present) < 2:
        # Not enough D1 features available → degrade gracefully to global mean.
        gm = float(y_tr.mean())
        logger.warning(
            f"  D1: only {len(cols_present)} of {len(feature_cols)} neighbour "
            f"features present — falling back to global mean"
        )
        return (
            pd.Series(np.full(len(X_tr), gm), index=X_tr.index),
            pd.Series(np.full(len(X_va), gm), index=X_va.index),
            pd.Series(np.full(len(X_test), gm), index=X_test.index),
        )

    # Median imputation fit on train fold only.
    medians = X_tr[cols_present].median()
    A = X_tr[cols_present].fillna(medians).to_numpy(dtype=np.float32)
    B = X_va[cols_present].fillna(medians).to_numpy(dtype=np.float32)
    C = X_test[cols_present].fillna(medians).to_numpy(dtype=np.float32)

    # Defensive: if any inf snuck through (e.g. APP_CREDIT_TERM division by zero
    # combined with median that was itself inf), replace with 0 in the fp32 view.
    A[~np.isfinite(A)] = 0.0
    B[~np.isfinite(B)] = 0.0
    C[~np.isfinite(C)] = 0.0

    k_eff = min(k, len(A))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1)
    nn.fit(A)

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

    Fits ``QuantileTransformer(output_distribution='normal')`` on numeric columns
    of ``X_tr`` only, then transforms ``X_va`` and ``X_test``. Columns ending in
    :data:`NAN_FLAG_SUFFIX` (``__isnan``) are left alone — they're already in {0, 1}.
    Integer-encoded categorical columns (suffix ``__nn``) are also left alone.

    Inf sanitization
    ----------------
    Some engineered features (APP_CREDIT_TERM = AMT_ANNUITY/AMT_CREDIT,
    PAYMENT_PERC = AMT_PAYMENT/AMT_INSTALMENT, etc.) can produce ``+inf``/``-inf``
    when the denominator is zero. ``QuantileTransformer`` rejects these. We
    replace ``inf``/``-inf`` with ``NaN`` before median-imputing — same treatment
    as XGBoost's ``missing`` mechanism (PLAN §1.4 sentinel policy).

    Returns copies — does not mutate inputs.
    """
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_test = X_test.copy()

    # Identify columns to RankGauss-transform: numeric, not a flag, not a label-encoded cat.
    numeric_cols = [
        c for c in X_tr.columns
        if pd.api.types.is_numeric_dtype(X_tr[c])
        and not c.endswith(NAN_FLAG_SUFFIX)
        and not c.endswith("__nn")
    ]
    if not numeric_cols:
        return X_tr, X_va, X_test

    # Step 1: Replace inf/-inf with NaN so median-impute and qt.fit don't choke.
    # We only mutate the slices we'll feed to qt — leaves flag/encoded cat cols intact.
    n_inf_total = 0
    for col in numeric_cols:
        for df in (X_tr, X_va, X_test):
            arr = df[col].to_numpy()
            inf_mask = np.isinf(arr)
            if inf_mask.any():
                n_inf_total += int(inf_mask.sum())
                # Cast to float64 to allow NaN assignment without dtype surprises.
                arr = arr.astype(np.float64, copy=True)
                arr[inf_mask] = np.nan
                df[col] = arr
    if n_inf_total:
        logger.info(f"  D2 RankGauss: replaced {n_inf_total} inf values with NaN before transform")

    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(X_tr)),
        output_distribution="normal",
        random_state=config.SEED,
    )
    # Median-impute before quantile transform (qt doesn't tolerate NaN).
    medians = X_tr[numeric_cols].median()
    # Defensive: if a column is all-NaN on train fold, median is NaN — fall back to 0.
    medians = medians.fillna(0.0)
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
    - 🆕 D1 neighbours target mean      — fold-aware NN index on 4 cols
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

    # For catboost matrix, pull cat_features list from the sidecar.
    # Caller-provided cat_features takes precedence.
    if cat_features is None and model.matrix == "catboost":
        cat_features = load_catboost_cat_features()
        if cat_features:
            logger.info(f"  catboost: loaded {len(cat_features)} cat features from sidecar")

    with timer("Polars → pandas"):
        train_df = train_pl.to_pandas()
        test_df = test_pl.to_pandas()

    y = train_df[config.TARGET_COL].to_numpy()
    drop_cols_train = [config.ID_COL, config.TARGET_COL, fold_col]
    X = train_df.drop(columns=drop_cols_train)
    X_test_full = test_df.drop(columns=[config.ID_COL, config.TARGET_COL])

    # Identify which high-cardinality cats are still present (for OOF target encoding).
    # On the catboost matrix these are kept as raw cats — DON'T target-encode them
    # (CatBoost has its own internal encoding).
    cat_cols_for_te: list[str] = []
    if model.matrix == "main":
        cat_cols_for_te = [c for c in HIGH_CARD_COLS if c in X.columns]
        if cat_cols_for_te:
            logger.info(
                f"  main matrix: {len(cat_cols_for_te)} high-card cats present for "
                f"per-fold target encoding: {cat_cols_for_te}"
            )

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
        if cat_cols_for_te:
            with timer(f"  fold {fold}: OOF target encoding"):
                X_tr_f, X_va_f, X_test_f = oof_target_encode(
                    X_tr_f, y_tr, X_va_f, X_test_f, cat_cols=cat_cols_for_te
                )
            # Drop the raw cat cols — they're string and would break the model.
            for c in cat_cols_for_te:
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
