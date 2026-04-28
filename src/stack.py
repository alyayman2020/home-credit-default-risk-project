"""
Stacking meta-learner — Plan B per PLAN_v2.md §5.2.

Triggered when rank-averaging (src/ensemble.py) lands below 0.795 OOF.

Approach
--------
Train a ``LogisticRegression(C=1.0, class_weight='balanced')`` on:

  - The 4 base-model OOF predictions (lgbm, xgb, catboost, nn_a)
    transformed to logits to give the linear model a smoother surface.
  - A handful of the strongest raw application features (EXT_SOURCE_*,
    AMT_INCOME_TOTAL, etc.) that capture signal the GBMs may have soft-pruned.

Why LogisticRegression and not another GBM
------------------------------------------
A second-level GBM on top of GBM OOF predictions tends to over-fit to fold
boundaries (it tries to memorize per-fold tendencies). A linear meta-learner
just learns optimal weights — equivalent to optimal blending — and adds the
ability to combine model predictions with raw features that didn't make the
first cut. Cheap, robust, well-calibrated.

Cross-validation
----------------
Uses the same fold mapping as level-1 (``data/features/folds.parquet``) so the
OOF for the stacker is honest. We re-fit the meta-learner per fold and score
on held-out rows.

Outputs
-------
Same convention as ``src/train.py``:

  - ``artifacts/predictions/stack/oof.npy``      — meta-learner OOF preds
  - ``artifacts/predictions/stack/test_mean.npy`` — meta-learner test preds
  - ``artifacts/predictions/stack/summary.json`` — per-fold + OOF AUC

Use ``python -m src.submit --source stack`` to write a Kaggle CSV.

CLI
---
``python -m src.stack``                                  # default 4 models + raw feats
``python -m src.stack --models lgbm,xgb,catboost``       # subset
``python -m src.stack --no-raw-features``                # OOF-only stacker
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.cv import load_main_folds
from src.utils import get_logger, set_seed, timer

logger = get_logger()


# ─── Strong raw features (PLAN §5.2 — top 5 by univariate AUC) ────────────────

# These survive in the main matrix as numerics. We pull them from the assembled
# parquet alongside the OOF predictions.
RAW_FEATURE_COLS: list[str] = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]


# ─── Loading ──────────────────────────────────────────────────────────────────


def _load_predictions(model_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load OOF + test predictions for the named models.

    Returns
    -------
    oof_matrix : (n_train, n_models)
    test_matrix : (n_test, n_models)
    """
    oof_cols, test_cols = [], []
    for name in model_names:
        oof_path = config.PREDICTIONS_DIR / name / "oof.npy"
        test_path = config.PREDICTIONS_DIR / name / "test_mean.npy"
        if not oof_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing predictions for model {name!r}.\n"
                f"  Expected: {oof_path}\n"
                f"  Expected: {test_path}\n"
                f"  Run `uv run python -m src.train --model {name}` first."
            )
        oof_cols.append(np.load(oof_path))
        test_cols.append(np.load(test_path))
    return np.column_stack(oof_cols), np.column_stack(test_cols)


def _load_y_and_ids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TARGET (train) and SK_ID_CURR (train, test) from the main feature matrix.

    Returns
    -------
    y : (n_train,)
    train_ids : (n_train,) — for sanity alignment
    test_ids : (n_test,)
    """
    main = pl.read_parquet(config.FEATURE_MATRICES["main"].parquet_path)
    train_part = main.filter(pl.col(config.TARGET_COL).is_not_null())
    test_part = main.filter(pl.col(config.TARGET_COL).is_null())
    y = train_part[config.TARGET_COL].to_numpy()
    return (
        y,
        train_part[config.ID_COL].to_numpy(),
        test_part[config.ID_COL].to_numpy(),
    )


def _load_raw_features() -> tuple[np.ndarray, np.ndarray]:
    """
    Pull the strong raw features from the *processed* application table
    (not the main matrix — those have been transformed). Aligned on SK_ID_CURR.

    Returns
    -------
    raw_train : (n_train, len(RAW_FEATURE_COLS))
    raw_test : (n_test, len(RAW_FEATURE_COLS))
    """
    train_app = pl.read_parquet(config.PROCESSED_DIR / "application_train.parquet")
    test_app = pl.read_parquet(config.PROCESSED_DIR / "application_test.parquet")

    cols_present = [c for c in RAW_FEATURE_COLS if c in train_app.columns]
    if not cols_present:
        logger.warning("  no raw features available — falling back to OOF-only stacker")
        return np.empty((train_app.height, 0)), np.empty((test_app.height, 0))

    if cols_present != RAW_FEATURE_COLS:
        missing = set(RAW_FEATURE_COLS) - set(cols_present)
        logger.warning(f"  raw features missing from application: {missing}")

    return (
        train_app.select(cols_present).to_numpy(),
        test_app.select(cols_present).to_numpy(),
    )


# ─── Stacking ────────────────────────────────────────────────────────────────


def _to_logits(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert probabilities to logits with epsilon clipping."""
    p = np.clip(p, eps, 1.0 - eps)
    return logit(p)


def stack_oof_with_logreg(
    oof_matrix: np.ndarray,
    test_matrix: np.ndarray,
    y: np.ndarray,
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    folds: np.ndarray,
    model_names: list[str],
    *,
    C: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Train a per-fold LogisticRegression meta-learner on
    [logit(OOF), standardized(raw_features)] → predict TARGET.

    Returns
    -------
    oof_pred : (n_train,)
    test_pred : (n_test,) — averaged across fold-trained meta-learners
    fold_aucs : list of per-fold AUCs
    """
    n_train, n_models = oof_matrix.shape
    n_test = test_matrix.shape[0]

    # Convert OOF + test predictions to logit space — gives LR a smoother surface.
    oof_logits = _to_logits(oof_matrix)
    test_logits = _to_logits(test_matrix)

    # Per-fold loop. Each fold trains LR on out-of-fold rows, scores held-out fold.
    oof_pred = np.zeros(n_train, dtype=np.float64)
    test_pred_per_fold: list[np.ndarray] = []
    fold_aucs: list[float] = []

    for fold in range(config.N_FOLDS):
        valid_mask = folds == fold
        train_mask = ~valid_mask

        # Build feature matrices for LR.
        X_tr_parts = [oof_logits[train_mask]]
        X_va_parts = [oof_logits[valid_mask]]
        X_te_parts = [test_logits]

        if raw_train.shape[1] > 0:
            # Median-impute then standardize raw features (LR is scale-sensitive).
            tr_raw = raw_train[train_mask].astype(np.float64).copy()
            va_raw = raw_train[valid_mask].astype(np.float64).copy()
            te_raw = raw_test.astype(np.float64).copy()

            # Median-impute using TRAIN-fold medians only (no leakage).
            medians = np.nanmedian(tr_raw, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            for arr in (tr_raw, va_raw, te_raw):
                nan_mask = np.isnan(arr)
                if nan_mask.any():
                    idx = np.where(nan_mask)
                    arr[idx] = medians[idx[1]]

            # Standardize on train fold only.
            scaler = StandardScaler()
            tr_raw = scaler.fit_transform(tr_raw)
            va_raw = scaler.transform(va_raw)
            te_raw = scaler.transform(te_raw)

            X_tr_parts.append(tr_raw)
            X_va_parts.append(va_raw)
            X_te_parts.append(te_raw)

        X_tr = np.column_stack(X_tr_parts)
        X_va = np.column_stack(X_va_parts)
        X_te = np.column_stack(X_te_parts)

        clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=config.SEED,
        )
        clf.fit(X_tr, y[train_mask])

        valid_pred = clf.predict_proba(X_va)[:, 1]
        oof_pred[valid_mask] = valid_pred
        fold_auc = float(roc_auc_score(y[valid_mask], valid_pred))
        fold_aucs.append(fold_auc)

        # Test predictions — average across folds.
        test_pred_per_fold.append(clf.predict_proba(X_te)[:, 1])

        # Log learned weights on the OOF logit columns (interpretability).
        w = clf.coef_[0]
        oof_w = w[:n_models]
        weights_str = ", ".join(f"{m}={oof_w[i]:.3f}" for i, m in enumerate(model_names))
        logger.info(
            f"  fold {fold}: AUC={fold_auc:.5f}  weights({weights_str})"
            + (f"  + {len(w) - n_models} raw" if len(w) > n_models else "")
        )

    test_pred = np.mean(test_pred_per_fold, axis=0)
    return oof_pred, test_pred, fold_aucs


# ─── CLI ─────────────────────────────────────────────────────────────────────


def run_stack(
    model_names: list[str],
    *,
    use_raw_features: bool = True,
    C: float = 1.0,
) -> dict:
    """End-to-end: load preds, stack, save outputs, log summary."""
    logger.info(config.summary())
    set_seed(config.SEED)
    logger.info(f"  Stacking models: {model_names}")
    logger.info(f"  Raw features: {'enabled' if use_raw_features else 'disabled'}")

    with timer("load OOF + test predictions"):
        oof_matrix, test_matrix = _load_predictions(model_names)
    logger.info(f"  oof_matrix: {oof_matrix.shape}  test_matrix: {test_matrix.shape}")

    with timer("load y + ids"):
        y, _, _ = _load_y_and_ids()
    if len(y) != oof_matrix.shape[0]:
        raise ValueError(
            f"OOF row count ({oof_matrix.shape[0]}) doesn't match TARGET length ({len(y)}). "
            "Did you re-run features without re-running training?"
        )

    raw_train = np.empty((len(y), 0))
    raw_test = np.empty((test_matrix.shape[0], 0))
    if use_raw_features:
        with timer("load raw features"):
            raw_train, raw_test = _load_raw_features()
        logger.info(f"  raw_train: {raw_train.shape}  raw_test: {raw_test.shape}")

    with timer("load fold mapping"):
        folds_pl = load_main_folds()
        folds = folds_pl["fold_main"].to_numpy()
    if len(folds) != len(y):
        raise ValueError(
            f"folds length ({len(folds)}) != y length ({len(y)}). "
            "Folds were built on a different application_train than the one in main matrix."
        )

    with timer("stack with LogisticRegression"):
        oof_pred, test_pred, fold_aucs = stack_oof_with_logreg(
            oof_matrix, test_matrix, y, raw_train, raw_test, folds,
            model_names, C=C,
        )

    oof_auc = float(roc_auc_score(y, oof_pred))
    fold_mean = float(np.mean(fold_aucs))
    fold_std = float(np.std(fold_aucs))

    logger.info("")
    logger.success(f"  stack: OOF AUC = {oof_auc:.5f}")
    logger.info(f"  stack: per-fold = {fold_mean:.5f} ± {fold_std:.5f}")

    # Compare to best single base model so we can decide whether to submit the stacker.
    base_aucs = []
    for name in model_names:
        summary_path = config.PREDICTIONS_DIR / name / "summary.json"
        if summary_path.exists():
            base_aucs.append((name, json.loads(summary_path.read_text())["oof_auc"]))
    if base_aucs:
        best_name, best_oof = max(base_aucs, key=lambda x: x[1])
        delta = oof_auc - best_oof
        if delta > 0:
            logger.success(
                f"  stack lift vs best base ({best_name}={best_oof:.5f}): +{delta:.5f}"
            )
        else:
            logger.warning(
                f"  stack vs best base ({best_name}={best_oof:.5f}): {delta:+.5f}  "
                "(meta-learner did not improve — submit the original blend instead)"
            )

    # Persist outputs in the same convention as src/train.py.
    out_dir = config.PREDICTIONS_DIR / "stack"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", oof_pred)
    np.save(out_dir / "test_mean.npy", test_pred)
    summary = {
        "model": "stack",
        "matrix": "meta",
        "base_models": model_names,
        "use_raw_features": use_raw_features,
        "C": C,
        "oof_auc": oof_auc,
        "fold_aucs": fold_aucs,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"  → wrote {out_dir / 'oof.npy'}")
    logger.info(f"  → wrote {out_dir / 'test_mean.npy'}")
    logger.info(f"  → wrote {out_dir / 'summary.json'}")
    logger.info(
        "  Submit with: uv run python -m src.submit --source stack"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Logistic-regression stacker (Plan B).")
    p.add_argument(
        "--models",
        type=str,
        default="lgbm,xgb,catboost,nn_a",
        help="Comma-separated list of base model names (default: lgbm,xgb,catboost,nn_a).",
    )
    p.add_argument(
        "--no-raw-features",
        action="store_true",
        help="Disable the strong-raw-features arm; stack on OOF predictions only.",
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="LogisticRegression inverse-regularization strength (default: 1.0).",
    )
    return p


def _main() -> None:
    args = _build_parser().parse_args()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    run_stack(
        model_names,
        use_raw_features=not args.no_raw_features,
        C=args.C,
    )


if __name__ == "__main__":
    _main()
