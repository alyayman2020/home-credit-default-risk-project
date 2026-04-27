"""
🆕 Phase 4.5 — Auxiliary GBM features (D7).

Reference: PLAN_v2.md §4.5.

**Gated**: only run if 3-GBM ensemble OOF AUC ≥ 0.792.

What this does
--------------
Train a *row-level* LightGBM on each of the two big monthly-ish tables:

1. ``installments_payments`` (13.6M rows) — predict TARGET on each installment
   row. Target leakage prevented by **GroupKFold by ``SK_ID_CURR``** (B1 / PLAN §3.1):
   each borrower lives in exactly one fold.
2. ``previous_application`` — same recipe at the application level.

Out-of-fold row-level predictions are then aggregated back to the
``SK_ID_CURR`` level as 4 features per source table:

    {prefix}_AUX_OOF_MEAN
    {prefix}_AUX_OOF_MAX
    {prefix}_AUX_OOF_MIN
    {prefix}_AUX_OOF_LAST3M_MEAN

→ 8 new features total. Expected lift: +0.001 to +0.003 OOF.

Smoke test (CRITICAL)
---------------------
After building groups, assert no ``SK_ID_CURR`` appears in both train and valid
within any fold. Without this, the auxiliary model leaks the target.

CLI
---
``python -m src.auxiliary --table installments``
``python -m src.auxiliary --table previous``
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from src import config
from src.cv import load_main_folds
from src.data import read_processed
from src.utils import get_logger, set_seed, timer

logger = get_logger()


GATE_OOF_AUC = 0.792


# ─── Smoke test ───────────────────────────────────────────────────────────────


def _assert_group_split(
    train_idx: np.ndarray, valid_idx: np.ndarray, groups: np.ndarray, fold: int
) -> None:
    """Assert no SK_ID_CURR appears in both train and valid for this fold."""
    train_ids = set(groups[train_idx].tolist())
    valid_ids = set(groups[valid_idx].tolist())
    overlap = train_ids & valid_ids
    if overlap:
        raise RuntimeError(
            f"GroupKFold fold {fold}: {len(overlap)} SK_ID_CURR overlap between "
            "train and valid. Auxiliary model would LEAK — abort."
        )


# ─── Per-table auxiliary builders ─────────────────────────────────────────────


def _train_row_level(
    rows: pl.DataFrame,
    *,
    prefix: str,
    months_col: str | None,
    drop_cols: list[str],
    n_splits: int = config.N_FOLDS,
) -> pl.DataFrame:
    """
    Run group-K-fold row-level LightGBM, aggregate OOF preds back to SK_ID_CURR.

    Parameters
    ----------
    rows
        Polars DataFrame with one row per monthly observation. Must include
        ``SK_ID_CURR`` and optionally a months column for last-3-month aggregation.
    prefix
        Output column prefix (e.g. ``"INS"`` or ``"PREV"``).
    months_col
        Column to filter for last-3-month aggregation, or None to skip.
    drop_cols
        Columns to drop from the feature matrix (IDs, target).
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm is required for auxiliary models.") from e

    set_seed(config.SEED)

    # Attach TARGET via application_train + the main fold map (training rows only).
    app = read_processed("application_train").select([config.ID_COL, config.TARGET_COL])
    rows = rows.join(app, on=config.ID_COL, how="inner")  # drops test-only borrowers
    logger.info(f"  {prefix}: row-level training set = {rows.height:,} rows")

    pdf = rows.to_pandas()
    y = pdf[config.TARGET_COL].to_numpy()
    groups = pdf[config.ID_COL].to_numpy()

    feat_cols = [c for c in pdf.columns if c not in drop_cols + [config.TARGET_COL]]
    X = pdf[feat_cols]

    oof = np.zeros(len(y), dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups)):
        _assert_group_split(tr_idx, va_idx, groups, fold)
        logger.info(f"  {prefix} fold {fold}: train={len(tr_idx)}, valid={len(va_idx)}")

        train_set = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        valid_set = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=train_set)
        booster = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.5,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "device_type": "gpu" if config.USE_GPU else "cpu",
                "scale_pos_weight": config.SCALE_POS_WEIGHT,
                "seed": config.SEED,
            },
            train_set,
            num_boost_round=2000,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = booster.predict(X.iloc[va_idx], num_iteration=booster.best_iteration)

    # Aggregate row-level OOF back to SK_ID_CURR
    pdf["_oof"] = oof
    agg_cols = {
        f"{prefix}_AUX_OOF_MEAN": ("_oof", "mean"),
        f"{prefix}_AUX_OOF_MAX": ("_oof", "max"),
        f"{prefix}_AUX_OOF_MIN": ("_oof", "min"),
    }
    pdf_curr = pdf.groupby(config.ID_COL).agg(**agg_cols).reset_index()  # type: ignore[arg-type]

    if months_col is not None and months_col in pdf.columns:
        # Detect whether this is a DAYS_* column (negative integers, ~ -1 to -3000)
        # or a MONTHS_BALANCE column (negative ints, ~ -1 to -96).
        # Last-3-months threshold: -90 days OR -3 months.
        col_min = pdf[months_col].min()
        if col_min is not None and col_min < -100:
            threshold = -90  # DAYS_*
        else:
            threshold = -3  # MONTHS_BALANCE
        recent = pdf[pdf[months_col] >= threshold]
        last3 = (
            recent.groupby(config.ID_COL)["_oof"]
            .mean()
            .rename(f"{prefix}_AUX_OOF_LAST3M_MEAN")
            .reset_index()
        )
        pdf_curr = pdf_curr.merge(last3, on=config.ID_COL, how="left")

    return pl.from_pandas(pdf_curr)


def build_installments_auxiliary() -> pl.DataFrame:
    """4 auxiliary features from row-level model on installments_payments."""
    ip = read_processed("installments_payments")
    return _train_row_level(
        ip,
        prefix="INS",
        months_col="DAYS_INSTALMENT",  # we'll convert: -90 ≈ last 3 months
        drop_cols=[config.ID_COL, "SK_ID_PREV"],
    )


def build_previous_auxiliary() -> pl.DataFrame:
    """4 auxiliary features from application-level model on previous_application."""
    prev = read_processed("previous_application")
    return _train_row_level(
        prev,
        prefix="PREV",
        months_col=None,
        drop_cols=[config.ID_COL, "SK_ID_PREV"],
    )


# ─── Gating ───────────────────────────────────────────────────────────────────


def gate_check() -> bool:
    """
    Check whether the gating threshold is met by inspecting the latest 3-GBM
    ensemble result.

    Reads ``artifacts/ensemble_log.csv`` (written by ``src/ensemble.py``) and
    compares the latest row's ``oof_auc`` to ``GATE_OOF_AUC`` (0.792 per PLAN §4.5).

    Returns
    -------
    bool
        True if latest ensemble OOF AUC ≥ GATE_OOF_AUC, False otherwise.
        Returns False if the log doesn't exist (auxiliary phase shouldn't run
        before the 3-GBM ensemble has been logged).
    """
    log_path = config.ARTIFACTS_DIR / "ensemble_log.csv"
    if not log_path.exists():
        logger.warning(
            f"  gate_check: no ensemble log at {log_path}. "
            "Run `python -m src.ensemble` first. Gate FAILED."
        )
        return False

    try:
        import pandas as pd
        log_df = pd.read_csv(log_path)
    except Exception as e:
        logger.error(f"  gate_check: could not read {log_path}: {e}. Gate FAILED.")
        return False

    if log_df.empty or "oof_auc" not in log_df.columns:
        logger.warning(f"  gate_check: empty or malformed log at {log_path}. Gate FAILED.")
        return False

    latest_auc = float(log_df.iloc[-1]["oof_auc"])
    passed = latest_auc >= GATE_OOF_AUC
    if passed:
        logger.success(
            f"  gate_check: PASSED — latest 3-GBM ensemble OOF = {latest_auc:.5f} "
            f"≥ {GATE_OOF_AUC:.3f}. Auxiliary phase will run."
        )
    else:
        logger.warning(
            f"  gate_check: FAILED — latest 3-GBM ensemble OOF = {latest_auc:.5f} "
            f"< {GATE_OOF_AUC:.3f}. Auxiliary phase skipped (use --skip-gate to force)."
        )
    return passed


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4.5 auxiliary GBM features.")
    parser.add_argument(
        "--table",
        choices=["installments", "previous", "both"],
        default="both",
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Skip the OOF≥0.792 gate (for dry-runs).",
    )
    args = parser.parse_args()

    if not args.skip_gate and not gate_check():
        logger.error(
            f"Gate not met (3-GBM OOF AUC < {GATE_OOF_AUC}); skipping auxiliary phase. "
            "Use --skip-gate to force."
        )
        return

    out_dir = config.FEATURES_DIR
    if args.table in ("installments", "both"):
        with timer("aux: installments"):
            ins_aux = build_installments_auxiliary()
        ins_aux.write_parquet(out_dir / "aux_installments.parquet", compression="snappy")
        logger.success(f"  → {out_dir}/aux_installments.parquet  ({ins_aux.shape})")

    if args.table in ("previous", "both"):
        with timer("aux: previous"):
            prev_aux = build_previous_auxiliary()
        prev_aux.write_parquet(out_dir / "aux_previous.parquet", compression="snappy")
        logger.success(f"  → {out_dir}/aux_previous.parquet  ({prev_aux.shape})")

    logger.info(
        "  Re-run `make features` (or merge directly in features/assemble.py) to fold "
        "these into the main matrix, then re-train and re-ensemble."
    )


if __name__ == "__main__":
    _main()
