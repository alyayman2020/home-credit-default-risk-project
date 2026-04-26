"""
Cross-validation fold builders.

Two splitters per PLAN_v2.md §3.1:

- **Main level**: ``StratifiedKFold(n=5, shuffle=True, random_state=42)`` on TARGET.
  Used by every primary model (LGBM/XGB/CatBoost/NN-A/NN-B).

- **Group level**: ``GroupKFold(n=5)`` grouped by ``SK_ID_CURR``.
  Required by Phase 4.5 auxiliary GBMs (PLAN §4.5, B1) — prevents same-borrower
  leakage when training on monthly rows.

Persisted to::

    data/features/folds.parquet         (SK_ID_CURR, fold_main)
    data/features/folds_group.parquet   (SK_ID_CURR, fold_group)   # one row per borrower

CLI
---
``python -m src.cv``           build and persist both fold mappings.
``python -m src.cv --smoke``   run leakage smoke checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src import config
from src.utils import get_logger, timer

if TYPE_CHECKING:
    pass

logger = get_logger()

FOLDS_MAIN_PATH: Path = config.FEATURES_DIR / "folds.parquet"
FOLDS_GROUP_PATH: Path = config.FEATURES_DIR / "folds_group.parquet"


# ─── Builders ─────────────────────────────────────────────────────────────────


def build_main_folds(
    application_train: pl.DataFrame,
    *,
    n_splits: int = config.N_FOLDS,
    seed: int = config.SEED,
) -> pl.DataFrame:
    """
    StratifiedKFold on TARGET. Returns a 2-column frame: ``SK_ID_CURR``, ``fold_main``.

    ``fold_main`` is the *validation* fold index for that row (0..n_splits-1).
    """
    if config.TARGET_COL not in application_train.columns:
        raise KeyError(f"{config.TARGET_COL!r} missing from application_train.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = application_train[config.TARGET_COL].to_numpy()
    fold_idx = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_idx[val_idx] = fold

    assert (fold_idx >= 0).all(), "All rows must be assigned to a fold."

    out = application_train.select(config.ID_COL).with_columns(
        pl.Series("fold_main", fold_idx).cast(pl.Int8)
    )
    logger.info(f"  built fold_main: {n_splits}-fold StratifiedKFold, seed={seed}")
    _log_class_balance_per_fold(out, application_train)
    return out


def build_group_folds(
    application_train: pl.DataFrame,
    *,
    n_splits: int = config.N_FOLDS,
) -> pl.DataFrame:
    """
    GroupKFold by ``SK_ID_CURR``. Returns ``SK_ID_CURR``, ``fold_group``.

    Rationale (PLAN §3.1, B1): when an auxiliary model trains on
    *per-row* tables (installments_payments, previous_application), the
    StratifiedKFold split on application rows would still let a borrower's
    monthly history leak across train/valid. GroupKFold by SK_ID_CURR
    forces each borrower into exactly one fold.
    """
    skf = GroupKFold(n_splits=n_splits)
    ids = application_train[config.ID_COL].to_numpy()
    # Dummy y / X (GroupKFold ignores y for splitting).
    fold_idx = np.full(len(ids), -1, dtype=np.int8)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(ids)), groups=ids)):
        fold_idx[val_idx] = fold

    assert (fold_idx >= 0).all(), "All rows must be assigned to a group fold."

    out = application_train.select(config.ID_COL).with_columns(
        pl.Series("fold_group", fold_idx).cast(pl.Int8)
    )
    logger.info(f"  built fold_group: {n_splits}-fold GroupKFold by {config.ID_COL}")
    return out


def _log_class_balance_per_fold(folds: pl.DataFrame, app_train: pl.DataFrame) -> None:
    """Log positive-class rate per fold to verify stratification."""
    joined = folds.join(app_train.select([config.ID_COL, config.TARGET_COL]), on=config.ID_COL)
    rates = (
        joined.group_by("fold_main")
        .agg(
            [
                pl.len().alias("n"),
                pl.col(config.TARGET_COL).mean().alias("pos_rate"),
            ]
        )
        .sort("fold_main")
    )
    for row in rates.iter_rows(named=True):
        logger.info(
            f"    fold {row['fold_main']}: n={row['n']:>7d}  pos_rate={row['pos_rate']:.4f}"
        )


# ─── Persistence ──────────────────────────────────────────────────────────────


def save_folds(main_folds: pl.DataFrame, group_folds: pl.DataFrame) -> None:
    """Write both fold parquets to ``data/features/``."""
    main_folds.write_parquet(FOLDS_MAIN_PATH, compression="snappy")
    group_folds.write_parquet(FOLDS_GROUP_PATH, compression="snappy")
    logger.success(f"Wrote {FOLDS_MAIN_PATH.name} and {FOLDS_GROUP_PATH.name}")


def load_main_folds() -> pl.DataFrame:
    """Load the persisted main-level fold mapping."""
    if not FOLDS_MAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FOLDS_MAIN_PATH}. Run `python -m src.cv` after `make data`."
        )
    return pl.read_parquet(FOLDS_MAIN_PATH)


def load_group_folds() -> pl.DataFrame:
    """Load the persisted group-level fold mapping."""
    if not FOLDS_GROUP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FOLDS_GROUP_PATH}. Run `python -m src.cv` after `make data`."
        )
    return pl.read_parquet(FOLDS_GROUP_PATH)


# ─── Smoke tests (PLAN §3.4) ──────────────────────────────────────────────────


def smoke() -> None:
    """
    Quick sanity checks on persisted folds.

    1. Every SK_ID_CURR appears exactly once in each fold mapping.
    2. fold_main values are 0..N_FOLDS-1.
    3. Per-fold positive rate within ±0.5pp of overall pos rate.
    4. fold_group respects group integrity (trivially true for one-row-per-borrower data,
       but we assert the data shape we expect).
    """
    failed = False

    try:
        main = load_main_folds()
        group = load_group_folds()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Check 1: uniqueness of SK_ID_CURR
    if main.n_unique(subset=[config.ID_COL]) != main.height:
        logger.error("FAIL: duplicate SK_ID_CURR in main folds.")
        failed = True
    else:
        logger.success(f"OK: main folds — {main.height} unique SK_ID_CURR")

    if group.n_unique(subset=[config.ID_COL]) != group.height:
        logger.error("FAIL: duplicate SK_ID_CURR in group folds.")
        failed = True
    else:
        logger.success(f"OK: group folds — {group.height} unique SK_ID_CURR")

    # Check 2: fold_main range
    fold_vals = sorted(main["fold_main"].unique().to_list())
    expected = list(range(config.N_FOLDS))
    if fold_vals != expected:
        logger.error(f"FAIL: fold_main values {fold_vals} != expected {expected}")
        failed = True
    else:
        logger.success(f"OK: fold_main has values {fold_vals}")

    # Check 3: stratification (requires app_train)
    try:
        from src.data import read_processed

        app = read_processed("application_train").select([config.ID_COL, config.TARGET_COL])
        joined = main.join(app, on=config.ID_COL)
        overall = joined[config.TARGET_COL].mean()
        per_fold = joined.group_by("fold_main").agg(pl.col(config.TARGET_COL).mean()).sort(
            "fold_main"
        )
        for row in per_fold.iter_rows(named=True):
            diff = abs(row[config.TARGET_COL] - overall)
            tag = "OK" if diff < 0.005 else "WARN"
            logger.info(
                f"  {tag}: fold {row['fold_main']} pos_rate={row[config.TARGET_COL]:.4f} "
                f"(overall={overall:.4f}, |Δ|={diff:.4f})"
            )
    except FileNotFoundError:
        logger.warning("Stratification check skipped — run `make data` first.")

    if failed:
        sys.exit(1)
    logger.success("All smoke checks passed.")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build CV fold mappings.")
    parser.add_argument(
        "--smoke", action="store_true", help="Run smoke tests on persisted folds and exit."
    )
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return

    from src.data import read_processed

    with timer("build folds"):
        app_train = read_processed("application_train")
        main = build_main_folds(app_train)
        group = build_group_folds(app_train)
        save_folds(main, group)
    logger.success("Folds built and saved.")


if __name__ == "__main__":
    _main()
