"""Tests for src/cv.py — fold builders and leakage protection.

API (from src/cv.py):
  build_main_folds(application_train, *, n_splits=5, seed=42) -> pl.DataFrame
    Returns a 2-column frame: SK_ID_CURR, fold_main (Int8, 0..n_splits-1)
  build_group_folds(application_train, *, n_splits=5) -> pl.DataFrame
    Returns SK_ID_CURR, fold_group. Groups by SK_ID_CURR (no group_col arg).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src import config, cv


def _toy_train(n: int = 1000) -> pl.DataFrame:
    """One-row-per-borrower toy frame with the schema build_*_folds expects."""
    rng = np.random.RandomState(config.SEED)
    return pl.DataFrame(
        {
            config.ID_COL: np.arange(n),
            config.TARGET_COL: rng.binomial(1, 0.08, n).astype(np.int8),
        }
    )


def test_main_folds_cover_all_rows() -> None:
    """Every row appears in exactly one validation fold."""
    df = _toy_train()
    folds = cv.build_main_folds(df, n_splits=5)
    arr = folds["fold_main"].to_numpy()
    assert len(arr) == df.height
    assert set(np.unique(arr)) == {0, 1, 2, 3, 4}


def test_main_folds_preserve_class_ratio() -> None:
    """Each fold keeps the positive class proportion within tolerance."""
    df = _toy_train(n=5000)
    folds = cv.build_main_folds(df, n_splits=5)
    joined = df.join(folds, on=config.ID_COL)
    overall = joined[config.TARGET_COL].mean()
    for k in range(5):
        fold_rate = joined.filter(pl.col("fold_main") == k)[config.TARGET_COL].mean()
        assert abs(fold_rate - overall) < 0.02, (
            f"fold {k} rate {fold_rate:.3f} drifts from overall {overall:.3f}"
        )


def test_group_folds_no_leakage() -> None:
    """build_group_folds groups by SK_ID_CURR — no ID in both train and valid of any fold."""
    df = _toy_train(n=2000)
    folds = cv.build_group_folds(df, n_splits=5)
    joined = df.join(folds, on=config.ID_COL).sort(config.ID_COL)
    fold_arr = joined["fold_group"].to_numpy()
    ids = joined[config.ID_COL].to_numpy()
    for k in range(5):
        train_ids = set(ids[fold_arr != k].tolist())
        val_ids = set(ids[fold_arr == k].tolist())
        assert not (train_ids & val_ids), f"fold {k}: ID overlap between train and valid"


def test_main_folds_seed_reproducible() -> None:
    """Same data + seed → same fold assignment."""
    df = _toy_train()
    a = cv.build_main_folds(df, n_splits=5, seed=42)["fold_main"].to_numpy()
    b = cv.build_main_folds(df, n_splits=5, seed=42)["fold_main"].to_numpy()
    np.testing.assert_array_equal(a, b)
