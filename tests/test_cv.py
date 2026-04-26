"""Tests for src/cv.py — fold builders and leakage protection."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src import config, cv


def _toy_train(n: int = 1000, n_groups: int = 200) -> pl.DataFrame:
    rng = np.random.RandomState(config.SEED)
    return pl.DataFrame(
        {
            config.ID_COL: np.arange(n),
            config.TARGET_COL: rng.binomial(1, 0.08, n),
            "GROUP_KEY": rng.randint(0, n_groups, n),
        }
    )


def test_stratified_folds_cover_all_rows() -> None:
    """Every row appears in exactly one validation fold."""
    if not hasattr(cv, "build_stratified_folds"):
        pytest.skip("build_stratified_folds not implemented")
    df = _toy_train()
    folds = cv.build_stratified_folds(df, n_splits=5, seed=config.SEED)
    # folds should be a Series/array of length n with values 0..4
    if hasattr(folds, "to_numpy"):
        folds = folds.to_numpy()
    assert len(folds) == df.height
    assert set(np.unique(folds)) == {0, 1, 2, 3, 4}


def test_stratified_preserves_class_ratio() -> None:
    """Each fold keeps the positive class proportion within tolerance."""
    if not hasattr(cv, "build_stratified_folds"):
        pytest.skip("build_stratified_folds not implemented")
    df = _toy_train(n=5000)
    folds = cv.build_stratified_folds(df, n_splits=5, seed=config.SEED)
    if hasattr(folds, "to_numpy"):
        folds = folds.to_numpy()
    overall_rate = df[config.TARGET_COL].mean()
    y = df[config.TARGET_COL].to_numpy()
    for k in range(5):
        fold_rate = y[folds == k].mean()
        assert abs(fold_rate - overall_rate) < 0.02, (
            f"fold {k} rate {fold_rate:.3f} drifts from overall {overall_rate:.3f}"
        )


def test_group_kfold_no_leakage() -> None:
    """GroupKFold guarantees no group appears in both train and validation of the same fold."""
    if not hasattr(cv, "build_group_folds"):
        pytest.skip("build_group_folds not implemented")
    df = _toy_train(n=2000, n_groups=300)
    folds = cv.build_group_folds(df, group_col="GROUP_KEY", n_splits=5)
    if hasattr(folds, "to_numpy"):
        folds = folds.to_numpy()

    groups = df["GROUP_KEY"].to_numpy()
    for k in range(5):
        train_groups = set(groups[folds != k])
        val_groups = set(groups[folds == k])
        leaked = train_groups & val_groups
        assert not leaked, f"GroupKFold leaked groups in fold {k}: {leaked}"


def test_fold_seed_reproducibility() -> None:
    """Same data + seed → same fold assignment."""
    if not hasattr(cv, "build_stratified_folds"):
        pytest.skip("build_stratified_folds not implemented")
    df = _toy_train()
    a = cv.build_stratified_folds(df, n_splits=5, seed=42)
    b = cv.build_stratified_folds(df, n_splits=5, seed=42)
    a_arr = a.to_numpy() if hasattr(a, "to_numpy") else a
    b_arr = b.to_numpy() if hasattr(b, "to_numpy") else b
    np.testing.assert_array_equal(a_arr, b_arr)
