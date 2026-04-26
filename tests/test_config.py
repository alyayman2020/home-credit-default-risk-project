"""Tests for src/config.py — paths, env loading, feature matrix specs."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config


def test_paths_resolved() -> None:
    """All declared paths are absolute Path objects."""
    for p in [
        config.PROJECT_ROOT,
        config.DATA_DIR,
        config.FEATURES_DIR,
        config.PROCESSED_DIR,
        config.ARTIFACTS_DIR,
        config.MODELS_DIR,
        config.PREDICTIONS_DIR,
        config.SUBMISSIONS_DIR,
    ]:
        assert isinstance(p, Path)
        assert p.is_absolute(), f"{p} should be absolute"


def test_write_dirs_exist() -> None:
    """Pipeline output directories are auto-created at import."""
    for p in [
        config.FEATURES_DIR,
        config.PROCESSED_DIR,
        config.ARTIFACTS_DIR,
        config.MODELS_DIR,
        config.PREDICTIONS_DIR,
        config.SUBMISSIONS_DIR,
    ]:
        assert p.exists(), f"{p} should be auto-created"


def test_raw_path_lookup() -> None:
    """raw_path() resolves known keys and rejects unknown keys."""
    p = config.raw_path("application_train")
    assert p.name == "application_train.csv"
    assert p.parent == config.DATA_DIR

    with pytest.raises(KeyError):
        config.raw_path("nonexistent")


def test_seed_constants() -> None:
    """Sanity on tunables — defaults are reasonable."""
    assert config.SEED == 42 or config.SEED > 0
    assert config.N_FOLDS == 5
    assert 0 < config.POS_RATE < 1
    assert config.SCALE_POS_WEIGHT > 1  # imbalanced positive class


def test_feature_matrices_registered() -> None:
    """Three matrices declared per PLAN §6.2."""
    assert set(config.FEATURE_MATRICES.keys()) == {"main", "catboost", "nn"}
    for name, spec in config.FEATURE_MATRICES.items():
        assert spec.name == name
        assert spec.parquet_path.suffix == ".parquet"
        assert spec.expected_min_cols < spec.expected_max_cols


def test_summary_includes_paths() -> None:
    s = config.summary()
    assert "PROJECT_ROOT" in s
    assert "DATA_DIR" in s
    assert "SEED" in s
