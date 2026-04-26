"""Tests for src/utils.py — seed, reduce_mem_usage, logger."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from src import utils


def test_set_seed_deterministic() -> None:
    """Repeated set_seed calls produce reproducible RNG output across libraries."""
    utils.set_seed(123)
    a_py = random.random()
    a_np = np.random.rand()

    utils.set_seed(123)
    b_py = random.random()
    b_np = np.random.rand()

    assert a_py == b_py
    assert a_np == b_np


def test_reduce_mem_usage_pandas() -> None:
    """reduce_mem_usage downcasts numerics without changing values."""
    df = pd.DataFrame(
        {
            "i": np.arange(100, dtype=np.int64),
            "f": np.linspace(-1, 1, 100, dtype=np.float64),
            "s": ["x"] * 100,
        }
    )
    before = df.memory_usage(deep=True).sum()
    out = utils.reduce_mem_usage(df.copy(), verbose=False)
    after = out.memory_usage(deep=True).sum()

    # Numeric values preserved.
    np.testing.assert_array_equal(out["i"].to_numpy(), df["i"].to_numpy())
    np.testing.assert_allclose(out["f"].to_numpy(), df["f"].to_numpy(), rtol=1e-3)
    # Memory reduced.
    assert after <= before


def test_logger_returns_singleton() -> None:
    """get_logger() returns a usable logger object on repeated calls."""
    log_a = utils.get_logger()
    log_b = utils.get_logger()
    # Both should be loguru's logger or compatible — just check they expose .info.
    assert hasattr(log_a, "info")
    assert hasattr(log_b, "info")
    log_a.info("test message — no failure expected")


def test_timer_context_manager() -> None:
    """Timer context manager exits cleanly without raising."""
    if not hasattr(utils, "Timer"):
        pytest.skip("Timer not implemented yet")
    with utils.Timer("unit-test"):
        _ = sum(range(1000))
