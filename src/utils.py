"""
Utility helpers used across the pipeline.

Includes:

- A ``loguru`` logger preconfigured with the level from ``config.LOG_LEVEL``.
- ``set_seed`` for reproducibility across numpy / torch / random.
- ``reduce_mem_usage`` for downcasting pandas dtypes (boundary helper for the
  Polars→pandas handoff before LightGBM/XGBoost).
- A ``timer`` context manager that logs elapsed wall-clock.

Reference: PLAN_v2.md §9 (output order — utilities are scaffolding for §3–§6).
"""

from __future__ import annotations

import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd
from loguru import logger

from src import config

# ─── Logger ───────────────────────────────────────────────────────────────────

# Reset default loguru handler so we control format and level.
logger.remove()
logger.add(
    sys.stderr,
    level=config.LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)


def get_logger() -> "logger.__class__":  # type: ignore[name-defined]
    """Return the configured loguru logger. (Indirection for testability.)"""
    return logger


# ─── Reproducibility ──────────────────────────────────────────────────────────


def set_seed(seed: int | None = None) -> int:
    """
    Seed numpy / random / PYTHONHASHSEED, and torch if available.

    Parameters
    ----------
    seed
        Integer seed. Defaults to ``config.SEED`` (42).

    Returns
    -------
    int
        The seed that was applied.
    """
    seed = int(seed) if seed is not None else config.SEED
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism flags — slight perf cost, big reproducibility win.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.debug(f"Seed set to {seed}")
    return seed


# ─── Memory reduction (boundary helper, not used inside Polars pipelines) ─────


def reduce_mem_usage(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns of a pandas DataFrame to the smallest-fitting dtype.

    This is the canonical Kaggle ``reduce_mem_usage`` (Will Koehrsen / ogrellier
    pattern). Apply only at the Polars→pandas boundary before LightGBM/XGBoost
    consumption (PLAN §A1, §A4 dtype assertion).

    Parameters
    ----------
    df
        Input DataFrame (modified in place is *not* guaranteed; assign the result).
    verbose
        If True, log the memory delta.

    Returns
    -------
    pd.DataFrame
        Same data, downcast dtypes.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object or isinstance(col_type, pd.CategoricalDtype):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if pd.api.types.is_integer_dtype(col_type):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(col_type):
            if (
                c_min >= np.finfo(np.float32).min
                and c_max <= np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        pct = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0.0
        logger.info(
            f"reduce_mem_usage: {start_mem:6.1f} MB → {end_mem:6.1f} MB "
            f"(saved {pct:4.1f}%)"
        )
    return df


# ─── Timers ───────────────────────────────────────────────────────────────────


@contextmanager
def timer(label: str = "block") -> Iterator[None]:
    """
    Context manager that logs wall-clock elapsed for a code block.

    Examples
    --------
    >>> with timer("load bureau"):
    ...     df = pl.read_csv(...)
    """
    start = time.perf_counter()
    logger.info(f"⏱  start: {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"⏱  done : {label}  ({elapsed:7.2f} sec)")


def fmt_bytes(n: int | float) -> str:
    """Format byte counts for human-readable logs."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:6.1f} {unit}"
        n /= 1024.0
    return f"{n:6.1f} PB"
