"""
Data loading layer.

Reads the 8 raw Kaggle CSVs with Polars (lazy + streaming where it matters),
applies the canonical Home Credit data-quirk fixes, and writes parquet files
to ``data/processed/`` for downstream feature builders.

Reference: PLAN_v2.md §1.4 (data quirks) and §6 (Polars memory budget).

CLI
---
``python -m src.data``  loads, cleans, and writes parquet for all tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src import config
from src.utils import get_logger, timer

logger = get_logger()

# ─── Sentinel constants (PLAN §1.4) ───────────────────────────────────────────

DAYS_EMPLOYED_SENTINEL = 365243
XNA_TOKENS = ("XNA", "XAP")


# ─── Column families ──────────────────────────────────────────────────────────

# Columns where 'XNA'/'XAP' should become NaN after load.
_APPLICATION_XNA_COLS = ["CODE_GENDER", "ORGANIZATION_TYPE", "NAME_FAMILY_STATUS"]
_PREVIOUS_XNA_COLS = ["NAME_CONTRACT_STATUS", "NAME_CASH_LOAN_PURPOSE", "CODE_REJECT_REASON"]


# ─── Loaders (Polars eager — switch to scan_csv for streaming where noted) ────


def _replace_xna(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Replace 'XNA' / 'XAP' tokens with null in the given string columns."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    return df.with_columns(
        [
            pl.when(pl.col(c).is_in(list(XNA_TOKENS))).then(None).otherwise(pl.col(c)).alias(c)
            for c in present
        ]
    )


def load_application(kind: str = "train") -> pl.DataFrame:
    """
    Load ``application_train.csv`` or ``application_test.csv`` and apply
    canonical fixes.

    Fixes applied (PLAN §1.4):
      - ``DAYS_EMPLOYED == 365243`` → null.
      - ``XNA`` / ``XAP`` → null in CODE_GENDER, ORGANIZATION_TYPE, NAME_FAMILY_STATUS.
    """
    if kind not in {"train", "test"}:
        raise ValueError(f"kind must be 'train' or 'test', got {kind!r}")
    path = config.raw_path(f"application_{kind}")
    with timer(f"load application_{kind}"):
        df = pl.read_csv(path, infer_schema_length=10_000)
        df = df.with_columns(
            pl.when(pl.col("DAYS_EMPLOYED") == DAYS_EMPLOYED_SENTINEL)
            .then(None)
            .otherwise(pl.col("DAYS_EMPLOYED"))
            .alias("DAYS_EMPLOYED")
        )
        df = _replace_xna(df, _APPLICATION_XNA_COLS)
    logger.info(f"  application_{kind}: shape={df.shape}")
    return df


def load_bureau() -> pl.DataFrame:
    """Load ``bureau.csv``."""
    with timer("load bureau"):
        df = pl.read_csv(config.raw_path("bureau"), infer_schema_length=10_000)
    logger.info(f"  bureau: shape={df.shape}")
    return df


def load_bureau_balance(*, lazy: bool = True) -> pl.LazyFrame | pl.DataFrame:
    """
    Load ``bureau_balance.csv`` (27.3M rows).

    Default returns a ``LazyFrame`` for streaming aggregation in
    ``features.bureau`` (PLAN §2.2 step 1, §6.1 row).
    """
    path = config.raw_path("bureau_balance")
    if lazy:
        logger.info(f"  bureau_balance: scan_csv (lazy) {path.name}")
        return pl.scan_csv(path)
    with timer("load bureau_balance (eager — ~376 MB)"):
        df = pl.read_csv(path, infer_schema_length=10_000)
    logger.info(f"  bureau_balance: shape={df.shape}")
    return df


def load_previous_application() -> pl.DataFrame:
    """Load ``previous_application.csv`` and clean XNA tokens."""
    with timer("load previous_application"):
        df = pl.read_csv(config.raw_path("previous_application"), infer_schema_length=10_000)
        df = df.with_columns(
            [
                pl.when(pl.col(c).is_in([365243.0, 365243])).then(None).otherwise(pl.col(c)).alias(c)
                for c in (
                    "DAYS_FIRST_DRAWING",
                    "DAYS_FIRST_DUE",
                    "DAYS_LAST_DUE_1ST_VERSION",
                    "DAYS_LAST_DUE",
                    "DAYS_TERMINATION",
                )
                if c in df.columns
            ]
        )
        df = _replace_xna(df, _PREVIOUS_XNA_COLS)
    logger.info(f"  previous_application: shape={df.shape}")
    return df


def load_installments_payments(*, lazy: bool = True) -> pl.LazyFrame | pl.DataFrame:
    """Load ``installments_payments.csv`` (13.6M rows). Lazy by default."""
    path = config.raw_path("installments_payments")
    if lazy:
        logger.info(f"  installments_payments: scan_csv (lazy) {path.name}")
        return pl.scan_csv(path)
    with timer("load installments_payments (eager — ~723 MB)"):
        df = pl.read_csv(path, infer_schema_length=10_000)
    logger.info(f"  installments_payments: shape={df.shape}")
    return df


def load_pos_cash_balance(*, lazy: bool = True) -> pl.LazyFrame | pl.DataFrame:
    """Load ``POS_CASH_balance.csv`` (10M rows). Lazy by default."""
    path = config.raw_path("pos_cash_balance")
    if lazy:
        logger.info(f"  pos_cash_balance: scan_csv (lazy) {path.name}")
        return pl.scan_csv(path)
    with timer("load pos_cash_balance"):
        df = pl.read_csv(path, infer_schema_length=10_000)
    logger.info(f"  pos_cash_balance: shape={df.shape}")
    return df


def load_credit_card_balance() -> pl.DataFrame:
    """Load ``credit_card_balance.csv`` (3.8M rows)."""
    with timer("load credit_card_balance"):
        df = pl.read_csv(config.raw_path("credit_card_balance"), infer_schema_length=10_000)
    logger.info(f"  credit_card_balance: shape={df.shape}")
    return df


# ─── Parquet round-trip ───────────────────────────────────────────────────────


def write_processed(df: pl.DataFrame, name: str) -> Path:
    """Write a DataFrame to ``data/processed/{name}.parquet``."""
    path = config.PROCESSED_DIR / f"{name}.parquet"
    with timer(f"write {path.name}"):
        df.write_parquet(path, compression="snappy")
    logger.info(f"  → {path}  ({path.stat().st_size / 1024**2:6.1f} MB)")
    return path


def read_processed(name: str) -> pl.DataFrame:
    """Read ``data/processed/{name}.parquet``."""
    path = config.PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed parquet not found: {path}. Run `make data` first."
        )
    return pl.read_parquet(path)


# ─── CLI entrypoint ───────────────────────────────────────────────────────────


def materialise_all() -> None:
    """
    Load every raw CSV (with sentinel cleanup) and write parquet to
    ``data/processed/``. Idempotent — running twice overwrites.
    """
    logger.info(config.summary())
    if not config.DATA_DIR.exists():
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {config.DATA_DIR}\n"
            "  Set DATA_DIR in .env (see .env.example) to the folder containing the raw CSVs."
        )

    # application_train + test
    write_processed(load_application("train"), "application_train")
    write_processed(load_application("test"), "application_test")

    # bureau family
    write_processed(load_bureau(), "bureau")
    bb = load_bureau_balance(lazy=False)  # type: ignore[arg-type]
    assert isinstance(bb, pl.DataFrame)
    write_processed(bb, "bureau_balance")

    # previous_application family
    write_processed(load_previous_application(), "previous_application")
    ip = load_installments_payments(lazy=False)  # type: ignore[arg-type]
    assert isinstance(ip, pl.DataFrame)
    write_processed(ip, "installments_payments")
    pc = load_pos_cash_balance(lazy=False)  # type: ignore[arg-type]
    assert isinstance(pc, pl.DataFrame)
    write_processed(pc, "pos_cash_balance")
    write_processed(load_credit_card_balance(), "credit_card_balance")

    logger.success("All raw tables materialised to data/processed/")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Materialise raw CSVs to processed parquet.")
    p.add_argument("--check", action="store_true", help="Print config summary and exit.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.check:
        print(config.summary())
    else:
        materialise_all()
