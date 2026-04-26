"""
Smoke-test submission writer — produces a Kaggle-format CSV WITHOUT training.

Use case
--------
Before running the full pipeline (steps 1–8 of RUNBOOK.md), you may want to
confirm that:
  1. ``DATA_DIR`` points to a valid raw-data folder
  2. ``application_test.csv`` is readable
  3. The submission CSV format matches Kaggle's exact spec

This script reads only ``application_test.csv``, writes one row per
``SK_ID_CURR`` with ``TARGET = 0.0807`` (the population positive rate), and
saves the file to ``submissions/``. Submitting it to Kaggle will score ~0.50
AUC — that's expected. The point is to verify the I/O path.

CLI
---
    uv run python -m src.dummy_submission
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from src import config
from src.utils import get_logger

logger = get_logger()

# Population positive rate (PLAN §1.3) — using this as a constant prediction.
DUMMY_TARGET: float = 0.0807


def main() -> None:
    test_path = config.raw_path("application_test")
    if not test_path.exists():
        raise FileNotFoundError(
            f"Cannot find {test_path}. "
            f"Set DATA_DIR in .env to the folder containing the Kaggle CSVs."
        )

    logger.info(f"Reading {test_path}")
    test = pl.read_csv(test_path, columns=[config.ID_COL])
    logger.info(f"  {test.height:,} test rows.")

    sub = test.with_columns(pl.lit(DUMMY_TARGET).alias(config.TARGET_COL))
    assert sub.columns == [config.ID_COL, config.TARGET_COL], (
        f"Expected exactly two columns; got {sub.columns}"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config.SUBMISSIONS_DIR / f"submission_{ts}_DUMMY_smoke.csv"
    sub.write_csv(out_path)

    size_kb = out_path.stat().st_size / 1024
    logger.success(f"Wrote {out_path}  ({sub.height:,} rows, {size_kb:.0f} KB)")
    logger.info(
        "This is a smoke-test submission. "
        "Expected Kaggle AUC: ~0.50. "
        "Run the full pipeline (RUNBOOK.md) for real predictions."
    )

    # Show the first 5 lines so the user can eyeball the format.
    with out_path.open() as f:
        head = "".join(f.readline() for _ in range(5))
    logger.info(f"\nFile preview:\n{head}")


if __name__ == "__main__":
    main()
