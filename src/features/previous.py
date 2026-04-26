"""
``previous_application`` feature builder.

Reference: PLAN_v2.md §2.3 — ~250 features (unchanged from v1).

Family
------
- Numeric aggregations per SK_ID_CURR: count / mean / sum / min / max / std on
  AMT_APPLICATION, AMT_CREDIT, AMT_DOWN_PAYMENT, AMT_GOODS_PRICE, RATE_*, CNT_PAYMENT,
  DAYS_DECISION, plus the cleaned DAYS_FIRST_DRAWING / DAYS_FIRST_DUE / etc.
- Approval / refusal stratification (NAME_CONTRACT_STATUS).
- Cash / Revolving / Consumer loan stratification (NAME_CONTRACT_TYPE).
- Categorical encodings.

This is a functional stub.
"""

from __future__ import annotations

import polars as pl

from src import config
from src.data import read_processed
from src.features.base import FeatureBuilder


class PreviousApplicationFeatures(FeatureBuilder):
    """Build features from previous_application."""

    name = "previous_application"
    prefix = "PREV_"

    def build(self) -> pl.DataFrame:
        prev = read_processed("previous_application")

        # TODO: per the plan, build:
        #   - all-rows aggregations
        #   - approved-only aggregations
        #   - refused-only aggregations
        #   - per-contract-type aggregations
        #   - last-N applications aggregations

        out = prev.group_by(config.ID_COL).agg(
            [
                pl.len().alias("PREV_COUNT"),
                pl.col("AMT_APPLICATION").mean().alias("PREV_AMT_APP_MEAN"),
                pl.col("AMT_APPLICATION").sum().alias("PREV_AMT_APP_SUM"),
                pl.col("AMT_CREDIT").mean().alias("PREV_AMT_CREDIT_MEAN"),
                pl.col("AMT_DOWN_PAYMENT").mean().alias("PREV_AMT_DOWN_PAYMENT_MEAN"),
                pl.col("DAYS_DECISION").max().alias("PREV_DAYS_DECISION_MAX"),
                pl.col("DAYS_DECISION").min().alias("PREV_DAYS_DECISION_MIN"),
                pl.col("CNT_PAYMENT").mean().alias("PREV_CNT_PAYMENT_MEAN"),
            ]
        )
        return out
