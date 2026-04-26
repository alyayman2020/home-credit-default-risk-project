"""
``bureau`` + ``bureau_balance`` feature builder.

Reference: PLAN_v2.md §2.2 — ~350 features.

Two-step build
--------------
1. **Polars lazy aggregation of bureau_balance per SK_ID_BUREAU** (~15 features).
   Streaming engine is mandatory (27.3M rows, 376 MB raw → ~80 MB streamed —
   PLAN §1.2 / §6.1).
2. **Merge into bureau, aggregate per SK_ID_CURR** (~335 features), stratified
   by ``CREDIT_ACTIVE`` and ``CREDIT_TYPE``.

This is a functional stub — flesh out the TODO blocks per the plan.
"""

from __future__ import annotations

import polars as pl

from src import config
from src.data import read_processed
from src.features.base import FeatureBuilder


class BureauFeatures(FeatureBuilder):
    """Build features from bureau + bureau_balance."""

    name = "bureau"
    prefix = "BUR_"

    def build(self) -> pl.DataFrame:
        # Step 1: aggregate bureau_balance per SK_ID_BUREAU using lazy/streaming.
        bb_per_bureau = self._aggregate_bureau_balance_lazy()

        # Step 2: merge with bureau, then aggregate per SK_ID_CURR.
        bureau = read_processed("bureau")
        bureau = bureau.join(bb_per_bureau, on="SK_ID_BUREAU", how="left")
        return self._aggregate_per_curr(bureau)

    def _aggregate_bureau_balance_lazy(self) -> pl.DataFrame:
        """
        ~15 features per ``SK_ID_BUREAU``.

        Build with ``pl.scan_parquet`` → group_by → ``.collect(streaming=True)``
        to keep peak RAM around ~80 MB on the 27M-row table.

        TODO: implement aggregations:
          - count of months observed
          - share of months in each STATUS code (X, 0, 1, 2, 3, 4, 5, C)
          - max consecutive months at status >= 1
          - DPD recency-weighted mean
        """
        bb_path = config.PROCESSED_DIR / "bureau_balance.parquet"
        lazy = pl.scan_parquet(bb_path)
        # Minimal placeholder: count + last MONTHS_BALANCE per SK_ID_BUREAU.
        agg = lazy.group_by("SK_ID_BUREAU").agg(
            [
                pl.len().alias("BUR_BB_COUNT"),
                pl.col("MONTHS_BALANCE").min().alias("BUR_BB_MONTHS_MIN"),
                pl.col("MONTHS_BALANCE").max().alias("BUR_BB_MONTHS_MAX"),
            ]
        )
        return agg.collect(streaming=True)

    def _aggregate_per_curr(self, bureau: pl.DataFrame) -> pl.DataFrame:
        """
        ~335 features per ``SK_ID_CURR``.

        TODO: implement, stratified by:
          - CREDIT_ACTIVE  ∈ {Active, Closed, Sold, Bad debt}
          - CREDIT_TYPE    (12 levels — top-5 stratification, rest pooled)

        For each numeric column: count / mean / sum / min / max / std (where defined).
        Plus categorical OHE aggregates.
        """
        # Placeholder: simple counts so the pipeline is wireable end-to-end.
        out = bureau.group_by(config.ID_COL).agg(
            [
                pl.len().alias("BUR_COUNT"),
                pl.col("AMT_CREDIT_SUM").sum().alias("BUR_AMT_CREDIT_SUM_SUM"),
                pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUR_AMT_CREDIT_DEBT_SUM"),
                pl.col("DAYS_CREDIT").max().alias("BUR_DAYS_CREDIT_MAX"),
                pl.col("DAYS_CREDIT").min().alias("BUR_DAYS_CREDIT_MIN"),
            ]
        )
        return out
