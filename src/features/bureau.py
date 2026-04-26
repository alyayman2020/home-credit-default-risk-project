"""
``bureau`` + ``bureau_balance`` feature builder.

Reference: PLAN_v2.md §2.2 — ~350 features.

Two-step build
--------------
1. **Polars lazy aggregation of bureau_balance per SK_ID_BUREAU** (~20 features).
   Streaming engine is mandatory (27.3M rows, 376 MB raw → ~80 MB streamed —
   PLAN §1.2 / §6.1).
2. **Merge into bureau, aggregate per SK_ID_CURR** (~330 features), stratified
   by ``CREDIT_ACTIVE`` (Active/Closed) and top-5 ``CREDIT_TYPE``.

Naming convention
-----------------
- ``BUR_BB_*``           = features derived from bureau_balance.
- ``BUR_*``              = features derived from bureau (after balance merge).
- ``BUR_ACTIVE_*`` / ``BUR_CLOSED_*`` = stratified aggregations.
- ``BUR_TYPE_<name>_*``  = per-credit-type aggregations.
"""

from __future__ import annotations

import polars as pl

from src import config
from src.data import read_processed
from src.features.base import FeatureBuilder

# Top-5 CREDIT_TYPE values cover ~99% of rows (PLAN §2.2).
TOP_CREDIT_TYPES: list[str] = [
    "Consumer credit",
    "Credit card",
    "Mortgage",
    "Car loan",
    "Microloan",
]

# Numeric columns we aggregate in bureau (PLAN §2.2).
BUREAU_NUMERIC_COLS: list[str] = [
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE",
    "AMT_CREDIT_MAX_OVERDUE",
    "AMT_ANNUITY",
    "DAYS_CREDIT",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "DAYS_CREDIT_UPDATE",
    "CREDIT_DAY_OVERDUE",
    "CNT_CREDIT_PROLONG",
]


class BureauFeatures(FeatureBuilder):
    """Build features from ``bureau`` + ``bureau_balance``."""

    name = "bureau"
    prefix = "BUR_"

    def build(self) -> pl.DataFrame:
        # Step 1: aggregate bureau_balance per SK_ID_BUREAU using lazy/streaming.
        bb_per_bureau = self._aggregate_bureau_balance_lazy()
        self.logger.info(f"  bureau_balance aggregated: {bb_per_bureau.shape}")

        # Step 2: merge with bureau, then build all per-borrower aggregations.
        bureau = read_processed("bureau")
        bureau = bureau.join(bb_per_bureau, on="SK_ID_BUREAU", how="left")
        self.logger.info(f"  bureau after balance merge: {bureau.shape}")

        # Build base + stratified + type-specific aggregations.
        base = self._aggregate_per_curr_base(bureau)
        active = self._aggregate_per_curr_stratified(bureau, "Active", "BUR_ACTIVE")
        closed = self._aggregate_per_curr_stratified(bureau, "Closed", "BUR_CLOSED")
        by_type = self._aggregate_per_credit_type(bureau)
        recency = self._recency_features(bureau)

        # Join all sub-frames on SK_ID_CURR.
        out = base
        for sub in (active, closed, by_type, recency):
            out = out.join(sub, on=config.ID_COL, how="left")

        # Cross-aggregate ratios (no separate frame needed).
        out = self._cross_ratios(out)
        return out

    # ─── Step 1: bureau_balance aggregation ──────────────────────────────────

    def _aggregate_bureau_balance_lazy(self) -> pl.DataFrame:
        """
        ~20 features per SK_ID_BUREAU.

        STATUS codes (from PLAN §2.2):
          0     = no DPD
          1..5  = months past due (severity)
          C     = closed
          X     = unknown
        """
        bb_path = config.PROCESSED_DIR / "bureau_balance.parquet"
        lazy = pl.scan_parquet(bb_path)

        # One-hot the STATUS codes inside the lazy frame so we can sum them per bureau.
        status_codes = ["0", "1", "2", "3", "4", "5", "C", "X"]
        lazy = lazy.with_columns(
            [
                (pl.col("STATUS") == s).cast(pl.Int32).alias(f"BB_STATUS_{s}")
                for s in status_codes
            ]
        )

        agg_exprs = [
            pl.len().alias("BUR_BB_COUNT"),
            pl.col("MONTHS_BALANCE").min().alias("BUR_BB_MONTHS_MIN"),
            pl.col("MONTHS_BALANCE").max().alias("BUR_BB_MONTHS_MAX"),
            pl.col("MONTHS_BALANCE").mean().alias("BUR_BB_MONTHS_MEAN"),
        ]
        # Status counts and shares.
        for s in status_codes:
            agg_exprs.append(pl.col(f"BB_STATUS_{s}").sum().alias(f"BUR_BB_STATUS_{s}_CNT"))
            agg_exprs.append(pl.col(f"BB_STATUS_{s}").mean().alias(f"BUR_BB_STATUS_{s}_PCT"))

        # Streaming collect for 27M rows; falls back to non-streaming if engine unavailable.
        try:
            return lazy.group_by("SK_ID_BUREAU").agg(agg_exprs).collect(streaming=True)
        except Exception:
            return lazy.group_by("SK_ID_BUREAU").agg(agg_exprs).collect()

    # ─── Step 2: per-borrower aggregations ───────────────────────────────────

    def _aggregate_per_curr_base(self, bureau: pl.DataFrame) -> pl.DataFrame:
        """Base (un-stratified) aggregations per SK_ID_CURR — ~80 features."""
        agg_exprs: list[pl.Expr] = [pl.len().alias("BUR_COUNT")]

        # Numeric column aggregations (mean/max/min/sum/std/count).
        for col in BUREAU_NUMERIC_COLS:
            if col not in bureau.columns:
                continue
            agg_exprs += [
                pl.col(col).mean().alias(f"BUR_{col}_MEAN"),
                pl.col(col).max().alias(f"BUR_{col}_MAX"),
                pl.col(col).min().alias(f"BUR_{col}_MIN"),
                pl.col(col).sum().alias(f"BUR_{col}_SUM"),
                pl.col(col).std().alias(f"BUR_{col}_STD"),
            ]

        # Bureau_balance roll-up aggregations (mean/max across borrower's bureaus).
        bb_cols = [c for c in bureau.columns if c.startswith("BUR_BB_")]
        for col in bb_cols:
            agg_exprs += [
                pl.col(col).mean().alias(f"{col}_MEAN"),
                pl.col(col).max().alias(f"{col}_MAX"),
                pl.col(col).sum().alias(f"{col}_SUM"),
            ]

        # Categorical n-uniques.
        for cat in ("CREDIT_TYPE", "CREDIT_CURRENCY", "CREDIT_ACTIVE"):
            if cat in bureau.columns:
                agg_exprs.append(pl.col(cat).n_unique().alias(f"BUR_CAT_{cat}_NUNIQUE"))

        return bureau.group_by(config.ID_COL).agg(agg_exprs)

    def _aggregate_per_curr_stratified(
        self, bureau: pl.DataFrame, status: str, prefix: str
    ) -> pl.DataFrame:
        """Same numeric aggregations but filtered to a single CREDIT_ACTIVE state."""
        sub = bureau.filter(pl.col("CREDIT_ACTIVE") == status)
        if sub.height == 0:
            # Build an empty frame keyed by SK_ID_CURR so the join doesn't fail.
            return bureau.select(config.ID_COL).unique().with_columns(
                pl.lit(None).alias(f"{prefix}_COUNT")
            )

        agg_exprs: list[pl.Expr] = [pl.len().alias(f"{prefix}_COUNT")]
        for col in BUREAU_NUMERIC_COLS:
            if col not in sub.columns:
                continue
            agg_exprs += [
                pl.col(col).mean().alias(f"{prefix}_{col}_MEAN"),
                pl.col(col).max().alias(f"{prefix}_{col}_MAX"),
                pl.col(col).sum().alias(f"{prefix}_{col}_SUM"),
            ]
        return sub.group_by(config.ID_COL).agg(agg_exprs)

    def _aggregate_per_credit_type(self, bureau: pl.DataFrame) -> pl.DataFrame:
        """Per-CREDIT_TYPE aggregations for the top-5 most common types."""
        result = bureau.select(config.ID_COL).unique()
        for credit_type in TOP_CREDIT_TYPES:
            sub = bureau.filter(pl.col("CREDIT_TYPE") == credit_type)
            if sub.height == 0:
                continue
            type_tag = credit_type.upper().replace(" ", "_")
            agg_exprs: list[pl.Expr] = [pl.len().alias(f"BUR_TYPE_{type_tag}_COUNT")]
            # Only the most informative numeric subsets per type (avoids feature explosion).
            for col in ("AMT_CREDIT_SUM", "DAYS_CREDIT", "AMT_CREDIT_SUM_DEBT"):
                if col not in sub.columns:
                    continue
                agg_exprs += [
                    pl.col(col).mean().alias(f"BUR_TYPE_{type_tag}_{col}_MEAN"),
                    pl.col(col).max().alias(f"BUR_TYPE_{type_tag}_{col}_MAX"),
                ]
            sub_agg = sub.group_by(config.ID_COL).agg(agg_exprs)
            result = result.join(sub_agg, on=config.ID_COL, how="left")
        return result

    def _recency_features(self, bureau: pl.DataFrame) -> pl.DataFrame:
        """Recency-engineered features (gap between credits, time since most recent)."""
        # Sort within borrower by DAYS_CREDIT descending to enable diff().
        sorted_b = bureau.sort([config.ID_COL, "DAYS_CREDIT"], descending=[False, True])
        # Diff between consecutive DAYS_CREDIT within a borrower's history.
        sorted_b = sorted_b.with_columns(
            pl.col("DAYS_CREDIT").diff().over(config.ID_COL).alias("_credit_gap")
        )
        return sorted_b.group_by(config.ID_COL).agg(
            [
                pl.col("DAYS_CREDIT").max().alias("BUR_RECENCY_DAYS_CREDIT_MAX"),
                pl.col("DAYS_CREDIT").min().alias("BUR_RECENCY_DAYS_CREDIT_MIN"),
                (pl.col("DAYS_CREDIT").max() - pl.col("DAYS_CREDIT").min()).alias(
                    "BUR_RECENCY_DAYS_CREDIT_RANGE"
                ),
                pl.col("_credit_gap").mean().alias("BUR_RECENCY_AVG_CREDIT_GAP"),
                pl.col("_credit_gap").std().alias("BUR_RECENCY_STD_CREDIT_GAP"),
            ]
        )

    def _cross_ratios(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cross-aggregate ratios (PLAN §2.2 final block)."""
        new_cols: list[pl.Expr] = []
        # Total debt / total credit sum
        if "BUR_AMT_CREDIT_SUM_DEBT_SUM" in df.columns and "BUR_AMT_CREDIT_SUM_SUM" in df.columns:
            new_cols.append(
                (pl.col("BUR_AMT_CREDIT_SUM_DEBT_SUM") / pl.col("BUR_AMT_CREDIT_SUM_SUM")).alias(
                    "BUR_DEBT_TO_CREDIT_RATIO"
                )
            )
        # Active/closed ratio
        if "BUR_ACTIVE_COUNT" in df.columns and "BUR_COUNT" in df.columns:
            new_cols.append(
                (pl.col("BUR_ACTIVE_COUNT") / pl.col("BUR_COUNT")).alias("BUR_ACTIVE_RATIO")
            )
        # Overdue/debt ratio
        if (
            "BUR_AMT_CREDIT_SUM_OVERDUE_SUM" in df.columns
            and "BUR_AMT_CREDIT_SUM_DEBT_SUM" in df.columns
        ):
            new_cols.append(
                (
                    pl.col("BUR_AMT_CREDIT_SUM_OVERDUE_SUM")
                    / pl.col("BUR_AMT_CREDIT_SUM_DEBT_SUM")
                ).alias("BUR_OVERDUE_TO_DEBT_RATIO")
            )
        return df.with_columns(new_cols) if new_cols else df
