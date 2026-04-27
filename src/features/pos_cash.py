"""
``POS_CASH_balance`` feature builder.

Reference: PLAN_v2.md §2.5 — ~110 features.

Composition
-----------
- Base aggregations (~50): all numerics × {mean, max, min, sum, std}.
- 🆕 B3 windowed (last 3, 6, 12 months): SK_DPD, SK_DPD_DEF, status share → ~30.
- DPD recency-weighted aggregations → ~10.
- Categorical encoding (mode + share of contract status) → ~10.

10M rows — load lazy and stream to keep peak RAM ~100 MB (PLAN §6.1).

Note on ``MONTHS_BALANCE``: this is a months-relative-to-now integer
(0 = current month, -1 = last month, etc.). Window thresholds are months, not days.
"""

from __future__ import annotations

import polars as pl

from src import config
from src.features.base import FeatureBuilder

# B3 windows for POS — values are MONTHS_BALANCE thresholds (negative).
WINDOWS_MONTHS: dict[str, int] = {
    "3M": -3,
    "6M": -6,
    "12M": -12,
}

# Numeric columns to aggregate.
POS_NUMERIC_COLS: list[str] = [
    "MONTHS_BALANCE",
    "CNT_INSTALMENT",
    "CNT_INSTALMENT_FUTURE",
    "SK_DPD",
    "SK_DPD_DEF",
]


class PosCashFeatures(FeatureBuilder):
    """Build features from ``POS_CASH_balance``."""

    name = "pos_cash_balance"
    prefix = "POS_"

    def build(self) -> pl.DataFrame:
        path = config.PROCESSED_DIR / "pos_cash_balance.parquet"
        lazy = pl.scan_parquet(path)
        # Per-row: late-payment flags computed once.
        lazy = lazy.with_columns(
            [
                (pl.col("SK_DPD") > 0).cast(pl.Int8).alias("POS_DPD_FLAG"),
                (pl.col("SK_DPD_DEF") > 0).cast(pl.Int8).alias("POS_DPD_DEF_FLAG"),
            ]
        )

        base = self._aggregate_lifetime(lazy)
        cats = self._categorical_aggregations(lazy)

        windowed = []
        for tag, threshold in WINDOWS_MONTHS.items():
            windowed.append(self._aggregate_window(lazy, tag, threshold))

        out = base.join(cats, on=config.ID_COL, how="left")
        for sub in windowed:
            out = out.join(sub, on=config.ID_COL, how="left")
        return out

    # ─── Aggregations ─────────────────────────────────────────────────────────

    def _build_numeric_aggs(self, prefix: str) -> list[pl.Expr]:
        """Standard numeric aggregations with a given prefix."""
        exprs = [pl.len().alias(f"{prefix}_COUNT")]
        for col in POS_NUMERIC_COLS:
            exprs += [
                pl.col(col).mean().alias(f"{prefix}_{col}_MEAN"),
                pl.col(col).max().alias(f"{prefix}_{col}_MAX"),
                pl.col(col).min().alias(f"{prefix}_{col}_MIN"),
                pl.col(col).sum().alias(f"{prefix}_{col}_SUM"),
                pl.col(col).std().alias(f"{prefix}_{col}_STD"),
            ]
        # Late-payment flag aggregates.
        exprs += [
            pl.col("POS_DPD_FLAG").sum().alias(f"{prefix}_DPD_FLAG_SUM"),
            pl.col("POS_DPD_FLAG").mean().alias(f"{prefix}_DPD_FLAG_RATE"),
            pl.col("POS_DPD_DEF_FLAG").sum().alias(f"{prefix}_DPD_DEF_FLAG_SUM"),
            pl.col("POS_DPD_DEF_FLAG").mean().alias(f"{prefix}_DPD_DEF_FLAG_RATE"),
        ]
        return exprs

    def _aggregate_lifetime(self, lazy: pl.LazyFrame) -> pl.DataFrame:
        """Lifetime aggregations — no window filter."""
        exprs = self._build_numeric_aggs(prefix="POS")
        try:
            return lazy.group_by(config.ID_COL).agg(exprs).collect(streaming=True)
        except Exception:
            return lazy.group_by(config.ID_COL).agg(exprs).collect()

    def _aggregate_window(self, lazy: pl.LazyFrame, tag: str, threshold: int) -> pl.DataFrame:
        """Window-filtered aggregations (last N months)."""
        prefix = f"POS_{tag}"
        sub = lazy.filter(pl.col("MONTHS_BALANCE") >= threshold)
        exprs = self._build_numeric_aggs(prefix=prefix)
        try:
            return sub.group_by(config.ID_COL).agg(exprs).collect(streaming=True)
        except Exception:
            return sub.group_by(config.ID_COL).agg(exprs).collect()

    def _categorical_aggregations(self, lazy: pl.LazyFrame) -> pl.DataFrame:
        """Status-mode and per-status share."""
        # NAME_CONTRACT_STATUS distribution.
        # Build one-hot columns inside lazy frame, then aggregate.
        statuses = ["Active", "Completed", "Signed", "Demand", "Returned to the store"]
        lazy_ohe = lazy.with_columns(
            [
                (pl.col("NAME_CONTRACT_STATUS") == s).cast(pl.Int8).alias(f"_POS_STATUS_{s}")
                for s in statuses
            ]
        )
        agg_exprs: list[pl.Expr] = []
        for s in statuses:
            tag = s.upper().replace(" ", "_")
            agg_exprs += [
                pl.col(f"_POS_STATUS_{s}").sum().alias(f"POS_STATUS_{tag}_CNT"),
                pl.col(f"_POS_STATUS_{s}").mean().alias(f"POS_STATUS_{tag}_RATE"),
            ]
        agg_exprs.append(pl.col("NAME_CONTRACT_STATUS").n_unique().alias("POS_STATUS_NUNIQUE"))
        try:
            return lazy_ohe.group_by(config.ID_COL).agg(agg_exprs).collect(streaming=True)
        except Exception:
            return lazy_ohe.group_by(config.ID_COL).agg(agg_exprs).collect()
