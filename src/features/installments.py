"""
``installments_payments`` feature builder.

Reference: PLAN_v2.md §2.4 — ~220 features (incl. B3 expanded windows).

Per-row engineering
-------------------
- DPD       = max(0, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)   (days late, clipped at 0)
- DBD       = max(0, DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT)   (days early, clipped at 0)
- PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT
- PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT                (under-payment)
- LATE_FLAG   = (DPD > 0).cast(int)

🆕 B3: expanded time windows
----------------------------
Per PLAN §2.4, build the same aggregations but filtered to:
  - last 1 month   (DAYS_INSTALMENT > -30)
  - last 3 months  (DAYS_INSTALMENT > -90)
  - last 6 months  (DAYS_INSTALMENT > -180)
  - last 12 months (DAYS_INSTALMENT > -365)

Streaming
---------
13.6M rows. We use ``pl.scan_parquet`` → lazy chain → ``.collect(streaming=True)``
to keep peak RAM ~150 MB.
"""

from __future__ import annotations

import polars as pl

from src import config
from src.features.base import FeatureBuilder

# Time windows for B3 features (in days, negative because DAYS_* are negative).
WINDOWS_DAYS: dict[str, int] = {
    "1M": -30,
    "3M": -90,
    "6M": -180,
    "12M": -365,
}


class InstallmentsFeatures(FeatureBuilder):
    """Build features from ``installments_payments``."""

    name = "installments_payments"
    prefix = "INS_"

    def build(self) -> pl.DataFrame:
        # Lazy scan + per-row engineering all done in one expression chain.
        ip_path = config.PROCESSED_DIR / "installments_payments.parquet"
        lazy = pl.scan_parquet(ip_path)
        lazy = self._engineer_per_row_lazy(lazy)

        # Base aggregations (lifetime, no window filter).
        base = self._aggregate_lifetime(lazy)

        # B3 windowed aggregations.
        windowed_frames = []
        for tag, threshold in WINDOWS_DAYS.items():
            windowed_frames.append(self._aggregate_window(lazy, tag, threshold))

        # Join everything on SK_ID_CURR.
        out = base
        for sub in windowed_frames:
            out = out.join(sub, on=config.ID_COL, how="left")
        return out

    # ─── Per-row engineering ──────────────────────────────────────────────────

    def _engineer_per_row_lazy(self, lazy: pl.LazyFrame) -> pl.LazyFrame:
        """Add DPD / DBD / PAYMENT_PERC / PAYMENT_DIFF / LATE_FLAG to a LazyFrame."""
        return lazy.with_columns(
            [
                # Days past due, clipped at 0.
                pl.max_horizontal([pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT"), pl.lit(0)])
                .alias("DPD"),
                # Days before due, clipped at 0.
                pl.max_horizontal([pl.col("DAYS_INSTALMENT") - pl.col("DAYS_ENTRY_PAYMENT"), pl.lit(0)])
                .alias("DBD"),
                # Payment ratio.
                (pl.col("AMT_PAYMENT") / pl.col("AMT_INSTALMENT")).alias("PAYMENT_PERC"),
                # Under-payment amount.
                (pl.col("AMT_INSTALMENT") - pl.col("AMT_PAYMENT")).alias("PAYMENT_DIFF"),
            ]
        ).with_columns(
            (pl.col("DPD") > 0).cast(pl.Int8).alias("LATE_FLAG"),
        )

    # ─── Aggregations ─────────────────────────────────────────────────────────

    def _build_agg_exprs(self, prefix: str) -> list[pl.Expr]:
        """Build the standard aggregation expression list with a given prefix."""
        return [
            # Per-borrower row count.
            pl.len().alias(f"{prefix}_COUNT"),
            # Days past due aggregations.
            pl.col("DPD").mean().alias(f"{prefix}_DPD_MEAN"),
            pl.col("DPD").max().alias(f"{prefix}_DPD_MAX"),
            pl.col("DPD").sum().alias(f"{prefix}_DPD_SUM"),
            pl.col("DPD").std().alias(f"{prefix}_DPD_STD"),
            # Days before due (early-payment buffer).
            pl.col("DBD").mean().alias(f"{prefix}_DBD_MEAN"),
            pl.col("DBD").max().alias(f"{prefix}_DBD_MAX"),
            # Payment ratio.
            pl.col("PAYMENT_PERC").mean().alias(f"{prefix}_PAYMENT_PERC_MEAN"),
            pl.col("PAYMENT_PERC").min().alias(f"{prefix}_PAYMENT_PERC_MIN"),
            pl.col("PAYMENT_PERC").max().alias(f"{prefix}_PAYMENT_PERC_MAX"),
            # Underpayment amount.
            pl.col("PAYMENT_DIFF").mean().alias(f"{prefix}_PAYMENT_DIFF_MEAN"),
            pl.col("PAYMENT_DIFF").sum().alias(f"{prefix}_PAYMENT_DIFF_SUM"),
            pl.col("PAYMENT_DIFF").max().alias(f"{prefix}_PAYMENT_DIFF_MAX"),
            # Late-payment flag.
            pl.col("LATE_FLAG").sum().alias(f"{prefix}_LATE_COUNT"),
            pl.col("LATE_FLAG").mean().alias(f"{prefix}_LATE_RATE"),
            # Raw amount aggregations.
            pl.col("AMT_INSTALMENT").mean().alias(f"{prefix}_AMT_INSTALMENT_MEAN"),
            pl.col("AMT_INSTALMENT").sum().alias(f"{prefix}_AMT_INSTALMENT_SUM"),
            pl.col("AMT_PAYMENT").mean().alias(f"{prefix}_AMT_PAYMENT_MEAN"),
            pl.col("AMT_PAYMENT").sum().alias(f"{prefix}_AMT_PAYMENT_SUM"),
        ]

    def _aggregate_lifetime(self, lazy: pl.LazyFrame) -> pl.DataFrame:
        """Lifetime aggregations — ~80 features (no window filter)."""
        agg_exprs = self._build_agg_exprs(prefix="INS")
        # Add a few extras only valid on the lifetime view.
        agg_exprs += [
            pl.col("NUM_INSTALMENT_VERSION").n_unique().alias("INS_NUM_VERSION_NUNIQUE"),
            pl.col("DAYS_INSTALMENT").min().alias("INS_DAYS_INSTALMENT_MIN"),
            pl.col("DAYS_INSTALMENT").max().alias("INS_DAYS_INSTALMENT_MAX"),
        ]
        try:
            return lazy.group_by(config.ID_COL).agg(agg_exprs).collect(streaming=True)
        except Exception:
            return lazy.group_by(config.ID_COL).agg(agg_exprs).collect()

    def _aggregate_window(self, lazy: pl.LazyFrame, tag: str, threshold: int) -> pl.DataFrame:
        """Aggregate filtered to a recency window."""
        prefix = f"INS_{tag}"
        # We use DAYS_INSTALMENT as the time anchor. Threshold is negative (e.g. -30 = last 30 days).
        sub_lazy = lazy.filter(pl.col("DAYS_INSTALMENT") > threshold)
        agg_exprs = self._build_agg_exprs(prefix=prefix)
        try:
            return sub_lazy.group_by(config.ID_COL).agg(agg_exprs).collect(streaming=True)
        except Exception:
            return sub_lazy.group_by(config.ID_COL).agg(agg_exprs).collect()
