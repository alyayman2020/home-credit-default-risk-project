"""
``installments_payments`` feature builder.

Reference: PLAN_v2.md §2.4 — ~220 features (+40 from v1 via B3).

Per-row engineering (unchanged)
-------------------------------
- DPD       = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT     (positive ⇒ late)
- DBD       = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT     (positive ⇒ early)
- PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT
- PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT
- LATE_FLAG    = (DPD > 0).cast(int)

🆕 B3: expanded time windows (+40 features)
-------------------------------------------
- Windows: last 1, 3, 6, 12 months (24-mo dropped — too redundant with lifetime).
- Per window: top aggregations on DPD / LATE_FLAG / PAYMENT_PERC only.
- 4 windows × 10 metrics = ~40 windowed features.

13.6M rows — load lazy (``pl.scan_parquet``) and stream to keep peak RAM
around ~150 MB (PLAN §6.1).

This is a functional stub.
"""

from __future__ import annotations

import polars as pl

from src import config
from src.features.base import FeatureBuilder


class InstallmentsFeatures(FeatureBuilder):
    """Build features from installments_payments."""

    name = "installments_payments"
    prefix = "INS_"

    def build(self) -> pl.DataFrame:
        ip_path = config.PROCESSED_DIR / "installments_payments.parquet"
        lazy = pl.scan_parquet(ip_path)

        # Per-row derived columns
        lazy = lazy.with_columns(
            [
                (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("INS_DPD"),
                (pl.col("DAYS_INSTALMENT") - pl.col("DAYS_ENTRY_PAYMENT")).alias("INS_DBD"),
                (pl.col("AMT_PAYMENT") / pl.col("AMT_INSTALMENT")).alias("INS_PAYMENT_PERC"),
                (pl.col("AMT_INSTALMENT") - pl.col("AMT_PAYMENT")).alias("INS_PAYMENT_DIFF"),
                (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT") > 0)
                .cast(pl.Int8)
                .alias("INS_LATE_FLAG"),
            ]
        )

        # Lifetime aggregations
        lifetime = (
            lazy.group_by(config.ID_COL)
            .agg(
                [
                    pl.len().alias("INS_COUNT"),
                    pl.col("INS_DPD").mean().alias("INS_DPD_MEAN"),
                    pl.col("INS_DPD").max().alias("INS_DPD_MAX"),
                    pl.col("INS_DPD").sum().alias("INS_DPD_SUM"),
                    pl.col("INS_DBD").mean().alias("INS_DBD_MEAN"),
                    pl.col("INS_PAYMENT_PERC").mean().alias("INS_PAYMENT_PERC_MEAN"),
                    pl.col("INS_PAYMENT_DIFF").mean().alias("INS_PAYMENT_DIFF_MEAN"),
                    pl.col("INS_LATE_FLAG").mean().alias("INS_LATE_FLAG_MEAN"),
                    pl.col("INS_LATE_FLAG").sum().alias("INS_LATE_FLAG_SUM"),
                ]
            )
            .collect(streaming=True)
        )

        # 🆕 B3 windowed features
        windowed = self._b3_windowed(lazy)

        return lifetime.join(windowed, on=config.ID_COL, how="left")

    def _b3_windowed(self, lazy: pl.LazyFrame) -> pl.DataFrame:
        """
        🆕 B3: windowed aggregations on DPD / LATE_FLAG / PAYMENT_PERC for the
        last 1, 3, 6, 12 months (PLAN §2.4 B3).

        Convention: ``DAYS_INSTALMENT`` is negative → "last 1 month" means
        ``DAYS_INSTALMENT >= -30``.

        TODO: expand metrics per the plan (target ~40 windowed features total).
        """
        windows = {"1M": -30, "3M": -90, "6M": -180, "12M": -365}

        # Build one aggregation per window, then join.
        out: pl.DataFrame | None = None
        for label, days_threshold in windows.items():
            agg = (
                lazy.filter(pl.col("DAYS_INSTALMENT") >= days_threshold)
                .group_by(config.ID_COL)
                .agg(
                    [
                        pl.len().alias(f"INS_W{label}_COUNT"),
                        pl.col("INS_DPD").mean().alias(f"INS_W{label}_DPD_MEAN"),
                        pl.col("INS_LATE_FLAG").mean().alias(f"INS_W{label}_LATE_RATE"),
                        pl.col("INS_PAYMENT_PERC").mean().alias(f"INS_W{label}_PAY_PERC"),
                    ]
                )
                .collect(streaming=True)
            )
            out = agg if out is None else out.join(agg, on=config.ID_COL, how="outer_coalesce")

        assert out is not None
        return out
