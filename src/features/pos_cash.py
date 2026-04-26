"""
``POS_CASH_balance`` feature builder.

Reference: PLAN_v2.md §2.5 — ~110 features (+30 from v1).

Composition
-----------
- Base aggregations (~50): unchanged from v1.
- 🆕 B3 windowed (last 3, 6, 12 months): SK_DPD, SK_DPD_DEF, status share → ~30 features.
- Recency-weighted DPD → ~10 features.
- Categorical encoding → ~10 features.

10M rows — load lazy and stream (PLAN §6.1: ~100 MB peak).

This is a functional stub.
"""

from __future__ import annotations

import polars as pl

from src import config
from src.features.base import FeatureBuilder


class PosCashFeatures(FeatureBuilder):
    """Build features from POS_CASH_balance."""

    name = "pos_cash_balance"
    prefix = "POS_"

    def build(self) -> pl.DataFrame:
        path = config.PROCESSED_DIR / "pos_cash_balance.parquet"
        lazy = pl.scan_parquet(path)

        # Lifetime aggregations
        lifetime = (
            lazy.group_by(config.ID_COL)
            .agg(
                [
                    pl.len().alias("POS_COUNT"),
                    pl.col("MONTHS_BALANCE").max().alias("POS_MONTHS_BALANCE_MAX"),
                    pl.col("MONTHS_BALANCE").min().alias("POS_MONTHS_BALANCE_MIN"),
                    pl.col("SK_DPD").mean().alias("POS_SK_DPD_MEAN"),
                    pl.col("SK_DPD").max().alias("POS_SK_DPD_MAX"),
                    pl.col("SK_DPD_DEF").mean().alias("POS_SK_DPD_DEF_MEAN"),
                    pl.col("SK_DPD_DEF").max().alias("POS_SK_DPD_DEF_MAX"),
                    pl.col("CNT_INSTALMENT").mean().alias("POS_CNT_INSTALMENT_MEAN"),
                    pl.col("CNT_INSTALMENT_FUTURE").mean().alias("POS_CNT_INST_FUTURE_MEAN"),
                ]
            )
            .collect(streaming=True)
        )

        # 🆕 B3: windowed
        windowed = self._b3_windowed(lazy)

        # TODO: implement recency-weighted DPD + status-share + categorical encoding
        # to reach the ~110-feature target.

        return lifetime.join(windowed, on=config.ID_COL, how="left")

    def _b3_windowed(self, lazy: pl.LazyFrame) -> pl.DataFrame:
        """
        🆕 B3 windowed aggregations on SK_DPD, SK_DPD_DEF, status share for last
        3 / 6 / 12 months. ~30 features total.

        TODO: expand status-share aggregations (NAME_CONTRACT_STATUS levels).
        """
        windows = {"3M": -3, "6M": -6, "12M": -12}
        out: pl.DataFrame | None = None
        for label, m in windows.items():
            agg = (
                lazy.filter(pl.col("MONTHS_BALANCE") >= m)
                .group_by(config.ID_COL)
                .agg(
                    [
                        pl.len().alias(f"POS_W{label}_COUNT"),
                        pl.col("SK_DPD").mean().alias(f"POS_W{label}_DPD_MEAN"),
                        pl.col("SK_DPD_DEF").mean().alias(f"POS_W{label}_DPD_DEF_MEAN"),
                    ]
                )
                .collect(streaming=True)
            )
            out = agg if out is None else out.join(agg, on=config.ID_COL, how="outer_coalesce")
        assert out is not None
        return out
