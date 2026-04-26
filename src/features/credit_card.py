"""
``credit_card_balance`` feature builder.

Reference: PLAN_v2.md §2.6 — ~150 features (+30 from v1).

Composition
-----------
- Base aggregations (~80): utilization, drawing patterns, overlimit.
- 🆕 B3 windowed (last 3, 6, 12 months) on utilization / balance / drawings → ~30.
- DPD aggregations + recency → ~10.
- Categorical encoding → ~10.

3.8M rows — light enough for eager Polars (~150 MB peak per PLAN §6.1).

This is a functional stub.
"""

from __future__ import annotations

import polars as pl

from src import config
from src.data import read_processed
from src.features.base import FeatureBuilder


class CreditCardFeatures(FeatureBuilder):
    """Build features from credit_card_balance."""

    name = "credit_card_balance"
    prefix = "CC_"

    def build(self) -> pl.DataFrame:
        cc = read_processed("credit_card_balance")

        # Per-row engineering: utilization
        cc = cc.with_columns(
            [
                (pl.col("AMT_BALANCE") / pl.col("AMT_CREDIT_LIMIT_ACTUAL"))
                .alias("CC_UTIL"),
                (pl.col("AMT_DRAWINGS_CURRENT") / pl.col("AMT_CREDIT_LIMIT_ACTUAL"))
                .alias("CC_DRAW_RATIO"),
            ]
        )

        # Lifetime
        lifetime = cc.group_by(config.ID_COL).agg(
            [
                pl.len().alias("CC_COUNT"),
                pl.col("AMT_BALANCE").mean().alias("CC_AMT_BALANCE_MEAN"),
                pl.col("AMT_BALANCE").max().alias("CC_AMT_BALANCE_MAX"),
                pl.col("AMT_CREDIT_LIMIT_ACTUAL").mean().alias("CC_LIMIT_MEAN"),
                pl.col("CC_UTIL").mean().alias("CC_UTIL_MEAN"),
                pl.col("CC_UTIL").max().alias("CC_UTIL_MAX"),
                pl.col("CC_DRAW_RATIO").mean().alias("CC_DRAW_RATIO_MEAN"),
                pl.col("SK_DPD").mean().alias("CC_SK_DPD_MEAN"),
                pl.col("SK_DPD").max().alias("CC_SK_DPD_MAX"),
                pl.col("SK_DPD_DEF").mean().alias("CC_SK_DPD_DEF_MEAN"),
                pl.col("MONTHS_BALANCE").min().alias("CC_MONTHS_MIN"),
            ]
        )

        # 🆕 B3 windowed
        windowed = self._b3_windowed(cc)

        # TODO: add categorical encoding (NAME_CONTRACT_STATUS levels)
        # and the rest of the ~80 base features.

        return lifetime.join(windowed, on=config.ID_COL, how="left")

    def _b3_windowed(self, cc: pl.DataFrame) -> pl.DataFrame:
        """🆕 B3 windowed aggregations on utilization / balance / drawings."""
        windows = {"3M": -3, "6M": -6, "12M": -12}
        out: pl.DataFrame | None = None
        for label, m in windows.items():
            agg = (
                cc.filter(pl.col("MONTHS_BALANCE") >= m)
                .group_by(config.ID_COL)
                .agg(
                    [
                        pl.col("CC_UTIL").mean().alias(f"CC_W{label}_UTIL_MEAN"),
                        pl.col("CC_UTIL").max().alias(f"CC_W{label}_UTIL_MAX"),
                        pl.col("AMT_BALANCE").mean().alias(f"CC_W{label}_BAL_MEAN"),
                        pl.col("AMT_DRAWINGS_CURRENT").sum().alias(f"CC_W{label}_DRAW_SUM"),
                    ]
                )
            )
            out = agg if out is None else out.join(agg, on=config.ID_COL, how="outer_coalesce")
        assert out is not None
        return out
