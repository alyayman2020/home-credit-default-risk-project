"""
``credit_card_balance`` feature builder.

Reference: PLAN_v2.md §2.6 — ~150 features.

Composition
-----------
- Base aggregations (~80): all numerics × {mean, max, min, sum, std}.
- Utilization features: AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL.
- Drawing-pattern features: count of months with drawings, mean drawing amount.
- Overlimit features: count of months with balance > limit, max overlimit ratio.
- 🆕 B3 windowed (last 3, 6, 12 months) on key metrics → ~30.
- DPD aggregations + recency → ~10.

3.8M rows — light enough for eager Polars (~150 MB peak per PLAN §6.1).
"""

from __future__ import annotations

import polars as pl

from src import config
from src.data import read_processed
from src.features.base import FeatureBuilder

# B3 windows (months — credit_card_balance uses MONTHS_BALANCE like POS).
WINDOWS_MONTHS: dict[str, int] = {
    "3M": -3,
    "6M": -6,
    "12M": -12,
}

# Numeric columns to aggregate.
CC_NUMERIC_COLS: list[str] = [
    "MONTHS_BALANCE",
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_ATM_CURRENT",
    "AMT_DRAWINGS_CURRENT",
    "AMT_DRAWINGS_OTHER_CURRENT",
    "AMT_DRAWINGS_POS_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "AMT_PAYMENT_CURRENT",
    "AMT_PAYMENT_TOTAL_CURRENT",
    "AMT_RECEIVABLE_PRINCIPAL",
    "AMT_RECIVABLE",
    "AMT_TOTAL_RECEIVABLE",
    "CNT_DRAWINGS_ATM_CURRENT",
    "CNT_DRAWINGS_CURRENT",
    "CNT_DRAWINGS_OTHER_CURRENT",
    "CNT_DRAWINGS_POS_CURRENT",
    "CNT_INSTALMENT_MATURE_CUM",
    "SK_DPD",
    "SK_DPD_DEF",
]


class CreditCardFeatures(FeatureBuilder):
    """Build features from ``credit_card_balance``."""

    name = "credit_card_balance"
    prefix = "CC_"

    def build(self) -> pl.DataFrame:
        cc = read_processed("credit_card_balance")
        cc = self._engineer_per_row(cc)

        base = self._aggregate(cc, prefix="CC")
        windowed = []
        for tag, threshold in WINDOWS_MONTHS.items():
            sub = cc.filter(pl.col("MONTHS_BALANCE") >= threshold)
            if sub.height == 0:
                continue
            windowed.append(self._aggregate(sub, prefix=f"CC_{tag}"))

        out = base
        for sub_agg in windowed:
            out = out.join(sub_agg, on=config.ID_COL, how="left")
        return out

    # ─── Per-row engineering ──────────────────────────────────────────────────

    def _engineer_per_row(self, cc: pl.DataFrame) -> pl.DataFrame:
        """Add per-row engineered columns: utilization, overlimit flag, drawing flag."""
        new_cols: list[pl.Expr] = []

        if "AMT_BALANCE" in cc.columns and "AMT_CREDIT_LIMIT_ACTUAL" in cc.columns:
            new_cols.append(
                (pl.col("AMT_BALANCE") / pl.col("AMT_CREDIT_LIMIT_ACTUAL")).alias("CC_UTILIZATION")
            )
            # Overlimit flag — month where balance exceeded limit.
            new_cols.append(
                (pl.col("AMT_BALANCE") > pl.col("AMT_CREDIT_LIMIT_ACTUAL"))
                .cast(pl.Int8)
                .alias("CC_OVERLIMIT_FLAG")
            )

        if "CNT_DRAWINGS_CURRENT" in cc.columns:
            new_cols.append(
                (pl.col("CNT_DRAWINGS_CURRENT") > 0)
                .cast(pl.Int8)
                .alias("CC_DRAWING_FLAG")
            )

        # DPD flags.
        if "SK_DPD" in cc.columns:
            new_cols.append(
                (pl.col("SK_DPD") > 0).cast(pl.Int8).alias("CC_DPD_FLAG")
            )
        if "SK_DPD_DEF" in cc.columns:
            new_cols.append(
                (pl.col("SK_DPD_DEF") > 0).cast(pl.Int8).alias("CC_DPD_DEF_FLAG")
            )

        return cc.with_columns(new_cols) if new_cols else cc

    # ─── Aggregations ─────────────────────────────────────────────────────────

    def _aggregate(self, cc: pl.DataFrame, prefix: str) -> pl.DataFrame:
        """Numeric + flag aggregations per SK_ID_CURR."""
        agg_exprs: list[pl.Expr] = [pl.len().alias(f"{prefix}_COUNT")]

        # Numeric aggregations (mean/max/min/sum/std).
        for col in CC_NUMERIC_COLS:
            if col not in cc.columns:
                continue
            agg_exprs += [
                pl.col(col).mean().alias(f"{prefix}_{col}_MEAN"),
                pl.col(col).max().alias(f"{prefix}_{col}_MAX"),
                pl.col(col).min().alias(f"{prefix}_{col}_MIN"),
                pl.col(col).sum().alias(f"{prefix}_{col}_SUM"),
                pl.col(col).std().alias(f"{prefix}_{col}_STD"),
            ]

        # Engineered flag aggregations.
        for flag_col, label in (
            ("CC_UTILIZATION", "UTILIZATION"),
            ("CC_OVERLIMIT_FLAG", "OVERLIMIT"),
            ("CC_DRAWING_FLAG", "DRAWING"),
            ("CC_DPD_FLAG", "DPD"),
            ("CC_DPD_DEF_FLAG", "DPD_DEF"),
        ):
            if flag_col not in cc.columns:
                continue
            if label == "UTILIZATION":
                agg_exprs += [
                    pl.col(flag_col).mean().alias(f"{prefix}_{label}_MEAN"),
                    pl.col(flag_col).max().alias(f"{prefix}_{label}_MAX"),
                    pl.col(flag_col).min().alias(f"{prefix}_{label}_MIN"),
                ]
            else:
                agg_exprs += [
                    pl.col(flag_col).sum().alias(f"{prefix}_{label}_SUM"),
                    pl.col(flag_col).mean().alias(f"{prefix}_{label}_RATE"),
                ]

        return cc.group_by(config.ID_COL).agg(agg_exprs)
