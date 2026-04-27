"""
``previous_application`` feature builder.

Reference: PLAN_v2.md §2.3 — ~250 features.

Composition
-----------
- Per-row engineered features: APP_CREDIT_PERC, downpayment ratio.
- Numeric aggregations per SK_ID_CURR (count / mean / sum / min / max / std).
- Approved/Refused stratification (NAME_CONTRACT_STATUS) — refusal stats are
  among the strongest signals in this competition.
- Top contract-type stratification (Cash, Revolving, Consumer).
- Categorical aggregations (n_unique, mode-via-frequency).
"""

from __future__ import annotations

import polars as pl

from src import config
from src.data import read_processed
from src.features.base import FeatureBuilder

# Numeric columns to aggregate over previous_application rows.
PREV_NUMERIC_COLS: list[str] = [
    "AMT_ANNUITY",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "AMT_DOWN_PAYMENT",
    "AMT_GOODS_PRICE",
    "HOUR_APPR_PROCESS_START",
    "RATE_DOWN_PAYMENT",
    "RATE_INTEREST_PRIMARY",
    "RATE_INTEREST_PRIVILEGED",
    "DAYS_DECISION",
    "CNT_PAYMENT",
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
    "NFLAG_INSURED_ON_APPROVAL",
]


class PreviousApplicationFeatures(FeatureBuilder):
    """Build features from ``previous_application``."""

    name = "previous_application"
    prefix = "PREV_"

    def build(self) -> pl.DataFrame:
        prev = read_processed("previous_application")

        # Per-row engineering before aggregation.
        prev = self._engineer_per_row(prev)

        # Base aggregations across all rows for each borrower.
        base = self._aggregate_base(prev)

        # Stratified by NAME_CONTRACT_STATUS = Approved.
        approved = self._aggregate_stratified(
            prev, filter_col="NAME_CONTRACT_STATUS", filter_val="Approved", prefix="PREV_APPROVED"
        )
        # Stratified by NAME_CONTRACT_STATUS = Refused.
        refused = self._aggregate_stratified(
            prev, filter_col="NAME_CONTRACT_STATUS", filter_val="Refused", prefix="PREV_REFUSED"
        )

        # Stratified by NAME_CONTRACT_TYPE for the three main types.
        type_aggs = self._aggregate_by_contract_type(prev)

        # Categorical aggregations + counts.
        cats = self._categorical_aggregations(prev)

        # Join everything on SK_ID_CURR.
        out = base
        for sub in (approved, refused, type_aggs, cats):
            out = out.join(sub, on=config.ID_COL, how="left")

        # Final cross-aggregate ratios.
        out = self._cross_ratios(out)
        return out

    # ─── Per-row engineering ──────────────────────────────────────────────────

    def _engineer_per_row(self, prev: pl.DataFrame) -> pl.DataFrame:
        """Add per-row features used by the aggregations."""
        new_cols: list[pl.Expr] = []

        if "AMT_APPLICATION" in prev.columns and "AMT_CREDIT" in prev.columns:
            # APP_CREDIT_PERC: how much of what they applied for did they actually receive?
            new_cols.append(
                (pl.col("AMT_APPLICATION") / pl.col("AMT_CREDIT")).alias("PREV_APP_CREDIT_PERC")
            )

        if "AMT_DOWN_PAYMENT" in prev.columns and "AMT_CREDIT" in prev.columns:
            new_cols.append(
                (pl.col("AMT_DOWN_PAYMENT") / pl.col("AMT_CREDIT")).alias(
                    "PREV_DOWNPAYMENT_TO_CREDIT_RATIO"
                )
            )

        # Refusal flag (per row) — used in cross_ratios for refusal rate.
        if "NAME_CONTRACT_STATUS" in prev.columns:
            new_cols.append(
                (pl.col("NAME_CONTRACT_STATUS") == "Refused")
                .cast(pl.Int8)
                .alias("_PREV_IS_REFUSED")
            )
            new_cols.append(
                (pl.col("NAME_CONTRACT_STATUS") == "Approved")
                .cast(pl.Int8)
                .alias("_PREV_IS_APPROVED")
            )

        return prev.with_columns(new_cols) if new_cols else prev

    # ─── Aggregations ─────────────────────────────────────────────────────────

    def _aggregate_base(self, prev: pl.DataFrame) -> pl.DataFrame:
        """Un-stratified aggregations per SK_ID_CURR (~120 features)."""
        agg_exprs: list[pl.Expr] = [pl.len().alias("PREV_COUNT")]

        # Numeric aggregations.
        for col in PREV_NUMERIC_COLS:
            if col not in prev.columns:
                continue
            agg_exprs += [
                pl.col(col).mean().alias(f"PREV_{col}_MEAN"),
                pl.col(col).max().alias(f"PREV_{col}_MAX"),
                pl.col(col).min().alias(f"PREV_{col}_MIN"),
                pl.col(col).sum().alias(f"PREV_{col}_SUM"),
                pl.col(col).std().alias(f"PREV_{col}_STD"),
            ]

        # Engineered per-row aggregations.
        if "PREV_APP_CREDIT_PERC" in prev.columns:
            agg_exprs += [
                pl.col("PREV_APP_CREDIT_PERC").mean().alias("PREV_APP_CREDIT_PERC_MEAN"),
                pl.col("PREV_APP_CREDIT_PERC").max().alias("PREV_APP_CREDIT_PERC_MAX"),
                pl.col("PREV_APP_CREDIT_PERC").min().alias("PREV_APP_CREDIT_PERC_MIN"),
            ]
        if "PREV_DOWNPAYMENT_TO_CREDIT_RATIO" in prev.columns:
            agg_exprs += [
                pl.col("PREV_DOWNPAYMENT_TO_CREDIT_RATIO")
                .mean()
                .alias("PREV_DOWNPAYMENT_RATIO_MEAN"),
                pl.col("PREV_DOWNPAYMENT_TO_CREDIT_RATIO")
                .max()
                .alias("PREV_DOWNPAYMENT_RATIO_MAX"),
            ]

        # Refusal / approval rates.
        if "_PREV_IS_REFUSED" in prev.columns:
            agg_exprs += [
                pl.col("_PREV_IS_REFUSED").sum().alias("PREV_REFUSED_COUNT"),
                pl.col("_PREV_IS_REFUSED").mean().alias("PREV_REFUSAL_RATE"),
            ]
        if "_PREV_IS_APPROVED" in prev.columns:
            agg_exprs += [
                pl.col("_PREV_IS_APPROVED").sum().alias("PREV_APPROVED_COUNT"),
                pl.col("_PREV_IS_APPROVED").mean().alias("PREV_APPROVAL_RATE"),
            ]

        return prev.group_by(config.ID_COL).agg(agg_exprs)

    def _aggregate_stratified(
        self, prev: pl.DataFrame, filter_col: str, filter_val: str, prefix: str
    ) -> pl.DataFrame:
        """Aggregate the same numerics but only on rows where filter_col == filter_val."""
        if filter_col not in prev.columns:
            # Empty fallback frame — keyed by SK_ID_CURR only.
            return prev.select(config.ID_COL).unique().with_columns(
                pl.lit(None).alias(f"{prefix}_COUNT")
            )

        sub = prev.filter(pl.col(filter_col) == filter_val)
        if sub.height == 0:
            return prev.select(config.ID_COL).unique().with_columns(
                pl.lit(0).alias(f"{prefix}_COUNT")
            )

        agg_exprs: list[pl.Expr] = [pl.len().alias(f"{prefix}_COUNT")]
        for col in PREV_NUMERIC_COLS:
            if col not in sub.columns:
                continue
            agg_exprs += [
                pl.col(col).mean().alias(f"{prefix}_{col}_MEAN"),
                pl.col(col).max().alias(f"{prefix}_{col}_MAX"),
                pl.col(col).sum().alias(f"{prefix}_{col}_SUM"),
            ]
        return sub.group_by(config.ID_COL).agg(agg_exprs)

    def _aggregate_by_contract_type(self, prev: pl.DataFrame) -> pl.DataFrame:
        """Per-contract-type aggregations: Cash loans, Revolving loans, Consumer loans."""
        result = prev.select(config.ID_COL).unique()
        if "NAME_CONTRACT_TYPE" not in prev.columns:
            return result

        for ctype, tag in (
            ("Cash loans", "PREV_CASH"),
            ("Revolving loans", "PREV_REVOLV"),
            ("Consumer loans", "PREV_CONSUMER"),
        ):
            sub = prev.filter(pl.col("NAME_CONTRACT_TYPE") == ctype)
            if sub.height == 0:
                continue
            agg_exprs: list[pl.Expr] = [pl.len().alias(f"{tag}_COUNT")]
            for col in ("AMT_CREDIT", "AMT_ANNUITY", "DAYS_DECISION"):
                if col not in sub.columns:
                    continue
                agg_exprs += [
                    pl.col(col).mean().alias(f"{tag}_{col}_MEAN"),
                    pl.col(col).max().alias(f"{tag}_{col}_MAX"),
                ]
            sub_agg = sub.group_by(config.ID_COL).agg(agg_exprs)
            result = result.join(sub_agg, on=config.ID_COL, how="left")
        return result

    def _categorical_aggregations(self, prev: pl.DataFrame) -> pl.DataFrame:
        """Categorical column n_uniques per borrower."""
        result = prev.select(config.ID_COL).unique()
        cat_cols = [
            "NAME_CONTRACT_TYPE",
            "NAME_PRODUCT_TYPE",
            "NAME_CASH_LOAN_PURPOSE",
            "CHANNEL_TYPE",
            "NAME_PORTFOLIO",
            "NAME_YIELD_GROUP",
            "CODE_REJECT_REASON",
        ]
        agg_exprs: list[pl.Expr] = []
        for col in cat_cols:
            if col in prev.columns:
                agg_exprs.append(pl.col(col).n_unique().alias(f"PREV_CAT_{col}_NUNIQUE"))
        if not agg_exprs:
            return result
        return prev.group_by(config.ID_COL).agg(agg_exprs)

    def _cross_ratios(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cross-aggregate features computed from the joined output."""
        new_cols: list[pl.Expr] = []
        # Approved-to-refused count ratio
        if "PREV_APPROVED_COUNT" in df.columns and "PREV_REFUSED_COUNT" in df.columns:
            new_cols.append(
                (
                    pl.col("PREV_APPROVED_COUNT") / (pl.col("PREV_REFUSED_COUNT") + 1)
                ).alias("PREV_APPROVED_TO_REFUSED_RATIO")
            )
        return df.with_columns(new_cols) if new_cols else df
