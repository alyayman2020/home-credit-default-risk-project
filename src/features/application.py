"""
``application`` table feature builder.

Reference: PLAN_v2.md §2.1 — ~270 features, target +20 from v1 via D1 + D4.

Feature families
----------------
- Domain ratios (~25): income / time / credit ratios. ``CREDIT_TERM = AMT_ANNUITY/AMT_CREDIT``.
- EXT_SOURCE engineering (~20): mean / std / prod / min / max of EXT_SOURCE_{1,2,3}.
- 🆕 D1 neighbours target mean (1): ``TARGET_NEIGHBORS_500_MEAN`` — fold-aware,
  built inside the CV loop. *Computed in train.py, NOT here.*
- 🆕 D4 divisive EXT × DAYS_BIRTH interactions (8 features).
- Document & flag aggregates (~5).
- Building aggregates (~12).
- Categorical encoding (~180): mix of frequency / OHE / target (target inside CV).

NOTE: This is a functional stub. The feature names below are placeholders that
illustrate the structure; flesh out each TODO block per the plan.
"""

from __future__ import annotations

import polars as pl

from src import config
from src.data import read_processed
from src.features.base import FeatureBuilder


class ApplicationFeatures(FeatureBuilder):
    """Build features from ``application_train.csv`` + ``application_test.csv``."""

    name = "application"
    prefix = "APP_"

    def build(self) -> pl.DataFrame:
        train = read_processed("application_train")
        test = read_processed("application_test")
        # Concatenate so we engineer both populations together (TARGET excluded for test).
        df = pl.concat([train.drop(config.TARGET_COL), test], how="vertical_relaxed")

        df = self._domain_ratios(df)
        df = self._ext_source_features(df)
        df = self._d4_divisive_interactions(df)
        df = self._document_flag_aggregates(df)
        df = self._building_aggregates(df)
        # Categorical encoding (frequency / OHE) — target encoding is fold-aware,
        # so it lives in src/train.py, not here.
        df = self._categorical_encoding(df)
        return df

    # ─── Subroutines ──────────────────────────────────────────────────────────

    def _domain_ratios(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        ~25 ratio features (PLAN §2.1).

        TODO: complete. Examples:
          APP_CREDIT_TERM         = AMT_ANNUITY / AMT_CREDIT             ⭐ top-importance
          APP_CREDIT_INCOME_RATIO = AMT_CREDIT  / AMT_INCOME_TOTAL
          APP_ANNUITY_INCOME_RATIO= AMT_ANNUITY / AMT_INCOME_TOTAL
          APP_INCOME_PER_PERSON   = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
          APP_DAYS_EMPLOYED_RATIO = DAYS_EMPLOYED / DAYS_BIRTH
        """
        return df.with_columns(
            [
                (pl.col("AMT_ANNUITY") / pl.col("AMT_CREDIT")).alias("APP_CREDIT_TERM"),
                (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL")).alias(
                    "APP_CREDIT_INCOME_RATIO"
                ),
                (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias(
                    "APP_ANNUITY_INCOME_RATIO"
                ),
                (pl.col("AMT_INCOME_TOTAL") / pl.col("CNT_FAM_MEMBERS")).alias(
                    "APP_INCOME_PER_PERSON"
                ),
                (pl.col("DAYS_EMPLOYED") / pl.col("DAYS_BIRTH")).alias(
                    "APP_DAYS_EMPLOYED_RATIO"
                ),
            ]
        )

    def _ext_source_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        EXT_SOURCE_{1,2,3} aggregates: mean / std / min / max / prod / nan-count.

        TODO: implement full set per PLAN §2.1.
        """
        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        return df.with_columns(
            [
                pl.mean_horizontal(ext_cols).alias("APP_EXT_MEAN"),
                pl.min_horizontal(ext_cols).alias("APP_EXT_MIN"),
                pl.max_horizontal(ext_cols).alias("APP_EXT_MAX"),
                (pl.col("EXT_SOURCE_1") * pl.col("EXT_SOURCE_2") * pl.col("EXT_SOURCE_3")).alias(
                    "APP_EXT_PROD"
                ),
            ]
        )

    def _d4_divisive_interactions(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        🆕 D4: 8 divisive EXT × DAYS_BIRTH interactions (PLAN §2.1 D4).

        TODO: implement the remaining 6 interactions:
          APP_CREDIT_TO_ANNUITY_RATIO_DIV_SCORE1_TO_BIRTH_RATIO
          APP_CREDIT_TO_ANNUITY_RATIO_DIV_DAYS_BIRTH
          APP_INCOME_PER_PERSON_DIV_EXT_MEAN
          APP_CREDIT_DIV_EXT_PROD
          APP_ANNUITY_DIV_EXT_MAX
          APP_DAYS_EMPLOYED_DIV_EXT_MIN
        """
        return df.with_columns(
            [
                (pl.col("APP_EXT_MEAN") / pl.col("DAYS_EMPLOYED")).alias(
                    "APP_EXT_MEAN_DIV_DAYS_EMPLOYED"
                ),
                (pl.col("APP_EXT_MEAN") * (-pl.col("DAYS_BIRTH") / 365)).alias(
                    "APP_EXT_MEAN_MUL_AGE_YEARS"
                ),
            ]
        )

    def _document_flag_aggregates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        ~5 features: sum / mean of FLAG_DOCUMENT_* columns.

        TODO: implement.
        """
        return df

    def _building_aggregates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        ~12 features: family means of *_AVG / *_MEDI / *_MODE columns
        (~50% missingness — aggregate to family means per PLAN §1.4).

        TODO: implement.
        """
        return df

    def _categorical_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Frequency encoding + OHE for low-cardinality categoricals
        (~180 features — PLAN §2.1).

        TODO: implement frequency encoding for high-cardinality cats
        (ORGANIZATION_TYPE has 58 levels), OHE for low-cardinality cats.
        Target encoding is fold-aware → lives in src/train.py.
        """
        return df
