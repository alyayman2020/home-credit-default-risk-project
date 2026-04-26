"""
``application`` table feature builder.

Reference: PLAN_v2.md §2.1 — ~270 features.

Feature families
----------------
- Domain ratios (~15): income / time / credit ratios. ``CREDIT_TERM = AMT_ANNUITY/AMT_CREDIT``.
- EXT_SOURCE engineering (~20): mean / std / prod / min / max + nullity flags + interactions.
- 🆕 D4 divisive EXT × DAYS_BIRTH interactions (8 features).
- Document & flag aggregates (~5).
- Building info aggregates (~15): family means + null counts of *_AVG / *_MEDI / *_MODE.
- Categorical encoding (~110): low-card OHE + frequency + count encoding.

NOT computed here (lives in train.py, fold-aware):
- D1 ``TARGET_NEIGHBORS_500_MEAN`` — needs OOF NearestNeighbors index.
- Target encoding for high-cardinality categoricals (ORGANIZATION_TYPE, OCCUPATION_TYPE).
"""

from __future__ import annotations

import polars as pl

from src import config
from src.data import read_processed
from src.features.base import FeatureBuilder

# ─── Categorical column registry ──────────────────────────────────────────────

# Low-cardinality (≤10 unique) — full one-hot.
LOW_CARD_OHE_COLS: list[str] = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_CONTRACT_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]

# High-cardinality — frequency + count encoding here, target encoding in train.py.
HIGH_CARD_COLS: list[str] = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]

# All categoricals (for frequency + count encoding).
ALL_CAT_COLS: list[str] = LOW_CARD_OHE_COLS + HIGH_CARD_COLS

# Building-info column families (~50% missing per PLAN §1.4).
BUILDING_SUFFIXES: list[str] = ["_AVG", "_MEDI", "_MODE"]
BUILDING_PREFIXES: list[str] = [
    "APARTMENTS",
    "BASEMENTAREA",
    "YEARS_BEGINEXPLUATATION",
    "YEARS_BUILD",
    "COMMONAREA",
    "ELEVATORS",
    "ENTRANCES",
    "FLOORSMAX",
    "FLOORSMIN",
    "LANDAREA",
    "LIVINGAPARTMENTS",
    "LIVINGAREA",
    "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA",
]


class ApplicationFeatures(FeatureBuilder):
    """Build features from ``application_train.csv`` + ``application_test.csv``."""

    name = "application"
    prefix = "APP_"

    def build(self) -> pl.DataFrame:
        train = read_processed("application_train")
        test = read_processed("application_test")

        # Concatenate so all engineering and OHE align across train+test.
        # TARGET column is dropped from train so the schemas match; the assembler
        # re-attaches TARGET via SK_ID_CURR.
        train_no_tgt = train.drop(config.TARGET_COL)
        df = pl.concat([train_no_tgt, test], how="vertical_relaxed")
        self.logger.info(f"  application combined: {df.shape}")

        df = self._domain_ratios(df)
        df = self._ext_source_features(df)
        df = self._d4_divisive_interactions(df)
        df = self._document_flag_aggregates(df)
        df = self._building_aggregates(df)
        df = self._categorical_encoding(df)
        df = self._select_outputs(df)

        return df

    # ─── Subroutines ──────────────────────────────────────────────────────────

    def _domain_ratios(self, df: pl.DataFrame) -> pl.DataFrame:
        """~15 ratio features (PLAN §2.1)."""
        return df.with_columns(
            [
                # Credit / annuity / income ratios
                (pl.col("AMT_ANNUITY") / pl.col("AMT_CREDIT")).alias("APP_CREDIT_TERM"),
                (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL")).alias("APP_CREDIT_INCOME_RATIO"),
                (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias("APP_ANNUITY_INCOME_RATIO"),
                (pl.col("AMT_GOODS_PRICE") / pl.col("AMT_CREDIT")).alias("APP_GOODS_PRICE_TO_CREDIT"),
                (pl.col("AMT_ANNUITY") / pl.col("AMT_GOODS_PRICE")).alias("APP_ANNUITY_TO_GOODS"),
                (pl.col("AMT_CREDIT") - pl.col("AMT_GOODS_PRICE")).alias("APP_CREDIT_GOODS_DIFF"),
                # Per-person normalisation
                (pl.col("AMT_INCOME_TOTAL") / pl.col("CNT_FAM_MEMBERS")).alias("APP_INCOME_PER_PERSON"),
                (pl.col("AMT_INCOME_TOTAL") / (pl.col("CNT_CHILDREN") + 1)).alias("APP_INCOME_PER_CHILD"),
                (pl.col("AMT_CREDIT") / pl.col("CNT_FAM_MEMBERS")).alias("APP_CREDIT_PER_PERSON"),
                # Time / age conversions (DAYS_* are negative integers)
                (-pl.col("DAYS_BIRTH") / 365.0).alias("APP_AGE_YEARS"),
                (-pl.col("DAYS_EMPLOYED") / 365.0).alias("APP_EMPLOYED_YEARS"),
                (pl.col("DAYS_EMPLOYED") / pl.col("DAYS_BIRTH")).alias("APP_DAYS_EMPLOYED_RATIO"),
                (-pl.col("DAYS_REGISTRATION") / 365.0).alias("APP_REGISTRATION_AGE_YEARS"),
                (-pl.col("DAYS_ID_PUBLISH") / 365.0).alias("APP_ID_PUBLISH_AGE_YEARS"),
                (pl.col("OWN_CAR_AGE") / (-pl.col("DAYS_BIRTH") / 365.0)).alias("APP_CAR_TO_AGE"),
            ]
        )

    def _ext_source_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """~20 EXT_SOURCE features (PLAN §2.1 EXT block)."""
        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        df = df.with_columns(
            [
                # Aggregations across the three sources
                pl.mean_horizontal(ext_cols).alias("APP_EXT_MEAN"),
                pl.min_horizontal(ext_cols).alias("APP_EXT_MIN"),
                pl.max_horizontal(ext_cols).alias("APP_EXT_MAX"),
                # Std across the three (manual since pl.std_horizontal isn't always available).
                # Build via concat_list → list.std.
                pl.concat_list(ext_cols).list.std().alias("APP_EXT_STD"),
                # Product (commonly cited as one of the strongest interactions).
                (pl.col("EXT_SOURCE_1") * pl.col("EXT_SOURCE_2") * pl.col("EXT_SOURCE_3")).alias(
                    "APP_EXT_PROD"
                ),
                # Weighted mean — weights from public-kernel consensus (E2 strongest).
                (
                    2 * pl.col("EXT_SOURCE_1")
                    + 3 * pl.col("EXT_SOURCE_2")
                    + 4 * pl.col("EXT_SOURCE_3")
                ).alias("APP_EXT_WEIGHTED"),
                # Pairwise products and sums
                (pl.col("EXT_SOURCE_1") * pl.col("EXT_SOURCE_2")).alias("APP_EXT12_PROD"),
                (pl.col("EXT_SOURCE_1") * pl.col("EXT_SOURCE_3")).alias("APP_EXT13_PROD"),
                (pl.col("EXT_SOURCE_2") * pl.col("EXT_SOURCE_3")).alias("APP_EXT23_PROD"),
                (pl.col("EXT_SOURCE_1") + pl.col("EXT_SOURCE_2")).alias("APP_EXT12_SUM"),
                (pl.col("EXT_SOURCE_2") + pl.col("EXT_SOURCE_3")).alias("APP_EXT23_SUM"),
                # Pairwise differences
                (pl.col("EXT_SOURCE_1") - pl.col("EXT_SOURCE_2")).alias("APP_EXT12_DIFF"),
                (pl.col("EXT_SOURCE_2") - pl.col("EXT_SOURCE_3")).alias("APP_EXT23_DIFF"),
                # Nullity indicators (missingness is itself predictive — PLAN §1.4)
                pl.col("EXT_SOURCE_1").is_null().cast(pl.Int8).alias("APP_EXT1_ISNULL"),
                pl.col("EXT_SOURCE_2").is_null().cast(pl.Int8).alias("APP_EXT2_ISNULL"),
                pl.col("EXT_SOURCE_3").is_null().cast(pl.Int8).alias("APP_EXT3_ISNULL"),
            ]
        )
        # Count of non-null EXT_SOURCE_*.
        df = df.with_columns(
            (
                3
                - pl.col("APP_EXT1_ISNULL")
                - pl.col("APP_EXT2_ISNULL")
                - pl.col("APP_EXT3_ISNULL")
            ).alias("APP_EXT_NONNULL_COUNT")
        )
        return df

    def _d4_divisive_interactions(self, df: pl.DataFrame) -> pl.DataFrame:
        """🆕 D4: 8 divisive EXT × DAYS interactions (PLAN §2.1 D4)."""
        return df.with_columns(
            [
                # All require columns built upstream (APP_EXT_MEAN etc.)
                (
                    pl.col("APP_CREDIT_TERM") / (pl.col("EXT_SOURCE_1") / pl.col("DAYS_BIRTH"))
                ).alias("APP_CREDIT_TERM_DIV_EXT1_TO_BIRTH"),
                (pl.col("APP_CREDIT_TERM") / pl.col("DAYS_BIRTH")).alias(
                    "APP_CREDIT_TERM_DIV_DAYS_BIRTH"
                ),
                (pl.col("APP_INCOME_PER_PERSON") / pl.col("APP_EXT_MEAN")).alias(
                    "APP_INCOME_PER_PERSON_DIV_EXT_MEAN"
                ),
                (pl.col("AMT_CREDIT") / pl.col("APP_EXT_PROD")).alias("APP_CREDIT_DIV_EXT_PROD"),
                (pl.col("AMT_ANNUITY") / pl.col("APP_EXT_MAX")).alias("APP_ANNUITY_DIV_EXT_MAX"),
                (pl.col("DAYS_EMPLOYED") / pl.col("APP_EXT_MIN")).alias(
                    "APP_DAYS_EMPLOYED_DIV_EXT_MIN"
                ),
                (pl.col("APP_EXT_MEAN") / (-pl.col("DAYS_EMPLOYED") + 1)).alias(
                    "APP_EXT_MEAN_DIV_DAYS_EMPLOYED"
                ),
                (pl.col("APP_EXT_MEAN") * (-pl.col("DAYS_BIRTH") / 365.0)).alias(
                    "APP_EXT_MEAN_MUL_AGE_YEARS"
                ),
            ]
        )

    def _document_flag_aggregates(self, df: pl.DataFrame) -> pl.DataFrame:
        """~5 features summarising the FLAG_DOCUMENT_* family (20 cols → 5 features)."""
        doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
        if not doc_cols:
            return df

        new_cols = [
            pl.sum_horizontal(doc_cols).alias("APP_DOCUMENT_COUNT"),
            pl.mean_horizontal(doc_cols).alias("APP_DOCUMENT_MEAN"),
        ]
        # FLAG_DOCUMENT_3 is the single strongest doc flag in public kernels.
        if "FLAG_DOCUMENT_3" in df.columns:
            new_cols.append(pl.col("FLAG_DOCUMENT_3").alias("APP_DOCUMENT_3_PRESENT"))

        df = df.with_columns(new_cols)

        # Region/contact flag aggregates (separate family).
        contact_cols = [
            c
            for c in df.columns
            if c.startswith("FLAG_") and c not in doc_cols and c not in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY")
        ]
        if contact_cols:
            df = df.with_columns(pl.sum_horizontal(contact_cols).alias("APP_FLAG_CONTACT_SUM"))
        return df

    def _building_aggregates(self, df: pl.DataFrame) -> pl.DataFrame:
        """~15 features from the *_AVG / *_MEDI / *_MODE building info family."""
        new_cols: list[pl.Expr] = []
        for suffix in BUILDING_SUFFIXES:
            family_cols = [
                f"{p}{suffix}" for p in BUILDING_PREFIXES if f"{p}{suffix}" in df.columns
            ]
            if not family_cols:
                continue
            # Family mean of all valid building-info columns.
            new_cols.append(pl.mean_horizontal(family_cols).alias(f"APP_BUILDING{suffix}_MEAN"))
            # Family null-count (informative — PLAN §1.4).
            null_count = pl.sum_horizontal(
                [pl.col(c).is_null().cast(pl.Int32) for c in family_cols]
            )
            new_cols.append(null_count.alias(f"APP_BUILDING{suffix}_NULLCOUNT"))
        # Cross-suffix mean of all building info (single robust signal).
        all_building = [
            f"{p}{s}"
            for p in BUILDING_PREFIXES
            for s in BUILDING_SUFFIXES
            if f"{p}{s}" in df.columns
        ]
        if all_building:
            new_cols.append(pl.mean_horizontal(all_building).alias("APP_BUILDING_GLOBAL_MEAN"))
        return df.with_columns(new_cols) if new_cols else df

    def _categorical_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        ~110 features:
          - One-hot encoding for low-cardinality categoricals (~80 columns).
          - Frequency encoding for ALL categoricals (~16 columns).
          - Count encoding for ALL categoricals (~16 columns).
        Target encoding for high-cardinality cats happens in train.py (fold-aware).
        """
        # Frequency + count encoding (combined — frequency = count / total, count = raw count)
        for col in ALL_CAT_COLS:
            if col not in df.columns:
                continue
            counts = df.group_by(col).agg(pl.len().alias("_n"))
            df = df.join(counts, on=col, how="left")
            total = df.height
            df = df.with_columns(
                [
                    pl.col("_n").alias(f"APP_CAT_COUNT_{col}"),
                    (pl.col("_n") / total).alias(f"APP_CAT_FREQ_{col}"),
                ]
            ).drop("_n")

        # One-hot encoding (Polars to_dummies). Resulting columns are prefixed by the source col.
        ohe_cols_present = [c for c in LOW_CARD_OHE_COLS if c in df.columns]
        if ohe_cols_present:
            df = df.to_dummies(columns=ohe_cols_present, separator="__")
            # Rename the produced dummies so they carry our APP_OHE_ prefix.
            rename_map: dict[str, str] = {}
            for src in ohe_cols_present:
                for c in df.columns:
                    if c.startswith(f"{src}__"):
                        rename_map[c] = f"APP_OHE_{c}"
            if rename_map:
                df = df.rename(rename_map)

        return df

    # ─── Output projection ────────────────────────────────────────────────────

    def _select_outputs(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Keep:
          - SK_ID_CURR (key)
          - All raw numeric columns (LightGBM/XGBoost/CatBoost handle NaN natively).
          - All engineered APP_* columns.

        Drop:
          - Raw FLAG_DOCUMENT_* columns (we keep the aggregates).
          - Raw building-info columns (we keep the family means).
          - Raw categorical string columns (replaced by OHE/freq/count encodings).

        Note: high-cardinality strings (ORGANIZATION_TYPE, OCCUPATION_TYPE) get
        kept by *catboost.parquet* via a separate path in assemble.py; here we
        drop them so the GBM/NN matrices are pure numeric.
        """
        # Drop the raw string categoricals — they were replaced by encodings.
        drop_cols = [c for c in ALL_CAT_COLS if c in df.columns]

        # Drop FLAG_DOCUMENT_* — the aggregates above replace them.
        drop_cols += [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]

        # Drop the raw building-info columns — we kept family aggregates.
        for p in BUILDING_PREFIXES:
            for s in BUILDING_SUFFIXES:
                col = f"{p}{s}"
                if col in df.columns:
                    drop_cols.append(col)

        # Always keep SK_ID_CURR.
        drop_cols = [c for c in set(drop_cols) if c != config.ID_COL]
        out = df.drop(drop_cols)

        # Make sure SK_ID_CURR is first column (cosmetic but helpful).
        cols = [config.ID_COL] + [c for c in out.columns if c != config.ID_COL]
        return out.select(cols)
