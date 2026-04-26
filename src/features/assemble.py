"""
Feature matrix assembly.

Joins all builder outputs onto ``application_train + application_test``,
then writes three parquet files per PLAN_v2.md §6.2:

- ``data/features/main.parquet``     — GBM matrix (LGBM, XGB), ~3.5 GB.
- ``data/features/catboost.parquet`` — raw categoricals preserved as ``category``, ~3.0 GB.
- ``data/features/nn.parquet``       — RankGauss-scaled + nan-flags, ~3.5 GB.

Plus mandatory pruning (PLAN §2.9):

1. Drop columns with >99% missingness.
2. Drop near-zero variance (variance < 1e-6 after scaling).
3. Correlation prune: |corr| > 0.98 → keep one with higher univariate AUC.
4. Auto-trigger null-importance pruning if column count > 1600.

CLI
---
``python -m src.features.assemble``  build all 3 matrices.
"""

from __future__ import annotations

import argparse

import polars as pl

from src import config
from src.data import read_processed
from src.features.application import ApplicationFeatures
from src.features.bureau import BureauFeatures
from src.features.credit_card import CreditCardFeatures
from src.features.installments import InstallmentsFeatures
from src.features.pos_cash import PosCashFeatures
from src.features.previous import PreviousApplicationFeatures
from src.utils import get_logger, timer

logger = get_logger()


# ─── Assembly ─────────────────────────────────────────────────────────────────


def build_base_frame() -> pl.DataFrame:
    """
    Concatenate ``application_train`` + ``application_test`` into a single
    base frame keyed by ``SK_ID_CURR``. TARGET is preserved (null for test).
    """
    train = read_processed("application_train")
    test = read_processed("application_test").with_columns(
        pl.lit(None).cast(pl.Int8).alias(config.TARGET_COL)
    )
    base = pl.concat([train, test], how="vertical_relaxed")
    logger.info(f"  base frame: {base.height} rows × {base.width} cols")
    return base


def join_all_features(base: pl.DataFrame) -> pl.DataFrame:
    """Run every FeatureBuilder and left-join onto the base frame."""
    builders = [
        ApplicationFeatures(),
        BureauFeatures(),
        PreviousApplicationFeatures(),
        InstallmentsFeatures(),
        PosCashFeatures(),
        CreditCardFeatures(),
    ]
    df = base
    for b in builders:
        feats = b.run()
        # ApplicationFeatures returns a re-derived application frame; replace
        # rather than join when the builder owns the application columns.
        if b.name == "application":
            # Drop overlapping cols on the base side (keep app-builder versions).
            overlap = [c for c in feats.columns if c in base.columns and c != config.ID_COL]
            df = df.drop(overlap).join(feats, on=config.ID_COL, how="left")
        else:
            df = df.join(feats, on=config.ID_COL, how="left")
        logger.info(f"  after join {b.name}: {df.width} cols")
    return df


# ─── Pruning (PLAN §2.9) ──────────────────────────────────────────────────────


def prune_high_missing(df: pl.DataFrame, threshold: float = 0.99) -> pl.DataFrame:
    """Drop columns with > ``threshold`` fraction missing."""
    n = df.height
    null_counts = df.null_count().row(0)
    drops = [c for c, k in zip(df.columns, null_counts) if k / n > threshold and c != config.ID_COL]
    if drops:
        logger.info(f"  prune_high_missing: dropped {len(drops)} columns (>{threshold:.0%} null)")
    return df.drop(drops)


def prune_near_zero_variance(df: pl.DataFrame, threshold: float = 1e-6) -> pl.DataFrame:
    """Drop numeric columns with variance < ``threshold``."""
    drops: list[str] = []
    for col in df.columns:
        if col in (config.ID_COL, config.TARGET_COL):
            continue
        if not df[col].dtype.is_numeric():
            continue
        var = df[col].drop_nulls().var()
        if var is not None and var < threshold:
            drops.append(col)
    if drops:
        logger.info(f"  prune_near_zero_variance: dropped {len(drops)} columns (var < {threshold})")
    return df.drop(drops)


def prune_high_correlation(df: pl.DataFrame, threshold: float = 0.98) -> pl.DataFrame:
    """
    Drop one of any pair with |corr| > ``threshold``.

    TODO: implement properly. Pseudocode:
      1. Compute correlation matrix on a sample (e.g. 50K rows for speed).
      2. For each pair above threshold, score both via univariate AUC against TARGET.
      3. Keep the higher-AUC member.

    For now, this is a no-op — wire when feature count exceeds the budget.
    """
    logger.warning(
        f"  prune_high_correlation: NOT IMPLEMENTED — TODO per PLAN §2.9 (threshold={threshold})"
    )
    return df


def maybe_null_importance_prune(df: pl.DataFrame, max_cols: int = 1600) -> pl.DataFrame:
    """
    Auto-trigger null-importance pruning when feature count exceeds ``max_cols``.

    PLAN §2.9 — train LGBM on shuffled target 5×, drop features whose real
    importance < 75th percentile of null importance.

    TODO: implement (uses LightGBM, depends on src/models/lgbm.py).
    """
    if df.width <= max_cols:
        logger.info(f"  null-importance prune: skipped (cols={df.width} ≤ {max_cols})")
        return df
    logger.warning(
        f"  null-importance prune: TRIGGERED (cols={df.width}) — TODO per PLAN §2.9"
    )
    return df


# ─── Matrix-specific transforms ───────────────────────────────────────────────


def to_main_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """
    GBM matrix (LGBM, XGB): all categoricals encoded (frequency / OHE),
    NaN preserved (LGBM/XGB handle natively).

    TODO: implement encodings — currently passes through.
    """
    return df


def to_catboost_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """
    CatBoost matrix: raw categoricals preserved as Polars ``Categorical`` so
    CatBoost handles them via internal encoding.

    TODO: cast string columns to ``pl.Categorical`` and emit a list of cat-col
    names alongside the parquet for the trainer to consume (PLAN §A4 dtype assertion).
    """
    return df


def to_nn_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """
    NN matrix:
      - 🆕 D2: RankGauss for all numerics (fit on train fold *inside* CV loop —
        do NOT precompute globally, that would leak).
      - 🆕 A3: ``{col}_is_nan`` flags for every numeric column with > 1% missingness.
      - Categoricals: label-encoded → embedding lookup at NN level.

    Critical: this assembler emits the *raw* features + nan-flags. RankGauss
    is fit per fold inside ``src/train.py``, never here.

    TODO: implement nan-flag generation and label encoding for cats.
    """
    return df


# ─── Orchestration ────────────────────────────────────────────────────────────


def assemble_all() -> None:
    """Build all three feature matrices end-to-end."""
    logger.info(config.summary())

    with timer("build base frame"):
        base = build_base_frame()

    with timer("join all builders"):
        full = join_all_features(base)

    with timer("prune"):
        full = prune_high_missing(full)
        full = prune_near_zero_variance(full)
        full = prune_high_correlation(full)
        full = maybe_null_importance_prune(full)

    logger.info(f"  post-prune shape: {full.shape}")

    # Emit three matrices.
    for spec_name, transform in [
        ("main", to_main_matrix),
        ("catboost", to_catboost_matrix),
        ("nn", to_nn_matrix),
    ]:
        spec = config.FEATURE_MATRICES[spec_name]
        with timer(f"assemble {spec_name}"):
            mat = transform(full)
            mat.write_parquet(spec.parquet_path, compression="snappy")
        size_mb = spec.parquet_path.stat().st_size / 1024**2
        logger.success(
            f"  → {spec.parquet_path.name}: {mat.shape}, {size_mb:.1f} MB"
        )

        # Soft check against expected column band (PLAN §6.2).
        if not (spec.expected_min_cols <= mat.width <= spec.expected_max_cols):
            logger.warning(
                f"    column count {mat.width} outside expected band "
                f"[{spec.expected_min_cols}, {spec.expected_max_cols}] for {spec_name}"
            )

    logger.success("All feature matrices assembled.")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Assemble feature matrices.")
    parser.add_argument(
        "--matrix",
        choices=["all", "main", "catboost", "nn"],
        default="all",
        help="Which matrix to build (default: all).",
    )
    args = parser.parse_args()

    if args.matrix != "all":
        logger.warning(
            "Single-matrix mode not yet implemented; falling back to --matrix all."
        )
    assemble_all()


if __name__ == "__main__":
    _main()
