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
from src.features.application import (
    ALL_CAT_COLS,
    HIGH_CARD_COLS,
    ApplicationFeatures,
)
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
    # Reorder test columns to match train's schema exactly.
    # `with_columns` appends TARGET to the end, but train has it after SK_ID_CURR.
    # `vertical_relaxed` allows dtype mismatches but NOT column-order mismatches.
    test = test.select(train.columns)
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


def prune_high_correlation(
    df: pl.DataFrame,
    threshold: float = 0.98,
    *,
    sample_size: int = 20_000,
    seed: int = config.SEED,
) -> pl.DataFrame:
    """
    Drop one of any numeric pair with |corr| > ``threshold``.

    🔥 Fast implementation (numpy + chunked ``corrcoef``):
      - Median-impute on the sample (NaN-tolerant correlation is slow in pandas).
      - Mean-center + scale to unit variance, then ``X.T @ X / n`` is the
        correlation matrix in one fused fp32 BLAS call (≈ 1–2 sec on 1500 × 20K).
      - Greedy upper-triangle scan, drop the lower-AUC member of each pair.

    Compared to the previous pandas-based implementation: ≈ 50–100× faster on
    Windows (no Python-loop NaN handling, single GEMM call instead of column-pair
    iteration). Wall time on the full Home Credit dataset: ~30 sec.

    Skipped when:
      - training rows < 1000 (sample isn't representative)
      - feature count < 200 (overhead exceeds benefit)
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    train_part = df.filter(pl.col(config.TARGET_COL).is_not_null())
    if train_part.height < 1000 or df.width < 200:
        logger.info(
            f"  prune_high_correlation: skipped (train_rows={train_part.height}, "
            f"cols={df.width})"
        )
        return df

    # Sample to control memory / wall time.
    n_sample = min(sample_size, train_part.height)
    sample = train_part.sample(n_sample, seed=seed)
    y = sample[config.TARGET_COL].to_numpy().astype(np.float32)

    # Numeric columns only (excluding ID + TARGET).
    numeric_cols = [
        c for c in sample.columns
        if c not in (config.ID_COL, config.TARGET_COL) and sample[c].dtype.is_numeric()
    ]
    if len(numeric_cols) < 2:
        logger.info("  prune_high_correlation: <2 numeric columns, skipping")
        return df

    # ── Pull to fp32 numpy in one shot ───────────────────────────────────────
    arr = sample.select(numeric_cols).to_numpy().astype(np.float32, copy=False)

    # ── Univariate AUC per column (used as tiebreaker) ───────────────────────
    aucs = np.full(len(numeric_cols), 0.5, dtype=np.float32)
    for j in range(arr.shape[1]):
        v = arr[:, j]
        mask = ~np.isnan(v)
        if mask.sum() < 100:
            continue
        if y[mask].std() == 0:
            continue
        try:
            a = roc_auc_score(y[mask], v[mask])
            aucs[j] = max(a, 1 - a)  # symmetric — sign agnostic
        except Exception:
            pass

    # ── Median-impute, then standardize, then correlation = X.T @ X / n ──────
    # NaN handling: replace with column median (computed ignoring NaN).
    medians = np.nanmedian(arr, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        # Vectorised fill — broadcast medians into the NaN positions.
        idx = np.where(nan_mask)
        arr[idx] = medians[idx[1]]

    # Mean / std per column.
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    # Avoid divide-by-zero for constants (variance pruning should have caught these).
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    arr -= mu
    arr /= sigma

    # Correlation matrix via single GEMM call.
    n = arr.shape[0]
    corr = (arr.T @ arr) / n  # shape (P, P)
    np.abs(corr, out=corr)

    # ── Greedy upper-triangle scan ───────────────────────────────────────────
    p = corr.shape[0]
    drops: set[int] = set()
    # Iterate by columns; for each, find any partner with |corr| > threshold
    # that we haven't already dropped, and drop the lower-AUC one.
    for i in range(p):
        if i in drops:
            continue
        # corr[i, j] for j > i — upper triangle.
        partners = np.where(corr[i, i + 1:] > threshold)[0] + (i + 1)
        for j in partners:
            if j in drops:
                continue
            if aucs[j] <= aucs[i]:
                drops.add(int(j))
            else:
                drops.add(int(i))
                break  # i is gone, no point continuing its row

    if drops:
        drop_names = [numeric_cols[k] for k in drops]
        logger.info(
            f"  prune_high_correlation: dropped {len(drop_names)} columns "
            f"(|corr|>{threshold}, sample n={n_sample})"
        )
        return df.drop(drop_names)
    logger.info(f"  prune_high_correlation: no pairs above {threshold} (sample n={n_sample})")
    return df


def maybe_null_importance_prune(
    df: pl.DataFrame,
    max_cols: int = 1600,
    *,
    n_runs: int = 5,
    sample_size: int = 100_000,
    pct_threshold: float = 75.0,
    seed: int = config.SEED,
) -> pl.DataFrame:
    """
    Auto-trigger null-importance pruning when feature count exceeds ``max_cols``.

    Algorithm (PLAN §2.9):
      1. Train one LGBM run on real (TARGET, X) → ``actual_imp`` per feature.
      2. Train ``n_runs`` LGBMs on shuffled-TARGET → ``null_imp`` distribution.
      3. For each feature, compute the ``pct_threshold``-percentile of its null
         importance. Drop features whose actual importance < that percentile.

    Skipped when ``df.width <= max_cols``.
    """
    if df.width <= max_cols:
        logger.info(f"  null-importance prune: skipped (cols={df.width} ≤ {max_cols})")
        return df

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("  null-importance prune: lightgbm not installed, skipping")
        return df

    import numpy as np

    train_part = df.filter(pl.col(config.TARGET_COL).is_not_null())
    n_sample = min(sample_size, train_part.height)
    sample = train_part.sample(n_sample, seed=seed)

    feat_cols = [
        c for c in sample.columns
        if c not in (config.ID_COL, config.TARGET_COL) and sample[c].dtype.is_numeric()
    ]
    X = sample.select(feat_cols).to_pandas()
    y = sample[config.TARGET_COL].to_numpy()

    base_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": seed,
    }

    def _run(y_target: np.ndarray) -> np.ndarray:
        booster = lgb.train(
            base_params,
            lgb.Dataset(X, label=y_target),
            num_boost_round=200,
        )
        return booster.feature_importance(importance_type="gain")

    logger.info(
        f"  null-importance prune: TRIGGERED (cols={df.width}) — "
        f"running {n_runs + 1} LGBMs on n={n_sample} sample"
    )
    actual = _run(y)

    rng = np.random.default_rng(seed)
    null_runs = np.zeros((n_runs, len(feat_cols)), dtype=np.float64)
    for i in range(n_runs):
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        null_runs[i] = _run(y_shuf)

    null_pct = np.percentile(null_runs, pct_threshold, axis=0)
    drops = [feat_cols[j] for j in range(len(feat_cols)) if actual[j] <= null_pct[j]]

    if drops:
        logger.info(
            f"  null-importance prune: dropped {len(drops)} features (real_imp ≤ "
            f"{int(pct_threshold)}th-pct of null_imp over {n_runs} shuffled runs)"
        )
    return df.drop(drops)


# ─── Matrix-specific transforms ───────────────────────────────────────────────


def to_main_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """
    GBM matrix (LGBM, XGB).

    Drops:
      - Any residual string columns (LGBM/XGB don't accept strings on the GBM matrix).
      - High-card cats ORGANIZATION_TYPE / OCCUPATION_TYPE if they leaked through
        from the base frame (the application builder doesn't always drop them).
        Their freq + count encodings already exist (APP_CAT_FREQ_*, APP_CAT_COUNT_*).
    """
    drops: list[str] = []

    # 1. Drop high-cardinality raw cats — main matrix is pure numeric. Their
    #    freq/count encodings (APP_CAT_FREQ_*, APP_CAT_COUNT_*) carry the signal.
    drops += [c for c in HIGH_CARD_COLS if c in df.columns]

    # 2. Drop any other residual string column.
    for c in df.columns:
        if c in (config.ID_COL, config.TARGET_COL):
            continue
        if df[c].dtype == pl.Utf8 and c not in drops:
            drops.append(c)

    if drops:
        logger.info(f"  to_main_matrix: dropping {len(drops)} string/raw-cat columns")
        df = df.drop(drops)
    return df


def to_catboost_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """
    CatBoost matrix.

    Strategy:
      1. Drop the OHE columns produced by the application builder (CatBoost has
         its own ordered target stats — better than our OHE for many cats).
      2. Drop already-present cat string columns from ``df`` to avoid duplicate
         columns on join.
      3. Re-attach the 16 raw string cats from ``application_train`` + ``application_test``.
      4. Cast to ``pl.Categorical`` so the pandas conversion lands as ``category`` dtype
         (required by the A4 dtype assertion in src/models/catboost.py).
      5. Emit ``data/features/cat_features.json`` listing the cat columns —
         ``src/train.py`` reads this when ``model.matrix == "catboost"``.
    """
    import json

    # Step 1: drop OHE columns produced by the application builder.
    ohe_cols = [c for c in df.columns if c.startswith("APP_OHE_")]
    if ohe_cols:
        df = df.drop(ohe_cols)
        logger.info(f"  to_catboost_matrix: dropped {len(ohe_cols)} OHE columns")

    # Step 2: drop any raw cat columns currently in df — we'll re-attach clean copies.
    existing_cats = [c for c in ALL_CAT_COLS if c in df.columns]
    if existing_cats:
        df = df.drop(existing_cats)

    # Step 3: re-attach raw cat columns from application_train + test.
    train = read_processed("application_train").select([config.ID_COL] + ALL_CAT_COLS)
    test = read_processed("application_test").select([config.ID_COL] + ALL_CAT_COLS)
    cats = pl.concat([train, test], how="vertical_relaxed")
    df = df.join(cats, on=config.ID_COL, how="left")

    # Step 4: cast to Categorical (Polars). Pandas conversion preserves this as 'category'.
    cat_cols_present = [c for c in ALL_CAT_COLS if c in df.columns]
    df = df.with_columns([pl.col(c).cast(pl.Categorical) for c in cat_cols_present])

    # Step 5: sidecar listing.
    sidecar_path = config.FEATURES_DIR / "cat_features.json"
    sidecar_path.write_text(json.dumps(cat_cols_present, indent=2))
    logger.info(
        f"  to_catboost_matrix: re-attached {len(cat_cols_present)} raw cats; "
        f"wrote sidecar {sidecar_path.name}"
    )
    return df


def to_nn_matrix(df: pl.DataFrame, *, nan_threshold: float = 0.01) -> pl.DataFrame:
    """
    NN matrix.

    Steps:
      1. 🆕 A3 nan-flags: for every numeric column with > ``nan_threshold`` (default 1%)
         missingness, add a ``{col}__isnan`` column (Int8).
      2. Drop OHE columns — the NN consumes label-encoded cats via embedding tables.
      3. Drop high-card raw cats from ``df`` (they leaked through), then re-attach
         clean copies from application_train+test and label-encode each to Int32.

    NOT done here:
      - 🆕 D2 RankGauss: fit per-fold inside ``src/train.py``. Doing it here would leak
        TARGET-correlated quantile boundaries from valid into train.

    Naming convention: nan-flags use the suffix ``__isnan`` (double underscore).
    Train.py's RankGauss step skips columns with this suffix.
    """
    # ── Step 1: nan-flags ────────────────────────────────────────────────────
    n = df.height
    null_counts = df.null_count().row(0)
    flag_cols: list[str] = []
    for col, k in zip(df.columns, null_counts):
        if col in (config.ID_COL, config.TARGET_COL):
            continue
        if not df[col].dtype.is_numeric():
            continue
        if k / n > nan_threshold:
            flag_cols.append(col)

    if flag_cols:
        df = df.with_columns(
            [pl.col(c).is_null().cast(pl.Int8).alias(f"{c}__isnan") for c in flag_cols]
        )
        logger.info(f"  to_nn_matrix: added {len(flag_cols)} __isnan flag columns")

    # ── Step 2: drop OHE — embeddings replace them ───────────────────────────
    ohe_cols = [c for c in df.columns if c.startswith("APP_OHE_")]
    if ohe_cols:
        df = df.drop(ohe_cols)

    # ── Step 3: drop existing raw cats, re-attach + label-encode ─────────────
    existing_cats = [c for c in ALL_CAT_COLS if c in df.columns]
    if existing_cats:
        df = df.drop(existing_cats)

    train = read_processed("application_train").select([config.ID_COL] + ALL_CAT_COLS)
    test = read_processed("application_test").select([config.ID_COL] + ALL_CAT_COLS)
    cats = pl.concat([train, test], how="vertical_relaxed")
    df = df.join(cats, on=config.ID_COL, how="left")

    nn_cat_cols: list[str] = []
    for col in ALL_CAT_COLS:
        if col not in df.columns:
            continue
        encoded = (
            pl.col(col)
            .cast(pl.Categorical)
            .to_physical()
            .cast(pl.Int32)
            .fill_null(-1)
            .alias(f"{col}__nn")
        )
        df = df.with_columns(encoded)
        nn_cat_cols.append(f"{col}__nn")

    # Drop the raw string cats — keep the integer-encoded versions only.
    df = df.drop([c for c in ALL_CAT_COLS if c in df.columns])

    logger.info(
        f"  to_nn_matrix: label-encoded {len(nn_cat_cols)} cats → integer columns"
    )
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
        with timer("  prune_high_correlation"):
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
