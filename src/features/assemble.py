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
    sample_size: int = 50_000,
    seed: int = config.SEED,
) -> pl.DataFrame:
    """
    Drop one of any numeric pair with |corr| > ``threshold``.

    Algorithm (PLAN §2.9):
      1. Take a stratified sample of training rows (test rows excluded — TARGET = null).
      2. Compute pairwise correlation matrix on numeric columns (numpy, fp32).
      3. For each pair above threshold, score both via univariate AUC against TARGET.
      4. Drop the lower-AUC member; keep the higher.

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

    # Sample to control memory.
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

    # Materialise to numpy in fp32 with NaN preserved.
    arr = sample.select(numeric_cols).to_numpy().astype(np.float32)

    # Univariate AUC per column (used as tiebreaker). Skip columns with all-null sample.
    aucs: dict[str, float] = {}
    for j, col in enumerate(numeric_cols):
        v = arr[:, j]
        mask = ~np.isnan(v)
        if mask.sum() < 100 or y[mask].std() == 0:
            aucs[col] = 0.5
            continue
        try:
            a = roc_auc_score(y[mask], v[mask])
            aucs[col] = max(a, 1 - a)  # symmetric — sign agnostic
        except Exception:
            aucs[col] = 0.5

    # Pearson correlation. NaN-aware: pandas is the path of least resistance.
    import pandas as pd
    corr = pd.DataFrame(arr, columns=numeric_cols).corr(numeric_only=True).abs()
    # Upper-triangular mask
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    corr_vals = corr.where(upper)

    drops: set[str] = set()
    # Iterate column-by-column; greedy drop lower-AUC member.
    for col in numeric_cols:
        if col in drops:
            continue
        # Find columns highly correlated with `col` that we haven't dropped yet.
        partners = corr_vals[col][corr_vals[col] > threshold].index.tolist()
        for partner in partners:
            if partner in drops:
                continue
            # Drop whichever has lower univariate AUC.
            if aucs.get(partner, 0.5) <= aucs.get(col, 0.5):
                drops.add(partner)
            else:
                drops.add(col)
                break  # `col` is gone; no point comparing it further

    if drops:
        logger.info(
            f"  prune_high_correlation: dropped {len(drops)} columns "
            f"(|corr|>{threshold}, sample n={n_sample})"
        )
    return df.drop(list(drops))


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

    The application builder has already produced OHE / frequency / count
    encodings for the 16 application categoricals (PLAN §2.1). All other
    builders aggregate to numeric. So `full` is already encoded — this
    transform's job is just to drop any residual string columns that slipped
    through and ensure NaN is preserved (LGBM/XGB handle natively).
    """
    string_cols = [
        c for c in df.columns
        if c not in (config.ID_COL, config.TARGET_COL) and df[c].dtype == pl.Utf8
    ]
    if string_cols:
        logger.info(f"  to_main_matrix: dropping {len(string_cols)} residual string columns")
        df = df.drop(string_cols)
    return df


def to_catboost_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """
    CatBoost matrix.

    The application builder dropped the raw 16 string cats in favour of OHE/freq/count
    encodings. CatBoost is happiest with the *raw* cats (it does its own ordered
    target statistics internally — better than our OHE in many cases). So we:

      1. Drop the OHE columns (``APP_OHE_*``) that duplicate raw cats.
      2. Re-attach the 16 raw string cats from ``application_train`` + ``application_test``.
      3. Cast them to ``pl.Categorical`` for downstream pandas-side ``category`` dtype
         (needed by the A4 dtype assertion in src/models/catboost.py).
      4. Emit a sidecar ``data/features/cat_features.json`` listing the cat columns,
         which ``src/train.py`` reads when ``model.matrix == "catboost"``.

    Net column delta: drops ~80 OHE cols, adds 16 raw cats. Catboost matrix is
    ~70 columns lighter than main.
    """
    import json
    from src.features.application import ALL_CAT_COLS

    # Step 1: drop OHE columns produced by the application builder.
    ohe_cols = [c for c in df.columns if c.startswith("APP_OHE_")]
    if ohe_cols:
        df = df.drop(ohe_cols)
        logger.info(f"  to_catboost_matrix: dropped {len(ohe_cols)} OHE columns")

    # Step 2: re-attach raw cat columns from application_train + test.
    train = read_processed("application_train").select([config.ID_COL] + ALL_CAT_COLS)
    test = read_processed("application_test").select([config.ID_COL] + ALL_CAT_COLS)
    cats = pl.concat([train, test], how="vertical_relaxed")
    df = df.join(cats, on=config.ID_COL, how="left")

    # Step 3: cast to Categorical (Polars). Pandas conversion preserves this as 'category'.
    cat_cols_present = [c for c in ALL_CAT_COLS if c in df.columns]
    df = df.with_columns([pl.col(c).cast(pl.Categorical) for c in cat_cols_present])

    # Step 4: sidecar listing.
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
      2. Label-encode the 16 application categoricals to small integers (so the NN
         can use them via embedding tables). Re-attaches raw cats first, then encodes.
      3. Drop the original string cats — keep only the integer-encoded versions.

    NOT done here:
      - 🆕 D2 RankGauss: fit per-fold inside ``src/train.py``. Doing it here would leak
        TARGET-correlated quantile boundaries from valid into train.
    """
    from src.features.application import ALL_CAT_COLS

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

    # ── Step 2 + 3: re-attach + label-encode application cats ────────────────
    # Drop OHE encodings — the NN gets per-cat embeddings instead.
    ohe_cols = [c for c in df.columns if c.startswith("APP_OHE_")]
    if ohe_cols:
        df = df.drop(ohe_cols)

    train = read_processed("application_train").select([config.ID_COL] + ALL_CAT_COLS)
    test = read_processed("application_test").select([config.ID_COL] + ALL_CAT_COLS)
    cats = pl.concat([train, test], how="vertical_relaxed")
    df = df.join(cats, on=config.ID_COL, how="left")

    # Label-encode each cat to a small Int32. Polars: cast to Categorical → to_physical()
    # gives the underlying integer code. Nulls become a sentinel −1 below.
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
