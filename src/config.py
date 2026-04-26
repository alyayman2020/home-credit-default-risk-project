"""
Central configuration for the Home Credit Default Risk pipeline.

All paths, seeds, and tunables are loaded from environment variables (with
sensible defaults) so a single ``.env`` file at the repo root drives the whole
project.  See ``.env.example`` for the canonical template.

Reference: PLAN_v2.md §6 (memory budget) and §3 (validation strategy).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import time. Override existing env vars.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=False)


# ─── Paths ────────────────────────────────────────────────────────────────────


def _get_project_root() -> Path:
    """Return the repository root, regardless of where Python was invoked."""
    if env := os.environ.get("PROJECT_ROOT"):
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _get_project_root()

# Data location — override with DATA_DIR in .env (e.g. G:/home-credit-default-risk).
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data" / "raw")).resolve()

# Pipeline outputs
FEATURES_DIR: Path = PROJECT_ROOT / "data" / "features"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR: Path = ARTIFACTS_DIR / "predictions"
SUBMISSIONS_DIR: Path = PROJECT_ROOT / "submissions"

# Ensure write-target dirs exist (raw is read-only, no need to create).
for _d in (FEATURES_DIR, PROCESSED_DIR, ARTIFACTS_DIR, MODELS_DIR, PREDICTIONS_DIR, SUBMISSIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─── Raw file map ─────────────────────────────────────────────────────────────

RAW_FILES: dict[str, str] = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "columns_description": "HomeCredit_columns_description.csv",
    "sample_submission": "sample_submission.csv",
}


def raw_path(key: str) -> Path:
    """Resolve a raw-CSV key (e.g. ``"bureau"``) to its absolute path."""
    if key not in RAW_FILES:
        raise KeyError(f"Unknown raw file key: {key!r}. Known keys: {sorted(RAW_FILES)}")
    return DATA_DIR / RAW_FILES[key]


# ─── Run-time tunables ────────────────────────────────────────────────────────

SEED: int = int(os.environ.get("SEED", "42"))
N_FOLDS: int = int(os.environ.get("N_FOLDS", "5"))
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
USE_GPU: bool = os.environ.get("USE_GPU", "1") == "1"

# Class balance: ~8.07% positive  →  scale_pos_weight ≈ 11.4 (PLAN §1.3)
POS_RATE: float = 0.0807
SCALE_POS_WEIGHT: float = (1 - POS_RATE) / POS_RATE  # ≈ 11.39

# Target column
TARGET_COL: str = "TARGET"
ID_COL: str = "SK_ID_CURR"


# ─── Feature-matrix specs (PLAN §6.2) ─────────────────────────────────────────


@dataclass(frozen=True)
class FeatureMatrixSpec:
    """Spec for one of the three feature matrices."""

    name: str
    parquet_path: Path
    description: str
    expected_min_cols: int = 1100
    expected_max_cols: int = 1700


FEATURE_MATRICES: dict[str, FeatureMatrixSpec] = {
    "main": FeatureMatrixSpec(
        name="main",
        parquet_path=FEATURES_DIR / "main.parquet",
        description="GBM matrix (LGBM, XGB) — all encodings applied",
        expected_min_cols=1300,
        expected_max_cols=1500,
    ),
    "catboost": FeatureMatrixSpec(
        name="catboost",
        parquet_path=FEATURES_DIR / "catboost.parquet",
        description="CatBoost matrix — raw categoricals preserved",
        expected_min_cols=1100,
        expected_max_cols=1300,
    ),
    "nn": FeatureMatrixSpec(
        name="nn",
        parquet_path=FEATURES_DIR / "nn.parquet",
        description="NN matrix — RankGauss + nan-flags",
        expected_min_cols=1300,
        expected_max_cols=1500,
    ),
}


# ─── Bundle ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Config:
    """Frozen view of resolved configuration. Useful for logging."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    seed: int = field(default_factory=lambda: SEED)
    n_folds: int = field(default_factory=lambda: N_FOLDS)
    use_gpu: bool = field(default_factory=lambda: USE_GPU)


CONFIG = Config()


def summary() -> str:
    """Multiline string summarising the resolved configuration. For CLI banners."""
    lines = [
        "─── Home Credit Default Risk — Configuration ───",
        f"  PROJECT_ROOT  : {PROJECT_ROOT}",
        f"  DATA_DIR      : {DATA_DIR}  (exists={DATA_DIR.exists()})",
        f"  FEATURES_DIR  : {FEATURES_DIR}",
        f"  ARTIFACTS_DIR : {ARTIFACTS_DIR}",
        f"  SEED          : {SEED}",
        f"  N_FOLDS       : {N_FOLDS}",
        f"  USE_GPU       : {USE_GPU}",
        f"  LOG_LEVEL     : {LOG_LEVEL}",
        "────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
