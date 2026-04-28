"""
XGBoost model wrapper.

Reference: PLAN_v2.md §4.1 (baseline) + §3.5 (tuning search space, B2 narrowed).

Baseline config (PLAN §4.1 step 2):
    max_depth=5, learning_rate=0.02, colsample_bytree=0.3,
    tree_method='hist', device='cuda'
Acceptance: OOF AUC ≥ 0.783

Notes
-----
1. **Inf sanitization.** XGBoost rejects ``inf`` / ``-inf`` values by default
   ("Input data contains `inf` or a value too large"). LightGBM tolerates them
   silently. Some of our engineered features (CREDIT_TERM = AMT_ANNUITY/AMT_CREDIT,
   PAYMENT_PERC = AMT_PAYMENT/AMT_INSTALMENT, etc.) can produce ``inf`` when the
   denominator is zero. We replace ``inf``/``-inf`` with ``NaN`` before training
   so XGBoost handles them via its ``missing`` mechanism (PLAN §1.4 sentinel
   policy — same treatment as DAYS_EMPLOYED 365243).

2. **GPU fallback.** If CUDA xgboost isn't available, ``fit_fold`` automatically
   falls back to ``device='cpu'`` and logs a warning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

from sklearn.metrics import roc_auc_score

from src import config
from src.models.base import FoldArtifacts, ModelBase
from src.utils import get_logger

logger = get_logger()


def _sanitize_inf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace +/-inf with NaN in numeric columns. XGBoost treats NaN as the
    "missing" sentinel by default; inf values cause a hard error.

    Returns a copy. Object/category dtype columns are left alone.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return out
    # Use np.where via mask — fast, in-place on the slice.
    mask = np.isinf(out[numeric_cols].to_numpy())
    if mask.any():
        n_inf = int(mask.sum())
        logger.info(f"  xgb: replaced {n_inf} inf values with NaN before training")
        # Apply replacement column-by-column to preserve dtypes.
        for col in numeric_cols:
            arr = out[col].to_numpy()
            inf_mask = np.isinf(arr)
            if inf_mask.any():
                arr = arr.astype(np.float64, copy=True)
                arr[inf_mask] = np.nan
                out[col] = arr
    return out


class XGBModel(ModelBase):
    name = "xgb"
    matrix = "main"

    def default_params(self) -> dict[str, Any]:
        """Baseline XGBoost (PLAN §4.1 step 2)."""
        return {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": "cuda" if config.USE_GPU else "cpu",
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_child_weight": 16,
            "n_estimators": 5000,
            "scale_pos_weight": config.SCALE_POS_WEIGHT,
            "seed": config.SEED,
            "verbosity": 1,
        }

    def _fit_with_fallback(
        self,
        clf: "xgb.XGBClassifier",
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
    ) -> "xgb.XGBClassifier":
        """Fit with the requested device; fall back to CPU if CUDA unavailable."""
        try:
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=100,
            )
            return clf
        except (xgb.core.XGBoostError, ValueError) as e:
            msg = str(e).lower()
            if "cuda" not in msg and "gpu" not in msg and "device" not in msg:
                raise
            logger.warning(
                "  XGBoost CUDA device not available — retrying on CPU. "
                f"Original error: {e}"
            )
            params = clf.get_params()
            params["device"] = "cpu"
            params["tree_method"] = "hist"
            cpu_clf = xgb.XGBClassifier(**params)
            cpu_clf.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=100,
            )
            return cpu_clf

    def fit_fold(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        X_test: pd.DataFrame,
        *,
        fold: int,
        cat_features: list[str] | None = None,
    ) -> FoldArtifacts:
        if xgb is None:
            raise ImportError("xgboost is not installed. Run `uv sync --extra dev`.")

        # Sanitize inf -> NaN. Done once on the largest frame, then we apply to all 3.
        # Logging happens inside _sanitize_inf, only once per call.
        X_train = _sanitize_inf(X_train)
        X_valid = _sanitize_inf(X_valid)
        X_test = _sanitize_inf(X_test)

        params = dict(self.params)
        n_estimators = params.pop("n_estimators", 5000)

        clf = xgb.XGBClassifier(
            **params,
            n_estimators=n_estimators,
            early_stopping_rounds=200,
            enable_categorical=True,
        )

        clf = self._fit_with_fallback(clf, X_train, y_train, X_valid, y_valid)

        valid_pred = clf.predict_proba(X_valid)[:, 1]
        test_pred = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_valid, valid_pred))

        model_path = config.MODELS_DIR / f"{self.name}_fold{fold}.json"
        clf.save_model(str(model_path))

        art = FoldArtifacts(
            fold=fold,
            valid_pred=valid_pred,
            test_pred=test_pred,
            valid_auc=auc,
            n_iterations=int(clf.best_iteration or 0),
            extra={"model_path": str(model_path)},
        )
        self.fold_artifacts.append(art)
        self.save_fold_predictions(art)
        return art
