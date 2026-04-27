"""
CatBoost model wrapper.

Reference: PLAN_v2.md §4.1 (baseline) + §3.5 (tuning search space, B2 narrowed).

Baseline config (PLAN §4.1 step 3):
    depth=5, learning_rate=0.03, task_type='GPU', iterations=5000
Acceptance: OOF AUC ≥ 0.785

🆕 A4 dtype assertion: categorical columns must be ``category`` dtype before
fit; assert this at the entry of ``fit_fold`` to fail fast on a Polars→pandas
boundary bug (PLAN §6 / §7 risk register).

Memory note: PLAN §6.1 puts CatBoost training at ~10.5 GB RAM, the tightest
spot in the budget. If OOM, fall back to ``border_count=64`` or ``task_type='CPU'``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None  # type: ignore[assignment,misc]
    Pool = None  # type: ignore[assignment,misc]

from sklearn.metrics import roc_auc_score

from src import config
from src.models.base import FoldArtifacts, ModelBase
from src.utils import get_logger

logger = get_logger()


class CatBoostModel(ModelBase):
    name = "catboost"
    matrix = "catboost"

    def default_params(self) -> dict[str, Any]:
        """Baseline CatBoost (PLAN §4.1 step 3)."""
        return {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "depth": 5,
            "learning_rate": 0.03,
            "iterations": 5000,
            "l2_leaf_reg": 3,
            "border_count": 128,
            "task_type": "GPU" if config.USE_GPU else "CPU",
            "auto_class_weights": "Balanced",
            "random_seed": config.SEED,
            "early_stopping_rounds": 200,
            "verbose": 100,
        }

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        """Pandas 2.2+ compatible categorical-dtype check."""
        return isinstance(series.dtype, pd.CategoricalDtype)

    def _assert_cat_dtypes(self, X: pd.DataFrame, cat_features: list[str] | None) -> None:
        """🆕 A4: assert categorical cols are ``category`` dtype, fail fast otherwise."""
        if not cat_features:
            return
        wrong = [
            c for c in cat_features
            if c in X.columns and not self._is_categorical(X[c])
        ]
        if wrong:
            raise TypeError(
                f"CatBoost A4 dtype assertion failed: {len(wrong)} cat features are "
                f"not pandas 'category' dtype. First 3: {wrong[:3]}. "
                "Fix in features/assemble.py:to_catboost_matrix."
            )

    def _normalize_cats_for_catboost(
        self, X: pd.DataFrame, cat_features: list[str] | None
    ) -> pd.DataFrame:
        """
        CatBoost requires cat columns to contain strings or integers (no NaN).
        We convert ``category`` cols to plain object/str and replace NaN with the
        sentinel string ``"__missing__"``.
        """
        if not cat_features:
            return X
        out = X.copy()
        for c in cat_features:
            if c not in out.columns:
                continue
            out[c] = out[c].astype("object")
            out[c] = out[c].fillna("__missing__")
            out[c] = out[c].astype(str)
        return out

    def _fit_with_fallback(
        self,
        params: dict[str, Any],
        train_pool: "Pool",
        valid_pool: "Pool",
    ) -> "CatBoostClassifier":
        """Fit with requested task_type; on GPU error, retry on CPU."""
        try:
            clf = CatBoostClassifier(**params)
            clf.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            return clf
        except Exception as e:
            msg = str(e).lower()
            if "gpu" not in msg and "cuda" not in msg and "device" not in msg:
                raise
            logger.warning(
                f"  CatBoost GPU not available, retrying on CPU. Original error: {e}"
            )
            cpu_params = dict(params)
            cpu_params["task_type"] = "CPU"
            clf = CatBoostClassifier(**cpu_params)
            clf.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            return clf

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
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed. Run `uv sync --extra dev`.")

        # Assert category dtype (caught before normalization to verify upstream contract).
        self._assert_cat_dtypes(X_train, cat_features)
        self._assert_cat_dtypes(X_valid, cat_features)
        self._assert_cat_dtypes(X_test, cat_features)

        # Normalize cat columns to string with sentinel for NaN.
        X_train = self._normalize_cats_for_catboost(X_train, cat_features)
        X_valid = self._normalize_cats_for_catboost(X_valid, cat_features)
        X_test = self._normalize_cats_for_catboost(X_test, cat_features)

        train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
        valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_features)

        clf = self._fit_with_fallback(self.params, train_pool, valid_pool)

        valid_pred = clf.predict_proba(X_valid)[:, 1]
        test_pred = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_valid, valid_pred))

        model_path = config.MODELS_DIR / f"{self.name}_fold{fold}.cbm"
        clf.save_model(str(model_path))

        art = FoldArtifacts(
            fold=fold,
            valid_pred=valid_pred,
            test_pred=test_pred,
            valid_auc=auc,
            n_iterations=clf.tree_count_ or 0,
            extra={"model_path": str(model_path)},
        )
        self.fold_artifacts.append(art)
        self.save_fold_predictions(art)
        return art
