"""
XGBoost model wrapper.

Reference: PLAN_v2.md §4.1 (baseline) + §3.5 (tuning search space, B2 narrowed).

Baseline config (PLAN §4.1 step 2):
    max_depth=5, learning_rate=0.02, colsample_bytree=0.3,
    tree_method='hist', device='cuda'
Acceptance: OOF AUC ≥ 0.783
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
            raise ImportError("xgboost is not installed. `make install` first.")

        params = dict(self.params)
        n_estimators = params.pop("n_estimators", 5000)

        clf = xgb.XGBClassifier(
            **params,
            n_estimators=n_estimators,
            early_stopping_rounds=200,
            enable_categorical=True,
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=100,
        )

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
