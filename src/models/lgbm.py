"""
LightGBM model wrapper.

Reference: PLAN_v2.md §4.1 (baseline) + §3.5 (tuning search space, B2 narrowed).

Baseline config (PLAN §4.1 step 1):
    num_leaves=48, lr=0.02, feature_fraction=0.3, bagging_fraction=0.8,
    n_estimators=5000, early_stopping=200
Acceptance: OOF AUC ≥ 0.785

Halt condition: if baseline LGBM < 0.78, the feature pipeline has a bug.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]

from sklearn.metrics import roc_auc_score

from src import config
from src.models.base import FoldArtifacts, ModelBase


class LGBMModel(ModelBase):
    name = "lgbm"
    matrix = "main"

    def default_params(self) -> dict[str, Any]:
        """Baseline LGBM (PLAN §4.1 step 1)."""
        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 48,
            "learning_rate": 0.02,
            "feature_fraction": 0.3,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 100,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "max_depth": -1,
            "n_estimators": 5000,
            "verbose": -1,
            "device_type": "gpu" if config.USE_GPU else "cpu",
            "scale_pos_weight": config.SCALE_POS_WEIGHT,
            "seed": config.SEED,
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
        if lgb is None:
            raise ImportError(
                "lightgbm is not installed. `make install` or `make install-gpu` first."
            )

        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=cat_features or "auto",
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_set,
            categorical_feature=cat_features or "auto",
            free_raw_data=False,
        )

        booster = lgb.train(
            self.params,
            train_set,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        valid_pred = booster.predict(X_valid, num_iteration=booster.best_iteration)
        test_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
        auc = float(roc_auc_score(y_valid, valid_pred))

        # Persist booster
        model_path = config.MODELS_DIR / f"{self.name}_fold{fold}.txt"
        booster.save_model(str(model_path), num_iteration=booster.best_iteration)

        art = FoldArtifacts(
            fold=fold,
            valid_pred=np.asarray(valid_pred),
            test_pred=np.asarray(test_pred),
            valid_auc=auc,
            n_iterations=booster.best_iteration or 0,
            extra={"model_path": str(model_path)},
        )
        self.fold_artifacts.append(art)
        self.save_fold_predictions(art)
        return art
