"""
LightGBM model wrapper.

Reference: PLAN_v2.md §4.1 (baseline) + §3.5 (tuning search space, B2 narrowed).

Baseline config (PLAN §4.1 step 1):
    num_leaves=48, lr=0.02, feature_fraction=0.3, bagging_fraction=0.8,
    n_estimators=5000, early_stopping=200
Acceptance: OOF AUC ≥ 0.785

Halt condition: if baseline LGBM < 0.78, the feature pipeline has a bug.

Notes on GPU
------------
The default ``pip``/``uv`` wheel of LightGBM on Windows is CPU-only. If
``USE_GPU=1`` but the GPU build isn't available, ``fit_fold`` will catch the
"GPU Tree Learner was not enabled" error and automatically retry on CPU,
logging a warning.
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
from src.utils import get_logger

logger = get_logger()


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

    def _train_with_fallback(
        self,
        params: dict[str, Any],
        train_set: "lgb.Dataset",
        valid_set: "lgb.Dataset",
    ) -> "lgb.Booster":
        """
        Train with the requested device. If the GPU build isn't available
        (common on Windows pip wheels), automatically retry on CPU.
        """
        try:
            return lgb.train(
                params,
                train_set,
                valid_sets=[valid_set],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(200, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            )
        except lgb.basic.LightGBMError as e:
            msg = str(e)
            if "GPU" in msg or "OpenCL" in msg:
                logger.warning(
                    "  LightGBM GPU build not available (pip wheel is CPU-only on "
                    "Windows). Retrying on CPU. To use GPU, install LightGBM from "
                    "source with --gpu (see https://lightgbm.readthedocs.io)."
                )
                cpu_params = dict(params)
                cpu_params["device_type"] = "cpu"
                return lgb.train(
                    cpu_params,
                    train_set,
                    valid_sets=[valid_set],
                    valid_names=["valid"],
                    callbacks=[
                        lgb.early_stopping(200, verbose=False),
                        lgb.log_evaluation(period=100),
                    ],
                )
            raise

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
                "lightgbm is not installed. Run `uv sync --extra dev`."
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

        booster = self._train_with_fallback(self.params, train_set, valid_set)

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
