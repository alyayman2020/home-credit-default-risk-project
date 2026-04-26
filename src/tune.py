"""
Hyperparameter tuning via Optuna.

Reference: PLAN_v2.md §3.5 (search spaces, B2 narrowed) + D6 (manual override).

Trial budgets per model (totals ~6.7 hrs on RTX 3060):

| Model     | Trials | Single-fold time | Total |
|-----------|-------:|-----------------:|------:|
| LightGBM  |     80 |          ~90 sec | ~2.0h |
| XGBoost   |     50 |         ~120 sec | ~1.7h |
| CatBoost  |     30 |         ~180 sec | ~1.5h |
| NN-A      |     30 |         ~180 sec | ~1.5h |

All search spaces have been narrowed toward shallow trees (B2):
- LightGBM: feature_fraction (0.2, 0.5), num_leaves (16, 64).
- XGBoost: colsample_bytree (0.2, 0.5), max_depth (4, 7).
- CatBoost: depth (4, 7).

🆕 D6: every trial's config + per-fold AUC + OOF AUC is appended to
``artifacts/tuning_log.csv``. After the sweep, the user reviews the top-10
manually and runs a few hand-picked configs around Optuna's best.

CLI
---
``python -m src.tune --model lgbm --trials 80``
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna

from src import config
from src.utils import get_logger

logger = get_logger()

TUNING_LOG_PATH: Path = config.ARTIFACTS_DIR / "tuning_log.csv"


# ─── Search spaces (PLAN §3.5, B2 narrowed) ───────────────────────────────────


def _suggest_lgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.5),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "max_depth": -1,
        "n_estimators": 5000,
        "device_type": "gpu" if config.USE_GPU else "cpu",
        "scale_pos_weight": config.SCALE_POS_WEIGHT,
        "seed": config.SEED,
        "verbose": -1,
    }


def _suggest_xgb(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda" if config.USE_GPU else "cpu",
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.5),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "n_estimators": 5000,
        "scale_pos_weight": config.SCALE_POS_WEIGHT,
        "seed": config.SEED,
    }


def _suggest_catboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "depth": trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "task_type": "GPU" if config.USE_GPU else "CPU",
        "iterations": 5000,
        "early_stopping_rounds": 200,
        "auto_class_weights": "Balanced",
        "random_seed": config.SEED,
        "verbose": False,
    }


def _suggest_nn_a(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "block_dims": trial.suggest_categorical(
            "block_dims",
            [(512, 256, 128), (768, 384, 192), (512, 256, 128)],
        ),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "lr_max": trial.suggest_float("lr_max", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "epochs": 30,
        "early_stopping_patience": 10,
        "seed": config.SEED,
    }


SUGGESTERS: dict[str, Any] = {
    "lgbm": _suggest_lgbm,
    "xgb": _suggest_xgb,
    "catboost": _suggest_catboost,
    "nn_a": _suggest_nn_a,
}


# ─── D6 logging ───────────────────────────────────────────────────────────────


def _ensure_log() -> None:
    """Create tuning_log.csv with header if missing."""
    if TUNING_LOG_PATH.exists():
        return
    TUNING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TUNING_LOG_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "model",
                "trial",
                "oof_auc",
                "fold_aucs_csv",
                "params_json",
            ]
        )


def _append_log_row(
    model_name: str,
    trial_number: int,
    oof_auc: float,
    fold_aucs: list[float],
    params: dict[str, Any],
) -> None:
    """Append one trial's record to tuning_log.csv (D6)."""
    import json

    _ensure_log()
    with TUNING_LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.utcnow().isoformat(timespec="seconds"),
                model_name,
                trial_number,
                f"{oof_auc:.6f}",
                ",".join(f"{a:.6f}" for a in fold_aucs),
                json.dumps(params, default=str),
            ]
        )


# ─── Objective ────────────────────────────────────────────────────────────────


def _build_objective(model_name: str):
    suggester = SUGGESTERS[model_name]

    def _objective(trial: optuna.Trial) -> float:
        params = suggester(trial)

        # Late import to avoid circularity at module load.
        from src.train import get_model, run_oof

        model = get_model(model_name, params=params)
        summary = run_oof(model)
        _append_log_row(
            model_name=model_name,
            trial_number=trial.number,
            oof_auc=summary["oof_auc"],
            fold_aucs=summary["fold_aucs"],
            params=params,
        )
        return summary["oof_auc"]

    return _objective


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuner.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(SUGGESTERS),
        help="Model to tune.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Trial count (defaults: lgbm=80, xgb=50, catboost=30, nn_a=30).",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL (e.g. sqlite:///optuna.db) for resume.",
    )
    args = parser.parse_args()

    default_trials = {"lgbm": 80, "xgb": 50, "catboost": 30, "nn_a": 30}
    n_trials = args.trials or default_trials[args.model]

    study = optuna.create_study(
        study_name=f"hcdr_{args.model}",
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(_build_objective(args.model), n_trials=n_trials, show_progress_bar=True)

    logger.success(
        f"Tuning complete: best AUC={study.best_value:.5f}  best_params={study.best_params}"
    )
    logger.info(
        f"  Top-10 trials logged to {TUNING_LOG_PATH} — review per D6 before "
        "promoting to `--mode tuned`."
    )


if __name__ == "__main__":
    _main()
