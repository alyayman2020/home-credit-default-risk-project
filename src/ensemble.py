"""
Ensemble — Level 1 rank averaging via Dirichlet grid + Nelder-Mead refinement.

Reference: PLAN_v2.md §5.1 — "A5 recipe replaces SLSQP".

Three steps
-----------
1. Sample 1000 weight vectors from ``Dirichlet(α=1)`` on N model dims.
2. Score each weighted rank-average by OOF AUC. Keep top 5.
3. From each top-5 starting point, run ``scipy.optimize.minimize(method='Nelder-Mead')``
   with a softmax reparameterization to keep weights in the simplex.

Then **D6 manual override**: log top-10 weight vectors + OOF AUC + LB AUC (when
known) to ``artifacts/ensemble_log.csv``. After 1–2 LB submissions, the user
picks the final blend (often 2nd-best OOF gives best LB).

Calibrated expectation (B4): lift +0.001 to +0.003 over best single model.
The score is won in features, not ensembling.

Plan B (stacking, only if rank-avg < 0.795): logistic regression on
4 OOF model preds + 5 strongest raw features. Implemented as a separate
function below, gated on the rank-avg result.

CLI
---
``python -m src.ensemble``                blend all available OOF predictions.
``python -m src.ensemble --models lgbm,xgb,catboost,nn_a``  explicit pick.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src import config
from src.utils import get_logger

logger = get_logger()

ENSEMBLE_LOG_PATH: Path = config.ARTIFACTS_DIR / "ensemble_log.csv"


# ─── Loading ──────────────────────────────────────────────────────────────────


def load_oof_and_test(model_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load OOF + test mean predictions for the named models.

    Returns
    -------
    oof_matrix
        shape (n_train, n_models) — column k is model k's OOF predictions.
    test_matrix
        shape (n_test, n_models) — column k is model k's mean-of-folds test predictions.
    y
        Ground-truth TARGET vector aligned with oof_matrix rows.
    """
    import polars as pl

    oof_cols, test_cols = [], []
    for name in model_names:
        oof_path = config.PREDICTIONS_DIR / name / "oof.npy"
        test_path = config.PREDICTIONS_DIR / name / "test_mean.npy"
        if not oof_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing predictions for model {name!r}.\n"
                f"  Expected: {oof_path}\n"
                f"  Expected: {test_path}\n"
                "  Run `python -m src.train --model {name}` first."
            )
        oof_cols.append(np.load(oof_path))
        test_cols.append(np.load(test_path))

    oof = np.column_stack(oof_cols)
    test = np.column_stack(test_cols)

    # Align y from the assembled main matrix
    main = pl.read_parquet(config.FEATURE_MATRICES["main"].parquet_path)
    train_part = main.filter(pl.col(config.TARGET_COL).is_not_null())
    y = train_part[config.TARGET_COL].to_numpy()
    if len(y) != oof.shape[0]:
        raise ValueError(
            f"OOF length ({oof.shape[0]}) does not match TARGET length ({len(y)}). "
            "Re-run `make features` and the train commands."
        )
    return oof, test, y


# ─── Rank average + scoring ───────────────────────────────────────────────────


def rank_normalize(matrix: np.ndarray) -> np.ndarray:
    """Column-wise rank, normalized to [0, 1]."""
    out = np.empty_like(matrix, dtype=np.float64)
    n = matrix.shape[0]
    for j in range(matrix.shape[1]):
        out[:, j] = (rankdata(matrix[:, j]) - 1) / (n - 1)
    return out


def weighted_rank_average(ranks: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Convex combination of pre-ranked column predictions."""
    return ranks @ weights


def auc_of_blend(weights: np.ndarray, ranks: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC of the weighted rank-average."""
    return float(roc_auc_score(y, weighted_rank_average(ranks, weights)))


# ─── Step 1: Dirichlet grid ───────────────────────────────────────────────────


def dirichlet_search(
    ranks: np.ndarray, y: np.ndarray, n_samples: int = 1000, seed: int = 42
) -> list[tuple[float, np.ndarray]]:
    """Sample Dirichlet(α=1), return list of (auc, weights) sorted high→low."""
    rng = np.random.default_rng(seed)
    n_models = ranks.shape[1]
    weights = rng.dirichlet(alpha=np.ones(n_models), size=n_samples)
    aucs = [auc_of_blend(w, ranks, y) for w in weights]
    pairs = sorted(zip(aucs, weights), key=lambda t: -t[0])
    logger.info(f"  Dirichlet step: top AUC = {pairs[0][0]:.5f}")
    return pairs


# ─── Step 2: Nelder-Mead refinement ───────────────────────────────────────────


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def nelder_mead_refine(
    starts: list[np.ndarray], ranks: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Refine each starting point with Nelder-Mead in softmax space.

    Returns the best (auc, weights) across all starting points.
    """
    best_auc = -np.inf
    best_weights = None
    for w0 in starts:
        # softmax-inverse: log(w + ε) — small offset for numerical stability
        z0 = np.log(np.clip(w0, 1e-9, None))

        def neg_auc(z: np.ndarray) -> float:
            return -auc_of_blend(_softmax(z), ranks, y)

        res = minimize(neg_auc, z0, method="Nelder-Mead", options={"xatol": 1e-4, "maxiter": 200})
        w = _softmax(res.x)
        auc = -res.fun
        if auc > best_auc:
            best_auc = float(auc)
            best_weights = w
    assert best_weights is not None
    logger.info(f"  Nelder-Mead step: best AUC = {best_auc:.5f}")
    return best_auc, best_weights


# ─── Step 3: D6 logging ───────────────────────────────────────────────────────


def _ensure_ensemble_log() -> None:
    if ENSEMBLE_LOG_PATH.exists():
        return
    ENSEMBLE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ENSEMBLE_LOG_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "models_csv", "weights_csv", "oof_auc", "public_lb", "private_lb"]
        )


def log_top_blends(
    model_names: list[str],
    pairs: list[tuple[float, np.ndarray]],
    *,
    top_k: int = 10,
) -> None:
    """Persist the top-K weight vectors + scores for D6 manual review."""
    _ensure_ensemble_log()
    with ENSEMBLE_LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        for auc, w in pairs[:top_k]:
            writer.writerow(
                [
                    datetime.utcnow().isoformat(timespec="seconds"),
                    ",".join(model_names),
                    ",".join(f"{x:.6f}" for x in w),
                    f"{auc:.6f}",
                    "",  # public_lb — fill in after submission
                    "",  # private_lb — fill in after final standings
                ]
            )
    logger.info(f"  Logged top-{top_k} blends to {ENSEMBLE_LOG_PATH}")


# ─── End-to-end driver ────────────────────────────────────────────────────────


def run_ensemble(model_names: list[str]) -> dict:
    """Run the full A5 recipe and emit the chosen weights + blended test pred."""
    logger.info(f"  Ensembling: {model_names}")
    oof, test, y = load_oof_and_test(model_names)

    oof_ranks = rank_normalize(oof)
    test_ranks = rank_normalize(test)

    # Step 1 — Dirichlet grid
    grid = dirichlet_search(oof_ranks, y)

    # Step 2 — Nelder-Mead refinement from top 5
    starts = [w for _, w in grid[:5]]
    nm_auc, nm_weights = nelder_mead_refine(starts, oof_ranks, y)

    # Step 3 — log for D6
    final_pairs = grid + [(nm_auc, nm_weights)]
    final_pairs.sort(key=lambda t: -t[0])
    log_top_blends(model_names, final_pairs)

    # Build final blended test predictions using the NM-refined weights
    blended_test = weighted_rank_average(test_ranks, nm_weights)

    # Save artifacts
    out = config.PREDICTIONS_DIR / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", weighted_rank_average(oof_ranks, nm_weights))
    np.save(out / "test_mean.npy", blended_test)
    np.save(out / "weights.npy", nm_weights)

    logger.success(
        f"  Final ensemble OOF AUC = {nm_auc:.5f} with weights "
        f"{dict(zip(model_names, [round(float(w), 4) for w in nm_weights]))}"
    )

    if nm_auc < 0.795:
        logger.warning(
            "  OOF < 0.795 — consider Plan B (stacking) per PLAN §5.2: "
            "LogisticRegression on 4 OOF preds + 5 strongest raw features."
        )

    return {
        "oof_auc": nm_auc,
        "weights": nm_weights.tolist(),
        "models": model_names,
    }


# ─── Plan B: Stacking (PLAN §5.2) ─────────────────────────────────────────────


def stacking_plan_b(model_names: list[str]) -> dict:
    """
    Plan B — only triggered if rank-avg OOF < 0.795 (PLAN §5.2).

    Logistic regression meta-learner on:
      - 4 OOF model predictions
      - 5 strongest raw features (highest univariate AUC)

    TODO: implement.
    """
    raise NotImplementedError(
        "Plan B stacking is not yet implemented — see PLAN §5.2 for the recipe."
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _default_models() -> list[str]:
    """Discover available models by inspecting predictions/."""
    avail = []
    for name in ("lgbm", "xgb", "catboost", "nn_a", "nn_b"):
        if (config.PREDICTIONS_DIR / name / "oof.npy").exists():
            avail.append(name)
    return avail


def _main() -> None:
    parser = argparse.ArgumentParser(description="Rank-average ensemble (A5).")
    parser.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated list (e.g. 'lgbm,xgb,catboost,nn_a'). "
            "Defaults to every model with persisted OOF predictions."
        ),
    )
    args = parser.parse_args()

    names = (
        [n.strip() for n in args.models.split(",") if n.strip()]
        if args.models
        else _default_models()
    )
    if len(names) < 2:
        raise SystemExit(
            f"Need ≥2 models with persisted predictions; found {names}. Run `make train` first."
        )
    run_ensemble(names)


if __name__ == "__main__":
    _main()
