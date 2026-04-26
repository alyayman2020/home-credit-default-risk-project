"""
Submission builder.

Writes ``submissions/submission_<YYYYMMDD_HHMMSS>_<label>_<oof>.csv`` in the
Kaggle-required format::

    SK_ID_CURR,TARGET
    100001,0.057
    100005,0.069
    ...

…and appends a row to ``submissions/log.csv`` so the user can correlate file
names with public/private LB scores after submission.

CLI
---
``python -m src.submit``                              # use ensemble blend
``python -m src.submit --source lgbm``                # single-model submission
``python -m src.submit --label tuned_3gbm_v1``        # custom label
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from src import config
from src.utils import get_logger

logger = get_logger()

LOG_PATH: Path = config.SUBMISSIONS_DIR / "log.csv"


def _ensure_log() -> None:
    if LOG_PATH.exists():
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "filename", "label", "source", "oof_auc", "public_lb", "private_lb", "notes"]
        )


def _append_log(filename: str, label: str, source: str, oof_auc: float | None) -> None:
    _ensure_log()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.utcnow().isoformat(timespec="seconds"),
                filename,
                label,
                source,
                f"{oof_auc:.6f}" if oof_auc is not None else "",
                "",  # public_lb — fill in after submission
                "",  # private_lb
                "",  # notes
            ]
        )


def _load_test_ids() -> np.ndarray:
    """Pull SK_ID_CURR for the test set from the assembled main matrix."""
    main = pl.read_parquet(config.FEATURE_MATRICES["main"].parquet_path)
    test_part = main.filter(pl.col(config.TARGET_COL).is_null())
    return test_part[config.ID_COL].to_numpy()


def _load_predictions(source: str) -> tuple[np.ndarray, float | None]:
    """Load test predictions for the given source ('ensemble' or model name)."""
    pred_path = config.PREDICTIONS_DIR / source / "test_mean.npy"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions not found at {pred_path}. "
            f"Run `make {'ensemble' if source == 'ensemble' else 'train'}` first."
        )
    preds = np.load(pred_path)

    # Try to read OOF AUC for filename labeling.
    oof_auc = None
    if source == "ensemble":
        oof_path = config.PREDICTIONS_DIR / "ensemble" / "oof.npy"
        if oof_path.exists():
            from sklearn.metrics import roc_auc_score

            oof = np.load(oof_path)
            main = pl.read_parquet(config.FEATURE_MATRICES["main"].parquet_path)
            train_part = main.filter(pl.col(config.TARGET_COL).is_not_null())
            y = train_part[config.TARGET_COL].to_numpy()
            oof_auc = float(roc_auc_score(y, oof))
    else:
        summary_path = config.PREDICTIONS_DIR / source / "summary.json"
        if summary_path.exists():
            import json

            oof_auc = float(json.loads(summary_path.read_text()).get("oof_auc", 0.0))

    return preds, oof_auc


def write_submission(
    *,
    source: str = "ensemble",
    label: str = "blend",
    out_dir: Path | None = None,
) -> Path:
    """
    Build and write a submission CSV.

    Returns
    -------
    Path
        Absolute path of the written CSV.
    """
    out_dir = out_dir or config.SUBMISSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ids = _load_test_ids()
    preds, oof_auc = _load_predictions(source)

    if len(preds) != len(test_ids):
        raise ValueError(
            f"Prediction length ({len(preds)}) != test ID count ({len(test_ids)}). "
            "Re-run the pipeline."
        )

    # Build filename: submission_<ts>_<label>_<oof>.csv
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    oof_tag = f"_oof{oof_auc:.4f}" if oof_auc else ""
    filename = f"submission_{ts}_{label}{oof_tag}.csv"
    out_path = out_dir / filename

    sub = pl.DataFrame({config.ID_COL: test_ids, config.TARGET_COL: preds})
    sub.write_csv(out_path)

    _append_log(filename, label, source, oof_auc)

    size_kb = out_path.stat().st_size / 1024
    logger.success(f"  → {out_path.name}  ({len(preds):,} rows, {size_kb:.0f} KB)")
    if oof_auc is not None:
        logger.info(f"  OOF AUC: {oof_auc:.5f}")
    logger.info(f"  Logged to {LOG_PATH}. Update public_lb / private_lb columns after submission.")

    return out_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Write a Kaggle submission CSV.")
    parser.add_argument(
        "--source",
        default="ensemble",
        help="Prediction source: 'ensemble' (default) or a single model name (lgbm/xgb/catboost/nn_a).",
    )
    parser.add_argument(
        "--label",
        default="blend",
        help="Free-form label embedded in the filename (e.g. 'tuned_3gbm_v1').",
    )
    args = parser.parse_args()
    write_submission(source=args.source, label=args.label)


if __name__ == "__main__":
    _main()
