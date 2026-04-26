# Home Credit Default Risk

> **Target:** Private LB ROC AUC ≥ 0.78 (stretch ≥ 0.80)
> **Constraint:** 16 GB RAM · RTX 3060 (8–12 GB VRAM)
> **Stack:** Python 3.11 · uv · Polars · LightGBM / XGBoost / CatBoost · PyTorch

End-to-end machine learning pipeline for the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition. Built around a **rank-averaged GBM + NN ensemble** with rigorous OOF validation, fold-aware feature engineering, and a memory budget that fits a 16 GB / RTX 3060 workstation.

Full design lives in **[`PLAN_v2.md`](PLAN_v2.md)**. Step-by-step CLI commands live in **[`RUNBOOK.md`](RUNBOOK.md)**.

> **Just want to verify the submission format?** Run `uv run python -m src.dummy_submission` after step 2 below — it writes a constant-prediction CSV to `submissions/` so you can confirm the file format matches Kaggle without running the full pipeline.

---

## Highlights

- **~1500 features** across 7 source tables, built with Polars streaming for the 27M-row `bureau_balance` and 13.6M-row `installments_payments`.
- **3 separate matrices**: GBM (encoded), CatBoost (raw categoricals), NN (RankGauss + nan-flags).
- **Fold-aware leakage controls**: OOF target encoding, fold-indexed nearest-neighbour target mean (D1), GroupKFold for per-row auxiliary models (D7).
- **Tuning**: Optuna 80/50/30/30 trials across LGBM/XGB/CatBoost/NN-A with manual override (D6).
- **Ensemble**: Dirichlet grid → Nelder-Mead refinement → manual review log (A5).

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/home-credit-default-risk.git
cd home-credit-default-risk

# CPU-only torch
make install

# GPU torch (CUDA 12.1)
make install-gpu
```

### 2. Configure data path

```bash
cp .env.example .env
# Edit .env and set DATA_DIR to where you unzipped the Kaggle CSVs.
# Example (Aly's machine): DATA_DIR=G:/home-credit-default-risk
```

The 10 raw CSVs expected:

```
application_train.csv          application_test.csv
bureau.csv                     bureau_balance.csv
previous_application.csv       installments_payments.csv
POS_CASH_balance.csv           credit_card_balance.csv
HomeCredit_columns_description.csv
sample_submission.csv
```

### 3. Run the pipeline

```bash
make data              # ~2 min   — Polars load + sentinel cleanup → parquet
make features          # 45–75 min — builds 3 feature matrices
make baseline          # ~3.5 hr  — LGBM + XGB + CatBoost + NN-A baselines
make tune              # ~6.7 hr  — Optuna overnight (LGBM 80, XGB 50, Cat 30, NN 30)
make train             # ~2 hr    — Refit tuned configs on 5 folds
make ensemble          # 15 min   — Dirichlet + Nelder-Mead weight search
make submit            # writes submissions/submission_<ts>_<label>_<oof>.csv
```

Or run the whole thing:

```bash
make all
```

---

## Repository Structure

```
home-credit-default-risk/
├── PLAN_v2.md                    # Full design doc
├── README.md                     # This file
├── pyproject.toml                # uv-managed deps
├── uv.lock                       # Pinned versions (commit this)
├── Makefile                      # All pipeline commands
├── .env.example                  # Copy → .env, set DATA_DIR
├── .gitignore
│
├── src/
│   ├── config.py                 # Paths, seeds, fold count (env-driven)
│   ├── utils.py                  # logger, set_seed, reduce_mem_usage, timers
│   ├── data.py                   # Polars loaders, sentinel replacement, parquet I/O
│   ├── cv.py                     # StratifiedKFold + GroupKFold builders + smoke
│   ├── features/
│   │   ├── base.py               # Polars-based ABC for all feature builders
│   │   ├── application.py        # ~270 features (incl. D1 neighbours, D4 interactions)
│   │   ├── bureau.py             # ~350 features (bureau + bureau_balance)
│   │   ├── previous.py           # ~250 features (previous_application)
│   │   ├── installments.py       # ~220 features (incl. B3 windowed: 1/3/6/12 mo)
│   │   ├── pos_cash.py           # ~110 features (incl. B3 windowed)
│   │   ├── credit_card.py        # ~150 features (incl. B3 windowed)
│   │   └── assemble.py           # Builds 3 matrices: main, catboost, nn
│   ├── models/
│   │   ├── base.py               # Abstract ModelBase
│   │   ├── lgbm.py               # LightGBM (GPU)
│   │   ├── xgb.py                # XGBoost (CUDA hist)
│   │   ├── catboost.py           # CatBoost (GPU, A4 dtype assertion)
│   │   └── nn.py                 # NN-A (gated residual MLP) + NN-B (smaller, ridge-selected)
│   ├── train.py                  # 5-fold OOF loop, GroupKFold variant
│   ├── tune.py                   # Optuna runners + D6 manual override logging
│   ├── ensemble.py               # Dirichlet + Nelder-Mead + manual log (A5)
│   ├── auxiliary.py              # Phase 4.5 — D7 auxiliary GBM features (gated)
│   └── submit.py                 # Build submission CSV + log row
│
├── notebooks/
│   ├── 01_eda_application.ipynb
│   ├── 02_eda_bureau_previous.ipynb
│   ├── 03_eda_monthly_tables.ipynb
│   └── 04_eda_target_leakage_check.ipynb   # PLAN §3.4 — D1 sanity + adversarial
│
├── data/
│   ├── raw/                      # Kaggle CSVs (gitignored, override via DATA_DIR)
│   ├── features/                 # main.parquet, catboost.parquet, nn.parquet
│   └── processed/                # Intermediate Polars parquet
│
├── artifacts/
│   ├── models/                   # Trained model binaries
│   ├── predictions/              # OOF + test parquets per model
│   ├── tuning_log.csv            # Optuna top-10 + D6 manual configs
│   └── ensemble_log.csv          # A5 weight vectors + scores
│
├── submissions/
│   ├── log.csv                   # ts, filename, label, oof, public_lb, private_lb
│   └── submission_<ts>_*.csv     # gitignored
│
└── tests/                        # Pytest suite (smoke + unit)
```

---

## Acceptance Gates

From PLAN_v2.md §10:

| Gate                          | Threshold                              |
| ----------------------------- | -------------------------------------- |
| Feature pipeline runs E2E     | ≥ 1300 cols, < 4 GB disk, < 12 GB RAM  |
| LGBM baseline OOF AUC         | ≥ 0.785                                |
| Tuned LGBM OOF AUC            | ≥ 0.790                                |
| 3-GBM ensemble OOF AUC        | ≥ 0.792                                |
| GBM + NN-A ensemble OOF AUC   | ≥ 0.794                                |
| Final ensemble OOF AUC        | ≥ 0.795                                |
| Final Public LB               | ≥ 0.795 (target ≥ 0.80)                |

CV-LB gap protocol: ≤ 0.005 acceptable · 0.005–0.01 investigate · > 0.01 halt and audit.

---

## References

- **ogrellier** — *LightGBM with Simple Features* (canonical aggregation pattern)
- **Will Koehrsen** — Kaggle Notebook series (per-table aggregation taxonomy)
- **Rishabh Rao** — top-7% writeup (D1 neighbours_target_mean_500)
- **17th-place solo gold writeup** — B1 GroupKFold, D7 auxiliary GBM, B2 shallow trees, B3 windows
- **DAE+NN top-4% writeup** — D2 RankGauss
- **Bojan Tunguz et al.** — 1st-place writeup (A5 manual ensemble weighting)
- **Aguiar** — stacking blueprint (Plan B fallback)

---

## Development

```bash
make format      # isort + black + ruff --fix
make lint        # ruff + black --check
make test        # pytest
make smoke       # CV + leakage smoke tests
```

---

## License

MIT — see `LICENSE` if present, or treat as MIT by default.

---

**Status:** Phase B scaffold complete. Feature builders and model wrappers are functional stubs; fill in per PLAN_v2.md sections referenced inside each file's TODO blocks.
