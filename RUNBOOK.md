# RUNBOOK — Home Credit Default Risk

End-to-end execution guide. Reference: `PLAN_v2.md` §9 (output order), §12 (time budget).

---

## 0. One-time setup (~20 min)

```powershell
# Windows PowerShell from project root
cd G:\home-credit-default-risk

# Install uv if you don't have it
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install deps (CPU torch first, then GPU torch via override)
uv sync --extra dev
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Configure data path
copy .env.example .env
# Edit .env so DATA_DIR=G:/home-credit-default-risk
# (or wherever the Kaggle CSVs actually live — your screenshot says they're already there)

# Sanity check
uv run python -m src.config
```

Expected: prints the resolved paths and confirms `DATA_DIR` exists.

---

## 1. Stage data (~5 min)

```powershell
make data
# OR equivalently:
uv run python -m src.data
```

What it does: reads each CSV with Polars, applies sentinel replacement (`DAYS_EMPLOYED == 365243` → NaN, `XNA`/`XAP` → NaN), downcasts numerics, writes parquet to `data/processed/`.

Acceptance: 8 parquet files written to `data/processed/`. Total disk ~600 MB.

---

## 2. Build features (~45–75 min) — PLAN §2

```powershell
make features
# OR:
uv run python -m src.features.assemble
```

Sequence inside the assembler:
1. `application` → in-memory dataframe.
2. `bureau` + `bureau_balance` → aggregated to `SK_ID_CURR`, written to `data/features/parts/bureau.parquet`.
3. `previous_application` → same.
4. `installments_payments` → same (with B3 windowed features).
5. `pos_cash_balance`, `credit_card_balance` → same.
6. Merge all to build the **3 final matrices**:
   - `data/features/main.parquet` (LGBM, XGB)
   - `data/features/catboost.parquet` (raw cats preserved)
   - `data/features/nn.parquet` (RankGauss + nan-flags)

Acceptance per `PLAN §10`: each matrix has ≥1100 columns, total disk ≤10 GB.

---

## 3. Build folds + leakage smoke (~5 min)

```powershell
uv run python -m src.cv --build
make smoke
```

What it does:
- Writes `data/features/folds.parquet` (StratifiedKFold) and `data/features/folds_group.parquet` (GroupKFold).
- Runs structural assertions (every row in exactly one fold; class ratio preserved; no group leakage).

Then run the leakage notebook interactively:
```powershell
uv run jupyter lab notebooks/04_eda_target_leakage_check.ipynb
```

**STOP if** adversarial AUC > 0.60 — apply rank-transform on `DAYS_*` per PLAN §7 D10.

---

## 4. Baselines (~3.5 hr active) — PLAN §4.1 + §4.3

Run sequentially, **stop and inspect after each**:

```powershell
make baseline-lgbm    # ~25 min, expect OOF AUC ≥ 0.785
make baseline-xgb     # ~30 min, expect OOF AUC ≥ 0.783
make baseline-cat     # ~50 min, expect OOF AUC ≥ 0.785
make baseline-nn      # ~40 min, expect OOF AUC ≥ 0.78
```

**Halt condition**: if LGBM baseline < 0.78, the feature pipeline has a bug. Stop, inspect feature importances + per-fold variance — DO NOT continue to tuning.

After each run, predictions are written to `artifacts/predictions/<model>/`:
- `oof.npy` — out-of-fold predictions on training data
- `test_mean.npy` — averaged test predictions across folds
- `summary.json` — `{oof_auc, fold_aucs, runtime_min}`

**First submission opportunity** (anchor CV-LB correlation):
```powershell
uv run python -m src.submit --source lgbm --label baseline_lgbm
```

---

## 5. Tuning (~6.7 hr — overnight) — PLAN §3.5

```powershell
make tune
```

Equivalent to:
```powershell
uv run python -m src.tune --model lgbm     --trials 80
uv run python -m src.tune --model xgb      --trials 50
uv run python -m src.tune --model catboost --trials 30
uv run python -m src.tune --model nn_a     --trials 30
```

Each saves to `artifacts/best_params/<model>.json` and appends every trial to `artifacts/tuning_log.csv` (D6 manual override support).

**After tuning**: review `artifacts/tuning_log.csv` and pick a final config per model. The chosen configs become the inputs to step 6.

---

## 6. Final retrain on best params (~2 hr)

```powershell
make train
```

Equivalent to running each model again in `--mode tuned`, which pulls the best params from `artifacts/best_params/`. Outputs the same `oof.npy` / `test_mean.npy` files (overwriting baseline outputs).

Acceptance per PLAN §10:
- Tuned LGBM OOF ≥ 0.790
- 3-GBM ensemble OOF ≥ 0.792
- + NN-A OOF ≥ 0.794

---

## 7. Ensemble (~15 min) — PLAN §5

```powershell
make ensemble
```

Pipeline:
1. Load all `oof.npy` and `test_mean.npy` from `artifacts/predictions/<model>/`.
2. Coarse Dirichlet grid (1000 weight vectors).
3. Multi-start Nelder-Mead from top 5 grid points.
4. Write final ensemble OOF + test predictions to `artifacts/predictions/ensemble/`.
5. Append top 10 weight candidates to `artifacts/ensemble_log.csv` for D6 manual override.

---

## 8. Submission (~30 sec)

```powershell
make submit
# OR with a custom label:
uv run python -m src.submit --label final_ensemble_v1
```

Writes `submissions/submission_<YYYYMMDD_HHMMSS>_<label>_oof<auc>.csv` in the exact Kaggle format:
```
SK_ID_CURR,TARGET
100001,0.057
100005,0.069
...
```

Appends a row to `submissions/log.csv`. Update `public_lb` and `private_lb` columns by hand after each Kaggle submission.

**Submission cadence** (PLAN §11):
1. Tuned LGBM single — anchors CV-LB correlation.
2. 3-GBM rank-average ensemble.
3. + NN-A.
4. Final pick from `artifacts/ensemble_log.csv` after seeing 1–2 LB scores.

---

## 9. Optional Phase 4.5 — D7 auxiliary GBM (gated on OOF ≥ 0.792)

```powershell
uv run python -m src.auxiliary --build
```

Then re-run `make train` (it will pick up the 8 new auxiliary features automatically) → `make ensemble` → `make submit`.

Skip entirely if step 7 ensemble OOF < 0.792 — the +0.001–0.003 lift isn't worth the risk.

---

## Wall-clock summary (RTX 3060 + 16 GB)

| Stage | Duration | Mode |
|---|---:|---|
| Setup + data | 25 min | Active |
| Features | 45–75 min | Active |
| Folds + smoke | 5 min | Active |
| Baselines (4 models) | ~3.5 hr | Active |
| Tuning | ~6.7 hr | Overnight |
| Tuned retrain | ~2 hr | Active |
| Ensemble + submit | 15 min | Active |
| **Total core** | **~13 hr** | 1 evening + 1 overnight + 1 morning |
| Optional 4.5 + 4.6 | +3.5 hr | Active |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError` on raw CSV | `DATA_DIR` not set or wrong | Edit `.env`, point to actual location |
| LGBM OOF < 0.78 on baseline | Feature pipeline bug | Inspect `artifacts/predictions/lgbm/summary.json`, check fold variance |
| CatBoost OOM (16 GB hit) | `border_count=254` too high | Override to 64 in `src/models/catboost.py` |
| GPU unavailable error | CUDA torch not installed | Re-run `uv pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| OOF AUC and Public LB diverge >0.01 | Leakage | Re-run notebook 04, audit top-importance features |
| Polars `streaming` errors | Polars version mismatch | Upgrade to ≥1.12 (`uv pip install -U polars`) |

---

## Sanity-check helper (no training, no Kaggle data needed)

If you just want to verify the submission path/format works **before** running the full pipeline:

```powershell
uv run python -m src.dummy_submission
```

This generates a constant-prediction submission (`TARGET = 0.0807` for every test row, the population positive rate). **It will score ~0.50 AUC on Kaggle** — purely a smoke test for the I/O path. Real predictions come from steps 1–8.
