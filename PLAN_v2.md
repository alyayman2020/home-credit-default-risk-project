# PLAN.md (v2) — Home Credit Default Risk

**Target:** Private LB ROC AUC ≥ 0.78 (stretch ≥ 0.80) | **Constraint:** 16 GB RAM, RTX 3060 (8–12 GB VRAM) | **Stack:** Python 3.11, uv, Polars (features), pandas (boundary), LightGBM/XGBoost/CatBoost/PyTorch

**v2 changelog vs v1:** +D1 neighbors feature, +D2 RankGauss, +A1 Polars in features/, +A3 nan-flags for NN, +A5 Dirichlet+NM ensemble, +B1 GroupKFold for aux models, +B2 shallower tree search, +B3 expanded time windows, +D7/D8 optional auxiliary phases. Dropped TabNet mention, SLSQP, vague kernel references.

---

## 1. Dataset Analysis (7 tables) — unchanged from v1

### 1.1 Topology

```
application_{train,test}  ──┐
   SK_ID_CURR (PK)          │
                            │
bureau ─────────────────────┤   SK_ID_CURR
   SK_ID_BUREAU (PK)        │
   ↑                        │
bureau_balance              │   SK_ID_BUREAU
                            │
previous_application ───────┤   SK_ID_CURR
   SK_ID_PREV (PK)          │
   ↑                        │
   ├── installments_payments │   SK_ID_PREV → SK_ID_CURR
   ├── POS_CASH_balance      │   SK_ID_PREV → SK_ID_CURR
   └── credit_card_balance   │   SK_ID_PREV → SK_ID_CURR
```

### 1.2 Table sizes & memory profile

| Table | Rows | Raw size | Polars-lazy peak | Notes |
|---|---:|---:|---:|---|
| application_train | 307K | 168 MB | <50 MB | Target here |
| application_test | 49K | 26 MB | <10 MB | |
| bureau | 1.7M | 170 MB | ~70 MB | |
| bureau_balance | 27.3M | 376 MB | **~80 MB streamed** | Polars streaming kills the v1 ceiling |
| previous_application | 1.7M | 404 MB | ~180 MB | |
| installments_payments | 13.6M | 723 MB | **~150 MB streamed** | Polars wins big here |
| POS_CASH_balance | 10M | 392 MB | ~100 MB | |
| credit_card_balance | 3.8M | 425 MB | ~150 MB | |

### 1.3 Class balance
~8.07% positive. `scale_pos_weight ≈ 11.4` for LGBM/XGB, `auto_class_weights='Balanced'` for CatBoost.

### 1.4 Data quirks (handled at load time in `data.py`)
- `DAYS_EMPLOYED == 365243` → NaN.
- `XNA`/`XAP` → NaN in `CODE_GENDER`, `ORGANIZATION_TYPE`, `NAME_CONTRACT_STATUS`.
- `AMT_INCOME_TOTAL` outliers (max 117M) → log1p transform OR winsorize at 99.5th percentile.
- `OWN_CAR_AGE`, `EXT_SOURCE_*` missingness is informative — `_is_nan` flags built (NN matrix only, A3).
- Building info `*_AVG/_MEDI/_MODE` ~50% missing — aggregate to family means.

---

## 2. Feature Engineering Catalog (~1500–1650 features GBM, ~300 features NN)

### 2.1 `application` table (~270 features, +20 from v1)

**Domain ratios (~25)** — unchanged from v1.
- Income ratios (7), time ratios (7), other ratios (~10). `CREDIT_TERM = AMT_ANNUITY/AMT_CREDIT` confirmed as a top-importance feature (D3).

**EXT_SOURCE engineering (~20) — unchanged**.

**🆕 D1: Neighbors target mean ⭐ MUST-ADD (+1 feature, expected +0.002–0.004 AUC)**
```
TARGET_NEIGHBORS_500_MEAN
  Feature: per-row, mean TARGET of 500 nearest neighbors in 4D space:
    [EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, CREDIT_ANNUITY_RATIO]
  LEAKAGE STATUS: OOF-ONLY — fold-aware NearestNeighbors index per fold.
  Implementation:
    For each fold k in 1..5:
      idx_k = NearestNeighbors(n=500).fit(train[fold != k][features])
      for row in train[fold == k]:
        neighbors = idx_k.kneighbors(row)
        TARGET_NEIGHBORS_500_MEAN[row] = mean(train[neighbors].TARGET)
    For test: build full-train index once, query test rows.
  Asserted: 5 distinct fold-indexed values per same neighbor candidate (smoke test).
```

**🆕 D4: Divisive EXT × DAYS_BIRTH interactions (+8 features)**
```
APP_CREDIT_TO_ANNUITY_RATIO_DIV_SCORE1_TO_BIRTH_RATIO
APP_CREDIT_TO_ANNUITY_RATIO_DIV_DAYS_BIRTH
APP_EXT_MEAN_DIV_DAYS_EMPLOYED
APP_EXT_MEAN_MUL_AGE_YEARS
APP_INCOME_PER_PERSON_DIV_EXT_MEAN
APP_CREDIT_DIV_EXT_PROD
APP_ANNUITY_DIV_EXT_MAX
APP_DAYS_EMPLOYED_DIV_EXT_MIN
```

**Document & flag aggregates (~5)**, **Building aggregates (~12)**, **Categorical encoding (~180)** — unchanged from v1.

### 2.2 `bureau` + `bureau_balance` (~350 features) — unchanged from v1

**Step 1**: Polars lazy aggregation of `bureau_balance` per `SK_ID_BUREAU` (~15 features). Streaming engine for the 27M rows.

**Step 2**: merge into `bureau`, aggregate per `SK_ID_CURR` (~335 features) — same stratification by `CREDIT_ACTIVE`, `CREDIT_TYPE` as v1.

### 2.3 `previous_application` (~250 features) — unchanged from v1

### 2.4 `installments_payments` (~220 features, +40 from v1 via B3)

**Per-row engineering** — unchanged: DPD, DBD, PAYMENT_PERC, PAYMENT_DIFF, LATE_FLAG.

**🆕 B3: Expanded time windows (+40 features)**
- Windows: **last 1, 3, 6, 12 months** (dropped 24 — too redundant with lifetime).
- Per window: top aggregations on `DPD`, `LATE_FLAG`, `PAYMENT_PERC` only (not all features — controls feature count).
- 4 windows × 10 metrics = ~40 windowed features.

### 2.5 `POS_CASH_balance` (~110 features, +30 from v1)

- Base aggregations (~50) — unchanged.
- **🆕 B3 windowed (last 3, 6, 12 months)**: SK_DPD, SK_DPD_DEF, status share → ~30 features.
- Recency-weighted DPD → ~10 features.
- Categorical encoding → ~10 features.

### 2.6 `credit_card_balance` (~150 features, +30 from v1)

- Base aggregations (~80) — unchanged: utilization, drawing patterns, overlimit.
- **🆕 B3 windowed (last 3, 6, 12 months)** on utilization, balance, drawings → ~30 features.
- DPD aggregations + recency → ~10 features.
- Categorical encoding → ~10 features.

### 2.7 Cross-table interactions (~30) — unchanged from v1

### 2.8 NN-only feature additions (NOT added to GBM matrix)

**🆕 A3: `_is_nan` flags for NN matrix (+~70 features, NN matrix only)**
- For every numeric column with >1% missingness, add `{col}_is_nan` ∈ {0, 1}.
- Reasoning: GBMs handle NaN natively (LGBM/XGB use missing-direction in splits); NNs cannot.
- Polluting GBM matrix with these wastes splits.

**🆕 D2: RankGauss-scaled features for NN matrix (replaces all numerics in NN matrix)**
- `sklearn.preprocessing.QuantileTransformer(output_distribution='normal', n_quantiles=10000, random_state=42)`.
- Fit on train fold only, transform train fold + valid fold inside CV loop.
- Replaces StandardScaler entirely. Better tail handling for `AMT_INCOME_TOTAL`-class outliers.

### 2.9 Feature selection

**Mandatory pruning (post-engineering, before training)**:
1. Drop columns with >99% missingness.
2. Drop near-zero variance (variance < 1e-6 after scaling).
3. Correlation prune: |corr| > 0.98 → keep one with higher univariate AUC.

**🆕 Aggressive pruning trigger**: If feature count > 1600 after step 3, run **null-importance method**:
- Train LGBM on shuffled target 5×, record feature importances.
- Drop features whose real importance < 75th percentile of null importance.
- Expected: prunes ~150–250 marginal features.

**🆕 D8: Ridge forward selection (NN-B path only, optional)**
- For NN-B variant (§4.3): forward selection w/ Ridge regression to pick top ~300 features.
- Skipped for NN-A and all GBMs.

**Estimated final counts**:
- GBM matrix (LGBM, XGB): **1300–1500 features**.
- CatBoost matrix (raw cats preserved): **1100–1300 features**.
- NN-A matrix (with nan-flags + RankGauss): **~1400 features**.
- NN-B matrix (post Ridge forward selection): **~300 features**.

### 2.10 Anti-leakage rules

| Encoding | Leakage status | Implementation |
|---|---|---|
| Frequency / count | Safe | Computed on full train+test |
| OHE | Safe | Stateless |
| Target encoding | OOF-ONLY | Inside CV loop, smoothed |
| Aggregations from child tables | Safe | No `TARGET` involved |
| **🆕 D1 neighbors feature** | **OOF-ONLY** | Fold-aware NN index, asserted |
| **🆕 D7 auxiliary GBM features** | **OOF + GroupKFold** | See §4.5 |

---

## 3. Validation Strategy

### 3.1 Splitter
- **Main level**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `TARGET`.
- **🆕 B1: Auxiliary level** (D7 monthly-record models): `GroupKFold(n_splits=5)` grouped by `SK_ID_CURR`. Prevents same-borrower leakage across folds when training on monthly rows.
- Persist both fold mappings to `data/features/folds.parquet` (`SK_ID_CURR, fold_main`) and `data/features/folds_group.parquet` (`SK_ID_PREV, fold_group`).

### 3.2 Metrics
- Primary: ROC AUC on concatenated OOF.
- Secondary: per-fold mean ± std. Flag if std > 0.005.
- Sanity: OOF AUC ≈ 5-fold mean within 0.001.

### 3.3 CV-LB correlation protocol
- Acceptable gap: |OOF − Public LB| ≤ 0.005.
- Investigate: 0.005 < |gap| ≤ 0.01.
- Halt and audit: |gap| > 0.01.

### 3.4 Leakage smoke tests (`notebooks/04_eda_target_leakage_check.ipynb`)
1. Univariate AUC scan — flag any single feature > 0.65.
2. Adversarial validation — train LGBM to distinguish train vs test.
3. Fold-encoding sanity — assert OOF target encodings differ across folds.
4. **🆕 D1 neighbor sanity** — assert that for the same row, the neighbor mean differs across the 5 fold indices.

### 3.5 Hyperparameter tuning (Optuna, 6–8 hr overnight budget)

| Model | Trials | Single-fold time | Total budget |
|---|---:|---:|---:|
| LightGBM | 80 | ~90 sec | ~2.0 hrs |
| XGBoost | 50 | ~120 sec | ~1.7 hrs |
| CatBoost | 30 | ~180 sec | ~1.5 hrs |
| NN-A | 30 | ~180 sec | ~1.5 hrs |
| **Total** | 190 | — | **~6.7 hrs** |

**🆕 B2: Search spaces narrowed toward shallow trees** (winner-confirmed regime):

```python
# LightGBM (NARROWED v2)
{
  "num_leaves": (16, 64),                    # was (16, 256)
  "learning_rate": (0.005, 0.05, log=True),
  "feature_fraction": (0.2, 0.5),            # was (0.5, 1.0) — KEY FIX
  "bagging_fraction": (0.5, 1.0),
  "bagging_freq": (1, 10),
  "min_child_samples": (20, 200),
  "reg_alpha": (1e-3, 10, log=True),
  "reg_lambda": (1e-3, 10, log=True),
  "max_depth": -1,
  "boosting_type": "gbdt",
  "objective": "binary",
  "metric": "auc",
  "n_estimators": 5000,
  "early_stopping_rounds": 200,
  "device_type": "gpu",   # RTX 3060
}

# XGBoost (NARROWED v2)
{
  "max_depth": (4, 7),                       # was (4, 10)
  "learning_rate": (0.005, 0.05, log=True),
  "subsample": (0.5, 1.0),
  "colsample_bytree": (0.2, 0.5),            # was (0.5, 1.0) — KEY FIX
  "gamma": (0, 5),
  "reg_alpha": (1e-3, 10, log=True),
  "reg_lambda": (1e-3, 10, log=True),
  "min_child_weight": (1, 100),
  "tree_method": "hist",
  "device": "cuda",
  "n_estimators": 5000,
  "early_stopping_rounds": 200,
}

# CatBoost (NARROWED v2)
{
  "depth": (4, 7),                           # was (4, 10)
  "learning_rate": (0.005, 0.05, log=True),
  "l2_leaf_reg": (1, 10),
  "bagging_temperature": (0, 5),
  "random_strength": (0, 10),
  "border_count": (32, 255),
  "task_type": "GPU",
  "iterations": 5000,
  "early_stopping_rounds": 200,
  "auto_class_weights": "Balanced",
}

# NN-A (MLP + entity embeddings + skip connections)
{
  "block_dims": [(512, 256, 128), (768, 384, 192), (512, 256, 128, 64)],
  "dropout": (0.2, 0.5),
  "lr_max": (1e-4, 5e-3, log=True),
  "weight_decay": (1e-6, 1e-3, log=True),
  "batch_size": [256, 512, 1024],
  "epochs": 30,
  "early_stopping_patience": 10,
}
```

**🆕 D6: Manual override step after Optuna**:
- Log all top-10 trials' configs + per-fold AUC + OOF AUC to `artifacts/tuning_log.csv`.
- After tuning, run 3 hand-picked configs around Optuna's best (e.g., ±20% on `num_leaves`, `feature_fraction`).
- Final config selected by user, not by `study.best_params` blindly.

---

## 4. Modeling Roadmap

### Phase 4.1 — Baselines (sequential, stop-and-report)

| Step | Model | Config | Acceptance |
|---|---|---|---|
| 1 | LightGBM | `num_leaves=48, lr=0.02, feature_fraction=0.3, bagging_fraction=0.8, n_est=5000, ES=200` | OOF AUC ≥ 0.785 |
| 2 | XGBoost | `max_depth=5, lr=0.02, colsample_bytree=0.3, tree_method=hist, device=cuda` | OOF AUC ≥ 0.783 |
| 3 | CatBoost | `depth=5, lr=0.03, task_type=GPU, iter=5000` | OOF AUC ≥ 0.785 |

**Halt condition**: if step 1 < 0.78, feature pipeline has a bug. Stop, debug.

### Phase 4.2 — Tuned GBMs (Optuna per §3.5 + D6 manual override)
**Acceptance**: best of three reaches OOF AUC ≥ 0.790.

### Phase 4.3 — Neural Networks

**🆕 A2: NN-A architecture (gated residual blocks)**

```
Inputs:
  ├── Numerics (~1100, RankGauss-scaled)
  ├── _is_nan flags (~70, raw 0/1)
  └── Categoricals (label-encoded → embeddings, dim = min(50, n_unique//2))

Block 1: Linear(input_dim → 512) + BN + GELU + Dropout
Block 2: Linear(512 → 256) + BN + GELU + Dropout
         + Skip projection (Linear(512 → 256))
Block 3: Linear(256 → 128) + BN + GELU + Dropout
         + Skip projection (Linear(256 → 128))
Output:  Linear(128 → 1)  # logit
```

**Training**:
- Loss: `BCEWithLogitsLoss(pos_weight=tensor(11.4))`.
- Optimizer: `AdamW`, weight_decay tuned by Optuna.
- Scheduler: `OneCycleLR(max_lr=tuned)`.
- Mixed precision: mandatory (`torch.cuda.amp`).
- Early stopping: patience=10 on fold AUC.
- Imputation: median for numerics (fit on train fold), `"missing"` token for categoricals.

**Expected OOF AUC**: 0.78–0.79.

**🆕 D5/D8: NN-B variant (Phase 4.6, optional, runs only if NN-A ≥ 0.78)**
- **Architecture**: smaller MLP `256 → 128 → 64`, dropout 0.5, no skip connections.
- **Features**: top ~300 selected via Ridge forward selection (`sklearn.feature_selection.SequentialFeatureSelector` with `Ridge(alpha=1.0)`).
- **Purpose**: feature-subset diversity for the ensemble. Different model + different features = different errors.
- **Skipped if**: NN-A underperforms (NN-B won't save it) or time budget tight.
- **Adds**: ~30 min training + ~60 min Ridge selection.

### Phase 4.4 — Class imbalance handling — unchanged from v1

### 🆕 Phase 4.5 — Auxiliary GBM features (D7, optional, gated on OOF ≥ 0.792)

**Skip if main pipeline OOF AUC < 0.792**. Only valuable when chasing 0.795+.

**Recipe**:
1. Build per-row dataset on `installments_payments` (13.6M rows): merge `TARGET` from application via `SK_ID_CURR`.
2. Train LightGBM with `GroupKFold(5)` grouped by `SK_ID_CURR` (B1 — prevents same-borrower leakage).
3. Same model trained on `previous_application` rows (1.7M rows, GroupKFold by `SK_ID_CURR`).
4. Aggregate row-level OOF predictions back to `SK_ID_CURR` level: `mean`, `max`, `min`, `last_3_months_mean`.
5. **Resulting features**: 4 from installments + 4 from previous = **8 new features**.
6. Add to GBM matrix only (not NN). Refit final GBMs.

**Expected lift**: +0.001 to +0.003 OOF AUC.
**Cost**: ~3 hours (1 hr per aux model build + tune, 1 hr final retrain).

**Smoke test**: assert no `SK_ID_CURR` appears in both train and validation within any fold of the aux model.

---

## 5. Ensembling

### 5.1 Level 1 — Rank averaging (PRIMARY) — 🆕 A5 recipe replaces SLSQP

**Step 1: Coarse Dirichlet grid search**
- Sample 1000 weight vectors from `Dirichlet(α=1)` on N model dimensions (N=3 if no NN-B, 4 if NN-B included).
- For each weight vector: compute weighted rank-average of OOF predictions, score with AUC.
- Keep top 5 weight vectors.

**Step 2: Multi-start Nelder-Mead refinement**
- From each of the top 5 grid points: `scipy.optimize.minimize(method='Nelder-Mead')` with constraint `weights ∈ [0,1], sum=1` (via softmax reparameterization).
- Take best across all 5 starts.

**Step 3: Manual override layer (D6 echo)**
- Log top 10 weight vectors + their OOF AUC + LB AUC (when available) to `artifacts/ensemble_log.csv`.
- After 1–2 LB submissions, user picks final blend (often 2nd-best OOF gives best LB).

**🆕 B4: Calibrated expectation**: lift +0.001 to +0.003 over best single model (NOT +0.003 to +0.008 as in v1). The score is won in features, not ensembling.

### 5.2 Level 2 — Stacking (Plan B, only if rank-avg < 0.795)
- Meta: `LogisticRegression(C=0.1, penalty='l2')`.
- Inputs: 4 OOF model preds + 5 strongest raw features.
- Strict OOF stacking.

---

## 6. Memory Budget (revised for v2 with Polars)

### 6.1 Per-step RAM ceiling

| Step | Peak RAM (v2) | vs v1 | Strategy |
|---|---:|---:|---|
| Load + downcast (Polars lazy) | 0.1 GB | -50% | `pl.scan_parquet` |
| Application features | 0.4 GB | unchanged | |
| **bureau_balance aggregation** | **0.3 GB** | **-25%** | Polars streaming on 27M rows (A1 win) |
| Bureau merge + aggregate | 0.9 GB | -10% | |
| previous_application | 1.3 GB | -15% | |
| **installments_payments** | **0.9 GB** | **-25%** | Polars streaming on 13.6M rows (A1 win) |
| POS / credit_card | 0.9 GB each | unchanged | |
| **🆕 D1 neighbors feature** | **+0.1 GB** | new | Per-fold NN index ~30MB each |
| Final assembly merge | 3.5 GB | unchanged | |
| **Train LGBM** (GBM matrix) | **8.5 GB** | +0.5 GB | More features |
| **Train XGB** | **9.5 GB** | +0.5 GB | |
| **Train CatBoost** | **10.5 GB** | +0.5 GB | Tightest. Watch carefully. |
| NN training | 4 GB RAM + 6 GB VRAM | unchanged | |

**Hard ceiling**: 14 GB RAM. CatBoost is at 10.5 GB — **at limit but feasible**. Mitigation if it OOMs: drop `border_count` to 64 or move to CPU.

### 6.2 Three feature matrices on disk

- `data/features/main.parquet` — GBM matrix (LGBM, XGB), all encodings, ~3.5 GB.
- `data/features/catboost.parquet` — raw categoricals preserved as `category`, ~3.0 GB.
- `data/features/nn.parquet` — RankGauss-scaled + nan-flags, ~3.5 GB.

Loaded one at a time. Total disk: ~10 GB.

---

## 7. Risk Register (v2 updates)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Target leakage via global target encoding | Medium | Catastrophic | OOF inside CV loop, asserted |
| **🆕 D1 neighbor feature leaks across folds** | Medium | Catastrophic | Fold-aware index, smoke test asserts different values per fold for same row |
| **🆕 D7 aux GBM leaks via SK_ID_CURR** | High if forgotten | -0.005 LB | GroupKFold mandatory, asserted |
| OOM on CatBoost training | Medium → Higher | Run halts | Fall back to `border_count=64` or CPU |
| **🆕 Polars-pandas boundary bugs** | Medium | Subtle correctness errors | Schema assertion at every `.to_pandas()` call |
| **🆕 D10 elevated**: `DAYS_*` adversarial drift | High | -0.005 to -0.01 LB | Adversarial AUC > 0.6 on `DAYS_*` family → **rank-transform** (don't drop) |
| `EXT_SOURCE` missingness not handled before ratio | Medium | NaN propagation | Fillna with median *after* nullity flag |
| `DAYS_EMPLOYED == 365243` not replaced | High if forgotten | -0.005 OOF AUC | Single point of replacement in `data.py` |
| Optuna search overfits to fold 0 | Medium | Top configs underperform 5-fold | D6 manual override + top-3 5-fold validation |
| NN-A underperforms 0.78 | Medium | Ensemble weaker | NN-B and stacking are non-blocking fallbacks |
| Feature count exceeds 1650 | Medium | OOM on CatBoost | Null-importance pruning auto-triggers at 1600 |

---

## 8. Reference Synthesis (v2)

Drawing on:
1. **Olivier "ogrellier"** — *LightGBM with Simple Features* — canonical aggregation pattern.
2. **Will Koehrsen** series — systematic per-table aggregation taxonomy.
3. **🆕 Rishabh Rao top-7% writeup** — D1 (neighbors_target_mean_500) source.
4. **🆕 17th-place solo gold writeup** — B1 (GroupKFold for monthly), D7 (auxiliary GBM features), B2 (shallow trees), B3 (expanded windows).
5. **🆕 DAE+NN top-4% writeup** — D2 (RankGauss) source. We adopt RankGauss without the full DAE (memory constraint).
6. **Bojan Tunguz et al. 1st-place writeup** — A5 (manual ensemble weighting beats optimizers).
7. **Aguiar** — stacking blueprint (Plan B).

---

## 9. Phase B Output Order (preview)

1. `pyproject.toml` (Polars, LightGBM-GPU, XGBoost-GPU, CatBoost-GPU, PyTorch-CUDA pinned), `Makefile`, `.gitignore`, `README.md`.
2. `src/config.py`, `src/utils.py` (logger, seed, `reduce_mem_usage`, timers), `src/data.py` (Polars loaders, sentinel replacement, parquet I/O), `src/cv.py` (StratifiedKFold + GroupKFold builders).
3. `src/features/base.py` (Polars-based ABC), then `application.py` (incl. D1 neighbors, D4 interactions), `bureau.py`, `previous.py`, `installments.py` (incl. B3 windows), `pos_cash.py`, `credit_card.py`, `assemble.py` (builds 3 matrices).
4. `src/models/base.py`, `lgbm.py`, `xgb.py`, `catboost.py` (with A4 dtype assertion), `nn.py` (NN-A + NN-B classes).
5. `src/train.py` (OOF loop, GroupKFold variant), `src/tune.py` (Optuna runners + D6 logging).
6. `src/ensemble.py` (Dirichlet + Nelder-Mead + manual log), `src/submit.py`.
7. `src/auxiliary.py` (Phase 4.5 D7 auxiliary GBMs, gated).
8. EDA notebook scaffolds (01–04, 04 includes D1 sanity + adversarial validation).
9. End-to-end runbook with exact CLI commands.

---

## 10. Acceptance Gates (recap)

| Gate | Threshold | If failed |
|---|---|---|
| Feature pipeline runs end-to-end | parquet ≥ 1300 cols, < 4 GB disk, < 12 GB RAM | Profile, prune, rebuild |
| LGBM baseline OOF AUC | ≥ 0.785 | Halt — feature bug suspected |
| Tuned LGBM OOF AUC | ≥ 0.790 | Extend search, re-run |
| 3-GBM ensemble OOF AUC | ≥ 0.792 | Re-weight, check correlation |
| GBM + NN-A ensemble OOF AUC | ≥ 0.794 | NN-B activation |
| **🆕 Phase 4.5 trigger** | OOF ≥ 0.792 | Skip 4.5 (cost not worth lift) |
| Final ensemble OOF AUC | ≥ 0.795 | Stacking Plan B |
| Final Public LB | ≥ 0.795 (target ≥ 0.80) | Audit CV-LB gap |

---

## 11. Submission File Specification

**Format** (per Kaggle):
```
SK_ID_CURR,TARGET
100001,0.1
100005,0.9
100013,0.2
...
```

**Generation** (`src/submit.py`):
- Loads test predictions from `artifacts/predictions/test_{model}.parquet`.
- Combines per `ensemble.py` weights.
- Writes to `submissions/submission_YYYYMMDD_HHMMSS_{label}_{cv_score}.csv`.
- Appends row to `submissions/log.csv`: `timestamp, filename, label, oof_auc, public_lb, private_lb, notes`.

**Submission cadence**:
1. **First submission**: tuned LGBM single model (anchors CV-LB correlation).
2. **Second**: 3-GBM rank-average ensemble.
3. **Third**: + NN-A.
4. **Fourth (optional)**: + NN-B and/or Phase 4.5 features.
5. Final pick from manual override log (A5 step 3).

---

## 12. Time Budget (realistic on RTX 3060 + 16 GB)

| Phase | Wall-clock | Run mode |
|---|---:|---|
| Feature engineering (full build) | 45–75 min | One-time |
| Folds + leakage smoke notebooks | 15 min | One-time |
| LGBM baseline (5-fold) | 20–30 min | Active |
| XGBoost baseline (5-fold, GPU) | 25–35 min | Active |
| CatBoost baseline (5-fold, GPU) | 40–60 min | Active |
| NN-A baseline (5-fold, GPU AMP) | 30–45 min | Active |
| **Subtotal — baselines** | **~3.5 hr** | One work session |
| Optuna LGBM (80 trials) | 2.0 hr | Overnight |
| Optuna XGB (50 trials) | 1.7 hr | Overnight |
| Optuna CatBoost (30 trials) | 1.5 hr | Overnight |
| Optuna NN-A (30 trials) | 1.5 hr | Overnight |
| **Subtotal — tuning** | **~6.7 hr** | Single overnight run |
| Tuned 5-fold final retrain (4 models) | ~2 hr | Active |
| Ensemble (Dirichlet + NM + manual review) | 15 min | Active |
| **Core pipeline total** | **~12.5 hr** | 1 work session + 1 overnight + 1 morning |
| Optional: Phase 4.5 (D7 aux GBMs) | +1.5–2 hr | Active, gated |
| Optional: NN-B + Ridge selection | +1.5 hr | Active, gated |
| **Full v2 chasing 0.80** | **~16 hr** | 2 overnights + 2 sessions |

**Suggested cadence**:
- Day 1 evening: feature pipeline + baselines (~4 hr) → submit baseline LGBM.
- Day 1 night: kick off Optuna (~7 hr overnight).
- Day 2 morning: final retrain + ensemble + submit (~2.5 hr).
- Day 2 evening (optional): Phase 4.5 + 4.6 if chasing gold.

---

**Status**: v2 plan complete with all approved modifications integrated. Feature count, memory budget, and time estimates re-calibrated. Awaiting "go" to begin Phase B.
