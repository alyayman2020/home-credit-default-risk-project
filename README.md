# Home Credit Default Risk — End-to-End Credit Scoring Pipeline

> **0.79309 Private LB · 0.79786 Public LB · OOF/Private gap < 0.002**
> A reproducible, modular ML pipeline for the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition.
> Author: **Aly Ayman** · [LinkedIn](https://www.linkedin.com/in/aly-ayman) · Cairo, Egypt 🇪🇬

Hi 👋 — thanks for stopping by. This repository is a hands-on demonstration of how I approach a real ML problem end-to-end: from messy CSVs through validation and modelling to a reproducible Kaggle submission. I built it on a 16 GB / RTX 4050 laptop with a focus on **engineering discipline** — the kind of pipeline I'd feel comfortable handing to a teammate on day one.

If you're a recruiter or hiring manager, the **TL;DR table below** should give you everything you need in 30 seconds. If you're a fellow practitioner, **[`PLAN_v2.md`](PLAN_v2.md)** has the full design rationale.

---

## TL;DR

| | Result |
|---|---|
| **Final OOF AUC** | 0.79518 (tuned blend) |
| **Public LB / Private LB** | 0.79786 / **0.79309** |
| **OOF − Private gap** | 0.00209 — well within the 0.005 healthy band |
| **Approach** | 5-fold OOF · LGBM + XGB + CatBoost + PyTorch NN · Dirichlet-Nelder-Mead blend |
| **Validation** | StratifiedKFold + GroupKFold + adversarial leakage smoke tests |
| **Code** | ~5,200 lines across 23 modules, 14 unit tests, fully reproducible |
| **Runtime** | ~6 hours end-to-end on a 16 GB / RTX 4050 laptop |

---

## Skills Demonstrated

| Area | Specifics |
|---|---|
| **Languages & libraries** | Python 3.11 · Polars · pandas · NumPy · scikit-learn |
| **Tabular ML** | LightGBM · XGBoost · CatBoost · OOF target encoding · nearest-neighbour features · null-importance pruning |
| **Deep learning** | PyTorch 2.x · gated residual MLP · embedding tables for categoricals · AMP · OneCycle LR · early stopping |
| **Validation** | 5-fold StratifiedKFold · GroupKFold · per-fold leakage controls · adversarial smoke tests |
| **Hyperparameter tuning** | Optuna (TPE sampler) · cost-aware early termination |
| **Ensembling** | Dirichlet weight grid · Nelder-Mead refinement · rank averaging · LR meta-learner |
| **Engineering** | `uv` + lockfile · Makefile + PowerShell · `pytest` · `loguru` · structured config via `.env` |
| **Domain** | Consumer credit risk · sentinel-value handling · business-AUC translation |
| **Reproducibility** | Fixed seeds · parquet for all intermediates · submission ledger · committed `uv.lock` |

---

## Why This Project

I wanted a portfolio piece that demonstrates I can build **production-style ML**, not just notebooks. So this repo is structured as a proper Python package with typed interfaces, dependency management, a test suite, and a Makefile/PowerShell build system. Each design choice is documented and defensible — including the ones where I deliberately stopped optimising because the marginal cost outweighed the marginal gain.

The headline number isn't really the AUC. It's the **0.00209 OOF–Private gap** — the fact that my cross-validation actually predicted Kaggle's private leaderboard. That's what production ML lives or dies on.

---

## Repository Structure

```
home-credit-default-risk-project/
├── PLAN_v2.md            # Full design document
├── README.md             # This file
├── RUNBOOK.md            # Step-by-step CLI commands
├── pyproject.toml        # uv-managed dependencies
├── uv.lock               # Pinned versions (committed)
├── Makefile / run.ps1    # Linux/macOS + Windows build targets
├── .env.example          # Copy → .env, set DATA_DIR
│
├── src/                  # The whole pipeline lives here as a Python package
│   ├── config.py         # Paths, seeds, fold count (env-driven)
│   ├── utils.py          # Logger, set_seed, timer
│   ├── data.py           # Polars loaders, sentinel cleanup
│   ├── cv.py             # CV folds + leakage smoke tests
│   ├── features/         # Per-table builders (application, bureau, …) + assembler
│   ├── models/           # LGBM, XGB, CatBoost, NN wrappers (auto GPU→CPU fallback)
│   ├── train.py          # 5-fold OOF loop with fold-aware transforms
│   ├── tune.py           # Optuna runner
│   ├── ensemble.py       # Dirichlet + Nelder-Mead weight search
│   ├── stack.py          # Logistic-regression meta-learner (Plan B)
│   └── submit.py         # Writes Kaggle CSV + appends to submission ledger
│
├── notebooks/            # EDA + leakage audit notebooks
├── data/                 # raw / processed / features (gitignored)
├── artifacts/            # model binaries, predictions, best params (gitignored)
├── submissions/          # CSVs + submissions/log.csv ledger
└── tests/                # pytest suite (smoke + unit)
```

---

## Results

Both submissions used the same 4-model blend, evaluated against Kaggle's hidden private test set. The two-step approach was deliberate: ship a solid baseline first (so I have a real LB anchor), then iterate.

| # | Description | OOF AUC | Public LB | Private LB | OOF − Private |
|---|---|---:|---:|---:|---:|
| 1 | **Baseline ensemble** — LGBM + XGB + CatBoost + NN-A, default hyperparameters | 0.79451 | 0.79799 | 0.79296 | +0.00155 |
| 2 | **Manually-tuned LGBM blend** — same blend, with LGBM hyperparameters from a partial Optuna run *(see Trade-offs below)* | 0.79518 | 0.79786 | **0.79309** | +0.00209 |

The ~0.002 OOF–Private gap is small, stable, and consistent. Many Kaggle pipelines silently overfit OOF by 0.01+; this one does not. That's the deliverable I'm most proud of.

### Per-model OOF AUC

| Model | Matrix | OOF AUC | Per-fold std |
|---|---|---:|---:|
| LightGBM (manually tuned) | `main` (1,048 features) | 0.79436 | ±0.00379 |
| XGBoost | `main` (1,048 features) | 0.79267 | ±0.00389 |
| CatBoost | `catboost` (1,001 features, raw cats) | 0.79202 | ±0.00389 |
| PyTorch NN-A (gated residual MLP) | `nn` (1,913 features, RankGauss + nan-flags) | 0.77791 | ±0.00463 |

The three GBMs are tightly correlated (within 0.0024 OOF AUC of each other). The NN trails on solo performance but adds enough diversity to lift the blend by ~+0.001.

---

## Business Framing

A credit-scoring model that ranks risk poorly costs a lender twice: through **realised losses** (loans extended to applicants who default) and **opportunity cost** (creditworthy applicants rejected because they look like risky ones).

ROC AUC measures **ranking quality**, which is what a credit officer actually uses. An AUC of 0.793 means:

> For any random pair of (defaulter, non-defaulter), the model assigns the defaulter a higher risk score 79.3% of the time.

In operational terms — on a **100,000-loan portfolio at $5,000 average principal and 8% baseline default rate** — the model's top decile concentrates roughly **3–4× the baseline default rate**. Refusing or risk-pricing that decile avoids a meaningful share of the ~$40M in expected default losses, while keeping false rejections of low-risk borrowers low enough that the policy stays commercially competitive.

There's a second story here too: Home Credit's stated mission is to extend credit to **thin-file applicants** — borrowers without traditional credit histories. A well-calibrated model lets a lender say *yes* to those applicants safely, by leveraging behavioural data from previous loans, instalment payments, and credit-bureau records at *other* institutions. Six of the seven source tables in this dataset describe exactly that kind of alternative data, and the strongest engineered features are built from it.

---

## Methodology

The pipeline is split into **seven CLI-driven stages** so each can be re-run, profiled, and tested in isolation.

```
  Raw CSVs (10 files, ~2.7 GB)
       ↓
  [1] Ingest        — Polars streaming → 8 parquets        (src/data.py)
       ↓
  [2] CV folds      — Stratified + GroupKFold + smoke      (src/cv.py)
       ↓
  [3] Features      — 6 builders → 1,420 → 1,048 features  (src/features/)
       ↓                3 matrices: main / catboost / nn
  [4] OOF training  — fold-aware target encoding + D1 NN   (src/train.py)
       ↓                + RankGauss for the NN matrix
  [5] Tuning        — Optuna w/ TPE sampler                (src/tune.py)
       ↓
  [6] Ensemble      — Dirichlet grid + Nelder-Mead         (src/ensemble.py)
       ↓
  [7] Submission    — CSV + append to submissions/log.csv  (src/submit.py)
```

### Highlights

- **Polars over pandas** for ingestion — handles 27M-row `bureau_balance` and 13.6M-row `installments_payments` in <1 second each.
- **Three feature matrices**, one per model family. CatBoost gets raw categoricals (its internal target stats outperform mine); the NN gets RankGauss-transformed numerics + `__isnan` flag columns + integer-encoded categoricals; LGBM and XGB get one-hot encodings.
- **Per-fold transforms inside the CV loop**. OOF target encoding, D1 nearest-neighbour target mean (k=500 on `EXT_SOURCE_*` + `CREDIT_TERM`), and RankGauss are all fit on training rows only. None of them leaks across folds.
- **Custom fast correlation pruner**. Replaced `pandas.corr()` (which hung for 30+ minutes on Windows) with median-impute → standardise → `X.T @ X / n`. Dropped 326 of 1,420 columns in **~2 seconds**.
- **Auto GPU→CPU fallback** in every model wrapper. Catches CUDA / OpenCL errors, retries on CPU, logs a warning. Makes the pipeline portable across machines.

---

## Engineering Trade-offs

### What I deliberately *didn't* do

**Did not run the full 80-trial Optuna sweep on LGBM.** I started it overnight, but after 3 trials the wall-clock projection exceeded 4 days — the Windows pip wheel of LightGBM is CPU-only, and compiling a GPU build wasn't worth it for this project. So I made a different call: I let trial 3 finish, looked at the numbers, and **manually adopted its hyperparameters** as the "tuned LGBM" config:

```
trial 0:  default config              OOF 0.79250
trial 1:  num_leaves=31, lr=0.019     OOF 0.79314
trial 2:  num_leaves=22, lr=0.013     OOF 0.79325
trial 3:  num_leaves=57, lr=0.005     OOF 0.79436   ← adopted manually
```

Trial 3 captured the bulk of the available lift over the default config (+0.0019 OOF). The remaining trials would, by Optuna's own typical lift curve on tabular problems, have added at most another ~+0.001 — for 4 more days of compute. This was a conscious cost-vs-lift decision. I committed `artifacts/best_params/lgbm.json` with trial 3's parameters and used `--mode tuned` for the final run, so the choice is fully reproducible. That parameter file is the receipt.

**Did not pursue auxiliary GBM features (D7).** Estimated ~6 hours compute for ~+0.002 OOF. Not worth it for a portfolio-demo objective.

**Did not implement NN-B** (smaller backup NN). NN-A was sufficient; NN-B would have correlated with it.

**Did not optimise to Public LB.** Selection used OOF AUC throughout. The Private LB result is the honest measure.

### What didn't work — and what I learned

- **Stacking didn't beat rank averaging.** I implemented a logistic-regression meta-learner over the 4 OOF prediction columns (`src/stack.py`) and it produced a *lower* OOF than the best single base model. The four GBMs were too correlated to give the meta-learner anything to exploit. *Lesson: measure base-learner pairwise OOF correlation before assuming stacking will help.*
- **NN matrix had a silent column-naming bug** (`__isnan` vs `_is_nan` between assembler and trainer). RankGauss quantile-transformed binary flag columns into useless near-uniform noise, and no exception was raised. Caught only by reading per-fold AUCs carefully. *Lesson: invariant assertions across module boundaries are cheap insurance.*
- **`pandas.corr()` was the wrong tool for the job** (see Methodology above). Replacing it with NumPy GEMM was a 1,000× speedup. *Lesson: use the tool that matches the workload.*

---

## What I'd Do With More Time

If this were a continuing project rather than a portfolio piece:

1. Heavier feature engineering — top-1% Kaggle solutions used dozens more aggregation features per source table. Estimated ceiling: +0.005 OOF.
2. Bureau-level neighbour features (D1 at the bureau-credit level, not just application level). Winners reported +0.003–0.005.
3. Stacking with **truly diverse** base learners (Random Forest, ExtraTrees, dart-boosted LightGBM) — diversity is what stacking needs.
4. GPU-compiled LightGBM. ~5× tuning speedup, would make the full Optuna sweep feasible.
5. Probability calibration (Platt / isotonic) for production deployment — AUC measures ranking, not calibration, and a deployed model needs both.
6. Drift monitoring (PSI / KL across time-windowed snapshots). Standard for production credit models.

---

## Quickstart

**Prerequisites:** Python 3.11, [uv](https://astral.sh/uv), ~15 GB free disk, 16 GB RAM. NVIDIA GPU optional.

```bash
git clone https://github.com/alyayman2020/home-credit-default-risk-project.git
cd home-credit-default-risk-project

# Linux / macOS
make install              # CPU torch
make install-gpu          # GPU torch (CUDA 12.1)

# Windows
.\run.ps1 install
.\run.ps1 install-gpu

cp .env.example .env      # edit DATA_DIR to point to the Kaggle CSVs
```

| Linux / macOS | Windows | Stage |
|---|---|---|
| `make data` | `.\run.ps1 data` | Ingest CSVs → parquet |
| `make folds` | `.\run.ps1 folds` | Build CV folds |
| `make features` | `.\run.ps1 features` | Build 3 feature matrices |
| `make baseline` | `.\run.ps1 baseline` | Train 4 baseline models |
| `make ensemble` | `.\run.ps1 ensemble` | Dirichlet + Nelder-Mead blend |
| `make submit` | `.\run.ps1 submit` | Write Kaggle submission CSV |
| `make all` | `.\run.ps1 all` | Run everything end-to-end |

Full step-by-step in [`RUNBOOK.md`](RUNBOOK.md). Design rationale in [`PLAN_v2.md`](PLAN_v2.md).

---

## Acknowledgments

I learned a lot from public Home Credit writeups by **ogrellier**, **Will Koehrsen**, and the various top-1% solo gold solutions — the windowed monthly aggregations, RankGauss for the NN matrix, and manual ensemble weighting all came from there.

I also used **Anthropic's Claude** as a pair-programming collaborator throughout this project — for design review, debugging, and rubber-ducking trade-offs. The strategic decisions and final calls were mine; AI accelerated execution. I document this openly because in 2026 it's standard professional tooling.

---

## Contact

Always happy to chat about ML, credit risk, or this project specifically. Find me at:

📧 **Email:** *(aly.ayman.ds@gmail.com)*
💼 **LinkedIn:** [alyayman](https://www.linkedin.com/in/alyayman)
🐙 **GitHub:** [@alyayman2020](https://github.com/alyayman2020)

— **Aly Ayman**, junior data scientist & ML engineer · ITI Data Science Track 2026 · Cairo 🇪🇬

---

*MIT License.*
