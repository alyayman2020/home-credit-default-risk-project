# ──────────────────────────────────────────────────────────────────────────────
# Home Credit Default Risk — Makefile
# Reference: PLAN_v2.md §9 (output order), §12 (time budget)
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help install install-gpu lock format lint test clean \
        data features baseline tune train ensemble submit \
        baseline-lgbm baseline-xgb baseline-cat baseline-nn \
        eda smoke all

# ─── Defaults ────────────────────────────────────────────────────────────────
PYTHON := uv run python
SRC    := src
PYTEST := uv run pytest

help:
	@echo "Home Credit Default Risk — pipeline targets"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install dependencies via uv (CPU torch)"
	@echo "  install-gpu    Install dependencies with CUDA 12.1 torch"
	@echo "  lock           Refresh uv.lock"
	@echo ""
	@echo "Quality:"
	@echo "  format         Run black + isort + ruff --fix"
	@echo "  lint           Run ruff (no fix) + black --check"
	@echo "  test           Run pytest"
	@echo "  smoke          Run leakage smoke tests"
	@echo ""
	@echo "Pipeline (PLAN_v2.md §9):"
	@echo "  data           Load + sentinel-replace + write parquet"
	@echo "  features       Build all 3 feature matrices (main, catboost, nn)"
	@echo "  baseline       All baselines: LGBM + XGB + CatBoost + NN-A"
	@echo "  baseline-lgbm  LGBM baseline only (PLAN §4.1 step 1)"
	@echo "  baseline-xgb   XGBoost baseline (PLAN §4.1 step 2)"
	@echo "  baseline-cat   CatBoost baseline (PLAN §4.1 step 3)"
	@echo "  baseline-nn    NN-A baseline (PLAN §4.3)"
	@echo "  tune           Optuna sweeps for all 4 models (~6.7 hr)"
	@echo "  ensemble       Dirichlet + Nelder-Mead blend"
	@echo "  submit         Write submission CSV to submissions/"
	@echo "  all            data → features → baseline → tune → ensemble → submit"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Remove caches and pyc files (keeps data/, artifacts/)"

# ─── Setup ───────────────────────────────────────────────────────────────────
install:
	uv sync --extra dev

install-gpu:
	uv sync --extra dev
	uv pip install torch --index-url https://download.pytorch.org/whl/cu121

lock:
	uv lock

# ─── Quality ─────────────────────────────────────────────────────────────────
format:
	uv run isort $(SRC) tests
	uv run black $(SRC) tests
	uv run ruff check --fix $(SRC) tests

lint:
	uv run ruff check $(SRC) tests
	uv run black --check $(SRC) tests

test:
	$(PYTEST)

smoke:
	$(PYTHON) -m src.cv --smoke
	@echo "→ Run notebooks/04_eda_target_leakage_check.ipynb for full leakage suite (PLAN §3.4)"

# ─── Pipeline ────────────────────────────────────────────────────────────────
data:
	$(PYTHON) -m src.data

features:
	$(PYTHON) -m src.features.assemble

baseline-lgbm:
	$(PYTHON) -m src.train --model lgbm --mode baseline

baseline-xgb:
	$(PYTHON) -m src.train --model xgb --mode baseline

baseline-cat:
	$(PYTHON) -m src.train --model catboost --mode baseline

baseline-nn:
	$(PYTHON) -m src.train --model nn_a --mode baseline

baseline: baseline-lgbm baseline-xgb baseline-cat baseline-nn

tune:
	$(PYTHON) -m src.tune --model lgbm     --trials 80
	$(PYTHON) -m src.tune --model xgb      --trials 50
	$(PYTHON) -m src.tune --model catboost --trials 30
	$(PYTHON) -m src.tune --model nn_a     --trials 30

train:
	$(PYTHON) -m src.train --model lgbm     --mode tuned
	$(PYTHON) -m src.train --model xgb      --mode tuned
	$(PYTHON) -m src.train --model catboost --mode tuned
	$(PYTHON) -m src.train --model nn_a     --mode tuned

ensemble:
	$(PYTHON) -m src.ensemble

submit:
	$(PYTHON) -m src.submit

all: data features baseline tune train ensemble submit

# ─── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
