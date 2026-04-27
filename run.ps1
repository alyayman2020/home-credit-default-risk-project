# ──────────────────────────────────────────────────────────────────────────────
# Home Credit Default Risk — PowerShell pipeline runner (Windows replacement
# for the Unix Makefile).
#
# Why this exists: `make` is not installed on Windows by default, so commands
# like `make features` fail. This script provides the same targets as the
# Makefile but in a way PowerShell understands.
#
# First-time setup (run ONCE in PowerShell, can be from any folder):
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# (PowerShell will ask "Yes to All?" — answer Y. This lets you run unsigned
# local scripts but still blocks scripts downloaded from the internet.)
#
# Then from the project root:
#     .\run.ps1 help
#     .\run.ps1 install
#     .\run.ps1 data
#     .\run.ps1 features
#     .\run.ps1 baseline-lgbm
#     .\run.ps1 all
# ──────────────────────────────────────────────────────────────────────────────

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"

# Use `uv run python` to match the Makefile workflow.
$Python = "uv run python"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Warning "uv not found on PATH. Install from https://astral.sh/uv first:"
    Write-Warning '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
}

function Invoke-Step {
    param([string]$Description, [string]$Command)
    Write-Host ""
    Write-Host "─── $Description ───" -ForegroundColor Cyan
    Write-Host "  $Command" -ForegroundColor DarkGray
    Write-Host ""
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Step failed: $Description (exit code $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

function Show-Help {
    Write-Host ""
    Write-Host "Home Credit Default Risk — pipeline targets (PowerShell)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Setup:"
    Write-Host "  install        Install dependencies via uv (CPU torch)"
    Write-Host "  install-gpu    Install dependencies with CUDA 12.1 torch"
    Write-Host "  lock           Refresh uv.lock"
    Write-Host ""
    Write-Host "Quality:"
    Write-Host "  format         Run isort + black + ruff --fix"
    Write-Host "  lint           Run ruff (no fix) + black --check"
    Write-Host "  test           Run pytest"
    Write-Host "  smoke          Run leakage smoke tests"
    Write-Host ""
    Write-Host "Pipeline (in order):"
    Write-Host "  data           Load CSVs -> processed parquet (~30 sec)"
    Write-Host "  folds          Build CV folds (~3 sec)"
    Write-Host "  features       Build all 3 feature matrices (~2 min)"
    Write-Host "  baseline-lgbm  LGBM baseline (~30 min on GPU, ~60 min CPU)"
    Write-Host "  baseline-xgb   XGBoost baseline"
    Write-Host "  baseline-cat   CatBoost baseline"
    Write-Host "  baseline-nn    NN-A baseline"
    Write-Host "  baseline       All four baselines in sequence"
    Write-Host "  tune           Optuna sweeps (~6.7 hr — overnight)"
    Write-Host "  train          Refit tuned models on 5 folds"
    Write-Host "  ensemble       Dirichlet + Nelder-Mead blend"
    Write-Host "  submit         Write submission CSV"
    Write-Host "  all            data -> folds -> features -> baseline -> tune -> train -> ensemble -> submit"
    Write-Host ""
    Write-Host "Cleanup:"
    Write-Host "  clean          Remove __pycache__, .pytest_cache, .ruff_cache"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 data"
    Write-Host "  .\run.ps1 folds"
    Write-Host "  .\run.ps1 features"
    Write-Host "  .\run.ps1 baseline-lgbm"
    Write-Host ""
}

switch ($Target) {
    "help"          { Show-Help }

    # ─── Setup ───────────────────────────────────────────────────────────
    "install"       { Invoke-Step "Install (CPU torch)" "uv sync --extra dev" }
    "install-gpu"   {
        Invoke-Step "Install dependencies"            "uv sync --extra dev"
        Invoke-Step "Install GPU torch (CUDA 12.1)"   "uv pip install torch --index-url https://download.pytorch.org/whl/cu121"
    }
    "lock"          { Invoke-Step "Refresh uv.lock" "uv lock" }

    # ─── Quality ─────────────────────────────────────────────────────────
    "format"        {
        Invoke-Step "isort"     "uv run isort src tests"
        Invoke-Step "black"     "uv run black src tests"
        Invoke-Step "ruff fix"  "uv run ruff check --fix src tests"
    }
    "lint"          {
        Invoke-Step "ruff"          "uv run ruff check src tests"
        Invoke-Step "black --check" "uv run black --check src tests"
    }
    "test"          { Invoke-Step "pytest" "uv run pytest" }
    "smoke"         {
        Invoke-Step "CV smoke" "$Python -m src.cv --smoke"
        Write-Host "-> Run notebooks/04_eda_target_leakage_check.ipynb for full leakage suite."
    }

    # ─── Pipeline ────────────────────────────────────────────────────────
    "data"          { Invoke-Step "Load raw CSVs -> processed parquet" "$Python -m src.data" }
    "folds"         { Invoke-Step "Build CV folds"                      "$Python -m src.cv" }
    "features"      { Invoke-Step "Assemble feature matrices"           "$Python -m src.features.assemble" }

    "baseline-lgbm" { Invoke-Step "LGBM baseline"     "$Python -m src.train --model lgbm --mode baseline" }
    "baseline-xgb"  { Invoke-Step "XGBoost baseline"  "$Python -m src.train --model xgb --mode baseline" }
    "baseline-cat"  { Invoke-Step "CatBoost baseline" "$Python -m src.train --model catboost --mode baseline" }
    "baseline-nn"   { Invoke-Step "NN-A baseline"     "$Python -m src.train --model nn_a --mode baseline" }
    "baseline"      {
        & $PSCommandPath baseline-lgbm
        & $PSCommandPath baseline-xgb
        & $PSCommandPath baseline-cat
        & $PSCommandPath baseline-nn
    }

    "tune"          {
        Invoke-Step "Tune LGBM (80 trials)"     "$Python -m src.tune --model lgbm     --trials 80"
        Invoke-Step "Tune XGB (50 trials)"      "$Python -m src.tune --model xgb      --trials 50"
        Invoke-Step "Tune CatBoost (30 trials)" "$Python -m src.tune --model catboost --trials 30"
        Invoke-Step "Tune NN-A (30 trials)"     "$Python -m src.tune --model nn_a     --trials 30"
    }

    "train"         {
        Invoke-Step "Refit LGBM (tuned)"     "$Python -m src.train --model lgbm     --mode tuned"
        Invoke-Step "Refit XGB (tuned)"      "$Python -m src.train --model xgb      --mode tuned"
        Invoke-Step "Refit CatBoost (tuned)" "$Python -m src.train --model catboost --mode tuned"
        Invoke-Step "Refit NN-A (tuned)"     "$Python -m src.train --model nn_a     --mode tuned"
    }

    "ensemble"      { Invoke-Step "Ensemble (Dirichlet + Nelder-Mead)" "$Python -m src.ensemble" }
    "submit"        { Invoke-Step "Write submission CSV"               "$Python -m src.submit" }

    "all"           {
        & $PSCommandPath data
        & $PSCommandPath folds
        & $PSCommandPath features
        & $PSCommandPath baseline
        & $PSCommandPath tune
        & $PSCommandPath train
        & $PSCommandPath ensemble
        & $PSCommandPath submit
    }

    # ─── Cleanup ─────────────────────────────────────────────────────────
    "clean"         {
        Get-ChildItem -Path . -Recurse -Force -Include "__pycache__", ".pytest_cache", ".ruff_cache" -Directory -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Host "Cleaned __pycache__, .pytest_cache, .ruff_cache, *.pyc" -ForegroundColor Green
    }

    default {
        Write-Host ""
        Write-Host "Unknown target: '$Target'" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
