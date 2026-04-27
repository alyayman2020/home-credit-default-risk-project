# ──────────────────────────────────────────────────────────────────────────────
# Home Credit Default Risk — PowerShell pipeline runner
# Windows replacement for the Unix Makefile.
#
# Usage (PowerShell, from project root):
#   .\run.ps1 help
#   .\run.ps1 install
#   .\run.ps1 data
#   .\run.ps1 features
#   .\run.ps1 baseline-lgbm
#   .\run.ps1 all
#
# If you get a "running scripts is disabled" error, run this once in an
# elevated PowerShell session:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# ──────────────────────────────────────────────────────────────────────────────

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"

# Use `uv run python` to match the Makefile. Falls back to plain `python` if
# uv is not on PATH (a warning is printed in that case).
$Python = "uv run python"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Warning "uv not found on PATH — falling back to plain 'python'. Install uv from https://astral.sh/uv for reproducible runs."
    $Python = "python"
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
    Write-Host "Pipeline (PLAN_v2.md `S9):"
    Write-Host "  data           Load + sentinel-replace + write parquet"
    Write-Host "  features       Build all 3 feature matrices"
    Write-Host "  baseline       All baselines: LGBM + XGB + CatBoost + NN-A"
    Write-Host "  baseline-lgbm  LGBM baseline only"
    Write-Host "  baseline-xgb   XGBoost baseline only"
    Write-Host "  baseline-cat   CatBoost baseline only"
    Write-Host "  baseline-nn    NN-A baseline only"
    Write-Host "  tune           Optuna sweeps (~6.7 hr)"
    Write-Host "  train          Refit tuned models on 5 folds"
    Write-Host "  ensemble       Dirichlet + Nelder-Mead blend"
    Write-Host "  submit         Write submission CSV"
    Write-Host "  all            data -> features -> baseline -> tune -> train -> ensemble -> submit"
    Write-Host ""
    Write-Host "Cleanup:"
    Write-Host "  clean          Remove __pycache__, .pytest_cache, .ruff_cache"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 data"
    Write-Host "  .\run.ps1 baseline-lgbm"
    Write-Host ""
}

switch ($Target) {
    "help"          { Show-Help }

    # ─── Setup ───────────────────────────────────────────────────────────
    "install"       { Invoke-Step "Install (CPU torch)" "uv sync --extra dev" }
    "install-gpu"   {
        Invoke-Step "Install (deps)" "uv sync --extra dev"
        Invoke-Step "Install GPU torch (CUDA 12.1)" "uv pip install torch --index-url https://download.pytorch.org/whl/cu121"
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
        Write-Host "→ Run notebooks/04_eda_target_leakage_check.ipynb for the full leakage suite."
    }

    # ─── Pipeline ────────────────────────────────────────────────────────
    "data"          { Invoke-Step "Load raw CSVs -> processed parquet" "$Python -m src.data" }
    "features"      { Invoke-Step "Assemble feature matrices"           "$Python -m src.features.assemble" }

    "baseline-lgbm" { Invoke-Step "LGBM baseline"     "$Python -m src.train --model lgbm --mode baseline" }
    "baseline-xgb"  { Invoke-Step "XGBoost baseline"  "$Python -m src.train --model xgb --mode baseline" }
    "baseline-cat"  { Invoke-Step "CatBoost baseline" "$Python -m src.train --model catboost --mode baseline" }
    "baseline-nn"   { Invoke-Step "NN-A baseline"     "$Python -m src.train --model nn_a --mode baseline" }
    "baseline"      {
        Invoke-Step "LGBM baseline"     "$Python -m src.train --model lgbm --mode baseline"
        Invoke-Step "XGBoost baseline"  "$Python -m src.train --model xgb --mode baseline"
        Invoke-Step "CatBoost baseline" "$Python -m src.train --model catboost --mode baseline"
        Invoke-Step "NN-A baseline"     "$Python -m src.train --model nn_a --mode baseline"
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
