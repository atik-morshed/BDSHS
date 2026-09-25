# BD-SHS Full Experimental Upgrade — Master Runner
# ==================================================
# Runs all phases in order. Each phase logs to its own file.
# Phase 0 (config) and checkpoint fix are already done.
#
# Usage:
#   .\run_upgrade.ps1              # run all phases
#   .\run_upgrade.ps1 -Phase 2    # run only Phase 2
#   .\run_upgrade.ps1 -StartAt 3  # start from Phase 3
#
# Phases:
#   1 = Alignment Validator
#   2 = Per-Example Faithfulness CSV   (longest — GPU required)
#   3 = Bootstrap CIs + Permutation Tests + Effect Sizes
#   4 = Fixed-Coverage Evaluation      (GPU required)
#   5 = Multi-Seed Training            (GPU required, ~2h)
#   6 = Multi-Seed Attribution         (GPU required, ~4h)
#   7 = Seed Stability + Final Tables + Figures
#   8 = Runtime Measurements

param(
    [int]$Phase   = 0,       # run only this phase (0 = all)
    [int]$StartAt = 1        # start from this phase
)

$PYTHON = "E:\BD SHS\.venv\Scripts\python.exe"
$ROOT   = "E:\BD SHS"

function Run-Phase {
    param([int]$PhaseNum, [string]$Script, [string]$Description, [hashtable]$Env = @{})

    if ($Phase -ne 0 -and $Phase -ne $PhaseNum) { return }
    if ($PhaseNum -lt $StartAt) { return }

    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "PHASE $PhaseNum — $Description" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "Script: $Script"
    Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    # Set environment variables for this phase
    $envBackup = @{}
    foreach ($kv in $Env.GetEnumerator()) {
        $envBackup[$kv.Key] = [System.Environment]::GetEnvironmentVariable($kv.Key)
        [System.Environment]::SetEnvironmentVariable($kv.Key, $kv.Value)
    }

    $t0 = Get-Date
    & $PYTHON -u "$ROOT\$Script"
    $exit_code = $LASTEXITCODE
    $elapsed = (Get-Date) - $t0

    # Restore environment
    foreach ($kv in $envBackup.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($kv.Key, $kv.Value)
    }

    if ($exit_code -eq 0) {
        Write-Host "Phase $PhaseNum completed in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    } else {
        Write-Host "Phase $PhaseNum FAILED (exit code $exit_code) after $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Red
        if ($Phase -eq 0) {
            Write-Host "Stopping. Fix the error and re-run with -StartAt $PhaseNum" -ForegroundColor Yellow
            exit $exit_code
        }
    }
}

# ── Phase 1: Alignment Validator ──────────────────────────────
Run-Phase -PhaseNum 1 `
    -Script "evaluation\alignment_validator.py" `
    -Description "Alignment Validator (fast — no GPU)"

# ── Phase 2: Per-Example Faithfulness CSV ─────────────────────
# This is the longest phase — re-runs all forward passes.
# Set FAITH_MAX_EXAMPLES=100 for a quick test first.
Run-Phase -PhaseNum 2 `
    -Script "evaluation\faithfulness_per_example.py" `
    -Description "Per-Example Faithfulness CSV (GPU — ~4-8h for full dataset)"

# ── Phase 3: Bootstrap + Permutation + Effect Sizes ───────────
Run-Phase -PhaseNum 3 `
    -Script "statistics\bootstrap.py" `
    -Description "Bootstrap CIs + Permutation Tests + Effect Sizes (CPU — ~10 min)"

# ── Phase 4: Fixed-Coverage Evaluation ────────────────────────
Run-Phase -PhaseNum 4 `
    -Script "evaluation\fixed_coverage_eval.py" `
    -Description "Fixed-Coverage Evaluation (GPU — ~4-8h)"

# ── Phase 5: Multi-Seed Training ──────────────────────────────
Run-Phase -PhaseNum 5 `
    -Script "training\train_multiseed.py" `
    -Description "Multi-Seed Training seeds 43+44 (GPU — ~2h)"

# ── Phase 6: Multi-Seed Attribution ───────────────────────────
Run-Phase -PhaseNum 6 `
    -Script "attribution\run_attribution_multiseed.py" `
    -Description "Multi-Seed Attribution IG+SHAP for seeds 43+44 (GPU — ~4h)"

# ── Phase 7: Seed Stability + Final Tables + Figures ──────────
Run-Phase -PhaseNum 7 `
    -Script "analysis\seed_stability.py" `
    -Description "Seed Stability + Final Tables + Figures (fast)"

# ── Phase 8: Runtime Measurements ─────────────────────────────
Run-Phase -PhaseNum 8 `
    -Script "analysis\runtime_measurements.py" `
    -Description "Runtime Measurements (GPU — ~5 min for 100 examples)"

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "All requested phases complete!" -ForegroundColor Green
Write-Host "Results in: $ROOT\results\" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
