"""
Phase 1 — Fix Effect Size Calculation (CRITICAL BUG FIX)
==========================================================

Issue: Effect sizes reported as 0.0000 when they should be non-zero.
Root cause: Bug in Cohen's d_z calculation.

This script:
1. Loads raw paired values for each comparison
2. Verifies example pairing by example_id
3. Calculates differences correctly
4. Computes Cohen's d_z: mean_diff / sd_diff
5. Adds sanity checks to prevent future bugs
6. Recalculates all 36 comparisons

Run: python statistics/phase1_fix_effect_size.py
Outputs: results/statistics/corrected_effect_sizes.json
"""

import os, sys, io, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "results" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase1_effect_size_fix_log.txt"
log_fh = open(LOG_FILE, "w", encoding="utf-8")

def log(message):
    """Write to both stdout and log file."""
    print(message)
    log_fh.write(message + "\n")
    log_fh.flush()

print("=" * 80)
log("=" * 80)
log("PHASE 1 — FIX EFFECT SIZE CALCULATION (CRITICAL BUG FIX)")
log("=" * 80)

# Load per-example results
csv_path = BASE_DIR / "results" / "evaluation" / "per_example" / "per_example_faithfulness_results.csv"

if not csv_path.exists():
    log(f"\n[ERROR] Per-example results not found: {csv_path}")
    sys.exit(1)

log(f"\nLoading per-example results from: {csv_path}")
df = pd.read_csv(csv_path)
log(f"Loaded {len(df)} rows")

# Define comparisons
strategies = ['sum', 'mean', 'max']
budgets = [10, 20, 30]
metrics = ['comprehensiveness', 'sufficiency', 'comp_efficiency', 'suff_efficiency']

all_results = []

log("\n" + "=" * 80)
log("INVESTIGATING EFFECT SIZE CALCULATION")
log("=" * 80)

# First, detailed investigation of one comparison to debug
log("\n" + "-" * 80)
log("DETAILED INVESTIGATION: Hierarchical vs Sum @ 10% Sufficiency Efficiency")
log("-" * 80)

hierarchical_mask = (df['strategy'] == 'hierarchical') & (df['budget_pct'] == 10)
baseline_mask = (df['strategy'] == 'sum') & (df['budget_pct'] == 10)

hierarchical_df = df[hierarchical_mask][['example_id', 'suff_efficiency']].set_index('example_id')
baseline_df = df[baseline_mask][['example_id', 'suff_efficiency']].set_index('example_id')

log(f"\nHierarchical examples: {len(hierarchical_df)}")
log(f"Baseline examples: {len(baseline_df)}")

# Verify example IDs match
h_ids = set(hierarchical_df.index)
b_ids = set(baseline_df.index)
common_ids = h_ids.intersection(b_ids)

log(f"Common example IDs: {len(common_ids)}")
log(f"Hierarchical-only IDs: {len(h_ids - b_ids)}")
log(f"Baseline-only IDs: {len(b_ids - h_ids)}")

if len(h_ids) != len(b_ids) or h_ids != b_ids:
    log("[WARNING] Example ID mismatch detected!")
    log("Proceeding with common IDs only.")

# Align by example_id
hierarchical_aligned = hierarchical_df.loc[list(common_ids)]['suff_efficiency'].values
baseline_aligned = baseline_df.loc[list(common_ids)]['suff_efficiency'].values

log(f"\nAligned Hierarchical values: {len(hierarchical_aligned)}")
log(f"Aligned Baseline values: {len(baseline_aligned)}")

# Calculate differences
differences = hierarchical_aligned - baseline_aligned

log(f"\nDifference statistics:")
log(f"  Mean: {differences.mean():.6f}")
log(f"  Std (ddof=1): {differences.std(ddof=1):.6f}")
log(f"  Std (ddof=0): {differences.std(ddof=0):.6f}")
log(f"  Min: {differences.min():.6f}")
log(f"  Max: {differences.max():.6f}")
log(f"  Median: {np.median(differences):.6f}")

# Calculate Cohen's d_z
mean_diff = differences.mean()
sd_diff = differences.std(ddof=1)

log(f"\nCohen's d_z calculation:")
log(f"  mean_diff = {mean_diff:.6f}")
log(f"  sd_diff = {sd_diff:.6f}")

if sd_diff == 0:
    log("[ERROR] Standard deviation is zero! Cannot calculate effect size.")
    dz = 0.0
else:
    dz = mean_diff / sd_diff
    log(f"  Cohen's d_z = {dz:.6f}")

# Sanity checks
log(f"\nSanity checks:")
log(f"  np.isfinite(dz): {np.isfinite(dz)}")
log(f"  sd_diff > 0: {sd_diff > 0}")
log(f"  abs(dz) < 1e-8: {abs(dz) < 1e-8}")
log(f"  abs(mean_diff) > 1e-3: {abs(mean_diff) > 1e-3}")

if abs(dz) < 1e-8 and abs(mean_diff) > 1e-3:
    log("[ERROR] Effect size suspiciously small given non-zero mean difference!")
    log("This indicates a bug in the calculation.")

# Now apply to all comparisons
log("\n" + "=" * 80)
log("APPLYING CORRECTED CALCULATION TO ALL COMPARISONS")
log("=" * 80)

for strategy in strategies:
    for budget in budgets:
        for metric in metrics:
            log(f"\n{'─'*80}")
            log(f"Comparing: Hierarchical vs {strategy}")
            log(f"Budget: {budget}%, Metric: {metric}")
            log(f"{'─'*80}")

            # Extract hierarchical and baseline values
            hierarchical_mask = (df['strategy'] == 'hierarchical') & (df['budget_pct'] == budget)
            baseline_mask = (df['strategy'] == strategy) & (df['budget_pct'] == budget)

            hierarchical_df = df[hierarchical_mask][['example_id', metric]].set_index('example_id')
            baseline_df = df[baseline_mask][['example_id', metric]].set_index('example_id')

            # Get common example IDs
            common_ids = hierarchical_df.index.intersection(baseline_df.index)
            
            if len(common_ids) == 0:
                log(f"[WARNING] No common examples for {strategy} @ {budget}%")
                continue

            hierarchical_aligned = hierarchical_df.loc[list(common_ids)][metric].values
            baseline_aligned = baseline_df.loc[list(common_ids)][metric].values

            # Calculate differences
            differences = hierarchical_aligned - baseline_aligned

            # Remove NaN values
            valid_mask = ~(np.isnan(hierarchical_aligned) | np.isnan(baseline_aligned))
            hierarchical_aligned = hierarchical_aligned[valid_mask]
            baseline_aligned = baseline_aligned[valid_mask]
            differences = differences[valid_mask]

            if len(differences) == 0:
                log(f"[WARNING] No valid data for {strategy} @ {budget}% - {metric}")
                continue

            log(f"  Valid examples: {len(differences)}")

            # Calculate statistics
            mean_diff = differences.mean()
            sd_diff = differences.std(ddof=1)

            # Calculate Cohen's d_z with sanity checks
            if sd_diff == 0 or not np.isfinite(sd_diff):
                log(f"[ERROR] Invalid standard deviation: {sd_diff}")
                dz = 0.0
            else:
                dz = mean_diff / sd_diff
                
                # Sanity check
                if abs(dz) < 1e-8 and abs(mean_diff) > 1e-3:
                    log(f"[ERROR] Suspicious effect size: dz={dz:.8f}, mean_diff={mean_diff:.6f}")
                else:
                    log(f"  Cohen's d_z: {dz:.6f}")

            log(f"  Mean difference: {mean_diff:.6f}")
            log(f"  Std difference: {sd_diff:.6f}")

            result = {
                'budget': budget,
                'baseline_strategy': strategy,
                'metric': metric,
                'n_examples': len(differences),
                'mean_hierarchical': float(np.mean(hierarchical_aligned)),
                'mean_baseline': float(np.mean(baseline_aligned)),
                'mean_difference': float(mean_diff),
                'std_difference': float(sd_diff),
                'cohens_dz': float(dz),
                'median_difference': float(np.median(differences)),
                'min_difference': float(differences.min()),
                'max_difference': float(differences.max())
            }

            all_results.append(result)

# Save corrected results
results_path = OUT_DIR / "corrected_effect_sizes.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

log(f"\n{'='*80}")
log(f"Corrected effect sizes saved to: {results_path}")
log(f"Total comparisons: {len(all_results)}")
log(f"{'='*80}")

# Generate summary
log("\n" + "=" * 80)
log("SUMMARY OF CORRECTED EFFECT SIZES")
log("=" * 80)

for metric in metrics:
    log(f"\n{metric.upper()}:")
    log("-" * 80)
    for budget in budgets:
        log(f"\nBudget {budget}%:")
        for strategy in strategies:
            match = next((r for r in all_results if r['budget'] == budget and 
                          r['baseline_strategy'] == strategy and r['metric'] == metric), None)
            if match:
                log(f"  vs {strategy:6s}: dz = {match['cohens_dz']:8.4f}, "
                    f"mean_diff = {match['mean_difference']:8.4f}")

log_fh.close()

print("\n" + "=" * 80)
print("Phase 1 complete. Check log file for details.")
print("=" * 80)
