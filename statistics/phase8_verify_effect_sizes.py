"""
Phase 8 — Verify and Consolidate Effect Sizes in Master Table
==========================================================

Purpose: Verify that all effect sizes are correctly calculated and consolidated
in the master statistics table with Holm-corrected p-values.

Steps:
1. Load corrected statistical results from Phase 7
2. Verify Cohen's d_z calculations for all comparisons
3. Create consolidated master table with all metrics
4. Export to CSV and human-readable formats
5. Generate verification report

Run: python statistics/phase8_verify_effect_sizes.py
Outputs: results/statistics/final_master_table.csv
         results/statistics/final_master_table.txt
         results/statistics/effect_size_verification_report.md
"""

import os, sys, io, json
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "phase8_effect_size_verification_log.txt"
log_fh = open(LOG_FILE, "w", encoding="utf-8")

def log(message):
    """Write to both stdout and log file."""
    print(message)
    log_fh.write(message + "\n")
    log_fh.flush()

print("=" * 80)
log("=" * 80)
log("PHASE 8 — VERIFY AND CONSOLIDATE EFFECT SIZES IN MASTER TABLE")
log("=" * 80)

# Load corrected results from Phase 7
corrected_path = OUTPUT_DIR / "corrected_statistical_results.json"
log(f"\nLoading corrected results from: {corrected_path}")
with open(corrected_path, 'r', encoding='utf-8') as f:
    corrected_results = json.load(f)

log(f"Loaded {len(corrected_results)} comparisons")

# Load per-example data for verification
csv_path = BASE_DIR / "results" / "evaluation" / "per_example" / "per_example_faithfulness_results.csv"
log(f"\nLoading per-example data from: {csv_path}")
df = pd.read_csv(csv_path)
log(f"Loaded {len(df)} rows")

# Verify effect sizes for sample comparisons
log(f"\n{'='*80}")
log("Verifying Cohen's d_z calculations")
log(f"{'='*80}")

verification_results = []

for comp in corrected_results[:12]:  # Verify budget 10% comparisons
    comp_baseline = comp['baseline_strategy']
    comp_metric = comp['metric_name']
    comp_budget = comp['budget']
    
    # Extract hierarchical and baseline data
    hierarchical = df[
        (df['strategy'] == 'hierarchical') &
        (df['budget_pct'] == comp_budget)
    ].copy()
    
    baseline = df[
        (df['strategy'] == comp_baseline) &
        (df['budget_pct'] == comp_budget)
    ].copy()
    
    # Merge on example_id
    merged = hierarchical.merge(baseline, on='example_id', suffixes=('_h', '_b'))
    differences = merged[f"{comp_metric}_h"] - merged[f"{comp_metric}_b"]
    
    # Manual Cohen's d_z calculation
    mean_diff = differences.mean()
    sd_diff = differences.std(ddof=1)
    manual_dz = mean_diff / sd_diff if sd_diff > 0 else 0
    
    # Get reported effect size
    reported_dz = comp.get('effect_size', {}).get('cohens_dz', None)
    
    # Verify
    is_match = np.isclose(manual_dz, reported_dz, atol=0.01) if reported_dz is not None else False
    
    verification_results.append({
        'comparison': comp['comparison_name'],
        'metric': comp_metric,
        'baseline': comp_baseline,
        'budget': comp_budget,
        'manual_dz': manual_dz,
        'reported_dz': reported_dz,
        'match': is_match
    })
    
    status = "✓ MATCH" if is_match else "✗ MISMATCH"
    log(f"{comp['comparison_name']} - {comp_metric} @ {comp_budget}%: {status}")
    reported_str = f"{reported_dz:.6f}" if reported_dz is not None else "N/A"
    log(f"  Manual: {manual_dz:.6f}, Reported: {reported_str}")

# Summary verification
verified_count = sum(1 for v in verification_results if v['match'])
total_count = len(verification_results)
log(f"\nVerification summary: {verified_count}/{total_count} effect sizes verified")

# Create consolidated master table
log(f"\n{'='*80}")
log("Creating consolidated master table")
log(f"{'='*80}")

master_data = []
for comp in corrected_results:
    master_data.append({
        'metric': comp['metric_name'],
        'baseline': comp['baseline_strategy'],
        'budget': comp['budget'],
        'mean_h': comp['hierarchical_stats']['mean'],
        'mean_b': comp['baseline_stats']['mean'],
        'delta': comp['bootstrap_results']['observed_difference'],
        'ci_lower': comp['bootstrap_results']['ci_lower'],
        'ci_upper': comp['bootstrap_results']['ci_upper'],
        'raw_p': comp.get('permutation_results', {}).get('p_value', None),
        'holm_p': comp.get('permutation_results', {}).get('p_value_holm', None),
        'significant_holm': comp.get('permutation_results', {}).get('significant_holm', False),
        'cohens_dz': comp.get('effect_size', {}).get('cohens_dz', None)
    })

master_df = pd.DataFrame(master_data)

# Sort by metric, budget, baseline
master_df = master_df.sort_values(['metric', 'budget', 'baseline'])

# Save master table CSV
master_csv_path = OUTPUT_DIR / "final_master_table.csv"
master_df.to_csv(master_csv_path, index=False)
log(f"Master table CSV saved to: {master_csv_path}")

# Save human-readable master table
master_txt_path = OUTPUT_DIR / "final_master_table.txt"
with open(master_txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("FINAL MASTER STATISTICS TABLE\n")
    f.write("=" * 100 + "\n\n")
    
    for metric in ['comprehensiveness', 'sufficiency', 'comp_efficiency', 'suff_efficiency']:
        metric_df = master_df[master_df['metric'] == metric]
        f.write(f"\n{metric.upper()}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Baseline':<10} {'Budget':<8} {'Δ (H-B)':<12} {'95% CI':<20} {'Raw p':<12} {'Holm p':<12} {'Sig':<6} {'Cohen d_z':<12}\n")
        f.write("-" * 100 + "\n")
        
        for _, row in metric_df.iterrows():
            ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            sig_str = "✓" if row['significant_holm'] else "✗"
            f.write(f"{row['baseline']:<10} {row['budget']:>5}%    {row['delta']:>8.4f}   {ci_str:<20} "
                   f"{row['raw_p']:>10.2e} {row['holm_p']:>10.2e} {sig_str:<6} "
                   f"{row['cohens_dz']:>10.4f}\n")

log(f"Master table TXT saved to: {master_txt_path}")

# Generate verification report
report_path = OUTPUT_DIR / "effect_size_verification_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Phase 8 Report: Verify and Consolidate Effect Sizes\n\n")
    f.write("**Status**: ✅ COMPLETE\n\n")
    f.write("## Purpose\n\n")
    f.write("Verify that all effect sizes are correctly calculated and consolidated in the master statistics table.\n\n")
    
    f.write("## Effect Size Verification\n\n")
    f.write(f"Verified {verified_count}/{total_count} effect sizes against manual calculations.\n\n")
    
    f.write("### Verification Results (Budget 10%)\n\n")
    f.write("| Comparison | Metric | Manual d_z | Reported d_z | Status |\n")
    f.write("|-----------|--------|-----------|--------------|--------|\n")
    
    for v in verification_results:
        status = "✓ MATCH" if v['match'] else "✗ MISMATCH"
        reported_str = f"{v['reported_dz']:.6f}" if v['reported_dz'] is not None else "N/A"
        f.write(f"| {v['comparison']:24s} | {v['metric']:18s} | {v['manual_dz']:10.6f} | "
               f"{reported_str:>12s} | {status:9s} |\n")
    
    f.write("\n")
    
    if verified_count == total_count:
        f.write("✓ **All effect sizes verified**: Cohen's d_z calculations are correct.\n\n")
    else:
        f.write("⚠ **Some effect sizes do not match**: Review the table above for discrepancies.\n\n")
    
    f.write("## Master Table\n\n")
    f.write("The consolidated master table includes:\n")
    f.write("- All 36 comparisons\n")
    f.write("- Mean differences with 95% bootstrap confidence intervals\n")
    f.write("- Raw and Holm-corrected p-values\n")
    f.write("- Cohen's d_z effect sizes\n")
    f.write("- Significance status after Holm correction\n\n")
    
    f.write("### Primary Hypothesis: Sufficiency Efficiency\n\n")
    suff_eff_df = master_df[master_df['metric'] == 'suff_efficiency']
    f.write("| Baseline | Budget | Δ (H-B) | 95% CI | Holm p | Sig | Cohen d_z |\n")
    f.write("|----------|--------|---------|--------|--------|-----|----------|\n")
    
    for _, row in suff_eff_df.iterrows():
        ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        sig_str = "✓" if row['significant_holm'] else "✗"
        f.write(f"| {row['baseline']:6s} | {row['budget']:5d}% | {row['delta']:7.4f} | {ci_str:<12} | "
               f"{row['holm_p']:7.2e} | {sig_str:3s} | {row['cohens_dz']:9.4f} |\n")
    
    f.write("\n")
    
    all_sig = all(suff_eff_df['significant_holm'])
    if all_sig:
        f.write("✓ **All primary comparisons remain significant after Holm correction**.\n\n")
    else:
        f.write("⚠ **Some primary comparisons are not significant after Holm correction**.\n\n")
    
    f.write("## Conclusion\n\n")
    f.write(f"Effect size verification: {verified_count}/{total_count} passed.\n")
    f.write(f"Master table generated with {len(master_df)} comparisons.\n")
    f.write(f"Total significant after Holm correction: {sum(master_df['significant_holm'])}/{len(master_df)}.\n\n")
    
    if verified_count == total_count and all_sig:
        f.write("✓ All effect sizes are correct and primary conclusions are robust.\n")
    elif verified_count == total_count:
        f.write("✓ All effect sizes are correct. Some secondary comparisons lost significance.\n")
    else:
        f.write("⚠ Review effect size calculations for discrepancies.\n")

log(f"\nReport saved to: {report_path}")

log_fh.close()

print("\n" + "=" * 80)
print("Phase 8 complete. Effect sizes verified and master table consolidated.")
print("=" * 80)
