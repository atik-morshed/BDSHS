"""
Phase 7 — Add Multiple-Comparison Correction (Holm-Bonferroni)
============================================================

Purpose: Apply Holm-Bonferroni correction to control family-wise error rate
across all statistical comparisons.

Steps:
1. Load statistical results from Phase 5
2. Extract p-values for all comparisons
3. Apply Holm-Bonferroni correction using statsmodels
4. Update master statistics table with corrected p-values
5. Report which comparisons remain significant after correction

Run: python statistics/phase7_multiple_comparison_correction.py
Outputs: results/statistics/corrected_statistical_results.json
         results/statistics/holm_correction_report.md
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

LOG_FILE = OUTPUT_DIR / "phase7_holm_correction_log.txt"
log_fh = open(LOG_FILE, "w", encoding="utf-8")

def log(message):
    """Write to both stdout and log file."""
    print(message)
    log_fh.write(message + "\n")
    log_fh.flush()

print("=" * 80)
log("=" * 80)
log("PHASE 7 — ADD MULTIPLE-COMPARISON CORRECTION (HOLM-BONFERRONI)")
log("=" * 80)

# Load Phase 5 results
phase5_path = OUTPUT_DIR / "rebuilt_statistical_results.json"
log(f"\nLoading Phase 5 results from: {phase5_path}")
with open(phase5_path, 'r', encoding='utf-8') as f:
    phase5_results = json.load(f)

log(f"Loaded {len(phase5_results)} comparisons")

# Extract p-values for correction
log(f"\n{'='*80}")
log("Extracting p-values for Holm-Bonferroni correction")
log(f"{'='*80}")

p_values = []
comparison_names = []

for comp in phase5_results:
    p_val = comp.get('permutation_results', {}).get('p_value', None)
    if p_val is not None:
        p_values.append(p_val)
        comparison_names.append(comp['comparison_name'])

log(f"Extracted {len(p_values)} p-values")
log(f"Min p-value: {min(p_values):.6f}")
log(f"Max p-value: {max(p_values):.6f}")

# Apply Holm-Bonferroni correction (manual implementation)
log(f"\n{'='*80}")
log("Applying Holm-Bonferroni correction")
log(f"{'='*80}")

alpha = 0.05

# Manual Holm-Bonferroni implementation
# 1. Sort p-values in ascending order
# 2. Compare each p-value to alpha / (m - i + 1) where m = total comparisons, i = rank
# 3. Reject all hypotheses with p-values less than their adjusted threshold

p_array = np.array(p_values)
m = len(p_array)
sorted_indices = np.argsort(p_array)
sorted_p = p_array[sorted_indices]

reject = np.zeros(m, dtype=bool)
pvals_corrected = np.zeros(m)

for i in range(m):
    # Holm-Bonferroni adjusted alpha
    adjusted_alpha = alpha / (m - i)
    
    # Holm-Bonferroni corrected p-value
    pvals_corrected[i] = min(1.0, sorted_p[i] * (m - i))
    
    # Check significance
    if sorted_p[i] <= adjusted_alpha:
        reject[i] = True
    else:
        # Once we fail to reject, all remaining larger p-values also fail
        for j in range(i, m):
            reject[j] = False
        break

# Restore original order
reject_original = np.zeros(m, dtype=bool)
pvals_corrected_original = np.zeros(m)
for i, idx in enumerate(sorted_indices):
    reject_original[idx] = reject[i]
    pvals_corrected_original[idx] = pvals_corrected[i]

reject = reject_original
pvals_corrected = pvals_corrected_original

log(f"Alpha level: {alpha}")
log(f"Correction method: Holm-Bonferroni (manual implementation)")
log(f"Total comparisons: {m}")
log(f"Significant before correction: {sum(p < alpha for p in p_values)}")
log(f"Significant after correction: {sum(reject)}")

# Update results with corrected p-values
log(f"\n{'='*80}")
log("Updating results with corrected p-values")
log(f"{'='*80}")

updated_results = []
for i, comp in enumerate(phase5_results):
    updated_comp = comp.copy()
    
    # Add Holm-corrected p-value
    if 'permutation_results' in updated_comp:
        updated_comp['permutation_results']['p_value_holm'] = float(pvals_corrected[i])
        updated_comp['permutation_results']['significant_holm'] = bool(reject[i])
    else:
        # If no permutation results, create structure
        updated_comp['permutation_results'] = {
            'p_value': None,
            'p_value_holm': float(pvals_corrected[i]),
            'significant_holm': bool(reject[i])
        }
    
    updated_results.append(updated_comp)

# Save corrected results
corrected_path = OUTPUT_DIR / "corrected_statistical_results.json"
with open(corrected_path, 'w', encoding='utf-8') as f:
    json.dump(updated_results, f, indent=2)

log(f"\nCorrected results saved to: {corrected_path}")

# Generate summary table
log(f"\n{'='*80}")
log("Generating Holm correction summary table")
log(f"{'='*80}")

summary_data = []
for comp in updated_results:
    summary_data.append({
        'comparison': comp['comparison_name'],
        'metric': comp['metric_name'],
        'baseline': comp['baseline_strategy'],
        'budget': comp['budget'],
        'raw_p': comp.get('permutation_results', {}).get('p_value', None),
        'holm_p': comp.get('permutation_results', {}).get('p_value_holm', None),
        'significant_raw': comp.get('permutation_results', {}).get('p_value', 1) < alpha,
        'significant_holm': comp.get('permutation_results', {}).get('significant_holm', False),
        'cohen_dz': comp.get('effect_size', {}).get('cohens_dz', None)
    })

summary_df = pd.DataFrame(summary_data)
summary_path = OUTPUT_DIR / "holm_correction_summary.csv"
summary_df.to_csv(summary_path, index=False)
log(f"Summary table saved to: {summary_path}")

# Generate markdown report
report_path = OUTPUT_DIR / "holm_correction_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Phase 7 Report: Multiple-Comparison Correction (Holm-Bonferroni)\n\n")
    f.write("**Status**: ✅ COMPLETE\n\n")
    f.write("## Purpose\n\n")
    f.write("Apply Holm-Bonferroni correction to control family-wise error rate across all statistical comparisons.\n\n")
    
    f.write("## Correction Details\n\n")
    f.write(f"- **Method**: Holm-Bonferroni\n")
    f.write(f"- **Alpha level**: {alpha}\n")
    f.write(f"- **Total comparisons**: {len(p_values)}\n")
    f.write(f"- **Significant before correction**: {sum(p < alpha for p in p_values)}\n")
    f.write(f"- **Significant after correction**: {sum(reject)}\n\n")
    
    f.write("## Results by Metric\n\n")
    
    for metric in ['comprehensiveness', 'sufficiency', 'comp_efficiency', 'suff_efficiency']:
        metric_df = summary_df[summary_df['metric'] == metric]
        f.write(f"### {metric}\n\n")
        f.write("| Comparison | Baseline | Budget | Raw p | Holm p | Sig Raw | Sig Holm | Cohen's d_z |\n")
        f.write("|-----------|----------|--------|-------|--------|---------|----------|------------|\n")
        
        for _, row in metric_df.iterrows():
            raw_sig = "✓" if row['significant_raw'] else "✗"
            holm_sig = "✓" if row['significant_holm'] else "✗"
            cohen_dz = row['cohen_dz'] if row['cohen_dz'] is not None else 0.0
            f.write(f"| {row['comparison']:24s} | {row['baseline']:6s} | {row['budget']:5d}% | "
                   f"{row['raw_p']:7.4e} | {row['holm_p']:7.4e} | {raw_sig:7s} | {holm_sig:8s} | "
                   f"{cohen_dz:9.4f} |\n")
        
        f.write("\n")
    
    f.write("## Primary Hypothesis: Sufficiency Efficiency\n\n")
    f.write("The primary hypothesis is that hierarchical aggregation improves sufficiency efficiency relative to flat baselines.\n\n")
    
    suff_eff_df = summary_df[summary_df['metric'] == 'suff_efficiency']
    f.write("### Sufficiency Efficiency Results\n\n")
    f.write("| Baseline | Budget | Raw p | Holm p | Sig Raw | Sig Holm | Cohen's d_z |\n")
    f.write("|----------|--------|-------|--------|---------|----------|------------|\n")
    
    for _, row in suff_eff_df.iterrows():
        raw_sig = "✓" if row['significant_raw'] else "✗"
        holm_sig = "✓" if row['significant_holm'] else "✗"
        cohen_dz = row['cohen_dz'] if row['cohen_dz'] is not None else 0.0
        f.write(f"| {row['baseline']:6s} | {row['budget']:5d}% | "
               f"{row['raw_p']:7.4e} | {row['holm_p']:7.4e} | {raw_sig:7s} | {holm_sig:8s} | "
               f"{cohen_dz:9.4f} |\n")
    
    f.write("\n")
    
    all_sig = all(suff_eff_df['significant_holm'])
    if all_sig:
        f.write("✓ **All primary comparisons remain significant after Holm-Bonferroni correction**.\n\n")
        f.write("The conclusion that hierarchical aggregation improves sufficiency efficiency is robust to multiple-comparison correction.\n")
    else:
        f.write("⚠ **Some primary comparisons are not significant after Holm-Bonferroni correction**.\n\n")
        f.write("Review the table above to see which comparisons remain significant.\n")
    
    f.write("\n## Conclusion\n\n")
    f.write(f"Holm-Bonferroni correction was applied to {len(p_values)} comparisons. ")
    f.write(f"{sum(reject)} comparisons remain significant at α = {alpha}.\n\n")
    
    if all_sig:
        f.write("The primary experimental conclusions are robust to multiple-comparison correction.\n")
    else:
        f.write("Some comparisons are not significant after correction. Review the summary table for details.\n")

log(f"\nReport saved to: {report_path}")

log_fh.close()

print("\n" + "=" * 80)
print("Phase 7 complete. Holm-Bonferroni correction applied.")
print("=" * 80)
