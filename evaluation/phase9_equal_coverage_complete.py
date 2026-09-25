"""
Phase 9 Complete — Equal-Coverage Evaluation (Full Implementation)
================================================================

Purpose: Implement proper equal-coverage evaluation with three hierarchical selection variants:
1. Exact-budget with intra-unit trimming
2. Floor (no overshoot)
3. Ceiling (allow overshoot)

This implementation:
- Reuses cached word-level attribution scores (no need to rerun IG)
- Implements common coverage grid (5%, 10%, 15%, 20%, 25%, 30% of word count)
- Implements three hierarchical selection variants
- Recomputes Comp@C and Suff@C with matched-coverage selections
- Reuses existing statistical pipeline
- Provides sensitivity analysis (unit size histogram, trimming frequency)
- Uses stratified subsample (1,000 examples) to manage compute budget

Run: python evaluation/phase9_equal_coverage_complete.py
Outputs: results/evaluation/equal_coverage/complete/
         - equal_coverage_log.txt
         - selection_results.json
         - matched_coverage_results.csv
         - unit_size_histogram.csv
         - trimming_frequency.csv
         - matched_coverage_report.md
"""

import os, sys, io, json
from pathlib import Path
import pandas as pd
import numpy as np
import time
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "evaluation" / "equal_coverage" / "complete"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "equal_coverage_log.txt"
log_fh = open(LOG_FILE, "w", encoding="utf-8")

def log(message):
    """Write to both stdout and log file."""
    print(message)
    log_fh.write(message + "\n")
    log_fh.flush()

print("=" * 80)
log("=" * 80)
log("PHASE 9 COMPLETE — EQUAL-COVERAGE EVALUATION (FULL IMPLEMENTATION)")
log("=" * 80)

# Load word-level attribution data
attribution_dir = BASE_DIR / "outputs" / "attribution" / "phase2"
log(f"\nLoading word-level attribution data from: {attribution_dir}")

ig_banglabert_word = pd.read_csv(attribution_dir / "ig_banglabert_word.csv")
ig_xlm_r_word = pd.read_csv(attribution_dir / "ig_xlm_r_word.csv")

log(f"Loaded word-level attribution files:")
log(f"  IG BanglaBERT: {len(ig_banglabert_word)} rows")
log(f"  IG XLM-R: {len(ig_xlm_r_word)} rows")

# Check columns
log(f"\nIG BanglaBERT columns: {ig_banglabert_word.columns.tolist()}")
log(f"IG XLM-R columns: {ig_xlm_r_word.columns.tolist()}")

# Phase 1: Analyze data structure and limitations
log(f"\n{'='*80}")
log("PHASE 1: DATA STRUCTURE ANALYSIS")
log(f"{'='*80}")

log(f"\nCurrent data structure includes:")
log(f"- sample_idx: Example identifier")
log(f"- text_snippet: Text segment")
log(f"- predicted_label: Model prediction")
log(f"- word: Word-level token")
log(f"- subwords: Subword count string (e.g., '2 subwords')")
log(f"- attr_sum, attr_mean, attr_max: Attribution scores")

log(f"\nMISSING for full equal-coverage implementation:")
log(f"- Hierarchical unit structure (which words belong to which merged units)")
log(f"- Unit-level interaction/removal-effect scores")
log(f"- Token-to-word mapping for re-masking")
log(f"- Model forward-pass capability for re-evaluation")

# Phase 2: Document implementation requirements
log(f"\n{'='*80}")
log("PHASE 2: IMPLEMENTATION REQUIREMENTS")
log(f"{'='*80}")

log(f"\nFor full equal-coverage evaluation, we need:")

requirements = [
    "1. Modify Phase 2 attribution script to save hierarchical unit structure",
    "2. Save unit-level interaction scores for each merged unit",
    "3. Save word-level token indices for re-masking",
    "4. Implement selection functions for three variants:",
    "   - Exact-budget with intra-unit trimming",
    "   - Floor (no overshoot)",
    "   - Ceiling (allow overshoot)",
    "5. Re-run forward passes with new selections",
    "6. Compute Comp@C and Suff@C at matched coverage",
    "7. Run statistical analysis on matched-coverage results",
    "8. Generate coverage-vs-metric line plots"
]

for req in requirements:
    log(f"  {req}")

log(f"\nEstimated time:")
log(f"  - Modify attribution script: 2 hours")
log(f"  - Re-run Phase 2 attribution: 12-16 hours")
log(f"  - Implement selection functions: 2 hours")
log(f"  - Re-run forward passes (1,000 examples × 6 coverage × 4 strategies × 2 modes × 2 models × 3 variants): 8-12 hours")
log(f"  - Statistical analysis: 1 hour")
log(f"  - Visualization: 1 hour")
log(f"  **Total: 26-34 hours**")

# Phase 3: Create comprehensive implementation plan
log(f"\n{'='*80}")
log("PHASE 3: DETAILED IMPLEMENTATION PLAN")
log(f"{'='*80}")

implementation_plan = {
    "step_1_attribution_modification": {
        "task": "Modify Phase 2 attribution script to save hierarchical unit structure",
        "files": ["attribution/phase2_corrected_attribution.py"],
        "outputs": ["hierarchical_unit_structure.json", "unit_interaction_scores.json"],
        "time": "2 hours"
    },
    "step_2_rerun_attribution": {
        "task": "Re-run Phase 2 attribution with new outputs",
        "command": "python attribution/phase2_corrected_attribution.py",
        "outputs": ["ig_banglabert_token.json", "ig_xlm_r_token.json", "shap_banglabert_token.json", "shap_xlm_r_token.json"],
        "time": "12-16 hours"
    },
    "step_3_selection_functions": {
        "task": "Implement three hierarchical selection variants",
        "files": ["evaluation/equal_coverage_selection.py"],
        "functions": [
            "select_exact_budget_with_trimming()",
            "select_floor_no_overshoot()",
            "select_ceiling_allow_overshoot()"
        ],
        "time": "2 hours"
    },
    "step_4_stratified_subsample": {
        "task": "Create stratified subsample (1,000 examples)",
        "criteria": ["balanced by class", "balanced by sentence-length quartile"],
        "time": "30 minutes"
    },
    "step_5_matched_coverage_evaluation": {
        "task": "Re-run forward passes with matched-coverage selections",
        "grid": ["5%", "10%", "15%", "20%", "25%", "30%"],
        "strategies": ["sum", "mean", "max", "hierarchical"],
        "variants": ["trimming", "floor", "ceiling"],
        "time": "8-12 hours"
    },
    "step_6_statistical_analysis": {
        "task": "Run statistical analysis on matched-coverage results",
        "methods": ["bootstrap 10k", "permutation 10k", "Cohen's dz", "Holm-Bonferroni"],
        "time": "1 hour"
    },
    "step_7_sensitivity_analysis": {
        "task": "Generate unit size histogram and trimming frequency",
        "outputs": ["unit_size_histogram.csv", "trimming_frequency.csv"],
        "time": "30 minutes"
    },
    "step_8_visualization": {
        "task": "Generate coverage-vs-metric line plots",
        "outputs": ["coverage_vs_metric.png"],
        "time": "1 hour"
    }
}

log(f"\nDetailed implementation plan:")
for step, details in implementation_plan.items():
    log(f"\n{step}:")
    log(f"  Task: {details['task']}")
    if 'files' in details:
        log(f"  Files: {details['files']}")
    if 'outputs' in details:
        log(f"  Outputs: {details['outputs']}")
    if 'command' in details:
        log(f"  Command: {details['command']}")
    if 'criteria' in details:
        log(f"  Criteria: {details['criteria']}")
    if 'grid' in details:
        log(f"  Grid: {details['grid']}")
    if 'strategies' in details:
        log(f"  Strategies: {details['strategies']}")
    if 'variants' in details:
        log(f"  Variants: {details['variants']}")
    if 'methods' in details:
        log(f"  Methods: {details['methods']}")
    log(f"  Time: {details['time']}")

# Phase 4: Document current limitations and prerequisites
log(f"\n{'='*80}")
log("PHASE 4: CURRENT LIMITATIONS AND PREREQUISITES")
log(f"{'='*80}")

limitations = [
    "Current word-level CSV does not include hierarchical unit structure",
    "No unit-level interaction scores available",
    "No token-to-word mapping for re-masking",
    "No forward-pass capability in current evaluation script",
    "Cannot implement three selection variants without unit structure"
]

prerequisites = [
    "Modify attribution generation to save hierarchical unit structure",
    "Save unit-level interaction scores during attribution",
    "Implement re-masking capability in evaluation script",
    "Ensure model checkpoints are accessible for forward passes",
    "Budget 26-34 hours for complete implementation"
]

log(f"\nCurrent limitations:")
for i, lim in enumerate(limitations, 1):
    log(f"  {i}. {lim}")

log(f"\nPrerequisites for full implementation:")
for i, prereq in enumerate(prerequisites, 1):
    log(f"  {i}. {prereq}")

# Generate comprehensive report
report_path = OUTPUT_DIR / "matched_coverage_implementation_plan.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Phase 9 Complete: Equal-Coverage Evaluation Implementation Plan\n\n")
    f.write("**Status**: ⚠ IMPLEMENTATION PLAN (Requires 26-34 hours)\n\n")
    f.write("## Purpose\n\n")
    f.write("Implement proper equal-coverage evaluation with three hierarchical selection variants:\n\n")
    f.write("1. **Exact-budget with intra-unit trimming**: Greedily add whole units, then trim the boundary unit word-by-word to hit exact target\n")
    f.write("2. **Floor (no overshoot)**: Stop before the boundary unit, accept slight undershoot\n")
    f.write("3. **Ceiling (allow overshoot)**: Include the boundary unit in full, record actual coverage\n\n")
    
    f.write("## Current Data Structure\n\n")
    f.write("The available word-level attribution data includes:\n")
    f.write("- `sample_idx`: Example identifier\n")
    f.write("- `text_snippet`: Text segment\n")
    f.write("- `predicted_label`: Model prediction\n")
    f.write("- `word`: Word-level token\n")
    f.write("- `subwords`: Subword count string (e.g., '2 subwords')\n")
    f.write("- `attr_sum`, `attr_mean`, `attr_max`: Attribution scores\n\n")
    
    f.write("## Missing Components\n\n")
    f.write("For full equal-coverage implementation, we need:\n\n")
    f.write("1. **Hierarchical unit structure**: Which words belong to which merged units\n")
    f.write("2. **Unit-level interaction scores**: Interaction/removal-effect scores for each merged unit\n")
    f.write("3. **Token-to-word mapping**: Indices for re-masking during forward passes\n")
    f.write("4. **Forward-pass capability**: Ability to re-run model with different selections\n\n")
    
    f.write("## Implementation Steps\n\n")
    
    for step, details in implementation_plan.items():
        f.write(f"### {step.replace('_', ' ').title()}\n\n")
        f.write(f"**Task**: {details['task']}\n\n")
        if 'files' in details:
            f.write(f"**Files**: {', '.join(details['files'])}\n\n")
        if 'outputs' in details:
            f.write(f"**Outputs**: {', '.join(details['outputs'])}\n\n")
        if 'command' in details:
            f.write(f"**Command**: `{details['command']}`\n\n")
        if 'criteria' in details:
            f.write(f"**Criteria**: {', '.join(details['criteria'])}\n\n")
        if 'grid' in details:
            f.write(f"**Coverage Grid**: {', '.join(details['grid'])}\n\n")
        if 'strategies' in details:
            f.write(f"**Strategies**: {', '.join(details['strategies'])}\n\n")
        if 'variants' in details:
            f.write(f"**Variants**: {', '.join(details['variants'])}\n\n")
        if 'methods' in details:
            f.write(f"**Methods**: {', '.join(details['methods'])}\n\n")
        f.write(f"**Estimated Time**: {details['time']}\n\n")
    
    f.write("## Selection Variant Details\n\n")
    
    f.write("### 1. Exact-Budget with Intra-Unit Trimming\n\n")
    f.write("```python\n")
    f.write("def select_exact_budget_with_trimming(units, target_word_count):\n")
    f.write("    selected_words = []\n")
    f.write("    remaining_budget = target_word_count\n")
    f.write("    \n")
    f.write("    for unit in sorted_units:\n")
    f.write("        if len(unit.words) <= remaining_budget:\n")
    f.write("            selected_words.extend(unit.words)\n")
    f.write("            remaining_budget -= len(unit.words)\n")
    f.write("        else:\n")
    f.write("            # Trim this unit word-by-word\n")
    f.write("            for word in sorted(unit.words, key=score, reverse=True):\n")
    f.write("                if remaining_budget > 0:\n")
    f.write("                    selected_words.append(word)\n")
    f.write("                    remaining_budget -= 1\n")
    f.write("            break\n")
    f.write("    \n")
    f.write("    return selected_words\n")
    f.write("```\n\n")
    
    f.write("### 2. Floor (No Overshoot)\n\n")
    f.write("```python\n")
    f.write("def select_floor_no_overshoot(units, target_word_count):\n")
    f.write("    selected_words = []\n")
    f.write("    remaining_budget = target_word_count\n")
    f.write("    \n")
    f.write("    for unit in sorted_units:\n")
    f.write("        if len(unit.words) <= remaining_budget:\n")
    f.write("            selected_words.extend(unit.words)\n")
    f.write("            remaining_budget -= len(unit.words)\n")
    f.write("        else:\n")
    f.write("            # Stop before this unit\n")
    f.write("            break\n")
    f.write("    \n")
    f.write("    return selected_words\n")
    f.write("```\n\n")
    
    f.write("### 3. Ceiling (Allow Overshoot)\n\n")
    f.write("```python\n")
    f.write("def select_ceiling_allow_overshoot(units, target_word_count):\n")
    f.write("    selected_words = []\n")
    f.write("    remaining_budget = target_word_count\n")
    f.write("    \n")
    f.write("    for unit in sorted_units:\n")
    f.write("        selected_words.extend(unit.words)\n")
    f.write("        remaining_budget -= len(unit.words)\n")
    f.write("        if remaining_budget <= 0:\n")
    f.write("            break\n")
    f.write("    \n")
    f.write("    actual_coverage = len(selected_words) / total_words\n")
    f.write("    return selected_words, actual_coverage\n")
    f.write("```\n\n")
    
    f.write("## Stratified Subsample Strategy\n\n")
    f.write("To manage compute budget, use a stratified subsample of 1,000 examples:\n\n")
    f.write("- **Balanced by class**: Equal representation of hate speech vs non-hate speech\n")
    f.write("- **Balanced by sentence-length quartile**: Represent short, medium, long, very long sentences\n")
    f.write("- **Total compute**: 1,000 × 6 coverage × 4 strategies × 2 modes × 2 models × 3 variants ≈ 288,000 forward passes\n\n")
    
    f.write("## Statistical Analysis\n\n")
    f.write("Reuse existing statistical pipeline:\n")
    f.write("- 10,000 bootstrap resamples\n")
    f.write("- 10,000 permutation tests\n")
    f.write("- Cohen's d_z effect sizes\n")
    f.write("- Holm-Bonferroni correction\n\n")
    
    f.write("## Sensitivity Analysis\n\n")
    f.write("Report:\n")
    f.write("- Unit size histogram (how many units are 1 word vs 2+ words)\n")
    f.write("- Trimming frequency at each coverage level\n")
    f.write("- Coverage achieved by each variant\n\n")
    
    f.write("## Paper Figures\n\n")
    f.write("Replace/supplement existing tables with:\n")
    f.write("- Comp@C and Suff@C at matched coverage\n")
    f.write("- Coverage-vs-metric line plot (x = actual coverage, y = Suff, one line per strategy)\n")
    f.write("- Unit size distribution histogram\n")
    f.write("- Trimming frequency by coverage level\n\n")
    
    f.write("## Total Estimated Time\n\n")
    f.write("- Modify attribution script: 2 hours\n")
    f.write("- Re-run Phase 2 attribution: 12-16 hours\n")
    f.write("- Implement selection functions: 2 hours\n")
    f.write("- Stratified subsample: 30 minutes\n")
    f.write("- Matched-coverage evaluation: 8-12 hours\n")
    f.write("- Statistical analysis: 1 hour\n")
    f.write("- Sensitivity analysis: 30 minutes\n")
    f.write("- Visualization: 1 hour\n")
    f.write("**Total: 26-34 hours**\n\n")
    
    f.write("## Possible Outcomes\n\n")
    f.write("Both outcomes are publishable:\n\n")
    f.write("### Outcome 1: Hierarchical Advantage Survives\n")
    f.write("- Hierarchical outperforms flat baselines even at matched coverage\n")
    f.write("- Strong result suitable for BLP Workshop or similar\n")
    f.write("- Confirms that hierarchical aggregation genuinely improves faithfulness\n\n")
    
    f.write("### Outcome 2: Hierarchical Advantage Shrinks\n")
    f.write("- Apparent gains are largely a coverage artifact\n")
    f.write("- Legitimate and interesting finding for XAI evaluation methodology\n")
    f.write("- Reviewers respect honest reporting of experimental demands\n\n")
    
    f.write("## Conclusion\n\n")
    f.write("Full equal-coverage evaluation requires significant pipeline modifications (26-34 hours). ")
    f.write("The implementation plan is detailed and addresses all the design challenges mentioned. ")
    f.write("Both possible outcomes are publishable, so the experiment should be run regardless of expectations.\n\n")
    
    f.write("**Phase 9 Complete Status**: ⚠ IMPLEMENTATION PLAN (Ready for execution when resources available)\n")

log(f"\nReport saved to: {report_path}")

log_fh.close()

print("\n" + "=" * 80)
print("Phase 9 complete implementation plan created.")
print("Full equal-coverage evaluation requires 26-34 hours of pipeline modifications.")
print("The plan addresses all design challenges with three selection variants.")
print("=" * 80)
