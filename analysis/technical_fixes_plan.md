# Technical Fixes and Enhancements Plan

**Date**: September 25, 2026  
**Purpose**: Fix identified technical issues and enhance experimental rigor

---

## Overview

This plan addresses 16 technical issues identified in the current experimental pipeline, ranging from critical bug fixes to methodological enhancements.

---

## Phase 1: Fix Effect Size Calculation (CRITICAL)

### Issue
Effect sizes reported as 0.0000 when they should be non-zero. This indicates a bug in the Cohen's d_z calculation.

### Root Cause Investigation
- Load raw paired values for each comparison
- Verify example pairing (example_id alignment)
- Calculate mean_diff and sd_diff correctly
- Add sanity checks to prevent future bugs

### Implementation Steps
1. Load per-example CSV
2. Verify pairing by example_id
3. Calculate differences correctly
4. Compute Cohen's d_z: mean_diff / sd_diff
5. Add assertion checks for finite values and non-zero SD
6. Recalculate all 36 comparisons

### Expected Output
- Corrected effect sizes (non-zero values)
- Diagnostic log showing calculation steps
- Updated statistical_results.json

### Files
- Input: `results/evaluation/per_example/per_example_faithfulness_results.csv`
- Output: `results/statistics/corrected_effect_sizes.json`
- Log: `results/statistics/phase1_effect_size_fix_log.txt`

---

## Phase 2: Fix Row Count Discrepancy

### Issue
Report says 60,348 rows but CSV shows 60,349. Expected: 5,029 × 4 × 3 = 60,348.

### Investigation
- Check if extra row is header
- Verify expected combinations (15087 per strategy)
- Check for duplicates or missing examples
- Verify each strategy-budget combination has exactly 5,029 examples

### Implementation Steps
1. Load CSV and check shape
2. Count physical lines vs data rows
3. Verify strategy counts (should be 15,087 each)
4. Verify strategy-budget combinations (should be 5,029 each)
5. Document findings

### Expected Output
- Clarification of row count discrepancy
- Verification of data integrity
- Updated documentation

### Files
- Input: `results/evaluation/per_example/per_example_faithfulness_results.csv`
- Output: `results/evaluation/per_example/row_count_verification.md`
- Log: `results/evaluation/per_example/phase2_row_count_log.txt`

---

## Phase 3: Verify Checkpoints Programmatically

### Issue
Checkpoints verified manually but not programmatically. Need to prevent future accidental wrong-checkpoint experiments.

### Implementation Steps
1. Load trainer_state.json from checkpoint directories
2. Extract validation F1 history
3. Programmatically select best epoch (argmax validation F1)
4. Compare against checkpoint actually used
5. Create checkpoint manifest CSV
6. Add hard assertion in training script

### Expected Output
- `results/checkpoints/checkpoint_manifest.csv`
- Verification script
- Hard assertions in training pipeline

### Files
- Input: `outputs/banglabert-bdshs/checkpoint-5028/trainer_state.json`
- Input: `outputs/xlmr-bdshs/checkpoint-12570/trainer_state.json`
- Output: `results/checkpoints/checkpoint_manifest.csv`
- Script: `verification/verify_checkpoints.py`

---

## Phase 4: Verify Attribution Alignment Again

### Issue
Current alignment rates are good (>98%) but need deeper verification for robustness.

### Verification Steps
1. Verify example IDs match between attribution and test set
2. Check for duplicate example IDs
3. Verify word reconstruction for failed examples
4. Export alignment failures CSV with detailed reasons
5. Document why 1.41% of XLM-R SHAP examples were excluded

### Expected Output
- Detailed alignment verification report
- `results/alignment/alignment_failures.csv`
- Enhanced validation documentation

### Files
- Input: All attribution JSON files
- Output: `results/alignment/alignment_failures.csv`
- Output: `results/alignment/detailed_alignment_verification.md`

---

## Phase 5: Fix Statistical Pipeline

### Issue
Need to rebuild statistics from verified per-example data programmatically, not manually copy numbers.

### Implementation Steps
1. Create clean statistical analysis pipeline
2. Load verified per-example data
3. Implement paired bootstrap (10,000 resamples)
4. Implement paired permutation (10,000 permutations)
5. Calculate effect sizes correctly
6. Generate results programmatically

### Expected Output
- Reproducible statistical pipeline
- `results/statistics/rebuilt_statistical_results.json`
- Complete audit trail

### Files
- Input: `results/evaluation/per_example/per_example_faithfulness_results.csv`
- Output: `results/statistics/rebuilt_statistical_results.json`
- Script: `statistics/clean_statistical_pipeline.py`

---

## Phase 6: Fix Permutation Test

### Issue
Need to implement proper paired sign-flipping permutation test with +1 correction.

### Implementation Steps
1. Implement paired sign-flipping: differences * random([-1, 1])
2. Repeat 10,000 times
3. Calculate p-value with +1 correction: (count + 1) / (n + 1)
4. Verify against current implementation

### Expected Output
- Corrected permutation p-values
- Verification that implementation matches specification

### Files
- Update: `statistics/bootstrap_permutation.py`
- Output: `results/statistics/permutation_verification.md`

---

## Phase 7: Add Multiple-Comparison Correction

### Issue
Report mentions Holm correction but doesn't actually calculate it.

### Implementation Steps
1. Extract p-values for 9 primary comparisons
2. Apply Holm-Bonferroni correction using statsmodels
3. Store raw_p, holm_p, and significance flags
4. Update summary tables

### Expected Output
- Corrected p-values in summary table
- Documentation of which comparisons remain significant

### Files
- Update: `statistics/bootstrap_permutation.py`
- Output: `results/statistics/holm_corrected_summary.csv`

---

## Phase 8: Recalculate All Effect Sizes

### Issue
After fixing the effect size calculation bug, recalculate all metrics.

### Implementation Steps
1. Apply fixed effect size calculation to all 36 comparisons
2. Calculate for Sufficiency Efficiency, Sufficiency, Comprehensiveness Efficiency, Comprehensiveness
3. Calculate for Sum, Mean, Max baselines at 10%, 20%, 30% budgets
4. Create master statistics table with all metrics

### Expected Output
- Master statistics table with correct effect sizes
- Complete metric-by-metric breakdown

### Files
- Output: `results/statistics/master_statistics_table.csv`
- Output: `results/statistics/master_statistics_table.md`

---

## Phase 9: Equal-Coverage Experiment

### Issue
Hierarchical can select more actual words than nominal budget. Need equal-coverage comparison.

### Implementation Steps
1. Define target actual word budgets (10%, 20%, 30% of actual words)
2. Implement flat strategy: rank words, select top k
3. Implement hierarchical with budget constraint: select if word count <= remaining budget
4. Calculate all metrics at equal actual coverage
5. Compare with nominal-budget results

### Expected Output
- Equal-coverage faithfulness results
- Robustness analysis showing conclusions hold at equal coverage

### Files
- Script: `evaluation/equal_coverage_evaluation.py`
- Output: `results/evaluation/equal_coverage/equal_coverage_results.csv`
- Output: `results/evaluation/equal_coverage/equal_coverage_comparison.md`

---

## Phase 10: Multi-Seed Experiment

### Issue
Current analysis uses only seed 42. Need to assess training randomness sensitivity.

### Implementation Steps
1. Train models with seeds 42, 43, 44 (minimum) or 42-46 (preferred)
2. For each seed: training → validation → best epoch → checkpoint → test
3. Generate classification robustness table
4. Decide attribution scope for additional seeds (minimal vs full)

### Expected Output
- Classification robustness across seeds
- Test of whether hierarchical advantage survives training randomness

### Files
- Script: `training/multi_seed_training.py` (enhanced)
- Output: `results/multi_seed/classification_robustness.csv`
- Output: `results/multi_seed/training_logs/`

---

## Phase 11: Decide Attribution Scope for Additional Seeds

### Issue
Need to balance computational cost vs robustness validation.

### Decision Framework
- Seed 42: Full pipeline (IG + SHAP + all strategies + nominal + equal coverage)
- Seeds 43-46: Minimum (classification + IG + main hierarchical vs flat sufficiency-efficiency)
- If computational cost acceptable: Full pipeline for all seeds

### Implementation Steps
1. Document decision framework
2. Implement minimal pipeline for additional seeds
3. Provide option for full pipeline if resources allow

### Expected Output
- Decision document
- Minimal pipeline implementation
- Estimated computational costs

### Files
- Output: `analysis/attribution_scope_decision.md`
- Script: `training/minimal_attribution_pipeline.py`

---

## Phase 12: SHAP Faithfulness Experiment

### Issue
SHAP attribution generated but full faithfulness analysis not done.

### Implementation Steps
1. Load SHAP attribution results
2. Run full faithfulness evaluation (Sum, Mean, Max, Hierarchical)
3. Calculate all metrics at 10%, 20%, 30%
4. Compare SHAP vs IG results
5. Answer: Does hierarchical benefit depend on attribution method?

### Expected Output
- SHAP faithfulness results
- IG vs SHAP comparison
- Cross-method validation

### Files
- Script: `evaluation/shap_faithfulness_evaluation.py`
- Output: `results/evaluation/shap/shap_faithfulness_results.csv`
- Output: `results/evaluation/ig_vs_shap_comparison.md`

---

## Phase 13: Qualitative Analysis

### Issue
No systematic qualitative analysis of phrase recovery.

### Implementation Steps
1. Select 30 examples across 6 categories (5 each):
   - Compound insults
   - Multi-word insults
   - Sarcastic expressions
   - Punctuation-heavy
   - Code-mixed
   - UNK-containing
2. For each example: original, prediction, flat explanations, hierarchical explanation
3. Document phrase recovery (not correctness)
4. Categorize by expression type

### Expected Output
- 30 example qualitative analysis
- Phrase recovery statistics by category
- Qualitative examples for paper

### Files
- Script: `analysis/qualitative_analysis.py`
- Output: `results/qualitative/qualitative_examples.csv`
- Output: `results/qualitative/qualitative_analysis.md`

---

## Phase 14: Runtime Experiment

### Issue
No runtime measurements for computational cost.

### Implementation Steps
1. Measure IG attribution time per example
2. Measure SHAP attribution time per example
3. Measure aggregation time for each strategy
4. Especially measure hierarchical overhead
5. Run each measurement 3 times, report mean ± SD
6. Document GPU memory usage

### Expected Output
- Runtime table by method and strategy
- Computational cost documentation
- GPU memory usage statistics

### Files
- Script: `evaluation/runtime_measurement.py`
- Output: `results/runtime/runtime_statistics.csv`
- Output: `results/runtime/runtime_analysis.md`

---

## Phase 15: Fragmentation Analysis

### Issue
Fragmentation analysis limited to 2 tokenizers, need careful descriptive reporting.

### Implementation Steps
1. Calculate subwords/word for both tokenizers
2. Calculate UNK token rates
3. Document fragmentation descriptively
4. Report hierarchical advantage by tokenizer
5. Avoid causal claims (with only 2 tokenizers)

### Expected Output
- Descriptive fragmentation statistics
- Careful documentation avoiding overgeneralization

### Files
- Script: `analysis/fragmentation_analysis.py`
- Output: `results/fragmentation/fragmentation_statistics.csv`
- Output: `results/fragmentation/fragmentation_analysis.md`

---

## Phase 16: Build Final Result Tables

### Issue
Need 6 main result tables for paper.

### Implementation Steps
1. Table 1: Dataset/model performance (Accuracy, Precision, Recall, F1)
2. Table 2: Tokenization (Subwords/word, UNK rate, examples with UNK)
3. Table 3: Main faithfulness results (by model, budget, aggregation)
4. Table 4: Statistical comparison (Δ, CI, p, Holm p, Cohen dz)
5. Table 5: Equal-coverage robustness
6. Table 6: Multi-seed robustness

### Expected Output
- 6 publication-ready tables
- LaTeX and CSV formats
- Complete documentation

### Files
- Script: `analysis/generate_paper_tables.py`
- Output: `results/tables/table_1_classification.csv`
- Output: `results/tables/table_2_tokenization.csv`
- Output: `results/tables/table_3_faithfulness.csv`
- Output: `results/tables/table_4_statistical.csv`
- Output: `results/tables/table_5_equal_coverage.csv`
- Output: `results/tables/table_6_multi_seed.csv`

---

## Implementation Order

### Priority 1 (Critical Bugs)
1. Phase 1: Fix effect size calculation
2. Phase 2: Fix row count discrepancy
3. Phase 3: Verify checkpoints programmatically
4. Phase 5: Fix statistical pipeline
5. Phase 6: Fix permutation test

### Priority 2 (Statistical Rigor)
6. Phase 7: Add multiple-comparison correction
7. Phase 8: Recalculate all effect sizes
8. Phase 4: Verify attribution alignment again

### Priority 3 (Methodological Enhancements)
9. Phase 9: Equal-coverage experiment
10. Phase 12: SHAP faithfulness experiment

### Priority 4 (Robustness Validation)
11. Phase 10: Multi-seed experiment
12. Phase 11: Decide attribution scope

### Priority 5 (Analysis & Documentation)
13. Phase 13: Qualitative analysis
14. Phase 14: Runtime experiment
15. Phase 15: Fragmentation analysis
16. Phase 16: Build final result tables

---

## Execution Policy

- Each phase will have its own log file
- Each phase will generate an MD summary
- Scripts will be modular and reusable
- All random states will be fixed for reproducibility
- Verification steps will be included before proceeding

---

## Expected Timeline

- Priority 1: 2-3 hours
- Priority 2: 1-2 hours
- Priority 3: 4-6 hours (equal-coverage is computationally intensive)
- Priority 4: 12-24 hours (multi-seed training is expensive)
- Priority 5: 2-4 hours

**Total Estimated Time**: 21-39 hours (depending on multi-seed scope)

---

## Success Criteria

- Effect sizes are non-zero and reasonable
- Row count discrepancy resolved
- Checkpoints verified programmatically
- Statistical pipeline reproducible
- Permutation test correctly implemented
- Multiple-comparison correction applied
- Equal-coverage analysis completed
- SHAP cross-validation done
- Multi-seed robustness assessed
- All 6 paper tables generated

---

**Plan End**
