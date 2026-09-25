# Phase 9 Complete: Equal-Coverage Evaluation Implementation Plan

**Status**: ⚠ IMPLEMENTATION PLAN (Requires 26-34 hours)

## Purpose

Implement proper equal-coverage evaluation with three hierarchical selection variants:

1. **Exact-budget with intra-unit trimming**: Greedily add whole units, then trim the boundary unit word-by-word to hit exact target
2. **Floor (no overshoot)**: Stop before the boundary unit, accept slight undershoot
3. **Ceiling (allow overshoot)**: Include the boundary unit in full, record actual coverage

## Current Data Structure

The available word-level attribution data includes:
- `sample_idx`: Example identifier
- `text_snippet`: Text segment
- `predicted_label`: Model prediction
- `word`: Word-level token
- `subwords`: Subword count string (e.g., '2 subwords')
- `attr_sum`, `attr_mean`, `attr_max`: Attribution scores

## Missing Components

For full equal-coverage implementation, we need:

1. **Hierarchical unit structure**: Which words belong to which merged units
2. **Unit-level interaction scores**: Interaction/removal-effect scores for each merged unit
3. **Token-to-word mapping**: Indices for re-masking during forward passes
4. **Forward-pass capability**: Ability to re-run model with different selections

## Implementation Steps

### Step 1 Attribution Modification

**Task**: Modify Phase 2 attribution script to save hierarchical unit structure

**Files**: attribution/phase2_corrected_attribution.py

**Outputs**: hierarchical_unit_structure.json, unit_interaction_scores.json

**Estimated Time**: 2 hours

### Step 2 Rerun Attribution

**Task**: Re-run Phase 2 attribution with new outputs

**Outputs**: ig_banglabert_token.json, ig_xlm_r_token.json, shap_banglabert_token.json, shap_xlm_r_token.json

**Command**: `python attribution/phase2_corrected_attribution.py`

**Estimated Time**: 12-16 hours

### Step 3 Selection Functions

**Task**: Implement three hierarchical selection variants

**Files**: evaluation/equal_coverage_selection.py

**Estimated Time**: 2 hours

### Step 4 Stratified Subsample

**Task**: Create stratified subsample (1,000 examples)

**Criteria**: balanced by class, balanced by sentence-length quartile

**Estimated Time**: 30 minutes

### Step 5 Matched Coverage Evaluation

**Task**: Re-run forward passes with matched-coverage selections

**Coverage Grid**: 5%, 10%, 15%, 20%, 25%, 30%

**Strategies**: sum, mean, max, hierarchical

**Variants**: trimming, floor, ceiling

**Estimated Time**: 8-12 hours

### Step 6 Statistical Analysis

**Task**: Run statistical analysis on matched-coverage results

**Methods**: bootstrap 10k, permutation 10k, Cohen's dz, Holm-Bonferroni

**Estimated Time**: 1 hour

### Step 7 Sensitivity Analysis

**Task**: Generate unit size histogram and trimming frequency

**Outputs**: unit_size_histogram.csv, trimming_frequency.csv

**Estimated Time**: 30 minutes

### Step 8 Visualization

**Task**: Generate coverage-vs-metric line plots

**Outputs**: coverage_vs_metric.png

**Estimated Time**: 1 hour

## Selection Variant Details

### 1. Exact-Budget with Intra-Unit Trimming

```python
def select_exact_budget_with_trimming(units, target_word_count):
    selected_words = []
    remaining_budget = target_word_count
    
    for unit in sorted_units:
        if len(unit.words) <= remaining_budget:
            selected_words.extend(unit.words)
            remaining_budget -= len(unit.words)
        else:
            # Trim this unit word-by-word
            for word in sorted(unit.words, key=score, reverse=True):
                if remaining_budget > 0:
                    selected_words.append(word)
                    remaining_budget -= 1
            break
    
    return selected_words
```

### 2. Floor (No Overshoot)

```python
def select_floor_no_overshoot(units, target_word_count):
    selected_words = []
    remaining_budget = target_word_count
    
    for unit in sorted_units:
        if len(unit.words) <= remaining_budget:
            selected_words.extend(unit.words)
            remaining_budget -= len(unit.words)
        else:
            # Stop before this unit
            break
    
    return selected_words
```

### 3. Ceiling (Allow Overshoot)

```python
def select_ceiling_allow_overshoot(units, target_word_count):
    selected_words = []
    remaining_budget = target_word_count
    
    for unit in sorted_units:
        selected_words.extend(unit.words)
        remaining_budget -= len(unit.words)
        if remaining_budget <= 0:
            break
    
    actual_coverage = len(selected_words) / total_words
    return selected_words, actual_coverage
```

## Stratified Subsample Strategy

To manage compute budget, use a stratified subsample of 1,000 examples:

- **Balanced by class**: Equal representation of hate speech vs non-hate speech
- **Balanced by sentence-length quartile**: Represent short, medium, long, very long sentences
- **Total compute**: 1,000 × 6 coverage × 4 strategies × 2 modes × 2 models × 3 variants ≈ 288,000 forward passes

## Statistical Analysis

Reuse existing statistical pipeline:
- 10,000 bootstrap resamples
- 10,000 permutation tests
- Cohen's d_z effect sizes
- Holm-Bonferroni correction

## Sensitivity Analysis

Report:
- Unit size histogram (how many units are 1 word vs 2+ words)
- Trimming frequency at each coverage level
- Coverage achieved by each variant

## Paper Figures

Replace/supplement existing tables with:
- Comp@C and Suff@C at matched coverage
- Coverage-vs-metric line plot (x = actual coverage, y = Suff, one line per strategy)
- Unit size distribution histogram
- Trimming frequency by coverage level

## Total Estimated Time

- Modify attribution script: 2 hours
- Re-run Phase 2 attribution: 12-16 hours
- Implement selection functions: 2 hours
- Stratified subsample: 30 minutes
- Matched-coverage evaluation: 8-12 hours
- Statistical analysis: 1 hour
- Sensitivity analysis: 30 minutes
- Visualization: 1 hour
**Total: 26-34 hours**

## Possible Outcomes

Both outcomes are publishable:

### Outcome 1: Hierarchical Advantage Survives
- Hierarchical outperforms flat baselines even at matched coverage
- Strong result suitable for BLP Workshop or similar
- Confirms that hierarchical aggregation genuinely improves faithfulness

### Outcome 2: Hierarchical Advantage Shrinks
- Apparent gains are largely a coverage artifact
- Legitimate and interesting finding for XAI evaluation methodology
- Reviewers respect honest reporting of experimental demands

## Conclusion

Full equal-coverage evaluation requires significant pipeline modifications (26-34 hours). The implementation plan is detailed and addresses all the design challenges mentioned. Both possible outcomes are publishable, so the experiment should be run regardless of expectations.

**Phase 9 Complete Status**: ⚠ IMPLEMENTATION PLAN (Ready for execution when resources available)
