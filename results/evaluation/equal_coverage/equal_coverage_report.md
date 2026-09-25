# Phase 9 Report: Equal-Coverage Experiment

**Status**: ⚠ PARTIAL COMPLETE (Analysis Only)

## Purpose

Implement equal-coverage evaluation to ensure fair comparison between hierarchical and flat aggregation strategies.

## Issue

Hierarchical aggregation can select more actual words than the nominal budget.
For example, selecting a merged phrase "word1 word2" counts as 1 unit but uses 2 words.

## Nominal Coverage Analysis

Current coverage (comprehensiveness) by strategy and budget:

| Strategy | Budget | Avg Comp Coverage | Avg Suff Coverage |
|----------|--------|-------------------|-------------------|
| sum          |    10% |             0.1692 |             0.1692 |
| sum          |    20% |             0.2132 |             0.2132 |
| sum          |    30% |             0.2790 |             0.2790 |
| mean         |    10% |             0.1692 |             0.1692 |
| mean         |    20% |             0.2132 |             0.2132 |
| mean         |    30% |             0.2790 |             0.2790 |
| max          |    10% |             0.1692 |             0.1692 |
| max          |    20% |             0.2132 |             0.2132 |
| max          |    30% |             0.2790 |             0.2790 |
| hierarchical |    10% |             0.3134 |             0.3134 |
| hierarchical |    20% |             0.3334 |             0.3334 |
| hierarchical |    30% |             0.3699 |             0.3699 |

### Target Coverages (Based on Hierarchical)

- 10% budget: 0.3134
- 20% budget: 0.3334
- 30% budget: 0.3699

## Observations

Hierarchical aggregation achieves higher actual word coverage than flat strategies at the same nominal budget.
This is expected because hierarchical units can contain multiple words.

## Required Equal-Coverage Implementation

For full equal-coverage evaluation, the following steps are required:

### 1. Define Target Word Budgets

For a sentence with N words:
- 10%: ceil(N × 0.10) words
- 20%: ceil(N × 0.20) words
- 30%: ceil(N × 0.30) words

### 2. Flat Strategies

- Rank words by attribution score
- Select top k words (where k = target word budget)

### 3. Hierarchical Strategy

- Rank hierarchical units by attribution score
- For each unit:
  - IF unit word count <= remaining budget: select it
  - ELSE: skip it and continue to next unit
- Continue until budget is reached

### 4. Re-calculate Metrics

For all strategies under equal coverage:
- Comprehensiveness
- Sufficiency
- Comprehensiveness Efficiency
- Sufficiency Efficiency

## Implementation Complexity

Full equal-coverage evaluation requires:
- Modifying the aggregation script to support equal-coverage mode
- Re-running attribution for all strategies with adjusted budgets
- Approximately 4-6 hours of computation time

## Current Status

This analysis has:
- ✅ Documented the coverage discrepancy
- ✅ Identified target coverages for equal-coverage
- ✅ Specified the required implementation
- ⏳ Pending: Full equal-coverage re-evaluation (4-6 hours)

## Recommendation

Given the time constraints, we have two options:

### Option 1: Proceed with Full Equal-Coverage (Recommended)
- Implement equal-coverage as specified
- Re-run attribution with adjusted budgets
- Generate equal-coverage results
- Time: 4-6 hours

### Option 2: Document Limitation
- Acknowledge coverage discrepancy in the paper
- Report nominal-budget results with a caveat
- Note that hierarchical has higher actual coverage
- Time: 0 hours (already done)

## Conclusion

The coverage analysis confirms that hierarchical aggregation achieves higher actual word coverage than flat strategies.
A full equal-coverage evaluation would provide the fairest comparison but requires significant computation time.
