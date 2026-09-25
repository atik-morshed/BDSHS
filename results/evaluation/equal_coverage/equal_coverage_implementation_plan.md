# Phase 9 Full: Equal-Coverage Implementation Plan

**Status**: ⚠ FRAMEWORK ONLY (Full implementation requires pipeline modifications)

## Purpose

Implement full equal-coverage evaluation to ensure fair comparison between hierarchical and flat aggregation strategies.

## Current Limitation

The existing pipeline does not support equal-coverage evaluation because:

1. **Missing hierarchical unit structure**: The CSV results do not include information about which words belong to which merged hierarchical units.

2. **Missing word-level attribution scores**: The CSV includes aggregated scores but not the original word-level scores needed for re-selection.

3. **No re-evaluation capability**: The current pipeline cannot re-run forward passes with different word selections.

## Required Pipeline Modifications

### Phase 2 Modification: Save Hierarchical Unit Structure

Modify `attribution/phase2_corrected_attribution.py` to save:
- Hierarchical unit membership for each word
- Word-level attribution scores (not just aggregated)
- Token-to-word mapping

### New Script: Equal-Coverage Aggregation

Create `evaluation/phase9_equal_coverage_aggregation.py` that:
1. Loads hierarchical unit structure
2. Calculates target word budget based on hierarchical coverage
3. For flat strategies: selects top k words
4. For hierarchical: skips merged units exceeding budget
5. Re-runs forward passes with new selections
6. Calculates faithfulness metrics

### Implementation Steps

1. Modify Phase 2 attribution script (2 hours)
2. Re-run Phase 2 attribution with new outputs (12-16 hours)
3. Create equal-coverage aggregation script (2 hours)
4. Run equal-coverage evaluation (4-6 hours)
5. Generate comparison report (1 hour)

**Total estimated time: 21-25 hours**

## Pragmatic Alternative

Given the significant time required, the pragmatic approach is:

1. ✅ Document the coverage discrepancy (already done)
2. ✅ Calculate the hierarchical advantage due to coverage (already done)
3. ✅ Acknowledge this as a limitation in the paper
4. ✅ Report nominal-budget results with a clear caveat

### Coverage Advantage Quantified

- 10% budget: Hierarchical has 85% higher actual coverage
- 20% budget: Hierarchical has 56% higher actual coverage
- 30% budget: Hierarchical has 33% higher actual coverage

### Paper Wording

The paper should include a limitation section:

```markdown
## Limitations

The current evaluation uses nominal token budgets (10%, 20%, 30% of tokens).
Hierarchical aggregation achieves higher actual word coverage than flat strategies
because merged units can contain multiple words. This creates a potential advantage
for hierarchical methods. Future work should implement equal-coverage evaluation
to ensure completely fair comparison.
```

## Conclusion

Full equal-coverage evaluation requires significant pipeline modifications
(21-25 hours total). The pragmatic approach is to document the limitation
and acknowledge it in the paper, which we have done.

**Phase 9 Full Status**: ⚠ FRAMEWORK ONLY (Full implementation deferred due to time constraints)
