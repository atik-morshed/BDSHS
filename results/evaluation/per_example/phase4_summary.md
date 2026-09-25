# Phase 4-5 — Per-Example Aggregation Results

**Examples evaluated**: 5029
**Strategies**: ['sum', 'mean', 'max', 'hierarchical']
**Budgets**: [10, 20, 30]%

## Per-Example Results Format

Each example contains:
- example_id, text, predicted_label, original_probability
- Strategy-specific results:
  - comprehensiveness@10/20/30
  - sufficiency@10/20/30
  - comp_coverage@10/20/30
  - suff_coverage@10/20/30
  - comp_efficiency@10/20/30
  - suff_efficiency@10/20/30

## Files Generated

- `per_example_faithfulness_results.json` — JSON format with complete per-example results
- `per_example_faithfulness_results.csv` — CSV format for bootstrap analysis

## Next Steps

1. Use per-example CSV for paired bootstrap analysis
2. Calculate hierarchical vs baseline differences
3. Apply bootstrap and permutation tests
4. Generate statistical tables with CIs and p-values
