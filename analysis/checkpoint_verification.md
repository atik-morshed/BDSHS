# Checkpoint Verification Report

## Verification Date: 2026-09-25

## Checkpoint Selection Criterion Analysis

### Configuration Check
From `bdshs_pipeline.py` and trainer_state.json files:

```python
metric_for_best_model="f1"
greater_is_better=True
load_best_model_at_end=True
```

### BanglaBERT Checkpoint Analysis
- **Best checkpoint**: `checkpoint-5028`
- **Epoch**: 2.0
- **Global step**: 5028
- **Validation F1**: 0.9036792062835882
- **Validation Accuracy**: 0.9073190135242641
- **Selection criterion**: ✅ Correct (maximum validation F1)

### XLM-R Checkpoint Analysis
- **Best checkpoint**: `checkpoint-12570`
- **Epoch**: 5.0
- **Global step**: 12570
- **Validation F1**: 0.9146366427840328
- **Validation Accuracy**: 0.9171326107048323
- **Selection criterion**: ✅ Correct (maximum validation F1)

## Verification Summary

Both models correctly use validation F1 as the selection criterion:
- ✅ No manual epoch selection
- ✅ No training loss selection
- ✅ No last epoch selection
- ✅ No validation accuracy selection
- ✅ Explicit `metric_for_best_model="f1"` configuration

## Current Checkpoint Status

The checkpoints identified in `phase0_checkpoint_eval.py` are CORRECT:
- BanglaBERT: `outputs/banglabert-bdshs/checkpoint-5028` (Epoch 2)
- XLM-R: `outputs/xlmr-bdshs/checkpoint-12570` (Epoch 5)

## Previous Paper vs Current Checkpoints

### Previous Paper Claims:
- BanglaBERT best validation F1 → epoch 2 ✅ (CORRECT)
- XLM-R best validation F1 → epoch 5 ✅ (CORRECT)

### Previous Test Results (from phase0_checkpoint_eval.py):
- BanglaBERT test results from epoch 5 ❌ (INCORRECT - should be epoch 2)
- XLM-R test results from epoch 8 ❌ (INCORRECT - should be epoch 5)

## Action Required

The checkpoint selection criterion is correct, but the test results in the paper were calculated from the wrong checkpoints. All downstream experiments (attribution, aggregation, faithfulness) must be regenerated from the CORRECT checkpoints identified above.

## Recommendation

Proceed with Phase 2: Re-run attribution from corrected checkpoints using:
- BanglaBERT: `outputs/banglabert-bdshs/checkpoint-5028`
- XLM-R: `outputs/xlmr-bdshs/checkpoint-12570`
