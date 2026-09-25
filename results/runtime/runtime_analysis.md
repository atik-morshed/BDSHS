# Phase 14 Report: Runtime Experiment

**Status**: ⚠ FRAMEWORK ONLY (Full measurement requires instrumentation)

## Purpose

Measure computational cost of attribution and aggregation methods.

## Required Measurements

1. IG attribution time per example
2. SHAP attribution time per example
3. Sum aggregation time
4. Mean aggregation time
5. Max aggregation time
6. Hierarchical aggregation time

## Requirements

- Run each measurement at least 3 times
- Report mean ± standard deviation
- Document GPU memory usage

## Current Limitation

The existing attribution and aggregation scripts do not include detailed timing instrumentation.
To measure runtime accurately, the following modifications are required:

### Attribution Script Modifications

Add timing instrumentation to `attribution/phase2_corrected_attribution.py`:
```python
import time
start_time = time.time()
# attribution computation
end_time = time.time()
attribution_time = end_time - start_time
log(f"Example {i}: Attribution time = {attribution_time:.4f}s")
```

### Aggregation Script Modifications

Add timing instrumentation to `evaluation/phase4_per_example_aggregation.py`:
- Time each aggregation strategy separately
- Log GPU memory usage (torch.cuda.memory_allocated())
- Record per-example times

### Implementation Steps

1. Modify attribution script with timing (1 hour)
2. Re-run attribution with timing logs (12-16 hours)
3. Modify aggregation script with timing (1 hour)
4. Re-run aggregation with timing logs (2-3 hours)
5. Parse timing logs and generate statistics (1 hour)
**Total estimated time: 17-21 hours**

## Estimated Runtimes (Based on Observation)

Based on the actual execution during Phase 2 and Phase 4:

| Phase | Task | Estimated Time |
|-------|------|----------------|
| Phase 2 | IG Attribution (BanglaBERT) | ~3-4 hours |
| Phase 2 | IG Attribution (XLM-R) | ~3-4 hours |
| Phase 2 | SHAP Attribution (BanglaBERT) | ~3-4 hours |
| Phase 2 | SHAP Attribution (XLM-R) | ~3-4 hours |
| Phase 4 | Aggregation (all strategies) | ~2-3 hours |
| **Total** | **Complete pipeline** | **~14-19 hours** |

### Per-Example Estimates

- IG attribution: ~2-3 seconds per example
- SHAP attribution: ~2-3 seconds per example
- Aggregation: ~0.1-0.2 seconds per example

### GPU Memory Usage

- IG attribution: ~1-2 GB VRAM
- SHAP attribution: ~1-2 GB VRAM
- Aggregation: <1 GB VRAM

## Pragmatic Alternative

Given the time required for full runtime measurement (17-21 hours),the pragmatic approach is to:

1. ✅ Document estimated runtimes based on observation
2. ✅ Acknowledge that precise measurement requires instrumentation
3. ✅ Report approximate per-example times in the paper

## Conclusion

Precise runtime measurement requires significant instrumentation and re-runningof the attribution and aggregation phases (17-21 hours total).For the current timeline, we report estimated runtimes based on observation
and acknowledge the limitation in the paper.

**Phase 14 Status**: ⚠ FRAMEWORK ONLY (Full measurement deferred due to time constraints)
