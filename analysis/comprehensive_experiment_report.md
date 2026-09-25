# BD-SHS Comprehensive Experimental Plan Report

**Experiment ID**: bdshs_bootstrap_v1
**Created**: 2026-09-25
**Status**: In Progress

## Executive Summary

This document outlines the comprehensive experimental plan for implementing bootstrap confidence intervals, permutation tests, and multi-seed analysis for the BD-SHS hate speech detection project. The plan addresses the checkpoint selection issue identified in Phase 0 and implements rigorous statistical validation for faithfulness evaluation.

## Implementation Status

### ✅ Completed Phases

#### Phase 0: Experimental Configuration
- **Status**: ✅ Completed
- **Files Created**:
  - `configs/experiment_config.yaml`
  - `configs/experiment_config.json`
- **Description**: Centralized configuration file containing all experimental parameters that should never be manually changed between experiments.

#### Phase 1: Checkpoint Verification
- **Status**: ✅ Completed
- **Files Created**:
  - `analysis/checkpoint_verification.md`
- **Key Findings**:
  - BanglaBERT: Correct checkpoint is `checkpoint-5028` (Epoch 2, val F1 = 0.9037)
  - XLM-R: Correct checkpoint is `checkpoint-12570` (Epoch 5, val F1 = 0.9146)
  - Previous paper used wrong checkpoints (epoch 5 for BanglaBERT, epoch 8 for XLM-R)
  - Selection criterion is correct (`metric_for_best_model="f1"`)

#### Phase 2: Corrected Attribution Framework
- **Status**: ✅ Framework Complete
- **Files Created**:
  - `attribution/phase2_corrected_attribution.py`
- **Description**: Attribution analysis script that uses the CORRECTED checkpoints. Ready to run.

#### Phase 3: Alignment Validation System
- **Status**: ✅ Framework Complete (CRITICAL ISSUE DISCOVERED)
- **Files Created**:
  - `evaluation/phase3_alignment_validator.py`
  - `analysis/alignment_issue_report.md`
- **Critical Finding**: Only 73/5,029 examples (1.45%) are cleanly aligned in current attribution files
- **Impact**: BLOCKS all downstream analysis until resolved
- **Resolution Required**: Must run Phase 2 corrected attribution

#### Phase 4-5: Per-Example Aggregation Framework
- **Status**: ✅ Framework Complete
- **Files Created**:
  - `evaluation/phase4_per_example_aggregation.py`
- **Description**: Per-example faithfulness evaluation framework for bootstrap analysis.

#### Phase 7-9: Statistical Analysis Framework
- **Status**: ✅ Framework Complete
- **Files Created**:
  - `statistics/bootstrap_permutation.py`
- **Description**: Implementation of paired bootstrap (10,000 resamples) and paired permutation tests (10,000 permutations) with effect size calculation.

#### Phase 10: Multi-Seed Training Framework
- **Status**: ✅ Framework Complete
- **Files Created**:
  - `training/multi_seed_training.py`
- **Description**: Training script for running experiments with multiple random seeds (42, 43, 44).

#### Phase 13: Comprehensive Experiment Report
- **Status**: ✅ Completed (this document)
- **Files Created**:
  - `analysis/comprehensive_experiment_report.md`
  - `run_comprehensive_experiment.ps1`

### 🚨 Critical Blocking Issues

#### Issue 1: GPU Availability ✅ RESOLVED
- **Status**: ✅ RESOLVED
- **Details**: GPU now configured (CUDA 12.1, PyTorch 2.5.0+cu121, NVIDIA RTX A6000)
- **Resolution**: Installed GPU-enabled PyTorch with CUDA 12.1 support
- **Timeline**: 36-66 hours (1.5-2.75 days) with GPU acceleration

#### Issue 2: Alignment Validation Failure
- **Status**: CRITICAL - BLOCKING
- **Details**: Current attribution files have 1.45% alignment rate (73/5,029 examples)
- **Root Cause**: Word reconstruction mismatch between attribution generation and validation
- **Required Action**: Run Phase 2 corrected attribution to regenerate files with proper alignment
- **Estimated Time (GPU)**: 12-24 hours

### ⏳ Pending Phases (Blocked by Alignment Issue)

#### Phase 4-5: Generate Per-Example Results
- **Status**: Blocked until alignment issue resolved
- **Requirements**: Run per-example aggregation after Phase 2 attribution is validated

#### Phase 6: Calculate Point Estimates with Statistics
- **Status**: Blocked until Phase 4-5 complete
- **Requirements**: Calculate mean, median, std, standard error for every condition

#### Phase 11: Generate Statistical Results Tables
- **Status**: Blocked until Phase 4-5 complete
- **Requirements**: Primary hypothesis table (18 comparisons), full statistical table (72 comparisons)

#### Phase 12: Create Visualization Figures
- **Status**: Blocked until Phase 4-5 complete
- **Requirements**: Difference plots, fragmentation analysis, seed stability plots

## Experimental Matrix

### Training Phase
```
2 models × 3 seeds = 6 training runs
```

### Attribution Phase
```
2 models × 3 seeds × 2 attribution methods = 12 attribution runs
```

### Aggregation Phase
```
Each run: 4 strategies × 3 budgets = 12 conditions
```

### Evaluation Phase
```
For every run:
- Comprehensiveness
- Sufficiency
- Coverage
- Comp-efficiency
- Suff-efficiency
```

### Statistical Analysis
```
10,000 bootstrap resamples + 10,000 permutation resamples per comparison
```

## Key Scientific Improvements

### 1. Checkpoint Correctness
- **Issue**: Previous paper used wrong checkpoints
- **Fix**: Explicit checkpoint selection based on validation F1
- **Impact**: All results will be scientifically valid

### 2. Statistical Rigor
- **Previous**: Point estimates only
- **New**: Paired bootstrap 95% CIs + permutation tests + effect sizes
- **Impact**: Quantified uncertainty and statistical significance

### 3. Seed Robustness
- **Previous**: Single seed (42)
- **New**: Multiple seeds (42, 43, 44)
- **Impact**: Assessment of training variability

### 4. Alignment Validation
- **Previous**: Informal spot checks
- **New**: Comprehensive alignment validation system
- **Impact**: Guaranteed correct token-to-word mapping

### 5. Per-Example Tracking
- **Previous**: Aggregate metrics only
- **New**: Per-example faithfulness results saved
- **Impact**: Enables paired statistical analysis

## Configuration Parameters

### Fixed Experimental Parameters
```yaml
dataset:
  name: "BD-SHS"
  test_size: 5029

models:
  - "BanglaBERT"
  - "XLM-R"

attribution:
  methods:
    - "Integrated Gradients"
    - "SHAP"

aggregation:
  strategies:
    - "Sum"
    - "Mean"
    - "Max"
    - "Hierarchical"

budgets:
  - 0.10  # 10%
  - 0.20  # 20%
  - 0.30  # 30%

seeds:
  - 42
  - 43
  - 44

bootstrap:
  n_resamples: 10000
  confidence_level: 0.95
  random_state: 42

permutation:
  n_permutations: 10000
  random_state: 42
```

## File Structure

```
BD SHS/
├── configs/
│   ├── experiment_config.yaml
│   └── experiment_config.json
├── attribution/
│   ├── phase2_corrected_attribution.py
│   └── bdshs_attribution.py (original)
├── evaluation/
│   ├── phase3_alignment_validator.py
│   └── aggregation_evaluation.py (original)
├── statistics/
│   └── bootstrap_permutation.py
├── training/
│   └── multi_seed_training.py
├── analysis/
│   ├── checkpoint_verification.md
│   └── comprehensive_experiment_report.md (this file)
├── results/
│   ├── alignment/
│   ├── statistics/
│   └── classification/
└── outputs/
    ├── attribution/
    │   └── phase2/
    └── training/
        └── seeds/
```

## Execution Plan (UPDATED - GPU Configured, Ready to Proceed)

### ✅ GPU Environment Configured
- **CUDA Available**: True
- **PyTorch Version**: 2.5.0+cu121
- **GPU**: NVIDIA RTX A6000
- **Status**: Ready for GPU-accelerated execution

### 📋 GPU Execution Plan

#### Step 1: Run Corrected Attribution (REQUIRED)
```bash
python attribution/phase2_corrected_attribution.py
```
- **Estimated Time (GPU)**: 12-24 hours
- **Purpose**: Regenerate attribution from CORRECTED checkpoints with proper alignment
- **Output**: `outputs/attribution/phase2/` directory with corrected attribution files

#### Step 2: Validate Corrected Attribution (REQUIRED)
```bash
python evaluation/phase3_alignment_validator.py
```
- **Expected Result**: >95% alignment rate
- **Purpose**: Ensure token-to-word alignment is correct before proceeding
- **Output**: `results/alignment/phase2/` validation report

#### Step 3: Run Per-Example Aggregation
```bash
python evaluation/phase4_per_example_aggregation.py
```
- **Estimated Time (GPU)**: 2-4 hours
- **Purpose**: Generate per-example faithfulness results for bootstrap analysis
- **Output**: `results/evaluation/per_example/` with CSV and JSON results

#### Step 4: Run Statistical Analysis
```bash
python statistics/bootstrap_permutation.py
```
- **Estimated Time (CPU)**: 1-2 hours
- **Purpose**: Apply paired bootstrap and permutation tests to per-example results
- **Output**: `results/statistics/` with confidence intervals and p-values

#### Step 5: Generate Results Tables
- Create comprehensive statistical tables
- Generate visualization figures
- **Estimated Time (CPU)**: 1-2 hours
- **Output**: `results/analysis/` with tables and figures

#### Step 6: Run Multi-Seed Training (OPTIONAL - Resource Intensive)
```bash
python training/multi_seed_training.py
```
- **Estimated Time (GPU)**: 24-42 hours
- **Purpose**: Assess training robustness across multiple seeds
- **Output**: `outputs/training/seeds/` with seed-specific results

#### Step 7: Compile Final Report
- Consolidate all results
- Create training logs
- Generate final comprehensive report
- **Output**: Updated `analysis/comprehensive_experiment_report.md`

### ⏱️ Timeline Comparison (GPU Configured)

| Phase | GPU Time | Status |
|-------|----------|--------|
| GPU Configuration | ✅ Complete | Done |
| Phase 2 Attribution | 12-24 hours | Ready to start |
| Phase 3 Validation | 0.2 hours | Blocked |
| Phase 4-5 Aggregation | 2-4 hours | Blocked |
| Phase 7-9 Statistics | 1-2 hours | Blocked |
| Phase 11-12 Tables/Figures | 1-2 hours | Blocked |
| Phase 10 Multi-Seed | 24-42 hours | Optional |
| **Total (Core)** | **16-32 hours** | - |
| **Total (With Multi-Seed)** | **40-74 hours** | - |

## Expected Outcomes

### Primary Hypothesis
Hierarchical aggregation improves coverage-normalized sufficiency relative to flat aggregation.

### Statistical Validation
- 18 primary comparisons with 95% CIs
- Paired permutation tests for significance
- Effect sizes (Cohen's d_z) for practical significance

### Seed Robustness
- Assessment of training variability across 3 seeds
- Confidence intervals for seed-level statistics

### Attribution Robustness
- Comparison of IG and SHAP attribution methods
- Cross-method validation of hierarchical improvement

## Computational Requirements

### GPU Resources
- Training: 6 runs × ~2-3 hours each = 12-18 hours
- Attribution: 12 runs × ~1-2 hours each = 12-24 hours
- Hierarchical aggregation: Additional forward passes

### Storage Requirements
- Checkpoints: ~2 GB per model × 3 seeds = 6 GB
- Attribution results: ~500 MB per run × 12 = 6 GB
- Per-example results: ~1 GB total
- Statistical results: ~100 MB

### Total Estimated Time
- GPU-intensive phases: 24-42 hours
- CPU-intensive phases: 4-8 hours
- Total: 28-50 hours

## Risk Mitigation

### Checkpoint Selection
- **Risk**: Using wrong checkpoints again
- **Mitigation**: Explicit checkpoint loading with verification

### Alignment Issues
- **Risk**: Token-to-word misalignment
- **Mitigation**: Comprehensive validation before faithfulness evaluation

### Computational Resources
- **Risk**: Insufficient GPU time
- **Mitigation**: Phased execution, checkpoint resumption

### Statistical Power
- **Risk**: Insufficient statistical power
- **Mitigation**: 10,000 resamples provide high power

## Quality Assurance

### Reproducibility
- Fixed random seeds for all stochastic operations
- Configuration file prevents parameter drift
- Detailed logging of all operations

### Validation
- Alignment validation before faithfulness evaluation
- Checkpoint verification before attribution
- Statistical sanity checks

### Documentation
- Training logs for each phase
- Comprehensive experiment report
- Statistical analysis documentation

## Next Steps

1. **Immediate**: Run alignment validation on current attribution
2. **Short-term**: Execute corrected attribution with proper checkpoints
3. **Medium-term**: Implement per-example result saving in aggregation
4. **Long-term**: Complete multi-seed training and statistical analysis

## References

- Bootstrap confidence intervals: Efron & Tibshirani (1993)
- Permutation tests: Good (2000)
- Effect sizes: Cohen (1988)
- Multiple comparison correction: Holm (1979)

---

**Document Version**: 1.0
**Last Updated**: 2026-09-25
**Author**: Devin AI Implementation
