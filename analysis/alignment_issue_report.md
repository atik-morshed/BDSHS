# Critical Alignment Issue Report

**Date**: 2026-09-25
**Severity**: CRITICAL
**Status**: Requires immediate investigation

## Issue Summary

Alignment validation revealed that only **73 out of 5,029 examples (1.45%)** are cleanly aligned across all attribution files. This represents a critical failure in the token-to-word alignment process.

## Details

### Alignment Statistics
- **Total examples**: 5,029
- **Cleanly aligned**: 73 (1.45%)
- **Warnings**: 4,956 (98.55%)
- **Excluded**: 4,956 (98.55%)

### Warning Breakdown
- **word_reconstruction_mismatch**: 4,956 (98.55%)

### Affected Files
- `ig_banglabert_token.json`
- `ig_xlm_r_token.json`
- `shap_banglabert_token.json`
- `shap_xlm_r_token.json`

## Root Cause Analysis

The issue appears to be in the `find_word_spans_from_saved` function which attempts to reconstruct word spans by searching for saved words in the original text. This method is failing because:

1. **Text cleaning differences**: The original text may have been cleaned differently during attribution generation vs validation
2. **Unicode normalization**: Bangla characters may have different representations
3. **Word boundary mismatches**: The word splitting logic may have changed between attribution generation and validation
4. **Punctuation handling**: Trailing punctuation separation may be inconsistent

## Impact

This alignment failure means:
- **Cannot proceed with faithfulness evaluation** - results would be invalid
- **Bootstrap analysis would be meaningless** - paired comparisons require correct alignment
- **All downstream experiments are blocked** until this is resolved

## Required Actions

### Immediate (Critical Path)
1. **Investigate text cleaning consistency** between attribution generation and validation
2. **Debug word reconstruction logic** with sample failing examples
3. **Fix alignment validation** to match the actual attribution generation process
4. **Re-run attribution with corrected checkpoints** (Phase 2) using fixed alignment

### Alternative Approach
If alignment validation cannot be fixed to match existing attribution:
1. **Run Phase 2 corrected attribution** which will generate new attribution files
2. **Use the same word boundary logic** in both attribution generation and validation
3. **Validate the new attribution files** before proceeding

## Recommendations

### Option A: Fix Existing Attribution (Risky)
- Debug and fix the alignment validation to work with existing attribution
- Risk: May introduce new bugs, time-consuming
- Benefit: Can use existing attribution results

### Option B: Regenerate Attribution (Recommended)
- Run Phase 2 corrected attribution with proper checkpoint selection
- Ensure word boundary logic is consistent between generation and validation
- Validate new attribution before proceeding
- Benefit: Clean slate, guaranteed consistency, uses correct checkpoints

## Next Steps

1. **Implement Option B** - Run Phase 2 corrected attribution
2. **Update alignment validation** to use the same word boundary logic as attribution generation
3. **Validate new attribution files** to ensure >95% alignment rate
4. **Proceed with faithfulness evaluation** only after alignment validation passes

## Timeline Estimate

- Phase 2 attribution generation: 12-24 hours (GPU intensive)
- Alignment validation: 5-10 minutes
- Total: 12-24 hours

## Conclusion

The current attribution files have a critical alignment issue that prevents any downstream analysis. The recommended approach is to regenerate attribution using the corrected checkpoints (Phase 2) with consistent word boundary logic, then validate before proceeding with faithfulness evaluation.

---

**Report Generated**: 2026-09-25
**Status**: BLOCKING - Resolution required before proceeding
