# Phase 3 — Alignment Validation Report (phase2_corrected)

**Attribution Type**: phase2_corrected
**Attribution Directory**: E:\BD SHS\outputs\attribution\phase2

## Summary

| Model_Method | Total | Cleanly_Aligned | Warnings | Excluded | Alignment_Rate |
|---|---|---|---|---|---|
| BanglaBERT_Integrated Gradients | 5029 | 5029 | 0 | 0 | 1.0000 |
| XLM-R_Integrated Gradients | 5029 | 5026 | 3 | 3 | 0.9994 |
| BanglaBERT_SHAP | 5029 | 4997 | 32 | 32 | 0.9936 |
| XLM-R_SHAP | 5029 | 4958 | 71 | 71 | 0.9859 |

## Detailed Warnings

### BanglaBERT_Integrated Gradients

- Total examples: 5029
- Cleanly aligned: 5029
- Warnings: 0
- Excluded: 0

### XLM-R_Integrated Gradients

- Total examples: 5029
- Cleanly aligned: 5026
- Warnings: 3
- Excluded: 3

#### Warning Types

- subword_count_mismatch: 3

#### Warning Details (first 20)

| Example ID | Warning Type | Details |
|-----------|--------------|---------|
| 1179 | subword_count_mismatch | Subwords count mismatch: subwords=42, content_tokens=43, total_tokens=45 |
| 2757 | subword_count_mismatch | Subwords count mismatch: subwords=19, content_tokens=20, total_tokens=22 |
| 2769 | subword_count_mismatch | Subwords count mismatch: subwords=23, content_tokens=24, total_tokens=26 |

### BanglaBERT_SHAP

- Total examples: 5029
- Cleanly aligned: 4997
- Warnings: 32
- Excluded: 32

#### Warning Types

- subword_count_mismatch: 32

#### Warning Details (first 20)

| Example ID | Warning Type | Details |
|-----------|--------------|---------|
| 3 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=201, total_tokens=203 |
| 167 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=258, total_tokens=260 |
| 274 | subword_count_mismatch | Subwords count mismatch: subwords=16, content_tokens=17, total_tokens=18 |
| 472 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=379, total_tokens=381 |
| 517 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=133, total_tokens=135 |
| 602 | subword_count_mismatch | Subwords count mismatch: subwords=8, content_tokens=9, total_tokens=10 |
| 802 | subword_count_mismatch | Subwords count mismatch: subwords=6, content_tokens=7, total_tokens=8 |
| 927 | subword_count_mismatch | Subwords count mismatch: subwords=11, content_tokens=12, total_tokens=13 |
| 1043 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=174, total_tokens=176 |
| 1082 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=166, total_tokens=168 |
| 1298 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=271, total_tokens=273 |
| 1409 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=195, total_tokens=197 |
| 1492 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=367, total_tokens=369 |
| 1636 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=237, total_tokens=239 |
| 1725 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=179, total_tokens=181 |
| 1759 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=188, total_tokens=190 |
| 1802 | subword_count_mismatch | Subwords count mismatch: subwords=14, content_tokens=15, total_tokens=16 |
| 2071 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=315, total_tokens=317 |
| 2163 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=167, total_tokens=169 |
| 2179 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=298, total_tokens=300 |

### XLM-R_SHAP

- Total examples: 5029
- Cleanly aligned: 4958
- Warnings: 71
- Excluded: 71

#### Warning Types

- subword_count_mismatch: 71

#### Warning Details (first 20)

| Example ID | Warning Type | Details |
|-----------|--------------|---------|
| 3 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=360, total_tokens=362 |
| 166 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=146, total_tokens=148 |
| 167 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=387, total_tokens=389 |
| 274 | subword_count_mismatch | Subwords count mismatch: subwords=24, content_tokens=25, total_tokens=26 |
| 472 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=603, total_tokens=605 |
| 517 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=227, total_tokens=229 |
| 532 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=148, total_tokens=150 |
| 602 | subword_count_mismatch | Subwords count mismatch: subwords=12, content_tokens=13, total_tokens=14 |
| 802 | subword_count_mismatch | Subwords count mismatch: subwords=11, content_tokens=12, total_tokens=13 |
| 866 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=182, total_tokens=184 |
| 927 | subword_count_mismatch | Subwords count mismatch: subwords=24, content_tokens=25, total_tokens=26 |
| 1043 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=250, total_tokens=252 |
| 1082 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=234, total_tokens=236 |
| 1179 | subword_count_mismatch | Subwords count mismatch: subwords=42, content_tokens=43, total_tokens=45 |
| 1198 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=153, total_tokens=155 |
| 1298 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=382, total_tokens=384 |
| 1304 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=159, total_tokens=161 |
| 1409 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=297, total_tokens=299 |
| 1492 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=517, total_tokens=519 |
| 1530 | subword_count_mismatch | Subwords count mismatch: subwords=126, content_tokens=217, total_tokens=219 |

