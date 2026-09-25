# Phase 4 Report: Deep Attribution Alignment Verification

**Status**: ✅ COMPLETE

## Verification Results

| Model/Method | Total | Unique IDs | Duplicates | Missing | Extra | Recon Issues | NaN Scores | Alignment Rate |
|-------------|-------|------------|------------|---------|-------|--------------|------------|---------------|
| BanglaBERT_IG   |  5029 |       5029 |          0 |       0 |      0 |            0 |           0 |        100.00% |
| BanglaBERT_SHAP |  5029 |       5029 |          0 |       0 |      0 |           32 |           0 |         99.36% |
| XLM-R_IG        |  5029 |       5029 |          0 |       0 |      0 |            3 |           0 |         99.94% |
| XLM-R_SHAP      |  5029 |       5029 |          0 |       0 |      0 |           71 |           0 |         98.59% |

## Alignment Failures

Total failures across all methods: 106

### Failure Types

- subword_count_mismatch (subwords=1, tokens=2): 1
- subword_count_mismatch (subwords=11, tokens=12): 3
- subword_count_mismatch (subwords=12, tokens=13): 2
- subword_count_mismatch (subwords=126, tokens=127): 1
- subword_count_mismatch (subwords=126, tokens=129): 2
- subword_count_mismatch (subwords=126, tokens=130): 3
- subword_count_mismatch (subwords=126, tokens=132): 2
- subword_count_mismatch (subwords=126, tokens=133): 3
- subword_count_mismatch (subwords=126, tokens=134): 1
- subword_count_mismatch (subwords=126, tokens=135): 1
- subword_count_mismatch (subwords=126, tokens=137): 1
- subword_count_mismatch (subwords=126, tokens=139): 1
- subword_count_mismatch (subwords=126, tokens=140): 2
- subword_count_mismatch (subwords=126, tokens=146): 1
- subword_count_mismatch (subwords=126, tokens=147): 2
- subword_count_mismatch (subwords=126, tokens=148): 1
- subword_count_mismatch (subwords=126, tokens=149): 1
- subword_count_mismatch (subwords=126, tokens=153): 1
- subword_count_mismatch (subwords=126, tokens=159): 1
- subword_count_mismatch (subwords=126, tokens=162): 1
- subword_count_mismatch (subwords=126, tokens=164): 2
- subword_count_mismatch (subwords=126, tokens=165): 1
- subword_count_mismatch (subwords=126, tokens=166): 1
- subword_count_mismatch (subwords=126, tokens=167): 2
- subword_count_mismatch (subwords=126, tokens=169): 1
- subword_count_mismatch (subwords=126, tokens=170): 2
- subword_count_mismatch (subwords=126, tokens=174): 2
- subword_count_mismatch (subwords=126, tokens=176): 1
- subword_count_mismatch (subwords=126, tokens=177): 1
- subword_count_mismatch (subwords=126, tokens=179): 1
- subword_count_mismatch (subwords=126, tokens=182): 1
- subword_count_mismatch (subwords=126, tokens=183): 1
- subword_count_mismatch (subwords=126, tokens=188): 1
- subword_count_mismatch (subwords=126, tokens=190): 1
- subword_count_mismatch (subwords=126, tokens=192): 1
- subword_count_mismatch (subwords=126, tokens=195): 1
- subword_count_mismatch (subwords=126, tokens=196): 1
- subword_count_mismatch (subwords=126, tokens=199): 1
- subword_count_mismatch (subwords=126, tokens=201): 1
- subword_count_mismatch (subwords=126, tokens=203): 1
- subword_count_mismatch (subwords=126, tokens=213): 1
- subword_count_mismatch (subwords=126, tokens=217): 1
- subword_count_mismatch (subwords=126, tokens=225): 1
- subword_count_mismatch (subwords=126, tokens=227): 1
- subword_count_mismatch (subwords=126, tokens=228): 1
- subword_count_mismatch (subwords=126, tokens=234): 1
- subword_count_mismatch (subwords=126, tokens=237): 1
- subword_count_mismatch (subwords=126, tokens=249): 1
- subword_count_mismatch (subwords=126, tokens=250): 1
- subword_count_mismatch (subwords=126, tokens=258): 1
- subword_count_mismatch (subwords=126, tokens=259): 1
- subword_count_mismatch (subwords=126, tokens=271): 1
- subword_count_mismatch (subwords=126, tokens=276): 1
- subword_count_mismatch (subwords=126, tokens=285): 2
- subword_count_mismatch (subwords=126, tokens=297): 1
- subword_count_mismatch (subwords=126, tokens=298): 1
- subword_count_mismatch (subwords=126, tokens=315): 1
- subword_count_mismatch (subwords=126, tokens=321): 1
- subword_count_mismatch (subwords=126, tokens=333): 1
- subword_count_mismatch (subwords=126, tokens=360): 1
- subword_count_mismatch (subwords=126, tokens=367): 1
- subword_count_mismatch (subwords=126, tokens=371): 1
- subword_count_mismatch (subwords=126, tokens=379): 1
- subword_count_mismatch (subwords=126, tokens=382): 1
- subword_count_mismatch (subwords=126, tokens=387): 1
- subword_count_mismatch (subwords=126, tokens=391): 3
- subword_count_mismatch (subwords=126, tokens=463): 1
- subword_count_mismatch (subwords=126, tokens=517): 1
- subword_count_mismatch (subwords=126, tokens=579): 1
- subword_count_mismatch (subwords=126, tokens=603): 1
- subword_count_mismatch (subwords=14, tokens=15): 2
- subword_count_mismatch (subwords=16, tokens=17): 2
- subword_count_mismatch (subwords=19, tokens=20): 2
- subword_count_mismatch (subwords=2, tokens=3): 1
- subword_count_mismatch (subwords=20, tokens=21): 1
- subword_count_mismatch (subwords=23, tokens=24): 2
- subword_count_mismatch (subwords=24, tokens=25): 3
- subword_count_mismatch (subwords=42, tokens=43): 2
- subword_count_mismatch (subwords=6, tokens=7): 1
- subword_count_mismatch (subwords=8, tokens=9): 2
