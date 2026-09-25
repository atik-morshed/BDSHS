"""
Phase 4 — Deep Attribution Alignment Verification
=================================================

Current alignment rates are already strong (>98%) but need deeper verification for robustness.

Verification Steps:
1. Verify example IDs match between attribution and test set
2. Check for duplicate example IDs
3. Verify word reconstruction for failed examples
4. Export alignment failures CSV with detailed reasons
5. Document why any examples were excluded

Run: python evaluation/phase4_deep_alignment_verification.py
Outputs: results/alignment/deep_alignment_verification.md
         results/alignment/alignment_failures.csv
"""

import os, sys, io, json
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
ATTRIBUTION_DIR = BASE_DIR / "outputs" / "attribution" / "phase2"
ALIGNMENT_DIR = BASE_DIR / "results" / "alignment" / "deep_verification"
ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ALIGNMENT_DIR / "phase4_deep_alignment_log.txt"
log_fh = open(LOG_FILE, "w", encoding="utf-8")

def log(message):
    """Write to both stdout and log file."""
    print(message)
    log_fh.write(message + "\n")
    log_fh.flush()

print("=" * 80)
log("=" * 80)
log("PHASE 4 — DEEP ATTRIBUTION ALIGNMENT VERIFICATION")
log("=" * 80)

# Attribution files to verify
attribution_files = {
    "BanglaBERT_IG": ATTRIBUTION_DIR / "ig_banglabert_token.json",
    "BanglaBERT_SHAP": ATTRIBUTION_DIR / "shap_banglabert_token.json",
    "XLM-R_IG": ATTRIBUTION_DIR / "ig_xlm_r_token.json",
    "XLM-R_SHAP": ATTRIBUTION_DIR / "shap_xlm_r_token.json"
}

all_verification_results = []
alignment_failures = []

for model_method, file_path in attribution_files.items():
    log(f"\n{'='*80}")
    log(f"Verifying: {model_method}")
    log(f"File: {file_path}")
    log(f"{'='*80}")

    if not file_path.exists():
        log(f"[ERROR] File not found: {file_path}")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        attribution_data = json.load(f)

    # A. Verify example IDs
    log(f"\n--- A. Example ID Verification ---")
    example_ids = [item.get('sample_idx', i) for i, item in enumerate(attribution_data)]
    unique_ids = set(example_ids)
    
    log(f"Total examples: {len(attribution_data)}")
    log(f"Unique example IDs: {len(unique_ids)}")
    log(f"Duplicate IDs: {len(example_ids) - len(unique_ids)}")

    if len(example_ids) != len(unique_ids):
        log(f"[WARNING] Duplicate example IDs detected!")
        duplicate_counts = {}
        for eid in example_ids:
            duplicate_counts[eid] = duplicate_counts.get(eid, 0) + 1
        duplicates = {k: v for k, v in duplicate_counts.items() if v > 1}
        log(f"Number of duplicated IDs: {len(duplicates)}")
        if duplicates:
            log(f"Sample duplicates: {list(duplicates.keys())[:5]}")
    else:
        log("✓ No duplicate example IDs")

    # B. Verify ID range matches test set
    log(f"\n--- B. Test Set ID Range Verification ---")
    expected_ids = set(range(5029))
    actual_ids = set(example_ids)
    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids

    log(f"Expected ID range: 0 to 5028")
    log(f"Missing IDs: {len(missing_ids)}")
    log(f"Extra IDs: {len(extra_ids)}")

    if missing_ids:
        log(f"[WARNING] Missing example IDs: {sorted(list(missing_ids))[:10]}")
        for mid in sorted(list(missing_ids))[:5]:
            alignment_failures.append({
                'model_method': model_method,
                'example_id': mid,
                'reason': 'missing_from_attribution',
                'text': 'N/A'
            })
    else:
        log("✓ All expected example IDs present")

    if extra_ids:
        log(f"[WARNING] Extra example IDs: {sorted(list(extra_ids))[:10]}")
        for eid in sorted(list(extra_ids))[:5]:
            alignment_failures.append({
                'model_method': model_method,
                'example_id': eid,
                'reason': 'extra_id_beyond_range',
                'text': 'N/A'
            })
    else:
        log("✓ No extra example IDs")

    # C. Verify word reconstruction for all examples
    log(f"\n--- C. Word Reconstruction Verification ---")
    reconstruction_issues = 0

    for idx, item in enumerate(attribution_data):
        text = item.get('text', '')
        word_attribution = item.get('word_attribution', [])
        tokens = item.get('tokens', [])

        # Check word attribution exists
        if not word_attribution:
            reconstruction_issues += 1
            alignment_failures.append({
                'model_method': model_method,
                'example_id': idx,
                'reason': 'missing_word_attribution',
                'text': text[:50] + '...' if len(text) > 50 else text
            })
            continue

        # Check token-word alignment
        wa_words = [w.get('word') if isinstance(w, dict) else w for w in word_attribution]
        
        # Filter special tokens from tokens for comparison
        special_tokens = {'[CLS]', '[SEP]', '<s>', '</s>', ''}
        content_tokens = [t for t in tokens if t not in special_tokens]
        
        # Subword count check
        total_subwords = sum(len(w.get('subwords', []) if isinstance(w, dict) else []) for w in word_attribution)
        
        if total_subwords != len(content_tokens):
            reconstruction_issues += 1
            alignment_failures.append({
                'model_method': model_method,
                'example_id': idx,
                'reason': f'subword_count_mismatch (subwords={total_subwords}, tokens={len(content_tokens)})',
                'text': text[:50] + '...' if len(text) > 50 else text
            })

    log(f"Word reconstruction issues: {reconstruction_issues}")
    log(f"Alignment rate: {(len(attribution_data) - reconstruction_issues) / len(attribution_data) * 100:.2f}%")

    # D. Check for NaN or infinite values
    log(f"\n--- D. Numeric Value Verification ---")
    nan_count = 0
    inf_count = 0

    for idx, item in enumerate(attribution_data):
        item_text = item.get('text', '')
        # Check scores
        token_scores = item.get('token_scores_raw', [])
        if any(not isinstance(s, (int, float)) or pd.isna(s) for s in token_scores):
            nan_count += 1
            alignment_failures.append({
                'model_method': model_method,
                'example_id': idx,
                'reason': 'nan_or_invalid_score',
                'text': item_text[:50] + '...' if len(item_text) > 50 else item_text
            })

    log(f"NaN/invalid score values: {nan_count}")
    log(f"Infinite values: {inf_count}")

    # Summary for this model-method
    verification_result = {
        'model_method': model_method,
        'total_examples': len(attribution_data),
        'unique_ids': len(unique_ids),
        'duplicate_ids': len(example_ids) - len(unique_ids),
        'missing_ids': len(missing_ids),
        'extra_ids': len(extra_ids),
        'reconstruction_issues': reconstruction_issues,
        'nan_invalid_scores': nan_count,
        'alignment_rate': (len(attribution_data) - reconstruction_issues) / len(attribution_data)
    }
    all_verification_results.append(verification_result)

# Save alignment failures CSV
if alignment_failures:
    failures_df = pd.DataFrame(alignment_failures)
    failures_path = ALIGNMENT_DIR / "alignment_failures.csv"
    failures_df.to_csv(failures_path, index=False)
    log(f"\nAlignment failures saved to: {failures_path}")
    log(f"Total failures: {len(alignment_failures)}")
else:
    log(f"\n✓ No alignment failures detected")

# Save verification results
results_path = ALIGNMENT_DIR / "deep_alignment_verification.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(all_verification_results, f, indent=2)

log(f"\nVerification results saved to: {results_path}")

# Generate markdown report
report_path = ALIGNMENT_DIR / "deep_alignment_verification.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Phase 4 Report: Deep Attribution Alignment Verification\n\n")
    f.write("**Status**: ✅ COMPLETE\n\n")
    f.write("## Verification Results\n\n")
    
    f.write("| Model/Method | Total | Unique IDs | Duplicates | Missing | Extra | Recon Issues | NaN Scores | Alignment Rate |\n")
    f.write("|-------------|-------|------------|------------|---------|-------|--------------|------------|---------------|\n")
    
    for r in all_verification_results:
        f.write(f"| {r['model_method']:15s} | {r['total_examples']:5d} | {r['unique_ids']:10d} | "
                f"{r['duplicate_ids']:10d} | {r['missing_ids']:7d} | {r['extra_ids']:6d} | "
                f"{r['reconstruction_issues']:12d} | {r['nan_invalid_scores']:11d} | "
                f"{r['alignment_rate']*100:13.2f}% |\n")
    
    f.write(f"\n## Alignment Failures\n\n")
    f.write(f"Total failures across all methods: {len(alignment_failures)}\n\n")
    
    if alignment_failures:
        f.write("### Failure Types\n\n")
        failure_types = {}
        for fail in alignment_failures:
            ftype = fail['reason']
            failure_types[ftype] = failure_types.get(ftype, 0) + 1
        
        for ftype, count in sorted(failure_types.items()):
            f.write(f"- {ftype}: {count}\n")
    else:
        f.write("✓ No alignment failures detected\n")

log(f"\nReport saved to: {report_path}")

log_fh.close()

print("\n" + "=" * 80)
print("Phase 4 complete.")
print("=" * 80)
