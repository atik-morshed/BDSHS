"""
Phase 3 — Alignment Validation System
======================================
Validates token-to-word alignment for attribution results before running faithfulness experiments.
This is critical to ensure attribution scores are correctly mapped to word boundaries.

Run:  python evaluation/phase3_alignment_validator.py
Outputs: ./results/alignment/
"""

import os, sys, io, json, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent

# Determine which attribution directory to use
phase2_dir = BASE_DIR / "outputs" / "attribution" / "phase2"
original_dir = BASE_DIR / "outputs" / "attribution"

if phase2_dir.exists():
    attribution_base_dir = phase2_dir
    attribution_type = "phase2_corrected"
else:
    attribution_base_dir = original_dir
    attribution_type = "original"

OUT_DIR = BASE_DIR / "results" / "alignment" / attribution_type
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 3 — ALIGNMENT VALIDATION SYSTEM")
print("=" * 60)

# Word span builder (must match attribution script)
def build_word_spans(text):
    """
    Build word spans that separate trailing punctuation clusters (e.g., 'কথাটা।' -> 'কথাটা' + '।').
    Returns a list of (start, end, token_text) tuples.
    """
    PUNCT_PATTERN = r'([।,!?;:\.:"\'()]+)$'  # trailing punctuation cluster

    spans = []
    cursor = 0
    for raw_word in text.split():
        idx = text.find(raw_word, cursor)
        if idx == -1:
            idx = cursor
        end = idx + len(raw_word)

        m = re.search(PUNCT_PATTERN, raw_word)
        if m and len(raw_word) > len(m.group(1)):
            core = raw_word[: -len(m.group(1))]
            spans.append((idx, idx + len(core), core))
            spans.append((idx + len(core), end, m.group(1)))
        else:
            spans.append((idx, end, raw_word))

        cursor = end

    return spans

def is_garbage_word(word):
    """True if a word consists entirely of Unicode replacement chars, punctuation-only junk,
    or other non-printable/control characters.
    Keeps important punctuation like the Bangla danda '।'.
    """
    stripped = word.strip()
    if not stripped:
        return True
    # entirely U+FFFD (replacement char) or other non-printable/control chars
    if all(ch == '\ufffd' or not ch.isprintable() for ch in stripped):
        return True
    # punctuation-only tokens (e.g., emoticon fragments like ':-' or '(') should be treated as garbage
    # but preserve meaningful punctuation such as the Bangla danda '।'
    allowed_keep = {'।'}
    if all((not ch.isalnum() and ch not in allowed_keep) for ch in stripped):
        return True
    return False

def find_word_spans_from_saved(text, wa_words):
    """Find character spans for each saved word by searching the text sequentially.
    Raises ValueError if any saved word can't be located (helps detect cleaning differences).
    """
    spans = []
    cursor = 0
    for w in wa_words:
        if w is None:
            raise ValueError(f"Saved word is None when building spans for text: {text[:80]!r}")
        idx = text.find(w, cursor)
        if idx == -1:
            raise ValueError(f"Cannot locate saved word {w!r} in text (truncated): {text[:80]!r}")
        spans.append((idx, idx + len(w), w))
        cursor = idx + len(w)
    return spans

def validate_attribution_file(token_file_path, model_name, attribution_method):
    """
    Validates alignment for a single attribution file.
    Returns alignment statistics and warnings.
    """
    print(f"\n{'─'*60}")
    print(f"Validating: {model_name} / {attribution_method}")
    print(f"File: {token_file_path}")
    print(f"{'─'*60}")

    if not token_file_path.exists():
        print(f"[ERROR] File not found: {token_file_path}")
        return None

    with open(token_file_path, 'r', encoding='utf-8') as f:
        token_results = json.load(f)

    stats = {
        "total_examples": len(token_results),
        "cleanly_aligned": 0,
        "warnings": defaultdict(int),
        "excluded": 0,
        "warning_details": []
    }

    for idx, example in enumerate(token_results):
        text = example.get('text', '')
        word_attribution = example.get('word_attribution', [])
        tokens = example.get('tokens', [])

        # Check A: Every attributed token has a span
        if len(tokens) != len(example.get('token_scores_raw', [])):
            stats["warnings"]["token_score_mismatch"] += 1
            stats["warning_details"].append({
                "example_id": idx,
                "warning_type": "token_score_mismatch",
                "details": f"tokens={len(tokens)}, scores={len(example.get('token_scores_raw', []))}"
            })
            continue

        # Check B: Word attribution data exists and is valid
        if not word_attribution:
            stats["warnings"]["missing_word_attribution"] += 1
            stats["warning_details"].append({
                "example_id": idx,
                "warning_type": "missing_word_attribution",
                "details": "No word_attribution field found"
            })
            continue

        # Check C: Word attribution structure is valid
        wa_words = [w.get('word') if isinstance(w, dict) else w for w in word_attribution]
        if not wa_words:
            stats["warnings"]["empty_word_attribution"] += 1
            stats["warning_details"].append({
                "example_id": idx,
                "warning_type": "empty_word_attribution",
                "details": "Empty word_attribution list"
            })
            continue

        # Check D: Each word attribution has required fields
        for w_idx, w in enumerate(word_attribution):
            if isinstance(w, dict):
                required_fields = ['word', 'subwords', 'sum', 'mean', 'max']
                for field in required_fields:
                    if field not in w:
                        stats["warnings"]["missing_field"] += 1
                        stats["warning_details"].append({
                            "example_id": idx,
                            "warning_type": "missing_field",
                            "details": f"Word {w_idx} missing field '{field}'"
                        })
                        break

        # Check E: Subwords list matches words (excluding special tokens and empty strings)
        wa_subwords = [w.get('subwords', []) if isinstance(w, dict) else [] for w in word_attribution]
        # Filter out special tokens and empty strings from token list for comparison
        content_tokens = [t.strip() for t in tokens if t not in ['[CLS]', '[SEP]', '<s>', '</s>', '']]
        total_subwords = sum(len(sw) for sw in wa_subwords)
        
        if total_subwords != len(content_tokens):
            stats["warnings"]["subword_count_mismatch"] += 1
            stats["warning_details"].append({
                "example_id": idx,
                "warning_type": "subword_count_mismatch",
                "details": f"Subwords count mismatch: subwords={total_subwords}, content_tokens={len(content_tokens)}, total_tokens={len(tokens)}"
            })
            continue

        # If we get here, the example is cleanly aligned
        stats["cleanly_aligned"] += 1

    stats["excluded"] = stats["total_examples"] - stats["cleanly_aligned"]

    # Print summary
    print(f"Total examples: {stats['total_examples']}")
    print(f"Cleanly aligned: {stats['cleanly_aligned']}")
    print(f"Warnings: {sum(stats['warnings'].values())}")
    print(f"Excluded: {stats['excluded']}")

    if stats['warnings']:
        print("\nWarning breakdown:")
        for warning_type, count in sorted(stats['warnings'].items(), key=lambda x: -x[1]):
            print(f"  {warning_type}: {count}")

    return stats

def main():
    print("\n" + "=" * 60)
    print("PHASE 3 — ALIGNMENT VALIDATION")
    print("=" * 60)

    print(f"Using attribution directory: {attribution_base_dir}")
    print(f"Attribution type: {attribution_type}")

    # Attribution files to validate
    attribution_files = [
        (attribution_base_dir / "ig_banglabert_token.json", "BanglaBERT", "Integrated Gradients"),
        (attribution_base_dir / "ig_xlm_r_token.json", "XLM-R", "Integrated Gradients"),
        (attribution_base_dir / "shap_banglabert_token.json", "BanglaBERT", "SHAP"),
        (attribution_base_dir / "shap_xlm_r_token.json", "XLM-R", "SHAP"),
    ]

    all_stats = {}
    for file_path, model, method in attribution_files:
        stats = validate_attribution_file(file_path, model, method)
        if stats:
            all_stats[f"{model}_{method}"] = stats

    # Generate summary report
    print("\n" + "=" * 60)
    print("ALIGNMENT VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Attribution type: {attribution_type}")
    print(f"Attribution directory: {attribution_base_dir}")

    summary_rows = []
    for key, stats in all_stats.items():
        summary_rows.append({
            "Model_Method": key,
            "Total": stats['total_examples'],
            "Cleanly_Aligned": stats['cleanly_aligned'],
            "Warnings": sum(stats['warnings'].values()),
            "Excluded": stats['excluded'],
            "Alignment_Rate": f"{stats['cleanly_aligned']/stats['total_examples']:.4f}"
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

    # Save detailed report
    report_path = OUT_DIR / "alignment_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase 3 — Alignment Validation Report ({attribution_type})\n\n")
        f.write(f"**Attribution Type**: {attribution_type}\n")
        f.write(f"**Attribution Directory**: {attribution_base_dir}\n\n")
        f.write("## Summary\n\n")
        f.write("| Model_Method | Total | Cleanly_Aligned | Warnings | Excluded | Alignment_Rate |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in summary_df.iterrows():
            f.write(f"| {row['Model_Method']} | {row['Total']} | {row['Cleanly_Aligned']} | {row['Warnings']} | {row['Excluded']} | {row['Alignment_Rate']} |\n")
        f.write("\n")

        f.write("## Detailed Warnings\n\n")
        for key, stats in all_stats.items():
            f.write(f"### {key}\n\n")
            f.write(f"- Total examples: {stats['total_examples']}\n")
            f.write(f"- Cleanly aligned: {stats['cleanly_aligned']}\n")
            f.write(f"- Warnings: {sum(stats['warnings'].values())}\n")
            f.write(f"- Excluded: {stats['excluded']}\n\n")

            if stats['warnings']:
                f.write("#### Warning Types\n\n")
                for warning_type, count in sorted(stats['warnings'].items(), key=lambda x: -x[1]):
                    f.write(f"- {warning_type}: {count}\n")
                f.write("\n")

            if stats['warning_details']:
                f.write("#### Warning Details (first 20)\n\n")
                f.write("| Example ID | Warning Type | Details |\n")
                f.write("|-----------|--------------|---------|\n")
                for detail in stats['warning_details'][:20]:
                    f.write(f"| {detail['example_id']} | {detail['warning_type']} | {detail['details']} |\n")
                f.write("\n")

    # Save detailed warnings to CSV
    all_warnings = []
    for key, stats in all_stats.items():
        model, method = key.rsplit('_', 1)
        for detail in stats['warning_details']:
            all_warnings.append({
                "model": model,
                "attribution_method": method,
                "example_id": detail['example_id'],
                "warning_type": detail['warning_type'],
                "details": detail['details']
            })

    if all_warnings:
        warnings_df = pd.DataFrame(all_warnings)
        warnings_path = OUT_DIR / "alignment_warnings.csv"
        warnings_df.to_csv(warnings_path, index=False, encoding="utf-8")
        print(f"\nDetailed warnings saved to: {warnings_path}")

    # Save alignment statistics
    stats_path = OUT_DIR / "alignment_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"Alignment statistics saved to: {stats_path}")

    print(f"\nFull report saved to: {report_path}")
    print("\n" + "=" * 60)
    print("Phase 3 Alignment Validation complete.")
    print("=" * 60)

    # Check if alignment rate is acceptable
    min_alignment_rate = 0.95  # 95% alignment required
    for key, stats in all_stats.items():
        alignment_rate = stats['cleanly_aligned'] / stats['total_examples']
        if alignment_rate < min_alignment_rate:
            print(f"\n[WARNING] {key} has low alignment rate: {alignment_rate:.4f} < {min_alignment_rate}")
            print("Consider investigating attribution generation before proceeding.")

if __name__ == "__main__":
    main()
