"""
Phase 1 — Alignment Validator
==============================
Validates every test example's token-to-word alignment for both models,
both attribution methods (IG and SHAP).

Replaces the vague "two warnings during spot-check" with a full quantitative report.

Outputs:
  results/alignment/alignment_warnings.csv
  results/alignment/alignment_summary.json
  results/alignment/alignment_validator.log

Run:
  .venv\Scripts\python.exe -u evaluation/alignment_validator.py
"""

import os, sys, io, json, re, logging
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "results" / "alignment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "alignment_validator.log"
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ATTR_DIR = BASE_DIR / "outputs" / "attribution"
TEST_CSV  = BASE_DIR / "test.csv"

MODEL_CONFIGS = {
    "BanglaBERT": {
        "ig_token_json":   ATTR_DIR / "ig_banglabert_token.json",
        "ig_word_csv":     ATTR_DIR / "ig_banglabert_word.csv",
        "shap_token_json": ATTR_DIR / "shap_banglabert_token.json",
        "shap_word_csv":   ATTR_DIR / "shap_banglabert_word.csv",
    },
    "XLM-R": {
        "ig_token_json":   ATTR_DIR / "ig_xlm_r_token.json",
        "ig_word_csv":     ATTR_DIR / "ig_xlm_r_word.csv",
        "shap_token_json": ATTR_DIR / "shap_xlm_r_token.json",
        "shap_word_csv":   ATTR_DIR / "shap_xlm_r_word.csv",
    },
}

TEXT_COL  = "sentence"
LABEL_COL = "hate speech"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def build_word_spans(text):
    """Character-level word spans; trims trailing punctuation into separate spans."""
    spans = []
    cursor = 0
    for raw_word in text.split():
        idx = text.find(raw_word, cursor)
        if idx == -1:
            idx = cursor
        end = idx + len(raw_word)
        m = re.search(r'([।,!?;:\.\"\'\(\)]+)$', raw_word)
        if m and len(raw_word) > len(m.group(1)):
            core = raw_word[: -len(m.group(1))]
            spans.append((idx, idx + len(core), core))
            spans.append((idx + len(core), end, m.group(1)))
        else:
            spans.append((idx, end, raw_word))
        cursor = end
    return spans


def classify_token(token: str, model_name: str) -> str:
    """Classify a token into one of the reporting categories."""
    if token in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]"):
        return "special_token"
    if token in ("[UNK]", "<unk>"):
        return "unk"
    # subword markers
    cleaned = token.replace("##", "").replace("▁", "").replace("Ġ", "").strip()
    if not cleaned:
        return "special_token"
    if all(not ch.isalnum() for ch in cleaned):
        return "punctuation"
    return "content"


# ─────────────────────────────────────────────────────────────
# Load test data
# ─────────────────────────────────────────────────────────────
logger.info("Loading test data …")
test_df = pd.read_csv(TEST_CSV)
test_df = test_df.dropna(subset=[TEXT_COL]).copy()
test_df[TEXT_COL] = test_df[TEXT_COL].astype(str).str.strip()
test_df = test_df[test_df[TEXT_COL].str.len() > 0].reset_index(drop=True)
logger.info(f"Test examples: {len(test_df):,}")

# ─────────────────────────────────────────────────────────────
# Validation function
# ─────────────────────────────────────────────────────────────
def validate_model_method(model_name: str, method: str,
                          token_json_path: Path, word_csv_path: Path,
                          test_df: pd.DataFrame):
    """
    For every example, check alignment and return a list of warning dicts.
    Returns (warnings_list, stats_dict).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating: {model_name} / {method}")
    logger.info(f"  Token JSON : {token_json_path}")
    logger.info(f"  Word CSV   : {word_csv_path}")

    # Load token-level data
    if not token_json_path.exists():
        logger.error(f"Token JSON not found: {token_json_path}")
        return [], {"error": "token_json_missing"}

    with open(token_json_path, encoding="utf-8") as f:
        token_data = json.load(f)

    # Build lookup by example index (token data may be list or dict keyed by str id)
    if isinstance(token_data, list):
        # list of dicts with 'id' or positional index
        tok_lookup = {}
        for i, rec in enumerate(token_data):
            key = rec.get("id", rec.get("example_id", i))
            tok_lookup[int(key)] = rec
    elif isinstance(token_data, dict):
        tok_lookup = {int(k): v for k, v in token_data.items()}
    else:
        logger.error("Unexpected token JSON format")
        return [], {"error": "bad_token_json_format"}

    # Load word-level CSV
    if not word_csv_path.exists():
        logger.error(f"Word CSV not found: {word_csv_path}")
        return [], {"error": "word_csv_missing"}

    word_df = pd.read_csv(word_csv_path)
    # Build lookup: text → list of (word, attribution) pairs
    # The CSV has columns: example_id or text, word, attribution/score
    # Figure out column names
    text_col_csv = None
    for c in ["text", "sentence", TEXT_COL]:
        if c in word_df.columns:
            text_col_csv = c
            break
    id_col_csv = None
    for c in ["example_id", "id"]:
        if c in word_df.columns:
            id_col_csv = c
            break

    warnings_list = []
    stats = {
        "total": 0,
        "clean": 0,
        "token_count_mismatch": 0,
        "word_reconstruction_fail": 0,
        "unexplained_tokens": 0,
        "excluded": 0,
    }

    for ex_idx, row in test_df.iterrows():
        text = row[TEXT_COL]
        stats["total"] += 1

        # ── Check A: token count == attribution count ──────────────
        tok_rec = tok_lookup.get(ex_idx)
        check_a_ok = True
        if tok_rec is not None:
            tokens = tok_rec.get("tokens", [])
            # Both IG and SHAP use 'token_scores_raw' in this codebase
            attrs = tok_rec.get("token_scores_raw",
                    tok_rec.get("attributions",
                    tok_rec.get("shap_values", [])))
            if isinstance(attrs, list) and len(attrs) > 0 and isinstance(attrs[0], list):
                # flatten nested lists (e.g., shap multi-class values)
                attrs = [float(a[0]) if isinstance(a, list) else float(a) for a in attrs]
            if len(tokens) != len(attrs):
                check_a_ok = False
                warnings_list.append({
                    "example_id": ex_idx,
                    "model": model_name,
                    "method": method,
                    "warning_type": "token_count_mismatch",
                    "details": f"tokens={len(tokens)}, attributions={len(attrs)}",
                })
                stats["token_count_mismatch"] += 1
        else:
            # Record that example has no token-level record
            warnings_list.append({
                "example_id": ex_idx,
                "model": model_name,
                "method": method,
                "warning_type": "missing_token_record",
                "details": f"example_id={ex_idx} not in token JSON",
            })

        # ── Check B: word reconstruction ───────────────────────────
        check_b_ok = True
        if tok_rec is not None:
            tokens = tok_rec.get("tokens", [])
            # filter special tokens and reconstruct
            content_tokens = [
                t for t in tokens
                if classify_token(t, model_name) == "content"
            ]
            # rough check: joined content tokens vs original words
            original_words = set(text.lower().split())
            reconstructed  = set()
            for t in content_tokens:
                clean = t.replace("##", "").replace("▁", "").replace("Ġ", "").lower().strip()
                if clean:
                    reconstructed.add(clean)
            # at least half the original words should appear in reconstructed
            if original_words:
                overlap = len(original_words & reconstructed) / len(original_words)
                if overlap < 0.5:
                    check_b_ok = False
                    warnings_list.append({
                        "example_id": ex_idx,
                        "model": model_name,
                        "method": method,
                        "warning_type": "word_reconstruction_fail",
                        "details": f"overlap={overlap:.2f}, original_words={len(original_words)}, reconstructed={len(reconstructed)}",
                    })
                    stats["word_reconstruction_fail"] += 1

        # ── Check C: unexplained tokens ────────────────────────────
        if tok_rec is not None:
            tokens = tok_rec.get("tokens", [])
            token_cats = [classify_token(t, model_name) for t in tokens]
            unk_count = token_cats.count("unk")
            if unk_count > 0:
                warnings_list.append({
                    "example_id": ex_idx,
                    "model": model_name,
                    "method": method,
                    "warning_type": "unk_token",
                    "details": f"unk_count={unk_count}, total_tokens={len(tokens)}",
                })
                stats["unexplained_tokens"] += 1

        if check_a_ok and check_b_ok:
            stats["clean"] += 1

    logger.info(f"  Total:   {stats['total']:,}")
    logger.info(f"  Clean:   {stats['clean']:,}  ({stats['clean']/max(1,stats['total'])*100:.1f}%)")
    logger.info(f"  Token count mismatch: {stats['token_count_mismatch']}")
    logger.info(f"  Word reconstruction fail: {stats['word_reconstruction_fail']}")
    logger.info(f"  Examples with UNK: {stats['unexplained_tokens']}")
    logger.info(f"  Warnings total: {len(warnings_list)}")

    return warnings_list, stats


# ─────────────────────────────────────────────────────────────
# Run validation for all combinations
# ─────────────────────────────────────────────────────────────
all_warnings = []
all_stats = {}

for model_name, paths in MODEL_CONFIGS.items():
    for method in ["ig", "shap"]:
        token_key = f"{method}_token_json"
        word_key  = f"{method}_word_csv"
        if token_key not in paths:
            logger.warning(f"No paths for {model_name}/{method}, skipping")
            continue

        warns, stats = validate_model_method(
            model_name=model_name,
            method=method.upper(),
            token_json_path=paths[token_key],
            word_csv_path=paths[word_key],
            test_df=test_df,
        )
        all_warnings.extend(warns)
        all_stats[f"{model_name}_{method.upper()}"] = stats

# ─────────────────────────────────────────────────────────────
# Save warnings CSV
# ─────────────────────────────────────────────────────────────
if all_warnings:
    warn_df = pd.DataFrame(all_warnings)
    warn_csv = OUT_DIR / "alignment_warnings.csv"
    warn_df.to_csv(warn_csv, index=False, encoding="utf-8")
    logger.info(f"\nWarnings CSV saved → {warn_csv}  ({len(warn_df):,} rows)")
else:
    logger.info("\nNo warnings recorded.")

# ─────────────────────────────────────────────────────────────
# Save summary JSON
# ─────────────────────────────────────────────────────────────
summary = {
    "test_size": len(test_df),
    "per_condition": all_stats,
    "total_warnings": len(all_warnings),
    "warning_breakdown": {},
}

if all_warnings:
    warn_df_full = pd.DataFrame(all_warnings)
    for wtype, grp in warn_df_full.groupby("warning_type"):
        summary["warning_breakdown"][wtype] = int(len(grp))

summary_path = OUT_DIR / "alignment_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
logger.info(f"Summary JSON saved → {summary_path}")

# ─────────────────────────────────────────────────────────────
# Print final report
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALIGNMENT VALIDATION SUMMARY")
print("=" * 60)
print(f"Total test examples : {len(test_df):,}")
for cond, stats in all_stats.items():
    clean_pct = stats.get("clean", 0) / max(1, stats.get("total", 1)) * 100
    print(f"\n{cond}:")
    print(f"  Clean     : {stats.get('clean', 0):,}  ({clean_pct:.1f}%)")
    print(f"  Warnings  : {stats.get('token_count_mismatch', 0)} token-mismatch  |  "
          f"{stats.get('word_reconstruction_fail', 0)} word-recon  |  "
          f"{stats.get('unexplained_tokens', 0)} UNK")
print(f"\nTotal warning events: {len(all_warnings):,}")
print(f"\nOutputs in: {OUT_DIR}")
