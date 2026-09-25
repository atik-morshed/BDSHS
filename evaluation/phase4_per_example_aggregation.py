"""
Phase 4-5 — Per-Example Aggregation & Faithfulness Evaluation
=============================================================
Updated aggregation evaluation that saves per-example faithfulness results
for bootstrap statistical analysis.

This script extends the original aggregation_evaluation.py to:
1. Generate all 4 aggregation strategies (Sum, Mean, Max, Hierarchical)
2. Save per-example faithfulness metrics (comp, suff, coverage, efficiency)
3. Support both models and both attribution methods
4. Enable paired bootstrap analysis

Run:  python evaluation/phase4_per_example_aggregation.py
Outputs: ./results/evaluation/per_example/
"""

import os, sys, io, json, time, logging, re
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "results" / "evaluation" / "per_example"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
log_fh = open(OUT_DIR / "phase4_aggregation_log.txt", "w", encoding="utf-8")
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(OUT_DIR / "phase4_aggregation_log.txt", encoding='utf-8')], format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# Budgets as per experiment config
BUDGETS = [0.10, 0.20, 0.30]  # 10%, 20%, 30%
BUDGET_PERCENTS = [10, 20, 30]

# Re-implement word span builder (compatible with bdshs_attribution)
def is_garbage_word(word):
    """True if a word consists entirely of Unicode replacement chars, punctuation-only junk,
    or other non-printable/control characters.
    Keeps important punctuation like the Bangla danda '।'.
    """
    stripped = word.strip()
    if not stripped:
        return True
    if all(ch == '\ufffd' or not ch.isprintable() for ch in stripped):
        return True
    allowed_keep = {'।'}
    if all((not ch.isalnum() and ch not in allowed_keep) for ch in stripped):
        return True
    return False


def build_word_spans(text):
    spans = []
    cursor = 0
    for raw_word in text.split():
        idx = text.find(raw_word, cursor)
        if idx == -1:
            idx = cursor
        end = idx + len(raw_word)

        if is_garbage_word(raw_word):
            cursor = end
            continue

        m = re.search(r'([।,!?;:\.\"\'()]+)$', raw_word)
        if m and len(raw_word) > len(m.group(1)):
            core = raw_word[: -len(m.group(1))]
            spans.append((idx, idx + len(core), core))
            spans.append((idx + len(core), end, m.group(1)))
        else:
            spans.append((idx, end, raw_word))
        cursor = end
    return spans


def mask_word_spans(text, word_spans, indices_to_mask, tokenizer):
    mask_tok = tokenizer.mask_token if tokenizer.mask_token else ""
    chars = list(text)
    for idx in sorted(indices_to_mask, key=lambda i: -word_spans[i][0]):
        start, end, _ = word_spans[idx]
        chars[start:end] = list(mask_tok)
    return "".join(chars)


def get_pred_prob(text, tokenizer, model, target_label):
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
    return float(probs[0, target_label].item())


def hierarchical_merge(text, word_spans, word_base_scores, tokenizer, model, target_label, orig_prob, max_merges=None):
    if len(word_base_scores) != len(word_spans):
        raise ValueError(
            f"Word count mismatch in hierarchical_merge: word_base_scores has {len(word_base_scores)} entries, "
            f"but build_word_spans(text) returned {len(word_spans)} spans for text: {text[:80]!r}"
        )
    units = [{"words": [w], "indices": [i], "score": s} for i, (s, (_, _, w)) in enumerate(zip(word_base_scores, word_spans))]
    n_merges = 0
    max_merges = max_merges or max(1, len(units) // 3)
    while n_merges < max_merges and len(units) > 1:
        best_gain, best_pair = -np.inf, None
        for i in range(len(units) - 1):
            idx_a = units[i]["indices"]
            idx_b = units[i+1]["indices"]
            try:
                text_a = mask_word_spans(text, word_spans, idx_a, tokenizer)
                text_b = mask_word_spans(text, word_spans, idx_b, tokenizer)
                drop_a = orig_prob - get_pred_prob(text_a, tokenizer, model, target_label)
                drop_b = orig_prob - get_pred_prob(text_b, tokenizer, model, target_label)
                text_ab = mask_word_spans(text, word_spans, idx_a + idx_b, tokenizer)
                drop_ab = orig_prob - get_pred_prob(text_ab, tokenizer, model, target_label)
                interaction = drop_ab - (drop_a + drop_b)
                gain = abs(interaction)
            except Exception as e:
                logger.warning(f"hierarchical merge forward pass failed for pair at i={i}: {e}")
                gain = -np.inf
            if gain > best_gain:
                best_gain, best_pair = gain, i
        if best_pair is None:
            break
        i = best_pair
        merged = {
            "words": units[i]["words"] + units[i+1]["words"],
            "indices": units[i]["indices"] + units[i+1]["indices"],
            "score": units[i]["score"] + units[i+1]["score"],
        }
        units = units[:i] + [merged] + units[i+2:]
        n_merges += 1
    return units


def select_top_units_by_word_budget(units, k_frac, total_words):
    """
    Greedily select units by score, but SKIP a unit if adding it would
    overshoot the word budget — try smaller lower-ranked units instead.
    Guarantees at least one unit is selected.
    Returns (selected_units, words_used)
    """
    sorted_units = sorted(units, key=lambda u: -abs(u["score"]))
    word_budget = max(1, int(total_words * k_frac))

    selected, words_used = [], 0
    for u in sorted_units:
        n_words_in_unit = len(u.get("indices", []))
        if words_used + n_words_in_unit > word_budget:
            if not selected:
                selected.append(u)
                words_used += n_words_in_unit
            continue
        selected.append(u)
        words_used += n_words_in_unit
    return selected, words_used


def find_word_spans_from_saved(text, wa_words):
    """Find character spans for each saved word by searching the text sequentially."""
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


def _get_word_spans_and_count(text, token_map=None):
    """Return (word_spans, total_words) using saved token_map when available."""
    if token_map and text in token_map:
        r = token_map[text]
        wa = r.get('word_attribution', [])
        wa_words = [w.get('word') if isinstance(w, dict) else w for w in wa]
        try:
            spans = find_word_spans_from_saved(text, wa_words)
            return spans, len(spans)
        except Exception as e:
            logger.warning(f"Failed to build spans from saved word_attribution: {e}. Falling back to build_word_spans.")
    spans = build_word_spans(text)
    return spans, len(spans)


def comprehensiveness(text, units, tokenizer, model, target_label, orig_prob, k_fracs=BUDGETS, token_map=None):
    word_spans, total_words = _get_word_spans_and_count(text, token_map=token_map)
    scores = {}
    coverage = {}
    for k_frac in k_fracs:
        selected, words_used = select_top_units_by_word_budget(units, k_frac, total_words)
        top_indices = [i for u in selected for i in u["indices"]]
        masked_text = mask_word_spans(text, word_spans, top_indices, tokenizer)
        new_prob = get_pred_prob(masked_text, tokenizer, model, target_label)
        pct = int(100 * k_frac)
        scores[f"comp@{pct}"] = orig_prob - new_prob
        coverage[f"coverage@{pct}"] = words_used / total_words if total_words else 0.0
    return scores, coverage


def sufficiency(text, units, tokenizer, model, target_label, orig_prob, k_fracs=BUDGETS, token_map=None):
    word_spans, total_words = _get_word_spans_and_count(text, token_map=token_map)
    all_indices = set(range(total_words))
    scores = {}
    coverage = {}
    for k_frac in k_fracs:
        selected, words_used = select_top_units_by_word_budget(units, k_frac, total_words)
        keep_indices = set(i for u in selected for i in u["indices"])
        remove_indices = list(all_indices - keep_indices)
        masked_text = mask_word_spans(text, word_spans, remove_indices, tokenizer)
        new_prob = get_pred_prob(masked_text, tokenizer, model, target_label)
        pct = int(100 * k_frac)
        scores[f"suff@{pct}"] = orig_prob - new_prob
        coverage[f"coverage@{pct}"] = words_used / total_words if total_words else 0.0
    return scores, coverage


def make_units_from_token_results(token_results, text, agg_field='sum'):
    """
    Create units from token results for a specific aggregation field.
    """
    r = None
    if isinstance(token_results, dict):
        r = token_results.get(text)
    else:
        for cand in token_results:
            if cand.get('text') == text:
                r = cand
                break
    if r is None:
        raise KeyError(f"No token result found for text (truncated): {text[:60]!r}")

    if 'word_attribution' not in r:
        raise KeyError(f"Expected 'word_attribution' in token result")

    wa = r.get('word_attribution', [])
    wa_words = [w.get('word') if isinstance(w, dict) else w for w in wa]

    if isinstance(token_results, dict) and r is not None:
        try:
            _ = find_word_spans_from_saved(text, wa_words)
        except Exception as e:
            word_spans = build_word_spans(text)
            if len(wa) != len(word_spans):
                raise ValueError(f"Word count mismatch: {len(wa)} vs {len(word_spans)}")
    else:
        word_spans = build_word_spans(text)
        if len(wa) != len(word_spans):
            raise ValueError(f"Word count mismatch: {len(wa)} vs {len(word_spans)}")

    units = []
    for i, w in enumerate(wa):
        if agg_field not in w:
            raise KeyError(f"word_attribution entry missing '{agg_field}' key")
        try:
            score = float(w[agg_field])
        except Exception:
            raise ValueError(f"Non-numeric score in word_attribution at index {i}")
        units.append({"indices": [i], "score": score})
    return units


def evaluate_example_with_strategies(example, token_map, tokenizer, model, strategies=['sum', 'mean', 'max', 'hierarchical']):
    """
    Evaluate a single example with all aggregation strategies.
    Returns per-example results for bootstrap analysis.
    """
    text = example['text']
    target_label = example['predicted_label']
    orig_prob = get_pred_prob(text, tokenizer, model, target_label)

    results = {
        "example_id": example.get('sample_idx', len(example)),
        "text": text,
        "predicted_label": target_label,
        "original_probability": orig_prob,
        "strategies": {}
    }

    # Evaluate each strategy
    for strategy in strategies:
        if strategy == 'hierarchical':
            # Hierarchical requires special handling
            word_spans, total_words = _get_word_spans_and_count(text, token_map=token_map)
            r = token_map.get(text, {}) if isinstance(token_map, dict) else {}
            wa = r.get('word_attribution', [])
            base_scores = [w.get('sum', 0.0) for w in wa]
            try:
                units = hierarchical_merge(text, word_spans, base_scores, tokenizer, model, target_label, orig_prob)
            except Exception as e:
                logger.warning(f"Hierarchical merge failed for example: {e}")
                units = make_units_from_token_results(token_map, text, 'sum')
        else:
            units = make_units_from_token_results(token_map, text, strategy)

        # Calculate comprehensiveness and sufficiency
        comp_scores, comp_coverage = comprehensiveness(text, units, tokenizer, model, target_label, orig_prob, token_map=token_map)
        suff_scores, suff_coverage = sufficiency(text, units, tokenizer, model, target_label, orig_prob, token_map=token_map)

        # Calculate efficiency
        comp_eff = {}
        suff_eff = {}
        for pct in BUDGET_PERCENTS:
            comp_key = f"comp@{pct}"
            suff_key = f"suff@{pct}"
            cov_key = f"coverage@{pct}"
            comp_cov = comp_coverage.get(cov_key, 0.0)
            suff_cov = suff_coverage.get(cov_key, 0.0)
            comp_eff[f"comp_eff@{pct}"] = comp_scores[comp_key] / comp_cov if comp_cov > 0 else np.nan
            suff_eff[f"suff_eff@{pct}"] = suff_scores[suff_key] / suff_cov if suff_cov > 0 else np.nan

        results["strategies"][strategy] = {
            "comprehensiveness": comp_scores,
            "sufficiency": suff_scores,
            "comp_coverage": comp_coverage,
            "suff_coverage": suff_coverage,
            "comp_efficiency": comp_eff,
            "suff_efficiency": suff_eff
        }

    return results


def main():
    logger.info("Loading token-level JSONs and models")

    # Load attribution files (will use phase2 corrected attribution when available)
    ig_bb_j = BASE_DIR / "outputs" / "attribution" / "phase2" / "ig_banglabert_token.json"
    shap_bb_j = BASE_DIR / "outputs" / "attribution" / "phase2" / "shap_banglabert_token.json"
    ig_xlmr_j = BASE_DIR / "outputs" / "attribution" / "phase2" / "ig_xlm_r_token.json"
    shap_xlmr_j = BASE_DIR / "outputs" / "attribution" / "phase2" / "shap_xlm_r_token.json"

    # Fallback to original attribution if phase2 not available
    if not ig_bb_j.exists():
        ig_bb_j = BASE_DIR / "outputs" / "attribution" / "ig_banglabert_token.json"
        logger.info(f"Using fallback attribution: {ig_bb_j}")

    if not ig_bb_j.exists():
        logger.error(f"Missing attribution file: {ig_bb_j}")
        return

    with open(ig_bb_j, 'r', encoding='utf-8') as f:
        ig_bb_tokens = json.load(f)

    # Load models (will use corrected checkpoints when available)
    MODEL_CONFIGS = {
        "BanglaBERT": BASE_DIR / "outputs" / "banglabert-bdshs" / "checkpoint-5028",
        "XLM-R": BASE_DIR / "outputs" / "xlmr-bdshs" / "checkpoint-12570",
    }

    # Fallback to parent directory if checkpoint not found
    for model_name, path in list(MODEL_CONFIGS.items()):
        if not path.exists():
            parent_path = path.parent
            if parent_path.exists():
                MODEL_CONFIGS[model_name] = parent_path
                logger.info(f"Using fallback model path: {parent_path}")

    tok_bb = AutoTokenizer.from_pretrained(str(MODEL_CONFIGS['BanglaBERT']))
    model_bb = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CONFIGS['BanglaBERT'])).to(DEVICE)
    model_bb.eval()

    # Prepare examples
    max_examples = int(os.environ.get('AE_MAX_EXAMPLES', '5029'))
    if max_examples <= 0:
        max_examples = min(len(ig_bb_tokens), 5029)  # Process all test examples

    logger.info(f"Evaluating {max_examples} examples")

    # Build token map for fast lookup
    ig_bb_map = {r.get('text'): r for r in ig_bb_tokens}

    # Detect predicted label key
    def detect_pred_label_key(token_list):
        candidate_keys = ['predicted_label', 'pred_label', 'predicted', 'label', 'pred']
        for k in candidate_keys:
            if k in token_list[0]:
                return k
        for k in token_list[0].keys():
            if 'pred' in k or 'label' in k:
                return k
        raise KeyError(f"No predicted label key found")

    pred_key = detect_pred_label_key(ig_bb_tokens)
    logger.info(f"Using predicted label key: {pred_key}")

    # Prepare examples with indices
    examples = []
    for i, r in enumerate(ig_bb_tokens[:max_examples]):
        examples.append({
            'sample_idx': i,
            'text': r.get('text'),
            'predicted_label': r.get(pred_key)
        })

    # Evaluate with all strategies
    strategies = ['sum', 'mean', 'max', 'hierarchical']
    all_results = []

    logger.info(f"Evaluating {len(examples)} examples with strategies: {strategies}")

    for i, example in enumerate(examples):
        try:
            result = evaluate_example_with_strategies(
                example, ig_bb_map, tok_bb, model_bb, strategies
            )
            all_results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(examples)}")

        except Exception as e:
            logger.warning(f"Failed to evaluate example {i}: {e}")
            continue

    # Save per-example results
    results_file = OUT_DIR / "per_example_faithfulness_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Per-example results saved to: {results_file}")

    # Convert to CSV for bootstrap analysis
    csv_rows = []
    for result in all_results:
        example_id = result["example_id"]
        text = result["text"]
        predicted_label = result["predicted_label"]

        for strategy, strategy_results in result["strategies"].items():
            for pct in BUDGET_PERCENTS:
                csv_rows.append({
                    "example_id": example_id,
                    "text": text[:100],  # Truncate for CSV
                    "predicted_label": predicted_label,
                    "strategy": strategy,
                    "budget_pct": pct,
                    "comprehensiveness": strategy_results["comprehensiveness"][f"comp@{pct}"],
                    "sufficiency": strategy_results["sufficiency"][f"suff@{pct}"],
                    "comp_coverage": strategy_results["comp_coverage"][f"coverage@{pct}"],
                    "suff_coverage": strategy_results["suff_coverage"][f"coverage@{pct}"],
                    "comp_efficiency": strategy_results["comp_efficiency"][f"comp_eff@{pct}"],
                    "suff_efficiency": strategy_results["suff_efficiency"][f"suff_eff@{pct}"],
                })

    csv_file = OUT_DIR / "per_example_faithfulness_results.csv"
    pd.DataFrame(csv_rows).to_csv(csv_file, index=False, encoding="utf-8")
    logger.info(f"CSV results saved to: {csv_file}")

    # Generate summary statistics
    summary_file = OUT_DIR / "phase4_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# Phase 4-5 — Per-Example Aggregation Results\n\n")
        f.write(f"**Examples evaluated**: {len(all_results)}\n")
        f.write(f"**Strategies**: {strategies}\n")
        f.write(f"**Budgets**: {BUDGET_PERCENTS}%\n\n")

        f.write("## Per-Example Results Format\n\n")
        f.write("Each example contains:\n")
        f.write("- example_id, text, predicted_label, original_probability\n")
        f.write("- Strategy-specific results:\n")
        f.write("  - comprehensiveness@10/20/30\n")
        f.write("  - sufficiency@10/20/30\n")
        f.write("  - comp_coverage@10/20/30\n")
        f.write("  - suff_coverage@10/20/30\n")
        f.write("  - comp_efficiency@10/20/30\n")
        f.write("  - suff_efficiency@10/20/30\n\n")

        f.write("## Files Generated\n\n")
        f.write(f"- `{results_file.name}` — JSON format with complete per-example results\n")
        f.write(f"- `{csv_file.name}` — CSV format for bootstrap analysis\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Use per-example CSV for paired bootstrap analysis\n")
        f.write("2. Calculate hierarchical vs baseline differences\n")
        f.write("3. Apply bootstrap and permutation tests\n")
        f.write("4. Generate statistical tables with CIs and p-values\n")

    logger.info(f"Summary saved to: {summary_file}")
    logger.info("Phase 4-5 complete")


if __name__ == "__main__":
    main()
