"""
Aggregation strategy comparison and faithfulness evaluation
Saves logs to outputs/attribution/aggregation_log.txt and outputs/attribution/aggregation_report.md
Run with GPU-enabled env; example one-liner provided by the assistant.
"""
from pathlib import Path
import os, json, time, logging
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
log_fh = open(OUT_DIR / "aggregation_log.txt", "w", encoding="utf-8")
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(OUT_DIR / "aggregation_log.txt", encoding='utf-8')], format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# Re-implement a punctuation-aware word span builder (compatible with bdshs_attribution)
import re

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


def build_word_spans(text):
    spans = []
    cursor = 0
    for raw_word in text.split():
        idx = text.find(raw_word, cursor)
        if idx == -1:
            idx = cursor
        end = idx + len(raw_word)

        # skip garbage-only tokens (e.g., runs of U+FFFD replacement characters)
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
    mask_tok = tokenizer.mask_token if tokenizer.mask_token else "[MASK]"
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
    # Ensure alignment between provided base scores and computed word spans
    if len(word_base_scores) != len(word_spans):
        raise ValueError(
            f"Word count mismatch in hierarchical_merge: word_base_scores has {len(word_base_scores)} entries, "
            f"but build_word_spans(text) returned {len(word_spans)} spans for text: {text[:80]!r}"
        )
    units = [{"words": [w], "indices": [i], "score": s} for i, (s, (_, _, w)) in enumerate(zip(word_base_scores, word_spans))]
    # Note: above zip ensures alignment: word_spans[i] corresponds to word
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
    overshoot the word budget — try smaller lower-ranked units instead,
    so hierarchical doesn't get a free pass on touching more text.
    Guarantees at least one unit is selected (can't do better with atomic units).
    Returns (selected_units, words_used)
    """
    sorted_units = sorted(units, key=lambda u: -abs(u["score"]))
    word_budget = max(1, int(total_words * k_frac))

    selected, words_used = [], 0
    for u in sorted_units:
        n_words_in_unit = len(u.get("indices", []))
        if words_used + n_words_in_unit > word_budget:
            if not selected:
                # nothing selected yet — must take at least one unit,
                # even if it overshoots (unavoidable with atomic merged units)
                selected.append(u)
                words_used += n_words_in_unit
            # skip this unit and keep looking for smaller ones that fit
            continue
        selected.append(u)
        words_used += n_words_in_unit
    return selected, words_used


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


def _get_word_spans_and_count(text, token_map=None):
    """Return (word_spans, total_words) using saved token_map when available,
    otherwise fall back to build_word_spans(text).
    """
    if token_map and text in token_map:
        r = token_map[text]
        wa = r.get('word_attribution', [])
        wa_words = [w.get('word') if isinstance(w, dict) else w for w in wa]
        try:
            spans = find_word_spans_from_saved(text, wa_words)
            return spans, len(spans)
        except Exception as e:
            logger.warning(f"Failed to build spans from saved word_attribution for text (trunc): {text[:80]!r}: {e}. Falling back to build_word_spans.")
    spans = build_word_spans(text)
    return spans, len(spans)


def comprehensiveness(text, units, tokenizer, model, target_label, orig_prob, k_fracs=(0.1,0.2,0.3), token_map=None):
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


def sufficiency(text, units, tokenizer, model, target_label, orig_prob, k_fracs=(0.1,0.2,0.3), token_map=None):
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
    token_results may be a list or a dict mapping text -> token record.
    This function finds the record for the given text, validates the schema,
    and returns a list of single-word units with scores taken from agg_field.
    Raises informative errors on schema mismatches instead of silently falling back.
    """
    r = None
    if isinstance(token_results, dict):
        r = token_results.get(text)
    else:
        # list: linear scan
        for cand in token_results:
            if cand.get('text') == text:
                r = cand
                break
    if r is None:
        # helpful debugging: show available sample keys and a short listing
        sample_keys = []
        if isinstance(token_results, list) and token_results:
            sample_keys = list(token_results[0].keys())
        raise KeyError(f"No token result found for text (truncated): {text[:60]!r}. Available top-level keys in token results sample: {sample_keys}")

    if 'word_attribution' not in r:
        raise KeyError(f"Expected 'word_attribution' in token result but it is missing. Available keys: {list(r.keys())}")

    wa = r.get('word_attribution', [])
    # If token_results is a dict (fast lookup), try to build spans from the saved words themselves
    # so we match the original attribution script's word boundaries exactly.
    wa_words = [w.get('word') if isinstance(w, dict) else w for w in wa]
    if isinstance(token_results, dict) and r is not None:
        try:
            _ = find_word_spans_from_saved(text, wa_words)
        except Exception as e:
            # If we can't build spans from saved words, fall back to the old check and warn
            word_spans = build_word_spans(text)
            if len(wa) != len(word_spans):
                raise ValueError(
                    f"Word count mismatch between saved word_attribution ({len(wa)}) and build_word_spans ({len(word_spans)}) for text: {text[:80]!r}.\n"
                    f"This indicates the tokenization/span logic diverged between the attribution script and the evaluator. Inspect the sentence manually."
                )
    else:
        # validate lengths against build_word_spans so downstream alignments cannot silently break
        word_spans = build_word_spans(text)
        if len(wa) != len(word_spans):
            raise ValueError(
                f"Word count mismatch between saved word_attribution ({len(wa)}) and build_word_spans ({len(word_spans)}) for text: {text[:80]!r}.\n"
                f"This indicates the tokenization/span logic diverged between the attribution script and the evaluator. Inspect the sentence manually."
            )

    units = []
    for i, w in enumerate(wa):
        # require the requested agg_field to be present — fail loudly if not
        if agg_field not in w:
            raise KeyError(f"word_attribution entry missing '{agg_field}' key. Available keys: {list(w.keys())}")
        try:
            score = float(w[agg_field])
        except Exception:
            raise ValueError(f"Non-numeric score in word_attribution at index {i}: {w}")
        units.append({"indices": [i], "score": score})
    return units


def run_hierarchical_on_examples(examples, tokenizer, model, sum_score_lookup, token_map=None, max_examples=50):
    results = []
    for ex in examples[:max_examples]:
        text = ex['text']
        target_label = ex['predicted_label']
        # try to obtain spans from saved token_map when possible
        if token_map and text in token_map:
            r = token_map[text]
            wa = r.get('word_attribution', [])
            wa_words = [w.get('word') if isinstance(w, dict) else w for w in wa]
            try:
                word_spans = find_word_spans_from_saved(text, wa_words)
            except Exception as e:
                logger.warning(f"Failed to build spans from saved word_attribution for hierarchical run on text (trunc): {text[:80]!r}: {e}. Falling back to build_word_spans.")
                word_spans = build_word_spans(text)
        else:
            word_spans = build_word_spans(text)
        base_scores = sum_score_lookup(text)
        orig_prob = get_pred_prob(text, tokenizer, model, target_label)
        units = hierarchical_merge(text, word_spans, base_scores, tokenizer, model, target_label, orig_prob)
        results.append({"text": text, "units": units, "target_label": target_label})
    return results


def evaluate_aggregation_strategy(examples, strategy_name, get_units_fn, tokenizer, model, k_fracs=(0.1,0.2,0.3), token_map=None):
    comp_records, suff_records, comp_cov_records, suff_cov_records = [], [], [], []
    for ex in examples:
        text, target_label = ex['text'], ex['predicted_label']
        orig_prob = get_pred_prob(text, tokenizer, model, target_label)
        units = get_units_fn(text, target_label)
        comp, comp_cov = comprehensiveness(text, units, tokenizer, model, target_label, orig_prob, k_fracs, token_map=token_map)
        suff, suff_cov = sufficiency(text, units, tokenizer, model, target_label, orig_prob, k_fracs, token_map=token_map)
        comp_records.append(comp)
        suff_records.append(suff)
        comp_cov_records.append(comp_cov)
        suff_cov_records.append(suff_cov)
    comp_df = pd.DataFrame(comp_records)
    suff_df = pd.DataFrame(suff_records)
    comp_cov_df = pd.DataFrame(comp_cov_records)
    suff_cov_df = pd.DataFrame(suff_cov_records)
    comp_mean = comp_df.mean().to_dict()
    suff_mean = suff_df.mean().to_dict()
    comp_cov_mean = comp_cov_df.mean().to_dict()
    suff_cov_mean = suff_cov_df.mean().to_dict()

    # compute efficiency = metric / coverage for each k (handle zero coverage)
    comp_eff = {}
    suff_eff = {}
    for comp_k, comp_v in comp_mean.items():
        # comp_k like 'comp@10'
        try:
            pct = comp_k.split('@')[1]
        except Exception:
            continue
        cov_key = f"coverage@{pct}"
        cov = comp_cov_mean.get(cov_key, None)
        if cov is None or cov == 0 or np.isnan(cov):
            comp_eff[f"comp_eff@{pct}"] = None
        else:
            comp_eff[f"comp_eff@{pct}"] = comp_v / cov
    for suff_k, suff_v in suff_mean.items():
        try:
            pct = suff_k.split('@')[1]
        except Exception:
            continue
        cov_key = f"coverage@{pct}"
        cov = suff_cov_mean.get(cov_key, None)
        if cov is None or cov == 0 or np.isnan(cov):
            suff_eff[f"suff_eff@{pct}"] = None
        else:
            suff_eff[f"suff_eff@{pct}"] = suff_v / cov

    return {
        "strategy": strategy_name,
        "comprehensiveness_mean": comp_mean,
        "sufficiency_mean": suff_mean,
        "comp_coverage_mean": comp_cov_mean,
        "suff_coverage_mean": suff_cov_mean,
        "comp_eff_mean": comp_eff,
        "suff_eff_mean": suff_eff,
        "n_examples": len(examples),
    }


def main():
    logger.info("Loading token-level JSONs and models")
    # files assumed in OUT_DIR
    ig_bb_j = OUT_DIR / 'ig_banglabert_token.json'
    shap_bb_j = OUT_DIR / 'shap_banglabert_token.json'
    ig_xlmr_j = OUT_DIR / 'ig_xlm_r_token.json'
    shap_xlmr_j = OUT_DIR / 'shap_xlm_r_token.json'
    if not ig_bb_j.exists():
        logger.error(f"Missing {ig_bb_j}")
        return
    with open(ig_bb_j, 'r', encoding='utf-8') as f:
        ig_bb_tokens = json.load(f)
    with open(shap_bb_j, 'r', encoding='utf-8') as f:
        shap_bb_tokens = json.load(f)
    with open(ig_xlmr_j, 'r', encoding='utf-8') as f:
        ig_xlmr_tokens = json.load(f)
    with open(shap_xlmr_j, 'r', encoding='utf-8') as f:
        shap_xlmr_tokens = json.load(f)

    # load models/tokenizers
    MODEL_CONFIGS = {
        "BanglaBERT": BASE_DIR / "outputs" / "banglabert-bdshs",
        "XLM-R": BASE_DIR / "outputs" / "xlmr-bdshs",
    }
    tok_bb = AutoTokenizer.from_pretrained(str(MODEL_CONFIGS['BanglaBERT']))
    model_bb = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CONFIGS['BanglaBERT'])).to(DEVICE)
    model_bb.eval()

    # Quick check: does this tokenizer preserve its mask token when inserted into raw text?
    try:
        if tok_bb.mask_token:
            test_text = "এটা " + tok_bb.mask_token + " একটা"
            tok_test = tok_bb(test_text, return_tensors='pt')
            tok_list = tok_bb.convert_ids_to_tokens(tok_test['input_ids'][0])
            if tok_bb.mask_token in tok_list:
                logger.info("Mask token preserved as a single token by tokenizer")
            else:
                logger.warning(f"Mask token was not preserved as a single token by tokenizer. Tokens: {tok_list}")
    except Exception as e:
        logger.warning(f"Mask-token tokenization check failed: {e}")

    # prepare eval examples (use the token JSON entry list for texts/pred labels; limit to 100)
    # Use the same 50 examples for all strategies for fair comparison
    # Detect which key holds the predicted label in the token JSONs
    def detect_pred_label_key(token_list):
        candidate_keys = ['predicted_label', 'pred_label', 'predicted', 'label', 'pred']
        for k in candidate_keys:
            if k in token_list[0]:
                return k
        # try to find any key that looks like a prediction key across first few entries
        for k in token_list[0].keys():
            if 'pred' in k or 'label' in k:
                return k
        raise KeyError(f"No predicted label key found in token JSON sample. Top-level keys: {list(token_list[0].keys())}")

    pred_key = detect_pred_label_key(ig_bb_tokens)
    logger.info(f"Using predicted label key '{pred_key}' from token JSON sample")

    # respect environment override for number of examples
    max_examples = int(os.environ.get('AE_MAX_EXAMPLES', '0'))
    if max_examples <= 0:
        max_examples = min(len(ig_bb_tokens), len(ig_xlmr_tokens))
    logger.info(f"Evaluating using max_examples={max_examples}")

    # build a quick lookup map by text to avoid repeated linear scans
    ig_bb_map = {r.get('text'): r for r in ig_bb_tokens}

    # validate schema on the first sample and show useful debug info
    sample = ig_bb_tokens[0]
    logger.info(f"Top-level sample keys: {list(sample.keys())}")
    if 'word_attribution' in sample:
        logger.info(f"First word attribution entry: {sample['word_attribution'][0]}")

    # construct eval_examples and fail loudly if a sample is missing the label key
    eval_examples = []
    for e in ig_bb_tokens[:max_examples]:
        if pred_key not in e:
            raise KeyError(f"Missing predicted label key '{pred_key}' in token entry. Available keys: {list(e.keys())}")
        eval_examples.append({"text": e['text'], "predicted_label": int(e[pred_key])})

    # One-time sanity spot-check: ensure saved word_attribution words match build_word_spans words
    mismatches = []
    for ex in eval_examples[:min(10, len(eval_examples))]:
        text = ex['text']
        r = ig_bb_map.get(text)
        if not r:
            logger.warning(f"No token entry found in map for text (truncated): {text[:60]!r}")
            continue
        wa = r.get('word_attribution', [])
        wa_words = [w.get('word') if isinstance(w, dict) else w for w in wa]
        span_words = [w for _, _, w in build_word_spans(text)]
        if wa_words != span_words:
            mismatches.append({'text': text, 'wa': wa_words, 'spans': span_words})
            logger.warning(f"Word alignment mismatch for text (trunc): {text[:80]!r}\n  wa: {wa_words}\n  spans: {span_words}")
    if mismatches:
        logger.warning(f"{len(mismatches)} mismatched examples found in spot-check (first {min(10, len(eval_examples))}).")

    results = []
    # sum/mean/max using IG token results (word_attribution)
    results.append(evaluate_aggregation_strategy(eval_examples, 'sum', lambda t, l: make_units_from_token_results(ig_bb_map, t, agg_field='sum'), tok_bb, model_bb, token_map=ig_bb_map))
    results.append(evaluate_aggregation_strategy(eval_examples, 'mean', lambda t, l: make_units_from_token_results(ig_bb_map, t, agg_field='mean'), tok_bb, model_bb, token_map=ig_bb_map))
    results.append(evaluate_aggregation_strategy(eval_examples, 'max', lambda t, l: make_units_from_token_results(ig_bb_map, t, agg_field='max'), tok_bb, model_bb, token_map=ig_bb_map))

    # hierarchical on the full selected evaluation set
    hier_results = run_hierarchical_on_examples(eval_examples[:max_examples], tok_bb, model_bb, sum_score_lookup=lambda text: [u['score'] for u in make_units_from_token_results(ig_bb_map, text, agg_field='sum')], token_map=ig_bb_map, max_examples=max_examples)
    hier_units_by_text = {r['text']: r['units'] for r in hier_results}
    results.append(evaluate_aggregation_strategy([ex for ex in eval_examples[:max_examples] if ex['text'] in hier_units_by_text], 'hierarchical', lambda t, l: hier_units_by_text[t], tok_bb, model_bb, token_map=ig_bb_map))

    # save comparison
    comparison_rows = []
    for r in results:
        row = {"strategy": r['strategy'], "n": r['n_examples']}
        # comprehensiveness and sufficiency metrics
        row.update({f"comp_{k}": v for k, v in r['comprehensiveness_mean'].items()})
        row.update({f"suff_{k}": v for k, v in r['sufficiency_mean'].items()})
        # coverage metrics (report actual word coverage at each k)
        row.update({f"comp_cov_{k}": v for k, v in r.get('comp_coverage_mean', {}).items()})
        row.update({f"suff_cov_{k}": v for k, v in r.get('suff_coverage_mean', {}).items()})
        # efficiency columns
        row.update({f"comp_eff_{k}": v for k, v in r.get('comp_eff_mean', {}).items()})
        row.update({f"suff_eff_{k}": v for k, v in r.get('suff_eff_mean', {}).items()})
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    csv_out = OUT_DIR / 'aggregation_strategy_comparison_banglabert.csv'
    comparison_df.to_csv(csv_out, index=False)
    logger.info(f"Saved comparison CSV -> {csv_out}")

    # write markdown report
    md = OUT_DIR / 'aggregation_report.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Aggregation strategy comparison (BanglaBERT)\n\n')
        f.write('## Summary\n\n')
        # to_markdown requires the 'tabulate' package. Fall back gracefully if it's not installed.
        try:
            md_table = comparison_df.to_markdown(index=False)
        except Exception:
            try:
                import tabulate  # noqa: F401
                md_table = comparison_df.to_markdown(index=False)
            except Exception:
                logger.warning("tabulate not installed; falling back to plain text table in MD. To get markdown tables, run: pip install tabulate")
                md_table = comparison_df.to_string(index=False)
        f.write(md_table + '\n\n')
        f.write(f'## Hierarchical sample details ({len(hier_results)} examples)\n\n')
        for r in hier_results:
            f.write('### Text:\n')
            f.write(r['text'] + '\n\n')
            f.write(str(r['units']) + '\n\n')
    logger.info(f"Saved MD report -> {md}")

    # ----- Repeat evaluation for XLM-R -----
    logger.info("Starting XLM-R evaluation")
    # load XLM-R tokenizer and model
    tok_xlmr = AutoTokenizer.from_pretrained(str(MODEL_CONFIGS['XLM-R']))
    model_xlmr = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CONFIGS['XLM-R'])).to(DEVICE)
    model_xlmr.eval()

    # mask token check for XLM-R
    try:
            if tok_xlmr.mask_token:
                test_text = "এটা " + tok_xlmr.mask_token + " একটা"
                tok_test = tok_xlmr(test_text, return_tensors='pt')
                tok_list = tok_xlmr.convert_ids_to_tokens(tok_test['input_ids'][0])
                if tok_xlmr.mask_token in tok_list:
                    logger.info("XLM-R mask token preserved as a single token by tokenizer")
                else:
                    logger.warning(f"XLM-R mask token was not preserved as a single token by tokenizer. Tokens: {tok_list}")
    except Exception as e:
            logger.warning(f"XLM-R mask-token tokenization check failed: {e}")

    # detect predicted label key for XLM-R tokens
    pred_key_xlmr = detect_pred_label_key(ig_xlmr_tokens)
    logger.info(f"Using predicted label key '{pred_key_xlmr}' for XLM-R token JSON sample")

    ig_xlmr_map = {r.get('text'): r for r in ig_xlmr_tokens}

    # construct eval examples for XLM-R
    eval_examples_xlmr = []
    for e in ig_xlmr_tokens[:max_examples]:
            if pred_key_xlmr not in e:
                raise KeyError(f"Missing predicted label key '{pred_key_xlmr}' in XLM-R token entry. Available keys: {list(e.keys())}")
            eval_examples_xlmr.append({"text": e['text'], "predicted_label": int(e[pred_key_xlmr])})

    results_x = []
    results_x.append(evaluate_aggregation_strategy(eval_examples_xlmr, 'sum', lambda t, l: make_units_from_token_results(ig_xlmr_map, t, agg_field='sum'), tok_xlmr, model_xlmr, token_map=ig_xlmr_map))
    results_x.append(evaluate_aggregation_strategy(eval_examples_xlmr, 'mean', lambda t, l: make_units_from_token_results(ig_xlmr_map, t, agg_field='mean'), tok_xlmr, model_xlmr, token_map=ig_xlmr_map))
    results_x.append(evaluate_aggregation_strategy(eval_examples_xlmr, 'max', lambda t, l: make_units_from_token_results(ig_xlmr_map, t, agg_field='max'), tok_xlmr, model_xlmr, token_map=ig_xlmr_map))

    hier_results_x = run_hierarchical_on_examples(eval_examples_xlmr[:max_examples], tok_xlmr, model_xlmr, sum_score_lookup=lambda text: [u['score'] for u in make_units_from_token_results(ig_xlmr_map, text, agg_field='sum')], token_map=ig_xlmr_map, max_examples=max_examples)
    hier_units_by_text_x = {r['text']: r['units'] for r in hier_results_x}
    results_x.append(evaluate_aggregation_strategy([ex for ex in eval_examples_xlmr[:max_examples] if ex['text'] in hier_units_by_text_x], 'hierarchical', lambda t, l: hier_units_by_text_x[t], tok_xlmr, model_xlmr, token_map=ig_xlmr_map))

    # save XLM-R comparison
    comparison_rows_x = []
    for r in results_x:
            row = {"strategy": r['strategy'], "n": r['n_examples']}
            row.update({f"comp_{k}": v for k, v in r['comprehensiveness_mean'].items()})
            row.update({f"suff_{k}": v for k, v in r['sufficiency_mean'].items()})
            row.update({f"comp_cov_{k}": v for k, v in r.get('comp_coverage_mean', {}).items()})
            row.update({f"suff_cov_{k}": v for k, v in r.get('suff_coverage_mean', {}).items()})
            # efficiency columns
            row.update({f"comp_eff_{k}": v for k, v in r.get('comp_eff_mean', {}).items()})
            row.update({f"suff_eff_{k}": v for k, v in r.get('suff_eff_mean', {}).items()})
            comparison_rows_x.append(row)
    comparison_df_x = pd.DataFrame(comparison_rows_x)
    csv_out_x = OUT_DIR / 'aggregation_strategy_comparison_xlmr.csv'
    comparison_df_x.to_csv(csv_out_x, index=False)
    logger.info(f"Saved XLM-R comparison CSV -> {csv_out_x}")

    # write XLM-R markdown report
    md_x = OUT_DIR / 'aggregation_report_xlmr.md'
    with open(md_x, 'w', encoding='utf-8') as f:
            f.write('# Aggregation strategy comparison (XLM-R)\n\n')
            f.write('## Summary\n\n')
            try:
                md_table = comparison_df_x.to_markdown(index=False)
            except Exception:
                try:
                    import tabulate  # noqa: F401
                    md_table = comparison_df_x.to_markdown(index=False)
                except Exception:
                    logger.warning("tabulate not installed; falling back to plain text table in MD. To get markdown tables, run: pip install tabulate")
                    md_table = comparison_df_x.to_string(index=False)
            f.write(md_table + '\n\n')
            f.write(f'## Hierarchical sample details ({len(hier_results_x)} examples)\n\n')
            for r in hier_results_x:
                f.write('### Text:\n')
                f.write(r['text'] + '\n\n')
                f.write(str(r['units']) + '\n\n')
    logger.info(f"Saved XLM-R MD report -> {md_x}")

    logger.info('XLM-R evaluation done')

if __name__ == '__main__':
    t0 = time.time()
    try:
        main()
    except Exception as e:
        logger.exception('Unhandled error in aggregation evaluation:')
    finally:
        log_fh.close()
