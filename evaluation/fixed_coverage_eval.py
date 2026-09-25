"""
Phase 6 — Fixed-Coverage Evaluation
======================================
Compares all four aggregation strategies (Sum, Mean, Max, Hierarchical)
at IDENTICAL actual word coverage, eliminating the coverage confound.

Problem: Hierarchical units span multiple words, so at a nominal 10% budget,
hierarchical might cover 31% of words while flat strategies cover only 17%.
This makes raw metric comparisons unfair.

Solution: For each example, we determine the actual coverage that each
strategy uses, then evaluate ALL strategies at that same actual coverage.

Approach:
  1. For each example, compute what actual coverage (fraction) Hierarchical
     achieves at the nominal budget.
  2. Evaluate Sum/Mean/Max at that SAME actual coverage.
  3. Compare metrics across all four strategies at equal coverage.

Also evaluates at fixed absolute coverage: 10%, 20%, 30% of actual words.

Outputs:
  results/faithfulness/fixed_coverage_faithfulness.csv
  results/faithfulness/fixed_coverage_summary.csv
  results/faithfulness/fixed_coverage_report.md
  results/faithfulness/fixed_coverage_eval.log

Run:
  .venv\Scripts\python.exe -u evaluation/fixed_coverage_eval.py
"""

import os, sys, io, json, re, time, logging
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"]              = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "faithfulness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "fixed_coverage_eval.log"
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

ATTR_DIR = BASE_DIR / "outputs" / "attribution"

CHECKPOINTS = {
    "BanglaBERT": BASE_DIR / "outputs" / "banglabert-bdshs" / "checkpoint-5028",
    "XLM-R":      BASE_DIR / "outputs" / "xlmr-bdshs"       / "checkpoint-12570",
}

ATTRIBUTION_FILES = {
    "BanglaBERT": {
        "IG":   ATTR_DIR / "ig_banglabert_token.json",
        "SHAP": ATTR_DIR / "shap_banglabert_token.json",
    },
    "XLM-R": {
        "IG":   ATTR_DIR / "ig_xlm_r_token.json",
        "SHAP": ATTR_DIR / "shap_xlm_r_token.json",
    },
}

NOMINAL_BUDGETS = [0.10, 0.20, 0.30]
MAX_EXAMPLES    = int(os.environ.get("FC_MAX_EXAMPLES", "0"))


# ─────────────────────────────────────────────────────────────
# Helpers (self-contained — same logic as faithfulness_per_example.py)
# ─────────────────────────────────────────────────────────────

def is_garbage_word(word):
    stripped = word.strip()
    if not stripped: return True
    if all(ch == '\ufffd' or not ch.isprintable() for ch in stripped): return True
    allowed_keep = {'।'}
    if all((not ch.isalnum() and ch not in allowed_keep) for ch in stripped): return True
    return False


def build_word_spans(text):
    spans, cursor = [], 0
    for raw_word in text.split():
        idx = text.find(raw_word, cursor)
        if idx == -1: idx = cursor
        end = idx + len(raw_word)
        if is_garbage_word(raw_word):
            cursor = end; continue
        m = re.search(r'([।,!?;:\.\"\'\(\)]+)$', raw_word)
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
        probs  = torch.softmax(logits, dim=-1)
    return float(probs[0, target_label].item())


def find_word_spans_from_saved(text, wa_words):
    spans, cursor = [], 0
    for w in wa_words:
        if w is None: raise ValueError(f"None word in: {text[:60]!r}")
        idx = text.find(w, cursor)
        if idx == -1: raise ValueError(f"Can't find {w!r} in: {text[:60]!r}")
        spans.append((idx, idx + len(w), w))
        cursor = idx + len(w)
    return spans


def get_word_spans_cached(text, tok_rec):
    if tok_rec is not None:
        wa = tok_rec.get('word_attribution', [])
        wa_words = [w.get('word') if isinstance(w, dict) else w for w in wa]
        try: return find_word_spans_from_saved(text, wa_words)
        except Exception: pass
    return build_word_spans(text)


def select_n_words(units, n_words_target):
    """
    Select units greedily (by |score|) until exactly n_words_target words
    are covered. Skip units that would overshoot; always select at least one.
    Returns (selected_units, words_actually_used).
    """
    sorted_units = sorted(units, key=lambda u: -abs(u["score"]))
    selected, words_used = [], 0
    for u in sorted_units:
        n = len(u.get("indices", []))
        if words_used + n > n_words_target:
            if not selected:
                selected.append(u)
                words_used += n
            continue
        selected.append(u)
        words_used += n
    return selected, words_used


def make_flat_units(tok_rec, agg_field):
    wa = tok_rec.get("word_attribution", [])
    return [{"indices": [i], "score": float(w[agg_field])} for i, w in enumerate(wa)
            if agg_field in w]


def hierarchical_merge(text, word_spans, base_scores, tokenizer, model,
                        target_label, orig_prob, max_merges=None):
    if len(base_scores) != len(word_spans):
        raise ValueError(f"Mismatch: {len(base_scores)} scores, {len(word_spans)} spans")
    units = [{"words": [w], "indices": [i], "score": s}
             for i, (s, (_, _, w)) in enumerate(zip(base_scores, word_spans))]
    max_merges = max_merges or max(1, len(units) // 3)
    for _ in range(max_merges):
        if len(units) < 2: break
        best_gain, best_pair = -np.inf, None
        for i in range(len(units) - 1):
            try:
                ia, ib = units[i]["indices"], units[i+1]["indices"]
                da  = orig_prob - get_pred_prob(mask_word_spans(text, word_spans, ia,    tokenizer), tokenizer, model, target_label)
                db  = orig_prob - get_pred_prob(mask_word_spans(text, word_spans, ib,    tokenizer), tokenizer, model, target_label)
                dab = orig_prob - get_pred_prob(mask_word_spans(text, word_spans, ia+ib, tokenizer), tokenizer, model, target_label)
                gain = abs(dab - (da + db))
            except Exception: gain = -np.inf
            if gain > best_gain: best_gain, best_pair = gain, i
        if best_pair is None: break
        i = best_pair
        units = (units[:i]
                 + [{"words":   units[i]["words"]   + units[i+1]["words"],
                     "indices": units[i]["indices"] + units[i+1]["indices"],
                     "score":   units[i]["score"]   + units[i+1]["score"]}]
                 + units[i+2:])
    return units


def eval_at_fixed_coverage(text, units, word_spans, tokenizer, model,
                            target_label, orig_prob, n_words_target):
    """
    Evaluate comp + suff at a fixed absolute word count.
    Returns dict with comp, suff, coverage.
    """
    total_words = len(word_spans)
    if total_words == 0:
        return None

    selected, words_used = select_n_words(units, n_words_target)
    top_indices = [i for u in selected for i in u["indices"]]

    # Comprehensiveness
    masked_comp = mask_word_spans(text, word_spans, top_indices, tokenizer)
    prob_comp   = get_pred_prob(masked_comp, tokenizer, model, target_label)
    comp = orig_prob - prob_comp

    # Sufficiency
    remove_indices = list(set(range(total_words)) - set(top_indices))
    masked_suff = mask_word_spans(text, word_spans, remove_indices, tokenizer)
    prob_suff   = get_pred_prob(masked_suff, tokenizer, model, target_label)
    suff = orig_prob - prob_suff

    coverage = words_used / total_words
    return {"comp": comp, "suff": suff, "coverage": coverage, "n_words": words_used}


# ─────────────────────────────────────────────────────────────
# Run one model × attribution method
# ─────────────────────────────────────────────────────────────

def run_fixed_coverage(model_name, attr_method, token_json_path,
                        tokenizer, model, max_examples):
    logger.info(f"\n{'='*60}")
    logger.info(f"Fixed-coverage eval: {model_name} / {attr_method}")

    with open(token_json_path, encoding="utf-8") as f:
        token_data = json.load(f)

    if isinstance(token_data, list):
        tok_map       = {r.get("text", ""): r for r in token_data}
        examples_list = token_data
    else:
        tok_map       = {}
        examples_list = []
        for r in token_data.values():
            tok_map[r.get("text", "")] = r
            examples_list.append(r)

    if max_examples > 0:
        examples_list = examples_list[:max_examples]
    logger.info(f"  Examples: {len(examples_list):,}")

    # Detect pred label key
    sample = examples_list[0]
    pred_key = next((k for k in ["predicted_label", "pred_label", "predicted", "label"]
                     if k in sample), None)
    if pred_key is None:
        raise KeyError(f"No pred label key. Keys: {list(sample.keys())}")

    rows = []
    t0   = time.time()

    for idx_e, e in enumerate(examples_list):
        text         = e.get("text", "")
        target_label = int(e.get(pred_key, 0))
        ex_id        = e.get("id", e.get("example_id", idx_e))
        tok_rec      = tok_map.get(text)
        if tok_rec is None: continue

        word_spans = get_word_spans_cached(text, tok_rec)
        total_words = len(word_spans)
        if total_words == 0: continue

        try:
            orig_prob = get_pred_prob(text, tokenizer, model, target_label)
        except Exception as ex:
            logger.debug(f"  get_pred_prob failed {idx_e}: {ex}")
            continue

        # Build units for each flat strategy
        flat_units = {}
        for agg_field in ["sum", "mean", "max"]:
            try:
                flat_units[agg_field] = make_flat_units(tok_rec, agg_field)
            except Exception: pass

        # Build hierarchical units using sum scores
        hier_units = None
        if "sum" in flat_units:
            try:
                base_scores = [u["score"] for u in flat_units["sum"]]
                if len(base_scores) == total_words:
                    hier_units = hierarchical_merge(
                        text, word_spans, base_scores, tokenizer, model,
                        target_label, orig_prob
                    )
            except Exception as ex:
                logger.debug(f"  Hierarchical failed {idx_e}: {ex}")

        # For each nominal budget: evaluate at the exact word count
        for k_frac in NOMINAL_BUDGETS:
            n_words_target = max(1, int(total_words * k_frac))
            pct = int(100 * k_frac)

            strategy_units_map = {
                "Sum":  flat_units.get("sum"),
                "Mean": flat_units.get("mean"),
                "Max":  flat_units.get("max"),
                "Hierarchical": hier_units,
            }

            for strat_name, units in strategy_units_map.items():
                if units is None: continue
                try:
                    m = eval_at_fixed_coverage(
                        text, units, word_spans, tokenizer, model,
                        target_label, orig_prob, n_words_target
                    )
                    if m is None: continue
                    cov = m["coverage"]
                    rows.append({
                        "example_id":       ex_id,
                        "model":            model_name,
                        "attribution":      attr_method,
                        "strategy":         strat_name,
                        "nominal_budget":   pct,
                        "n_words_target":   n_words_target,
                        "total_words":      total_words,
                        "comp":             m["comp"],
                        "suff":             m["suff"],
                        "coverage":         cov,
                        "comp_eff":         m["comp"] / cov if cov > 0 else None,
                        "suff_eff":         m["suff"] / cov if cov > 0 else None,
                    })
                except Exception as ex:
                    logger.debug(f"  {strat_name}@{pct}% failed {idx_e}: {ex}")

        if (idx_e + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = (idx_e + 1) / elapsed
            eta     = (len(examples_list) - idx_e - 1) / max(rate, 1e-9)
            logger.info(f"  {idx_e+1}/{len(examples_list)}  "
                        f"{rate:.1f} ex/s  ETA {eta/60:.1f} min  rows: {len(rows):,}")

    logger.info(f"  Done. {len(rows):,} rows  {(time.time()-t0)/60:.1f} min")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    t_global  = time.time()
    all_frames = []

    for model_name, ckpt_path in CHECKPOINTS.items():
        logger.info(f"\n{'#'*60}")
        logger.info(f"Model: {model_name}  checkpoint: {ckpt_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        mdl       = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path))
        mdl       = mdl.to(DEVICE).eval()

        for attr_method, json_path in ATTRIBUTION_FILES[model_name].items():
            if not json_path.exists():
                logger.error(f"Attribution file not found: {json_path}")
                continue
            frame = run_fixed_coverage(
                model_name=model_name,
                attr_method=attr_method,
                token_json_path=json_path,
                tokenizer=tokenizer,
                model=mdl,
                max_examples=MAX_EXAMPLES,
            )
            all_frames.append(frame)

            interim = RESULTS_DIR / f"fixed_cov_{model_name.lower().replace('-','_')}_{attr_method.lower()}.csv"
            frame.to_csv(interim, index=False, encoding="utf-8")
            logger.info(f"  Interim → {interim}")

        del mdl
        torch.cuda.empty_cache()

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        out = RESULTS_DIR / "fixed_coverage_faithfulness.csv"
        combined.to_csv(out, index=False, encoding="utf-8")
        logger.info(f"\nFull fixed-coverage CSV → {out}  ({len(combined):,} rows)")

        # Summary
        summary = (combined
                   .groupby(["model", "attribution", "strategy", "nominal_budget"])
                   [["comp", "suff", "coverage"]]
                   .mean().round(4))
        summary_path = RESULTS_DIR / "fixed_coverage_summary.csv"
        summary.reset_index().to_csv(summary_path, index=False, encoding="utf-8")
        logger.info(f"Summary → {summary_path}")
        logger.info(f"\n{summary.to_string()}")

        # Markdown report
        report_path = RESULTS_DIR / "fixed_coverage_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Fixed-Coverage Faithfulness Evaluation\n\n")
            f.write(
                "Each strategy is evaluated at the **same absolute word count** "
                "(nominal_budget × total_words) per example, removing the coverage "
                "confound from hierarchical multi-word unit selection.\n\n"
            )
            f.write("## Aggregate Results\n\n")
            try:
                f.write(summary.reset_index().to_markdown(index=False))
            except Exception:
                f.write(summary.reset_index().to_string(index=False))
        logger.info(f"Report → {report_path}")

    logger.info(f"\nTotal runtime: {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled error in fixed_coverage_eval.py:")
        sys.exit(1)
