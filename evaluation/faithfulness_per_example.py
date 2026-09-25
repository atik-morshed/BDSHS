"""
Phase 2 — Per-Example Faithfulness CSV
========================================
Generates per-example comprehensiveness, sufficiency, and coverage for ALL
strategies (Sum, Mean, Max, Hierarchical) × ALL budgets (10%, 20%, 30%)
× BOTH models (BanglaBERT, XLM-R) × BOTH attribution methods (IG, SHAP).

This is the central dataset that drives all downstream statistical tests.

Output schema:
  example_id, model, attribution, strategy, budget,
  comp, suff, coverage, comp_eff, suff_eff

Output files:
  results/faithfulness/per_example_faithfulness.csv
  results/faithfulness/per_example_faithfulness_ig.csv    (IG-only subset)
  results/faithfulness/per_example_faithfulness_shap.csv  (SHAP-only subset)
  results/faithfulness/faithfulness_run.log

Run:
  .venv\Scripts\python.exe -u evaluation/faithfulness_per_example.py

Environment overrides:
  FAITH_MAX_EXAMPLES=N   limit to first N examples (for testing)
  FAITH_SKIP_HIERARCHICAL=1   skip expensive hierarchical forward passes
"""

import os, sys, io, json, re, time, logging, traceback
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"]              = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "faithfulness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "faithfulness_run.log"
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
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

ATTR_DIR = BASE_DIR / "outputs" / "attribution"

# Best checkpoints (verified in phase0_checkpoint_eval.json)
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

K_FRACS   = [0.10, 0.20, 0.30]
MAX_EXAMPLES = int(os.environ.get("FAITH_MAX_EXAMPLES", "0"))
SKIP_HIER    = os.environ.get("FAITH_SKIP_HIERARCHICAL", "0") == "1"

# ─────────────────────────────────────────────────────────────
# Shared helpers (duplicated from aggregation_evaluation.py
# so this script is fully self-contained)
# ─────────────────────────────────────────────────────────────
def is_garbage_word(word):
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


def select_top_units_by_word_budget(units, k_frac, total_words):
    """Greedy budget-aware unit selection (same as original aggregation_evaluation.py)."""
    sorted_units = sorted(units, key=lambda u: -abs(u["score"]))
    word_budget  = max(1, int(total_words * k_frac))
    selected, words_used = [], 0
    for u in sorted_units:
        n_words = len(u.get("indices", []))
        if words_used + n_words > word_budget:
            if not selected:
                selected.append(u)
                words_used += n_words
            continue
        selected.append(u)
        words_used += n_words
    return selected, words_used


def find_word_spans_from_saved(text, wa_words):
    spans = []
    cursor = 0
    for w in wa_words:
        if w is None:
            raise ValueError(f"Saved word is None in text: {text[:80]!r}")
        idx = text.find(w, cursor)
        if idx == -1:
            raise ValueError(f"Cannot locate {w!r} in text: {text[:80]!r}")
        spans.append((idx, idx + len(w), w))
        cursor = idx + len(w)
    return spans


def get_word_spans_cached(text, token_rec):
    """Use saved word_attribution words for spans when available."""
    if token_rec is not None:
        wa = token_rec.get('word_attribution', [])
        wa_words = [w.get('word') if isinstance(w, dict) else w for w in wa]
        try:
            return find_word_spans_from_saved(text, wa_words)
        except Exception:
            pass
    return build_word_spans(text)


# ─────────────────────────────────────────────────────────────
# Compute per-example comp + suff for a single strategy
# ─────────────────────────────────────────────────────────────
def compute_example_metrics(text, units, word_spans, tokenizer, model,
                             target_label, orig_prob, k_fracs):
    """
    Returns a dict:
      { '10': {'comp': ..., 'suff': ..., 'coverage': ...},
        '20': {...}, '30': {...} }
    """
    total_words = len(word_spans)
    results = {}
    for k_frac in k_fracs:
        pct = int(100 * k_frac)
        selected, words_used = select_top_units_by_word_budget(units, k_frac, total_words)
        top_indices = [i for u in selected for i in u["indices"]]

        # Comprehensiveness: mask selected, measure drop
        masked_text_comp = mask_word_spans(text, word_spans, top_indices, tokenizer)
        prob_comp = get_pred_prob(masked_text_comp, tokenizer, model, target_label)
        comp = orig_prob - prob_comp

        # Sufficiency: keep selected, mask rest
        all_indices   = set(range(total_words))
        keep_indices  = set(top_indices)
        remove_indices = list(all_indices - keep_indices)
        masked_text_suff = mask_word_spans(text, word_spans, remove_indices, tokenizer)
        prob_suff = get_pred_prob(masked_text_suff, tokenizer, model, target_label)
        suff = orig_prob - prob_suff

        coverage = words_used / total_words if total_words > 0 else 0.0
        results[str(pct)] = {
            "comp":     comp,
            "suff":     suff,
            "coverage": coverage,
        }
    return results


# ─────────────────────────────────────────────────────────────
# Hierarchical merge (same logic as original)
# ─────────────────────────────────────────────────────────────
def hierarchical_merge(text, word_spans, word_base_scores, tokenizer, model,
                        target_label, orig_prob, max_merges=None):
    if len(word_base_scores) != len(word_spans):
        raise ValueError(
            f"Score/span mismatch: {len(word_base_scores)} scores vs "
            f"{len(word_spans)} spans for: {text[:60]!r}"
        )
    units = [
        {"words": [w], "indices": [i], "score": s}
        for i, (s, (_, _, w)) in enumerate(zip(word_base_scores, word_spans))
    ]
    n_merges  = 0
    max_merges = max_merges or max(1, len(units) // 3)
    while n_merges < max_merges and len(units) > 1:
        best_gain, best_pair = -np.inf, None
        for i in range(len(units) - 1):
            try:
                idx_a = units[i]["indices"]
                idx_b = units[i+1]["indices"]
                drop_a  = orig_prob - get_pred_prob(mask_word_spans(text, word_spans, idx_a,       tokenizer), tokenizer, model, target_label)
                drop_b  = orig_prob - get_pred_prob(mask_word_spans(text, word_spans, idx_b,       tokenizer), tokenizer, model, target_label)
                drop_ab = orig_prob - get_pred_prob(mask_word_spans(text, word_spans, idx_a+idx_b, tokenizer), tokenizer, model, target_label)
                gain = abs(drop_ab - (drop_a + drop_b))
            except Exception as e:
                logger.debug(f"Hierarchical merge forward pass failed at i={i}: {e}")
                gain = -np.inf
            if gain > best_gain:
                best_gain, best_pair = gain, i
        if best_pair is None:
            break
        i = best_pair
        units = (units[:i]
                 + [{"words":   units[i]["words"]   + units[i+1]["words"],
                     "indices": units[i]["indices"] + units[i+1]["indices"],
                     "score":   units[i]["score"]   + units[i+1]["score"]}]
                 + units[i+2:])
        n_merges += 1
    return units


# ─────────────────────────────────────────────────────────────
# Build flat word-level units from token JSON record
# ─────────────────────────────────────────────────────────────
def make_flat_units(token_rec, agg_field):
    """Returns list of {indices: [i], score: float} for flat strategies."""
    wa = token_rec.get("word_attribution", [])
    units = []
    for i, w in enumerate(wa):
        if agg_field not in w:
            raise KeyError(
                f"Missing '{agg_field}' in word_attribution entry {i}. "
                f"Available keys: {list(w.keys())}"
            )
        units.append({"indices": [i], "score": float(w[agg_field])})
    return units


# ─────────────────────────────────────────────────────────────
# Run one model × one attribution method
# ─────────────────────────────────────────────────────────────
def run_model_attribution(model_name, attr_method, token_json_path,
                           tokenizer, model, max_examples):
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {model_name} / {attr_method}")

    with open(token_json_path, encoding="utf-8") as f:
        token_data = json.load(f)

    # Build lookup by text
    if isinstance(token_data, list):
        tok_map = {r.get("text", ""): r for r in token_data}
        examples_list = token_data
    else:
        tok_map       = {}
        examples_list = []
        for r in token_data.values():
            tok_map[r.get("text", "")] = r
            examples_list.append(r)

    if max_examples > 0:
        examples_list = examples_list[:max_examples]
    logger.info(f"  Examples to process: {len(examples_list):,}")

    # Detect predicted label key
    sample = examples_list[0]
    pred_key = None
    for k in ["predicted_label", "pred_label", "predicted", "label"]:
        if k in sample:
            pred_key = k
            break
    if pred_key is None:
        raise KeyError(f"No predicted label key found. Keys: {list(sample.keys())}")
    logger.info(f"  Predicted label key: '{pred_key}'")

    # ── Pre-compute hierarchical units for all examples ────
    hier_units_map = {}
    if not SKIP_HIER:
        logger.info("  Pre-computing hierarchical units (this takes a while)…")
        t_hier_start = time.time()
        for idx_e, e in enumerate(examples_list):
            text         = e.get("text", "")
            target_label = int(e.get(pred_key, 0))
            tok_rec      = tok_map.get(text)
            if tok_rec is None:
                continue
            word_spans = get_word_spans_cached(text, tok_rec)
            if not word_spans:
                continue
            try:
                flat_units = make_flat_units(tok_rec, "sum")
                base_scores = [u["score"] for u in flat_units]
                if len(base_scores) != len(word_spans):
                    raise ValueError(
                        f"base_scores ({len(base_scores)}) != word_spans ({len(word_spans)})"
                    )
                orig_prob = get_pred_prob(text, tokenizer, model, target_label)
                hier_units = hierarchical_merge(
                    text, word_spans, base_scores, tokenizer, model, target_label, orig_prob
                )
                hier_units_map[text] = hier_units
            except Exception as ex:
                logger.warning(f"  Hierarchical failed for example {idx_e}: {ex}")

            if (idx_e + 1) % 200 == 0:
                elapsed = time.time() - t_hier_start
                rate = (idx_e + 1) / elapsed
                eta  = (len(examples_list) - idx_e - 1) / max(rate, 1e-9)
                logger.info(f"  Hierarchical: {idx_e+1}/{len(examples_list)} "
                            f"  {rate:.1f} ex/s  ETA {eta/60:.1f} min")
        logger.info(f"  Hierarchical units computed for {len(hier_units_map):,} examples "
                    f"in {(time.time()-t_hier_start)/60:.1f} min")

    # ── Per-example faithfulness loop ──────────────────────
    rows = []
    t0   = time.time()
    strategies = {
        "Sum":  "sum",
        "Mean": "mean",
        "Max":  "max",
    }

    for idx_e, e in enumerate(examples_list):
        text         = e.get("text", "")
        target_label = int(e.get(pred_key, 0))
        ex_id        = e.get("id", e.get("example_id", idx_e))
        tok_rec      = tok_map.get(text)
        if tok_rec is None:
            logger.debug(f"  No token record for example {idx_e}")
            continue

        word_spans = get_word_spans_cached(text, tok_rec)
        if not word_spans:
            logger.debug(f"  Empty word spans for example {idx_e}")
            continue

        try:
            orig_prob = get_pred_prob(text, tokenizer, model, target_label)
        except Exception as ex:
            logger.warning(f"  get_pred_prob failed for example {idx_e}: {ex}")
            continue

        # Flat strategies
        for strat_name, agg_field in strategies.items():
            try:
                units = make_flat_units(tok_rec, agg_field)
                metrics = compute_example_metrics(
                    text, units, word_spans, tokenizer, model,
                    target_label, orig_prob, K_FRACS
                )
                for pct_str, m in metrics.items():
                    cov = m["coverage"]
                    rows.append({
                        "example_id":  ex_id,
                        "model":       model_name,
                        "attribution": attr_method,
                        "strategy":    strat_name,
                        "budget":      int(pct_str),
                        "comp":        m["comp"],
                        "suff":        m["suff"],
                        "coverage":    cov,
                        "comp_eff":    m["comp"]  / cov if cov > 0 else None,
                        "suff_eff":    m["suff"]  / cov if cov > 0 else None,
                    })
            except Exception as ex:
                logger.debug(f"  {strat_name} failed for example {idx_e}: {ex}")

        # Hierarchical
        if not SKIP_HIER and text in hier_units_map:
            try:
                units = hier_units_map[text]
                metrics = compute_example_metrics(
                    text, units, word_spans, tokenizer, model,
                    target_label, orig_prob, K_FRACS
                )
                for pct_str, m in metrics.items():
                    cov = m["coverage"]
                    rows.append({
                        "example_id":  ex_id,
                        "model":       model_name,
                        "attribution": attr_method,
                        "strategy":    "Hierarchical",
                        "budget":      int(pct_str),
                        "comp":        m["comp"],
                        "suff":        m["suff"],
                        "coverage":    cov,
                        "comp_eff":    m["comp"] / cov if cov > 0 else None,
                        "suff_eff":    m["suff"] / cov if cov > 0 else None,
                    })
            except Exception as ex:
                logger.debug(f"  Hierarchical eval failed for example {idx_e}: {ex}")

        if (idx_e + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = (idx_e + 1) / elapsed
            eta     = (len(examples_list) - idx_e - 1) / max(rate, 1e-9)
            logger.info(f"  Progress: {idx_e+1}/{len(examples_list)} "
                        f"  {rate:.1f} ex/s  ETA {eta/60:.1f} min  rows so far: {len(rows):,}")

    logger.info(f"  Done. {len(rows):,} rows in {(time.time()-t0)/60:.1f} min")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t_global = time.time()
    all_frames = []

    for model_name, ckpt_path in CHECKPOINTS.items():
        logger.info(f"\n{'#'*60}")
        logger.info(f"Loading model: {model_name}  checkpoint: {ckpt_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        model     = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path))
        model     = model.to(DEVICE).eval()

        for attr_method, json_path in ATTRIBUTION_FILES[model_name].items():
            if not json_path.exists():
                logger.error(f"Attribution file not found: {json_path}")
                continue

            df = run_model_attribution(
                model_name    = model_name,
                attr_method   = attr_method,
                token_json_path = json_path,
                tokenizer     = tokenizer,
                model         = model,
                max_examples  = MAX_EXAMPLES,
            )
            all_frames.append(df)

            # Save intermediate result
            interim_path = RESULTS_DIR / f"faithfulness_{model_name.lower().replace('-','_')}_{attr_method.lower()}.csv"
            df.to_csv(interim_path, index=False, encoding="utf-8")
            logger.info(f"  Interim CSV saved → {interim_path}  ({len(df):,} rows)")

        # Free GPU memory between models
        del model
        torch.cuda.empty_cache()

    # ── Combine and save ──────────────────────────────────
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        out_path = RESULTS_DIR / "per_example_faithfulness.csv"
        combined.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"\nFull CSV saved → {out_path}  ({len(combined):,} rows)")

        # Save IG-only and SHAP-only subsets
        for method in ["IG", "SHAP"]:
            subset = combined[combined["attribution"] == method]
            subset_path = RESULTS_DIR / f"per_example_faithfulness_{method.lower()}.csv"
            subset.to_csv(subset_path, index=False, encoding="utf-8")
            logger.info(f"  {method} subset → {subset_path}  ({len(subset):,} rows)")

        # Print aggregate summary to verify it matches existing paper tables
        logger.info("\n" + "="*60)
        logger.info("AGGREGATE SUMMARY (should reproduce paper tables):")
        logger.info("="*60)
        summary = (combined
                   .groupby(["model", "attribution", "strategy", "budget"])
                   [["comp", "suff", "coverage", "comp_eff", "suff_eff"]]
                   .mean()
                   .round(4))
        logger.info(f"\n{summary.to_string()}")

        summary_path = RESULTS_DIR / "faithfulness_aggregate_summary.csv"
        summary.reset_index().to_csv(summary_path, index=False, encoding="utf-8")
        logger.info(f"\nAggregate summary CSV → {summary_path}")
    else:
        logger.error("No data frames produced — check errors above.")

    logger.info(f"\nTotal runtime: {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled error in faithfulness_per_example.py:")
        sys.exit(1)
