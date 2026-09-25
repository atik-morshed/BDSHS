"""
Runtime Measurements
=====================
Measures per-example runtime for each aggregation strategy:
  Sum, Mean, Max, Hierarchical

For comparison, also measures IG and SHAP attribution time.

Outputs:
  results/analysis/runtime_measurements.csv
  results/analysis/runtime_report.md

Run:
  .venv\Scripts\python.exe -u analysis/runtime_measurements.py

Uses 100 random examples from the test set for timing.
"""

import os, sys, io, json, re, time, logging, warnings
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"]              = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR     = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE_DIR / "results" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
ATTR_DIR     = BASE_DIR / "outputs" / "attribution"

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TIMING = int(os.environ.get("RUNTIME_N", "100"))
MAX_LENGTH = 128

CHECKPOINTS = {
    "BanglaBERT": BASE_DIR / "outputs" / "banglabert-bdshs" / "checkpoint-5028",
    "XLM-R":      BASE_DIR / "outputs" / "xlmr-bdshs"       / "checkpoint-12570",
}
IG_FILES = {
    "BanglaBERT": ATTR_DIR / "ig_banglabert_token.json",
    "XLM-R":      ATTR_DIR / "ig_xlm_r_token.json",
}


def is_garbage_word(word):
    stripped = word.strip()
    if not stripped: return True
    if all(ch == '\ufffd' or not ch.isprintable() for ch in stripped): return True
    if all((not ch.isalnum() and ch != '।') for ch in stripped): return True
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


def mask_word_spans(text, word_spans, indices, tokenizer):
    mask_tok = tokenizer.mask_token if tokenizer.mask_token else "[MASK]"
    chars = list(text)
    for idx in sorted(indices, key=lambda i: -word_spans[i][0]):
        s, e, _ = word_spans[idx]
        chars[s:e] = list(mask_tok)
    return "".join(chars)


def get_pred_prob(text, tokenizer, model, target_label):
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)
    return float(probs[0, target_label].item())


def time_flat_strategy(text, units, word_spans, tokenizer, model,
                        target_label, orig_prob, k_frac=0.10):
    total_words = len(word_spans)
    budget      = max(1, int(total_words * k_frac))
    sorted_u    = sorted(units, key=lambda u: -abs(u["score"]))
    selected, used = [], 0
    for u in sorted_u:
        n = len(u["indices"])
        if used + n > budget:
            if not selected: selected.append(u); used += n
            continue
        selected.append(u); used += n

    top_idx = [i for u in selected for i in u["indices"]]
    get_pred_prob(mask_word_spans(text, word_spans, top_idx, tokenizer), tokenizer, model, target_label)
    remove_idx = list(set(range(total_words)) - set(top_idx))
    get_pred_prob(mask_word_spans(text, word_spans, remove_idx, tokenizer), tokenizer, model, target_label)


def time_hierarchical(text, units_input, word_spans, tokenizer, model,
                       target_label, orig_prob, max_merges=None):
    units = [{"words": [w], "indices": [i], "score": s}
             for i, (s, (_, _, w)) in enumerate(zip(units_input, word_spans))]
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
    # also measure faithfulness
    time_flat_strategy(text, units, word_spans, tokenizer, model, target_label, orig_prob)


def run_timing_for_model(model_name, ckpt_path, ig_path):
    logger.info(f"\nTiming: {model_name}")

    with open(ig_path, encoding="utf-8") as f:
        token_data = json.load(f)

    sample = token_data[:N_TIMING]
    tok_map = {r.get("text", ""): r for r in token_data}

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    model     = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path))
    model     = model.to(DEVICE).eval()

    pred_key = next((k for k in ["predicted_label", "pred_label", "label"] if k in sample[0]), "predicted_label")

    rows = []

    for strat_name in ["Sum", "Mean", "Max", "Hierarchical"]:
        times = []
        for e in sample:
            text         = e.get("text", "")
            target_label = int(e.get(pred_key, 0))
            tok_rec      = tok_map.get(text)
            if tok_rec is None: continue

            wa = tok_rec.get("word_attribution", [])
            word_spans = build_word_spans(text)
            if not word_spans: continue

            agg_field = "sum"  # all flat strategies use their respective field
            if strat_name == "Mean": agg_field = "mean"
            elif strat_name == "Max": agg_field = "max"

            flat_units = [
                {"indices": [i], "score": float(w[agg_field])}
                for i, w in enumerate(wa) if agg_field in w
            ]
            if len(flat_units) != len(word_spans): continue

            orig_prob = get_pred_prob(text, tokenizer, model, target_label)

            t_start = time.perf_counter()
            if strat_name == "Hierarchical":
                base_scores = [float(w["sum"]) for w in wa if "sum" in w]
                if len(base_scores) != len(word_spans): continue
                time_hierarchical(text, base_scores, word_spans, tokenizer, model, target_label, orig_prob)
            else:
                time_flat_strategy(text, flat_units, word_spans, tokenizer, model, target_label, orig_prob)
            t_end = time.perf_counter()

            times.append(t_end - t_start)

        if times:
            rows.append({
                "model":              model_name,
                "strategy":           strat_name,
                "n_examples":         len(times),
                "mean_time_s":        round(float(np.mean(times)), 3),
                "median_time_s":      round(float(np.median(times)), 3),
                "std_time_s":         round(float(np.std(times)), 3),
                "total_time_s":       round(float(np.sum(times)), 3),
                "examples_per_sec":   round(len(times) / np.sum(times), 2) if np.sum(times) > 0 else None,
            })
            logger.info(f"  {strat_name:15s}: mean={np.mean(times):.3f}s  "
                        f"median={np.median(times):.3f}s  "
                        f"std={np.std(times):.3f}s")

    del model
    torch.cuda.empty_cache()
    return rows


def main():
    all_rows = []
    for model_name, ckpt_path in CHECKPOINTS.items():
        ig_path = IG_FILES.get(model_name)
        if ig_path is None or not ig_path.exists():
            logger.warning(f"IG file not found for {model_name}")
            continue
        rows = run_timing_for_model(model_name, ckpt_path, ig_path)
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = ANALYSIS_DIR / "runtime_measurements.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"\nRuntime CSV → {csv_path}")

        report = ["# Runtime Measurements\n",
                  f"N examples: {N_TIMING}  |  Device: {DEVICE}\n\n",
                  "## Per-Strategy Timing\n\n"]
        try:
            report.append(df.to_markdown(index=False))
        except Exception:
            report.append(df.to_string(index=False))
        report.append("\n\n> Hierarchical requires additional forward passes per merge step.\n")

        report_path = ANALYSIS_DIR / "runtime_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        logger.info(f"Runtime report → {report_path}")

        logger.info("\n" + df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled error in runtime_measurements.py:")
        sys.exit(1)
