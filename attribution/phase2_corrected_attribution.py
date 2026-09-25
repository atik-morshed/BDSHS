"""
Phase 2 — Corrected Attribution Analysis
==========================================
Re-runs Integrated Gradients and SHAP attribution from CORRECTED checkpoints:
  BanglaBERT : checkpoint-5028  (Epoch 2, step 5028,  val F1 = 0.9037)
  XLM-R      : checkpoint-12570 (Epoch 5, step 12570, val F1 = 0.9146)

This replaces the previous attribution which used incorrect checkpoints.

Run:  python attribution/phase2_corrected_attribution.py
Outputs: ./outputs/attribution/phase2/
"""

import os, sys, io, json, random, warnings, re
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"]              = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline,
)

try:
    from captum.attr import LayerIntegratedGradients
    CAPTUM_OK = True
except ImportError:
    print("[WARN] captum not installed - IG skipped.  pip install captum")
    CAPTUM_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    print("[WARN] shap not installed - SHAP skipped.  pip install shap")
    SHAP_OK = False

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 60)
print("PHASE 2 — CORRECTED ATTRIBUTION ANALYSIS")
print("=" * 60)
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
    print("VRAM            :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("PyTorch version :", torch.__version__)
print("=" * 60)

# ─────────────────────────────────────────────
# PATHS & LOGGING
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "outputs" / "attribution" / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "phase2_attribution_log.txt"

class Tee:
    def __init__(self, stream, fh): self.stream, self.fh = stream, fh
    def write(self, data): self.stream.write(data); self.fh.write(data)
    def flush(self): self.stream.flush(); self.fh.flush()
    def isatty(self): return False
    def fileno(self): return self.stream.fileno()
    def readable(self): return False
    @property
    def encoding(self): return self.stream.encoding

_log_fh    = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log_fh)

# CORRECTED checkpoint paths based on verification
MODEL_CONFIGS = {
    "BanglaBERT": BASE_DIR / "outputs" / "banglabert-bdshs" / "checkpoint-5028",
    "XLM-R":      BASE_DIR / "outputs" / "xlmr-bdshs" / "checkpoint-12570",
}

TEXT_COL  = "sentence"
LABEL_COL = "hate speech"

print("\nCORRECTED CHECKPOINT CONFIGURATION:")
for name, path in MODEL_CONFIGS.items():
    print(f"  {name:15s}: {path}")
    if not path.exists():
        print(f"  [ERROR] Checkpoint not found: {path}")
        sys.exit(1)

# ─────────────────────────────────────────────
# 1.  LOAD TEST DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1 - LOAD TEST DATA")
print("=" * 60)

test_df = pd.read_csv(BASE_DIR / "test.csv")
test_df = test_df.dropna(subset=[TEXT_COL]).copy()
test_df[TEXT_COL] = test_df[TEXT_COL].astype(str).str.strip()
test_df = test_df[test_df[TEXT_COL].str.len() > 0].reset_index(drop=True)

random.seed(SEED)
shap_eval_texts  = test_df[TEXT_COL].tolist()
ig_preview_texts = test_df[TEXT_COL].head(5).tolist()

print(f"Test set size      : {len(test_df):,}")
print(f"SHAP sample size   : {len(shap_eval_texts)} (full test split)")
print(f"IG preview samples : {len(ig_preview_texts)}")

# ─────────────────────────────────────────────
# 2.  WORD-BOUNDARY RECONSTRUCTION
# ─────────────────────────────────────────────
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


def align_subwords_to_words(text, tokenizer, raw_scores, normalized_scores=None):
    """
    Align tokenizer subword offsets to word spans and attach both raw and normalized scores.
    raw_scores: iterable of per-token raw attribution scores (e.g., sum over embedding dim)
    normalized_scores: optional iterable of the same shape with normalized scores
    Returns a list of groups: {word, start, end, subwords, raw_scores, normalized_scores}
    """
    try:
        enc       = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
        offsets   = enc["offset_mapping"]
        input_ids = enc["input_ids"]
    except Exception:
        enc       = tokenizer(text, truncation=True, max_length=128)
        input_ids = enc["input_ids"]
        offsets   = [(0, 0)] * len(input_ids)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    words = build_word_spans(text)

    word_groups = [
        {"word": w, "start": s, "end": e, "subwords": [], "raw_scores": [], "normalized_scores": []}
        for s, e, w in words
    ]

    unmatched = 0
    norm_iterable = normalized_scores if normalized_scores is not None else raw_scores
    for tok, (os, oe), raw_s, norm_s in zip(tokens, offsets, raw_scores, norm_iterable):
        if os == oe:
            continue
        matched = False
        for wg in word_groups:
            if os >= wg["start"] and oe <= wg["end"]:
                wg["subwords"].append(tok)
                wg["raw_scores"].append(float(raw_s))
                wg["normalized_scores"].append(float(norm_s))
                matched = True
                break
        if not matched:
            unmatched += 1

    if unmatched:
        print(f"[WARN] {unmatched} unmatched tokens in: {text[:60]}...")

    return word_groups

def word_agg(groups):
    """
    Aggregate subword scores into word-level metrics.
    Accepts groups with either 'raw_scores' (preferred), 'normalized_scores', or legacy 'scores'.
    """
    out = []
    for g in groups:
        # Prefer raw_scores if present, fall back to normalized_scores or legacy 'scores'
        sc_list = None
        if g.get("raw_scores") is not None:
            sc_list = g["raw_scores"]
        elif g.get("normalized_scores") is not None:
            sc_list = g["normalized_scores"]
        elif g.get("scores") is not None:
            sc_list = g["scores"]
        # Guard against numpy array truth-value ambiguity and empty lists
        if sc_list is None:
            continue
        try:
            length_ok = len(sc_list) > 0
        except Exception:
            # If object has no len, coerce to array and check size
            sc_arr_temp = np.array(sc_list)
            if sc_arr_temp.size == 0:
                continue
            length_ok = True
        if not length_ok:
            continue
        sc = np.array(sc_list, dtype=float)
        out.append({
            "word": g["word"],
            "subwords": g["subwords"],
            "sum":  round(float(np.sum(sc)),  6),
            "mean": round(float(np.mean(sc)), 6),
            "max":  round(float(np.max(sc)),  6),
        })
    return out

# ─────────────────────────────────────────────
# 3.  INTEGRATED GRADIENTS
# ─────────────────────────────────────────────
def run_ig(model_name, model_path, texts, n_steps=50, max_steps=300, delta_threshold=0.05):
    if not CAPTUM_OK:
        print(f"[SKIP] captum unavailable for {model_name}"); return {}
    print(f"\n{'─'*60}\nIG: {model_name}  ({len(texts)} samples, start_steps={n_steps})\n{'─'*60}")

    tok   = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), use_safetensors=True).to(DEVICE)
    model.eval()
    lig = LayerIntegratedGradients(
        lambda ids, mask: torch.softmax(model(input_ids=ids, attention_mask=mask).logits, dim=-1),
        model.get_input_embeddings())

    results = []
    for i, text in enumerate(texts):
        try:
            enc  = tok(text, return_tensors="pt", truncation=True, max_length=128)
            ids  = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)
            base = torch.full_like(ids, tok.pad_token_id)
            base[:, 0] = ids[:, 0]; base[:, -1] = ids[:, -1]

            with torch.no_grad():
                logits     = model(input_ids=ids, attention_mask=mask).logits
                probs      = torch.softmax(logits, dim=-1)
                target     = int(torch.argmax(probs, dim=-1).item())
                pred_score = float(probs[0, target].item())

            steps = n_steps
            attr, delta = lig.attribute(
                inputs=ids, baselines=base,
                additional_forward_args=(mask,),
                target=target, n_steps=steps, return_convergence_delta=True)
            delta_val = float(delta.item())

            # retry with more steps if convergence is poor
            while abs(delta_val) > delta_threshold * max(abs(pred_score), 1e-6) and steps < max_steps:
                steps = min(steps * 2, max_steps)
                attr, delta = lig.attribute(
                    inputs=ids, baselines=base,
                    additional_forward_args=(mask,),
                    target=target, n_steps=steps, return_convergence_delta=True)
                delta_val = float(delta.item())

            converged = abs(delta_val) <= delta_threshold * max(abs(pred_score), 1e-6)

            raw_attributions = attr.sum(dim=-1).squeeze(0)
            normalized_attributions = raw_attributions / (raw_attributions.norm() + 1e-9)

            raw_scores = raw_attributions.detach().cpu().numpy()
            norm_scores = normalized_attributions.detach().cpu().numpy()

            tokens = tok.convert_ids_to_tokens(ids.squeeze(0).tolist())
            wa = word_agg(align_subwords_to_words(text, tok, raw_scores, norm_scores))

            results.append({
                "text": text,
                "predicted_label": target,
                "pred_score": round(pred_score, 6),
                "convergence_delta": delta_val,
                "n_steps_used": int(steps),
                "converged": bool(converged),
                "tokens": tokens,
                "token_scores_raw": [float(x) for x in raw_scores],
                "token_scores_normalized": [float(x) for x in norm_scores],
                "word_attribution": wa,
            })

            if i < 5:
                print(f"\n[IG #{i+1}] label={target} score={pred_score:.6f} delta={delta_val:.6f} steps={steps} converged={converged}")
                for t, s in zip(tokens, norm_scores):
                    print(f"  {t:>22s}: {'+' if s>=0 else '-'}{abs(s):.4f}  {'#'*int(abs(s)*30)}")
            if (i+1) % 50 == 0: print(f"  IG progress: {i+1}/{len(texts)}")
        except Exception as e:
            print(f"  [WARN] IG sample {i}: {e}"); continue

    jf = OUT_DIR / f"ig_{model_name.lower().replace('-','_')}_token.json"
    with open(jf, "w", encoding="utf-8") as f: json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  IG token JSON  -> {jf}  ({len(results)} samples)")

    rows = [{"sample_idx": i, "text_snippet": e["text"][:60],
              "predicted_label": e["predicted_label"],
              "word": w["word"], "subwords": "|".join(w["subwords"]),
              "attr_sum": w["sum"], "attr_mean": w["mean"], "attr_max": w["max"]}
             for i, e in enumerate(results) for w in e["word_attribution"]]
    if rows:
        cf = OUT_DIR / f"ig_{model_name.lower().replace('-','_')}_word.csv"
        pd.DataFrame(rows).to_csv(cf, index=False, encoding="utf-8")
        print(f"  IG word CSV    -> {cf}")

    del model; torch.cuda.empty_cache()
    return {"token_results": results}

# ─────────────────────────────────────────────
# 4.  SHAP
# ─────────────────────────────────────────────
def run_shap_analysis(model_name, model_path, texts):
    if not SHAP_OK:
        print(f"[SKIP] shap unavailable for {model_name}"); return {}
    print(f"\n{'─'*60}\nSHAP: {model_name}  ({len(texts)} samples)\n{'─'*60}")

    tok   = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), use_safetensors=True).to(DEVICE)
    model.eval()

    clf = hf_pipeline("text-classification", model=model, tokenizer=tok,
                       device=0 if torch.cuda.is_available() else -1,
                       top_k=None, truncation=True, max_length=128)
    explainer = shap.Explainer(clf)
    results, bs = [], 10

    for s in range(0, len(texts), bs):
        batch = texts[s:s+bs]
        try:
            sv = explainer(batch)
            for j, text in enumerate(batch):
                try:
                    vals = sv[j].values
                    # Ensure numeric numpy array
                    vals = np.array(vals)
                    if vals.size == 0:
                        print(f"  [WARN] SHAP sample empty values at batch {s} idx {j}");
                        continue
                    agg = vals.sum(axis=0) if vals.ndim > 1 else vals
                    target = int(np.argmax(agg))
                    sc = vals[:, target] if vals.ndim > 1 else vals
                    sc = np.array(sc, dtype=float)
                    norm_sc = sc / (np.linalg.norm(sc) + 1e-9)
                    wa = word_agg(align_subwords_to_words(text, tok, sc, norm_sc))
                    tokens = None
                    try:
                        tokens = list(sv[j].data)
                    except Exception:
                        # Fallback: tokenize the text
                        tokens = tok.tokenize(text)
                    results.append({
                        "text": text,
                        "predicted_label": target,
                        "tokens": tokens,
                        "token_scores_raw": [float(x) for x in sc],
                        "token_scores_normalized": [float(x) for x in norm_sc],
                        "word_attribution": wa,
                    })
                except Exception as e_s:
                    print(f"  [WARN] SHAP sample [{s + j}] processing error: {e_s}");
                    continue
        except Exception as e:
            print(f"  [WARN] SHAP batch [{s}:{s+bs}]: {e}"); continue
        done = min(s+bs, len(texts))
        if done % 50 == 0 or done >= len(texts):
            print(f"  SHAP progress: {done}/{len(texts)}")

    jf = OUT_DIR / f"shap_{model_name.lower().replace('-','_')}_token.json"
    with open(jf, "w", encoding="utf-8") as f: json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  SHAP token JSON -> {jf}  ({len(results)} samples)")

    rows = [{"sample_idx": i, "text_snippet": e["text"][:60],
              "predicted_label": e["predicted_label"],
              "word": w["word"], "subwords": "|".join(w["subwords"]),
              "shap_sum": w["sum"], "shap_mean": w["mean"], "shap_max": w["max"]}
             for i, e in enumerate(results) for w in e["word_attribution"]]
    if rows:
        cf = OUT_DIR / f"shap_{model_name.lower().replace('-','_')}_word.csv"
        pd.DataFrame(rows).to_csv(cf, index=False, encoding="utf-8")
        print(f"  SHAP word CSV   -> {cf}")

    del model; torch.cuda.empty_cache()
    return {"token_results": results}

# ─────────────────────────────────────────────
# 5.  RUN BOTH MODELS
# ─────────────────────────────────────────────
summary = {}
for mname, mpath in MODEL_CONFIGS.items():
    if not mpath.exists():
        print(f"\n[SKIP] Not found: {mpath}"); continue
    print(f"\n{'='*60}\nMODEL : {mname}\nPATH  : {mpath}\n{'='*60}")
    # Run IG on the same full test set as SHAP for direct comparability.
    ig_r   = run_ig(mname, mpath, shap_eval_texts, n_steps=50, max_steps=300)
    shap_r = run_shap_analysis(mname, mpath, shap_eval_texts)
    summary[mname] = {"ig_samples": len(ig_r.get("token_results", [])),
                      "shap_samples": len(shap_r.get("token_results", []))}

# ─────────────────────────────────────────────
# 6.  ATTRIBUTION REPORT (Markdown)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 - PHASE 2 ATTRIBUTION REPORT")
print("=" * 60)

rp = OUT_DIR / "phase2_attribution_report.md"
with open(rp, "w", encoding="utf-8") as f:
    f.write("# Phase 2 — Corrected Attribution Analysis Report\n\n")
    f.write("## Important Note\n\n")
    f.write("This analysis uses **CORRECTED checkpoints**:\n")
    f.write("- BanglaBERT: `checkpoint-5028` (Epoch 2, step 5028, val F1 = 0.9037)\n")
    f.write("- XLM-R: `checkpoint-12570` (Epoch 5, step 12570, val F1 = 0.9146)\n\n")
    f.write("These are the checkpoints selected by `metric_for_best_model=\"f1\"`.\n\n")
    f.write("## Run Summary\n\n| Model | IG Samples | SHAP Samples |\n| --- | --- | --- |\n")
    for mn, s in summary.items():
        f.write(f"| {mn} | {s['ig_samples']} | {s['shap_samples']} |\n")
    f.write("\n## Methods\n\n")
    f.write("### Integrated Gradients\n- captum LayerIntegratedGradients\n")
    f.write("- Baseline: all PAD (CLS/SEP kept)\n- 50 steps, L2-normalised\n\n")
    f.write("### SHAP\n- shap.Explainer + HF pipeline\n- full cleaned test split\n\n")
    f.write("## Word Aggregation\n\n| Strategy | Description |\n| --- | --- |\n")
    f.write("| sum | Total attribution |\n| mean | Length-normalised |\n| max | Peak signal |\n\n")
    f.write("## Output Files (in outputs/attribution/phase2/)\n\n")
    for m in ["banglabert", "xlm_r"]:
        for method in ["ig", "shap"]:
            f.write(f"- {method}_{m}_token.json — token scores\n")
            f.write(f"- {method}_{m}_word.csv   — word aggregations\n")
    f.write("- phase2_attribution_log.txt — full log\n")

print(f"\n  Report -> {rp}")
print(f"  Log    -> {LOG_FILE}")
print("\n" + "=" * 60)
print("Phase 2 Attribution Analysis complete.")
print("=" * 60)
_log_fh.close()
