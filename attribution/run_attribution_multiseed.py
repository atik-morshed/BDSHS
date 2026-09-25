"""
Phase 8 — Multi-Seed Attribution (Seeds 43, 44)
=================================================
Runs IG + SHAP attribution for the new seed checkpoints (43, 44).
Follows the exact same protocol as bdshs_attribution.py.

Attribution is run from the best checkpoint (by val F1) for each seed.

Outputs:
  results/seeds/seed_{43,44}/attribution/
    ig_{model}_token.json
    ig_{model}_word.csv
    shap_{model}_token.json
    shap_{model}_word.csv
    attribution.log

Run:
  .venv\Scripts\python.exe -u attribution/run_attribution_multiseed.py

To run only specific seeds or models:
  ATTR_SEEDS=43 ATTR_MODELS=BanglaBERT .venv\Scripts\python.exe -u attribution/run_attribution_multiseed.py
"""

import os, sys, io, json, random, re, warnings, time, logging
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"]              = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache"
os.environ.setdefault("HF_HOME",              str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE",str(_CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE",   str(_CACHE_ROOT / "huggingface" / "transformers"))

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
    print("[WARN] captum not installed — IG skipped.  pip install captum")
    CAPTUM_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    print("[WARN] shap not installed — SHAP skipped.  pip install shap")
    SHAP_OK = False

BASE_DIR    = Path(__file__).resolve().parent.parent
SEEDS_DIR   = BASE_DIR / "results" / "seeds"
ATTR_SEEDS  = [int(s) for s in os.environ.get("ATTR_SEEDS", "43,44").split(",")]
ATTR_MODELS = os.environ.get("ATTR_MODELS", "BanglaBERT,XLM-R").split(",")

TEXT_COL  = "sentence"
LABEL_COL = "hate speech"
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EXAMPLES = int(os.environ.get("ATTR_MAX_EXAMPLES", "0"))

# IG convergence protocol
IG_INIT_STEPS       = 50
IG_STEP_INCREMENTS  = [50, 100, 200, 300]
IG_CONV_THRESHOLD   = 0.05


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def subword_attrs_to_word_attrs(tokens, attrs, tokenizer, model_name):
    """
    Aggregate subword-level attributions to word level using sum/mean/max.
    Returns list of dicts: [{word, sum, mean, max, token_count}].
    """
    word_attrs = []
    current_word_tokens, current_word_attrs = [], []

    def is_continuation(token, prev_token, model_name):
        # BanglaBERT uses ##; XLM-R uses ▁ to mark word-start
        if "xlm" in model_name.lower() or "roberta" in model_name.lower():
            return not token.startswith("▁") and token not in ["<s>", "</s>", "<pad>"]
        return token.startswith("##")

    def flush(tokens_buf, attrs_buf):
        if not tokens_buf: return
        word_str = "".join(t.replace("##", "").replace("▁", "") for t in tokens_buf)
        word_attrs.append({
            "word":        word_str,
            "sum":         float(sum(attrs_buf)),
            "mean":        float(np.mean(attrs_buf)),
            "max":         float(max(attrs_buf, key=abs)),
            "token_count": len(tokens_buf),
        })

    for token, attr in zip(tokens, attrs):
        if token in ["[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]"]:
            flush(current_word_tokens, current_word_attrs)
            current_word_tokens, current_word_attrs = [], []
            continue
        if is_continuation(token, current_word_tokens[-1] if current_word_tokens else None, model_name):
            current_word_tokens.append(token)
            current_word_attrs.append(attr)
        else:
            flush(current_word_tokens, current_word_attrs)
            current_word_tokens = [token]
            current_word_attrs  = [attr]

    flush(current_word_tokens, current_word_attrs)
    return word_attrs


# ─────────────────────────────────────────────────────────────
# IG attribution
# ─────────────────────────────────────────────────────────────

def run_ig_on_model(model_name, tokenizer, model, test_texts, test_labels, out_dir, logger):
    if not CAPTUM_OK:
        logger.warning("captum not available — skipping IG")
        return

    logger.info(f"  Running IG for {model_name} …")

    # Find embedding layer
    embedding_layer = None
    for name, mod in model.named_modules():
        if "embedding" in name.lower() and hasattr(mod, "weight"):
            embedding_layer = mod
            break
    if embedding_layer is None:
        logger.error("  Could not find embedding layer")
        return

    lig = LayerIntegratedGradients(
        lambda input_ids, attention_mask, token_type_ids=None: (
            model(input_ids=input_ids, attention_mask=attention_mask,
                  token_type_ids=token_type_ids)
            if token_type_ids is not None
            else model(input_ids=input_ids, attention_mask=attention_mask)
        ).logits,
        embedding_layer,
    )

    token_records, word_records = [], []
    t0 = time.time()

    for idx, (text, true_label) in enumerate(zip(test_texts, test_labels)):
        try:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH, padding=False).to(DEVICE)
            input_ids      = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            token_type_ids = enc.get("token_type_ids")

            with torch.no_grad():
                logits = (model(**enc).logits if token_type_ids is None
                          else model(input_ids=input_ids, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids).logits)
                probs = torch.softmax(logits, dim=-1)
            predicted_label = int(torch.argmax(probs).item())
            predicted_prob  = float(probs[0, predicted_label].item())

            # PAD baseline
            baseline_ids = torch.zeros_like(input_ids)
            baseline_ids[0, 0]  = tokenizer.cls_token_id or 0
            baseline_ids[0, -1] = tokenizer.sep_token_id or 2

            # Adaptive IG steps
            attr, delta = None, float("inf")
            for n_steps in IG_STEP_INCREMENTS:
                kwargs = dict(
                    inputs          = input_ids,
                    baselines       = baseline_ids,
                    target          = predicted_label,
                    n_steps         = n_steps,
                    additional_forward_args=(attention_mask,
                                             token_type_ids) if token_type_ids is not None
                                            else (attention_mask,),
                    return_convergence_delta=True,
                )
                try:
                    attr, delta = lig.attribute(**kwargs)
                except Exception:
                    attr, delta_t = lig.attribute(
                        inputs=input_ids,
                        baselines=baseline_ids,
                        target=predicted_label,
                        n_steps=n_steps,
                        additional_forward_args=(attention_mask,),
                        return_convergence_delta=True,
                    )
                    delta = float(delta_t.mean().item())
                delta = float(delta.mean().item()) if hasattr(delta, "mean") else float(delta)
                if abs(delta) < IG_CONV_THRESHOLD:
                    break

            attrs = attr.sum(dim=-1).squeeze(0).cpu().numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
            word_attribution = subword_attrs_to_word_attrs(tokens, attrs, tokenizer, model_name)

            token_records.append({
                "id":                idx,
                "text":              text,
                "true_label":        true_label,
                "predicted_label":   predicted_label,
                "probability":       predicted_prob,
                "tokens":            tokens,
                "attributions":      attrs,
                "convergence_delta": delta,
                "word_attribution":  word_attribution,
            })
            for wa in word_attribution:
                word_records.append({
                    "example_id":    idx,
                    "text":          text,
                    "predicted_label": predicted_label,
                    "word":          wa["word"],
                    "sum":           wa["sum"],
                    "mean":          wa["mean"],
                    "max":           wa["max"],
                    "token_count":   wa["token_count"],
                })

        except Exception as ex:
            logger.warning(f"  IG failed for example {idx}: {ex}")

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta  = (len(test_texts) - idx - 1) / max(rate, 1e-9)
            logger.info(f"  IG {idx+1}/{len(test_texts)}  {rate:.1f} ex/s  ETA {eta/60:.1f} min")

    model_slug = model_name.lower().replace("-", "_").replace(" ", "_")
    tok_path = out_dir / f"ig_{model_slug}_token.json"
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(token_records, f, ensure_ascii=False)

    word_df = pd.DataFrame(word_records)
    word_df.to_csv(out_dir / f"ig_{model_slug}_word.csv", index=False, encoding="utf-8")

    logger.info(f"  IG done: {len(token_records):,} examples saved → {tok_path}")


# ─────────────────────────────────────────────────────────────
# SHAP attribution
# ─────────────────────────────────────────────────────────────

def run_shap_on_model(model_name, tokenizer, model, test_texts, test_labels, out_dir, logger):
    if not SHAP_OK:
        logger.warning("shap not available — skipping SHAP")
        return

    logger.info(f"  Running SHAP for {model_name} …")

    clf_pipeline = hf_pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=True,
        max_length=MAX_LENGTH,
        truncation=True,
    )

    masker   = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(clf_pipeline, masker)

    token_records, word_records = [], []
    batch_size = 32
    t0 = time.time()

    for batch_start in range(0, len(test_texts), batch_size):
        batch_texts  = test_texts[batch_start:batch_start + batch_size]
        batch_labels = test_labels[batch_start:batch_start + batch_size]
        try:
            shap_values = explainer(batch_texts)
        except Exception as ex:
            logger.warning(f"  SHAP batch {batch_start}–{batch_start+len(batch_texts)} failed: {ex}")
            continue

        for i, (text, true_label) in enumerate(zip(batch_texts, batch_labels)):
            try:
                tokens = shap_values[i].data
                values = shap_values[i].values  # shape: (n_tokens, n_classes)
                # Get target-class values
                pred_label = int(np.argmax(np.mean(values, axis=0)))
                target_vals = [float(v[pred_label]) for v in values]

                word_attribution = subword_attrs_to_word_attrs(
                    tokens, target_vals, tokenizer, model_name
                )

                idx = batch_start + i
                token_records.append({
                    "id":              idx,
                    "text":            text,
                    "true_label":      true_label,
                    "predicted_label": pred_label,
                    "tokens":          list(tokens),
                    "shap_values":     target_vals,
                    "word_attribution": word_attribution,
                })
                for wa in word_attribution:
                    word_records.append({
                        "example_id":    idx,
                        "text":          text,
                        "predicted_label": pred_label,
                        "word":          wa["word"],
                        "sum":           wa["sum"],
                        "mean":          wa["mean"],
                        "max":           wa["max"],
                        "token_count":   wa["token_count"],
                    })
            except Exception as ex:
                logger.warning(f"  SHAP example {batch_start+i} failed: {ex}")

        if (batch_start + len(batch_texts)) % 500 == 0:
            elapsed = time.time() - t0
            done    = batch_start + len(batch_texts)
            rate    = done / elapsed
            eta     = (len(test_texts) - done) / max(rate, 1e-9)
            logger.info(f"  SHAP {done}/{len(test_texts)}  {rate:.1f} ex/s  ETA {eta/60:.1f} min")

    model_slug = model_name.lower().replace("-", "_").replace(" ", "_")
    tok_path = out_dir / f"shap_{model_slug}_token.json"
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(token_records, f, ensure_ascii=False)

    word_df = pd.DataFrame(word_records)
    word_df.to_csv(out_dir / f"shap_{model_slug}_word.csv", index=False, encoding="utf-8")

    logger.info(f"  SHAP done: {len(token_records):,} examples → {tok_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    t_global = time.time()

    # Load test data
    test_df = pd.read_csv(BASE_DIR / "test.csv")
    test_df = test_df.dropna(subset=[TEXT_COL]).copy()
    test_df[TEXT_COL] = test_df[TEXT_COL].astype(str).str.strip()
    test_df = test_df[test_df[TEXT_COL].str.len() > 0].reset_index(drop=True)

    label_values = sorted(test_df[LABEL_COL].unique())
    raw_to_id = {lbl: idx for idx, lbl in enumerate(label_values)}
    test_df["labels"] = test_df[LABEL_COL].map(raw_to_id).astype(int)

    test_texts  = test_df[TEXT_COL].tolist()
    test_labels = test_df["labels"].tolist()
    if MAX_EXAMPLES > 0:
        test_texts  = test_texts[:MAX_EXAMPLES]
        test_labels = test_labels[:MAX_EXAMPLES]

    logger.info(f"Test examples: {len(test_texts):,}")

    for seed in ATTR_SEEDS:
        seed_dir = SEEDS_DIR / f"seed_{seed}"
        for model_name in ATTR_MODELS:
            model_slug = model_name.lower().replace("-", "_").replace(" ", "_")
            ckpt_dir   = BASE_DIR / "results" / "classification" / f"seed_{seed}" / model_slug / "best_model"

            if not ckpt_dir.exists():
                logger.error(
                    f"Checkpoint not found for {model_name} seed {seed}: {ckpt_dir}\n"
                    "Run training/train_multiseed.py first."
                )
                continue

            out_dir = seed_dir / "attribution"
            out_dir.mkdir(parents=True, exist_ok=True)

            log_handler = logging.FileHandler(out_dir / "attribution.log", encoding="utf-8")
            logger.addHandler(log_handler)

            set_all_seeds(seed)
            logger.info(f"\n{'#'*60}")
            logger.info(f"Attribution: {model_name}  seed={seed}")
            logger.info(f"Checkpoint: {ckpt_dir}")

            tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
            model     = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
            model     = model.to(DEVICE).eval()

            run_ig_on_model(model_name, tokenizer, model, test_texts, test_labels, out_dir, logger)
            run_shap_on_model(model_name, tokenizer, model, test_texts, test_labels, out_dir, logger)

            del model
            torch.cuda.empty_cache()
            logger.removeHandler(log_handler)

    logger.info(f"\nTotal attribution runtime: {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(SEEDS_DIR.parent / "attribution_multiseed.log", encoding="utf-8"),
        ],
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    try:
        main()
    except Exception:
        logger.exception("Unhandled error in run_attribution_multiseed.py:")
        sys.exit(1)
