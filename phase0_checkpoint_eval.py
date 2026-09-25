"""
Phase 0 — Checkpoint Fix & Table I Regeneration
================================================
Explicitly loads the best checkpoints identified from trainer_state.json:
  BanglaBERT : checkpoint-5028  (Epoch 2, step 5028,  val F1 = 0.9037)
  XLM-R      : checkpoint-12570 (Epoch 5, step 12570, val F1 = 0.9146)

Evaluates each on the full test set and regenerates Table I.
Writes corrected results to:
  outputs/phase0_checkpoint_eval.json
  outputs/table_I_corrected.md
"""

import os, sys, io, json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_CACHE_ROOT   = _PROJECT_ROOT / ".cache"
os.environ.setdefault("HF_HOME",               str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE",     str(_CACHE_ROOT / "huggingface" / "transformers"))
os.environ.setdefault("HF_HUB_CACHE",          str(_CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TMP",  str(_CACHE_ROOT / "tmp"))
os.environ.setdefault("TEMP", str(_CACHE_ROOT / "tmp"))
os.environ["TRANSFORMERS_NO_TF"]              = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]            = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR   = _PROJECT_ROOT
OUT_DIR    = BASE_DIR / "outputs"
TEXT_COL   = "sentence"
LABEL_COL  = "hate speech"
MAX_LENGTH = 128
BATCH_SIZE = 32

print("=" * 70)
print("PHASE 0 — CHECKPOINT FIX & TABLE I REGENERATION")
print("=" * 70)
print(f"Device  : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
print()

# Best checkpoints from trainer_state.json inspection
BEST_CHECKPOINTS = {
    "BanglaBERT": {
        "checkpoint_dir"  : str(OUT_DIR / "banglabert-bdshs" / "checkpoint-5028"),
        "epoch"           : 2,
        "global_step"     : 5028,
        "val_f1_reported" : 0.9037,
        "val_acc_reported": 0.9073,
    },
    "XLM-R": {
        "checkpoint_dir"  : str(OUT_DIR / "xlmr-bdshs" / "checkpoint-12570"),
        "epoch"           : 5,
        "global_step"     : 12570,
        "val_f1_reported" : 0.9146,
        "val_acc_reported": 0.9171,
    },
}

print("Best checkpoints (from trainer_state.json):")
for name, info in BEST_CHECKPOINTS.items():
    print(f"  {name:15s}: {info['checkpoint_dir']}")
    print(f"               epoch={info['epoch']}, step={info['global_step']}, "
          f"val_F1={info['val_f1_reported']}")
print()

# ── Load & clean test data ─────────────────────────────────────────────────
print("Loading test.csv ...")
test_df = pd.read_csv(BASE_DIR / "test.csv")
test_df = test_df.dropna(subset=[TEXT_COL, LABEL_COL])
test_df[TEXT_COL] = test_df[TEXT_COL].astype(str).str.strip()
test_df = test_df[test_df[TEXT_COL].str.len() > 0]
test_df = test_df.drop_duplicates(subset=[TEXT_COL])

label_values = sorted(test_df[LABEL_COL].unique().tolist())
raw_to_id    = {lbl: idx for idx, lbl in enumerate(label_values)}
label2id     = {str(lbl): int(idx) for idx, lbl in enumerate(label_values)}
id2label     = {int(idx): str(lbl) for idx, lbl in enumerate(label_values)}
num_labels   = len(label2id)

test_df["labels"] = test_df[LABEL_COL].map(raw_to_id).astype(int)
print(f"Test set : {len(test_df):,} examples | num_labels={num_labels}")
print(f"Label distribution: {test_df['labels'].value_counts().to_dict()}\n")

# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    avg   = "binary" if num_labels == 2 else "macro"
    return {
        "accuracy" : float(accuracy_score(labels, preds)),
        "f1"       : float(f1_score(labels, preds, average=avg, zero_division=0)),
        "precision": float(precision_score(labels, preds, average=avg, zero_division=0)),
        "recall"   : float(recall_score(labels, preds, average=avg, zero_division=0)),
    }

# ── Evaluation function ────────────────────────────────────────────────────
def eval_checkpoint(name, info):
    ckpt_dir  = info["checkpoint_dir"]
    ckpt_path = Path(ckpt_dir)

    print(f"\n{'─'*70}")
    print(f"Evaluating : {name}")
    print(f"  Checkpoint : {ckpt_dir}")
    print(f"  Epoch {info['epoch']}  |  Global step {info['global_step']}")
    print(f"{'─'*70}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
    sf = ckpt_path / "model.safetensors"
    if not sf.exists():
        raise FileNotFoundError(f"model.safetensors missing in {ckpt_dir}")
    print(f"  model.safetensors: {sf.stat().st_size/1e6:.1f} MB  OK")

    parent_dir = ckpt_path.parent
    print(f"  Tokenizer from  : {parent_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(parent_dir))

    print("  Loading model weights ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )
    model.eval()

    def tokenize_fn(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=MAX_LENGTH)

    test_hf  = Dataset.from_pandas(test_df[[TEXT_COL, "labels"]].reset_index(drop=True))
    test_tok = test_hf.map(tokenize_fn, batched=True)
    test_tok = test_tok.remove_columns([TEXT_COL])
    test_tok.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_args = TrainingArguments(
        output_dir=str(OUT_DIR / "_phase0_tmp"),
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    results    = trainer.evaluate(metric_key_prefix="test")
    pred_out   = trainer.predict(test_tok)
    preds      = np.argmax(pred_out.predictions, axis=-1)
    labels_np  = test_df["labels"].values

    cm = confusion_matrix(labels_np, preds)
    cr = classification_report(
        labels_np, preds,
        target_names=[f"class_{k}({id2label[k]})" for k in sorted(id2label)],
        digits=4,
    )

    print(f"\n  TEST METRICS ({name} — Epoch {info['epoch']}, step {info['global_step']}):")
    for raw_key, lbl in [("test_accuracy","Accuracy"), ("test_f1","F1"),
                         ("test_precision","Precision"), ("test_recall","Recall")]:
        print(f"    {lbl:12s}: {results.get(raw_key, float('nan')):.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{cr}")

    return {
        "model"             : name,
        "checkpoint_dir"    : ckpt_dir,
        "epoch"             : info["epoch"],
        "global_step"       : info["global_step"],
        "val_f1_at_epoch"   : info["val_f1_reported"],
        "val_acc_at_epoch"  : info["val_acc_reported"],
        "test_accuracy"     : round(results.get("test_accuracy",  float("nan")), 6),
        "test_f1"           : round(results.get("test_f1",        float("nan")), 6),
        "test_precision"    : round(results.get("test_precision", float("nan")), 6),
        "test_recall"       : round(results.get("test_recall",    float("nan")), 6),
        "test_loss"         : round(results.get("test_loss",      float("nan")), 6),
        "confusion_matrix"  : cm.tolist(),
        "classification_report": cr,
        "n_test"            : int(len(test_df)),
    }

# ── Run evaluations ────────────────────────────────────────────────────────
phase0_results = {}
for model_name, ckpt_info in BEST_CHECKPOINTS.items():
    res = eval_checkpoint(model_name, ckpt_info)
    phase0_results[model_name] = res

# ── Sanity delta check ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SANITY CHECK — Corrected vs. previously reported test results")
print("=" * 70)

previous = {
    "BanglaBERT": {"test_f1": 0.9047, "test_accuracy": 0.9071,
                   "test_precision": 0.8922, "test_recall": 0.9176, "epoch": 5.0},
    "XLM-R"     : {"test_f1": 0.9172, "test_accuracy": 0.9191,
                   "test_precision": 0.9023, "test_recall": 0.9325, "epoch": 8.0},
}

delta_rows = []
for name in ["BanglaBERT", "XLM-R"]:
    corrected = phase0_results[name]
    prev      = previous[name]
    for metric in ["test_f1", "test_accuracy", "test_precision", "test_recall"]:
        delta = corrected[metric] - prev[metric]
        delta_rows.append({
            "Model"    : name,
            "Metric"   : metric,
            "Previous" : round(prev[metric], 4),
            "Corrected": round(corrected[metric], 4),
            "Delta"    : round(delta, 4),
        })

delta_df       = pd.DataFrame(delta_rows)
material_change = delta_df["Delta"].abs().max()
print(delta_df.to_string(index=False))
print(f"\nMax |delta| : {material_change:.4f}")
if material_change > 0.005:
    print("*** MATERIAL CHANGE (>0.005) — update paper accordingly ***")
else:
    print("OK: change is immaterial (<0.005) — checkpoints are consistent.")

# ── Generate Table I ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE I — Classification Performance (corrected checkpoints)")
print("=" * 70)

rows = []
for name in ["BanglaBERT", "XLM-R"]:
    r = phase0_results[name]
    rows.append({
        "Model"            : name,
        "Best Epoch (step)": f"{r['epoch']} ({r['global_step']})",
        "Accuracy"         : f"{r['test_accuracy']:.4f}",
        "Precision"        : f"{r['test_precision']:.4f}",
        "Recall"           : f"{r['test_recall']:.4f}",
        "F1"               : f"{r['test_f1']:.4f}",
    })
table_df = pd.DataFrame(rows)
print(table_df.to_string(index=False))

# ── Persist JSON ───────────────────────────────────────────────────────────
out_json = OUT_DIR / "phase0_checkpoint_eval.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "description"              : "Phase 0 — explicit re-evaluation from best checkpoints",
        "banglabert_best"          : "checkpoint-5028 (Epoch 2, step 5028)",
        "xlmr_best"                : "checkpoint-12570 (Epoch 5, step 12570)",
        "results"                  : phase0_results,
        "delta_vs_previous"        : delta_df.to_dict(orient="records"),
        "material_change_threshold": 0.005,
        "material_change_detected" : bool(material_change > 0.005),
        "max_delta"                : float(material_change),
    }, f, indent=2, ensure_ascii=False)
print(f"\nPhase 0 JSON saved   -> {out_json}")

# ── Persist Table I markdown ───────────────────────────────────────────────
table_I_path = OUT_DIR / "table_I_corrected.md"
with open(table_I_path, "w", encoding="utf-8") as f:
    f.write("# Table I — Classification Performance on BD-SHS Test Set\n\n")
    f.write("> **Corrected checkpoints** — loaded explicitly from `trainer_state.json` best.\n\n")
    f.write("| Model | Best Epoch | Global Step | Val F1 (at epoch) | "
            "Test Accuracy | Test Precision | Test Recall | Test F1 |\n")
    f.write("|-------|:----------:|:-----------:|:-----------------:|"
            ":-------------:|:--------------:|:-----------:|:-------:|\n")
    for name in ["BanglaBERT", "XLM-R"]:
        r = phase0_results[name]
        f.write(f"| {name} | {r['epoch']} | {r['global_step']:,} | "
                f"{r['val_f1_at_epoch']:.4f} | {r['test_accuracy']:.4f} | "
                f"{r['test_precision']:.4f} | {r['test_recall']:.4f} | "
                f"{r['test_f1']:.4f} |\n")
    f.write("\n\n## Delta vs. Previously Reported Numbers\n\n")
    f.write("| Model | Metric | Previous | Corrected | Delta |\n")
    f.write("|-------|--------|:--------:|:---------:|:-----:|\n")
    for _, row in delta_df.iterrows():
        flag = " ⚠" if abs(row["Delta"]) > 0.005 else ""
        f.write(f"| {row['Model']} | {row['Metric']} | {row['Previous']:.4f} | "
                f"{row['Corrected']:.4f} | {row['Delta']:+.4f}{flag} |\n")
    f.write(f"\n**Max |Δ|** = `{material_change:.4f}` — ")
    if material_change > 0.005:
        f.write("**MATERIAL change** — update paper with corrected numbers.\n")
    else:
        f.write("immaterial change (<0.005) — checkpoints are consistent.\n")
    f.write("\n\n## Checkpoint Provenance\n\n")
    f.write("- **BanglaBERT** `outputs/banglabert-bdshs/checkpoint-5028`\n")
    f.write("  - Epoch **2**, global step 5028\n")
    f.write("  - `load_best_model_at_end=True`, `metric_for_best_model=\"f1\"`\n")
    f.write("- **XLM-R** `outputs/xlmr-bdshs/checkpoint-12570`\n")
    f.write("  - Epoch **5**, global step 12570\n")
    f.write("  - `load_best_model_at_end=True`, `metric_for_best_model=\"f1\"`\n")
    f.write("\n> All downstream attribution, aggregation, and faithfulness results\n")
    f.write("> **must be regenerated** from these corrected checkpoint weights.\n")

print(f"Table I (corrected) saved -> {table_I_path}")
print("\n" + "=" * 70)
print("PHASE 0 COMPLETE")
print("=" * 70)
