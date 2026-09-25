"""
BD-SHS Hate Speech Detection — Full Training Pipeline
======================================================
Dataset:  train.csv / val.csv / test.csv  (columns: sentence, target, type, hate speech)
Models:   csebuetnlp/banglabert  |  xlm-roberta-base
GPU:      NVIDIA RTX A6000  (use E:\\BD SHS\\.venv)

Run:  .\\run_train.ps1
      or:  .\\.venv\\Scripts\\python.exe -u bdshs_pipeline.py
Outputs are saved to ./outputs/

Retrains both models on the FULL train/val/test splits.
Uses a high epoch cap with early stopping; epoch metrics are written
to the run log, per-model CSV/MD, and training_summary_report.md.
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS & REPRODUCIBILITY
# ─────────────────────────────────────────────
import os, sys, io, random, json, warnings
from pathlib import Path

# Keep Hugging Face + temp caches on E: so C: is not filled.
_PROJECT_ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = _PROJECT_ROOT / ".cache"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_ROOT / "huggingface" / "transformers"))
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TMP", str(_CACHE_ROOT / "tmp"))
os.environ.setdefault("TEMP", str(_CACHE_ROOT / "tmp"))

# ── Disable TF/Keras integration in transformers (we use PyTorch only) ──────
# Must be set BEFORE importing transformers to avoid Keras 3 / tf-keras conflict.
os.environ["TRANSFORMERS_NO_TF"]              = "1"   # skip all TF-dependent code paths
os.environ["TF_ENABLE_ONEDNN_OPTS"]           = "0"   # silence oneDNN info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]            = "3"   # suppress TF C++ logs if TF loads
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # suppress Windows symlinks warning
import numpy as np
import pandas as pd
import torch

# Redirect stdout to UTF-8 so Bangla characters print correctly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Full-dataset fine-tuning: high epoch cap; early stopping picks the real stop.
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 60)
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
    print("VRAM            :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("PyTorch version :", torch.__version__)
print(f"Max epochs      : {NUM_EPOCHS}")
print(f"Early stopping  : patience={EARLY_STOPPING_PATIENCE} (val F1)")
print(f"Batch / LR      : {BATCH_SIZE} / {LEARNING_RATE}")
print("=" * 60)

# ─────────────────────────────────────────────
# PATHS & RUN LOGGER
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "outputs"
LOG_FILE = OUT_DIR / "run_log.txt"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Tee output to a log file as well
class Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh
    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
    def flush(self):
        self.stream.flush()
        self.fh.flush()
    def isatty(self):
        return False   # not a real TTY — suppress ANSI/colour output
    def fileno(self):
        return self.stream.fileno()
    def readable(self):
        return False
    @property
    def encoding(self):
        return self.stream.encoding

_log_fh = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log_fh)

# ─────────────────────────────────────────────
# 1.  LOAD & INSPECT DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1 — DATA INSPECTION")
print("=" * 60)

train_df = pd.read_csv(BASE_DIR / "train.csv")
val_df   = pd.read_csv(BASE_DIR / "val.csv")
test_df  = pd.read_csv(BASE_DIR / "test.csv")

# Column mapping confirmed from pre-run inspection:
#   Text  -> "sentence"
#   Label -> "hate speech"  (int64: 0=non-hate, 1=hate)
TEXT_COL  = "sentence"
LABEL_COL = "hate speech"

print("\nColumn names  :", train_df.columns.tolist())
print("Shapes — train:", train_df.shape, "| val:", val_df.shape, "| test:", test_df.shape)
print("\nDtypes:\n", train_df.dtypes.to_string())
print("\nNunique:\n", train_df.nunique().to_string())
print("\nNull counts (train):\n", train_df.isnull().sum().to_string())

for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{split_name} label distribution:")
    vc  = df[LABEL_COL].value_counts(dropna=False)
    pct = df[LABEL_COL].value_counts(normalize=True, dropna=False) * 100
    combined = pd.DataFrame({"count": vc, "pct (%)": pct.round(2)})
    print(combined.to_string())

print("\nTYPE distribution (train, all values):")
print(train_df["type"].value_counts(dropna=False).to_string())

print("\nTARGET distribution (train, all values):")
print(train_df["target"].value_counts(dropna=False).to_string())

print("\nSentence word-length stats (train):")
wlen = train_df[TEXT_COL].astype(str).str.split().str.len()
print(wlen.describe().to_string())
print(f"  95th percentile : {wlen.quantile(0.95):.0f} words")
print(f"  99th percentile : {wlen.quantile(0.99):.0f} words")
print(f"  max             : {wlen.max()} words")

# ─────────────────────────────────────────────
# 2.  CLEAN & ENCODE LABELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — CLEANING & LABEL ENCODING")
print("=" * 60)

def clean_df(df):
    df = df.copy()
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL].str.len() > 0]
    df = df.drop_duplicates(subset=[TEXT_COL])
    return df

train_df = clean_df(train_df)
val_df   = clean_df(val_df)
test_df  = clean_df(test_df)

print("After cleaning — train:", train_df.shape, "| val:", val_df.shape, "| test:", test_df.shape)

# Labels are already int64 (0=non-hate, 1=hate) — build label<->id maps
label_values = sorted(train_df[LABEL_COL].unique().tolist())
raw_to_id    = {lbl: idx for idx, lbl in enumerate(label_values)}
label2id     = {str(lbl): int(idx) for idx, lbl in enumerate(label_values)}
id2label     = {int(idx): str(lbl) for idx, lbl in enumerate(label_values)}
num_labels   = len(label2id)

print("Label mapping  :", label2id)
print("num_labels     :", num_labels)

for df in [train_df, val_df, test_df]:
    df["labels"] = df[LABEL_COL].map(raw_to_id).astype(int)

# ─────────────────────────────────────────────
# CLASS IMBALANCE CHECK
# ─────────────────────────────────────────────
print("\n--- Class imbalance check (train) ---")
class_counts = train_df["labels"].value_counts().sort_index()
total        = len(train_df)
for cls, cnt in class_counts.items():
    print(f"  Class {cls} ({id2label[cls]}): {cnt:,}  ({cnt/total*100:.1f}%)")

majority_ratio = class_counts.max() / class_counts.min()
print(f"\nMajority/minority ratio: {majority_ratio:.2f}x")

if majority_ratio > 1.5:
    print("[INFO] Imbalance detected — computing inverse-frequency class weights.")
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(label2id.values())),
        y=train_df["labels"].values
    )
    class_weights     = torch.tensor(class_weights_arr, dtype=torch.float)
    USE_WEIGHTED_LOSS = True
    print("  Class weights:", class_weights.tolist())
else:
    print("[INFO] Distribution is balanced — standard cross-entropy.")
    class_weights     = None
    USE_WEIGHTED_LOSS = False

# ─────────────────────────────────────────────
# 3.  HUGGING FACE DATASETS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — HUGGING FACE DATASETS")
print("=" * 60)

def to_hf_dataset(df):
    return Dataset.from_pandas(df[[TEXT_COL, "labels"]].reset_index(drop=True))

raw_datasets = DatasetDict({
    "train"     : to_hf_dataset(train_df),
    "validation": to_hf_dataset(val_df),
    "test"      : to_hf_dataset(test_df),
})
print(raw_datasets)

# ─────────────────────────────────────────────
# 4.  TOKENIZER FRAGMENTATION ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — TOKENIZER FRAGMENTATION ANALYSIS")
print("=" * 60)

def avg_tokens_per_word(tokenizer, texts, sample_size=1000):
    sample = texts[:sample_size]
    ratios = []
    for text in sample:
        words = text.split()
        if not words:
            continue
        n_subwords = len(tokenizer.tokenize(text))
        ratios.append(n_subwords / len(words))
    return float(np.mean(ratios)), float(np.std(ratios))

print("Loading tokenizers for fragmentation probing ...")
bangla_tok_probe = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
xlmr_tok_probe   = AutoTokenizer.from_pretrained("xlm-roberta-base")

sample_texts = train_df[TEXT_COL].tolist()
bb_mean,   bb_std   = avg_tokens_per_word(bangla_tok_probe, sample_texts)
xlmr_mean, xlmr_std = avg_tokens_per_word(xlmr_tok_probe,  sample_texts)

print(f"\nBanglaBERT : {bb_mean:.3f} subwords/word  (std {bb_std:.3f})")
print(f"XLM-R      : {xlmr_mean:.3f} subwords/word  (std {xlmr_std:.3f})")
print(f"XLM-R fragments {xlmr_mean/bb_mean:.2f}x more than BanglaBERT")
print(
    "\nInterpretation:\n"
    "  BanglaBERT was pre-trained on 18.6 GB of Bangla text with a Bangla-specific\n"
    "  SentencePiece vocabulary, representing common Bangla words as single/few\n"
    "  subword tokens. XLM-R uses a 250k multilingual shared vocabulary where Bangla\n"
    "  receives a smaller share, causing systematically heavier fragmentation.\n"
    "  This confirms the two distinct fragmentation regimes used in the paper."
)

frag_results = {
    "BanglaBERT_mean": round(bb_mean, 4),
    "BanglaBERT_std" : round(bb_std,  4),
    "XLM_R_mean"     : round(xlmr_mean, 4),
    "XLM_R_std"      : round(xlmr_std,  4),
    "ratio"          : round(xlmr_mean / bb_mean, 4),
}
with open(OUT_DIR / "fragmentation_results.json", "w", encoding="utf-8") as f:
    json.dump(frag_results, f, indent=2)
print(f"\nFragmentation results saved → {OUT_DIR / 'fragmentation_results.json'}")

# ─────────────────────────────────────────────
# 5.  METRICS & TRAINING HELPERS
# ─────────────────────────────────────────────
def df_to_markdown_str(df, include_index=True):
    """Pure-Python markdown table generator without tabulate dependency."""
    df_reset = df.reset_index() if include_index else df.copy()
    headers = [str(col) for col in df_reset.columns]
    rows = [[str(val) for val in row] for row in df_reset.values]
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join([header_line, sep_line] + row_lines)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    avg = "binary" if num_labels == 2 else "macro"
    return {
        "accuracy" : float(accuracy_score(labels, preds)),
        "f1"       : float(f1_score(labels, preds, average=avg, zero_division=0)),
        "precision": float(precision_score(labels, preds, average=avg, zero_division=0)),
        "recall"   : float(recall_score(labels, preds, average=avg, zero_division=0)),
    }

def tokenize_datasets(tokenizer, raw_datasets, max_length=128):
    def tokenize_fn(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length)
    tokenized = raw_datasets.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns([TEXT_COL])
    tokenized.set_format("torch")
    return tokenized

# Custom Trainer with optional weighted cross-entropy loss
class WeightedTrainer(Trainer):
    """Injects class-weighted CrossEntropyLoss to handle class imbalance."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(DEVICE) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights if self.class_weights is not None else None
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

from transformers import TrainerCallback

class EpochLoggingCallback(TrainerCallback):
    """Print and persist validation metrics after every training epoch."""
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.epoch_records = []
        self.last_train_loss = None
        self.csv_path = self.output_dir / "epoch_metrics.csv"
        self.md_path = self.output_dir / "epoch_metrics.md"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def _write_epoch_files(self):
        if not self.epoch_records:
            return
        df_all = pd.DataFrame(self.epoch_records)
        df_all.to_csv(self.csv_path, index=False)
        with open(self.md_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Epoch-by-Epoch Validation — {self.model_name}\n\n")
            fh.write(f"Max epochs: {NUM_EPOCHS}  |  Early stopping patience: {EARLY_STOPPING_PATIENCE}\n\n")
            fh.write(df_to_markdown_str(df_all, include_index=False) + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        # Skip the post-training TEST evaluate() call (uses metric_key_prefix="test").
        if any(k.startswith("test_") for k in metrics):
            return
        if "eval_f1" not in metrics:
            return

        epoch = metrics.get("epoch", state.epoch)
        train_loss = self.last_train_loss
        for entry in reversed(state.log_history):
            if "loss" in entry and entry.get("epoch") is not None and abs(entry["epoch"] - epoch) < 0.1:
                train_loss = entry["loss"]
                break

        record = {
            "Epoch": int(round(epoch)) if epoch is not None else len(self.epoch_records) + 1,
            "Train Loss": round(float(train_loss), 4) if train_loss is not None else "-",
            "Val Loss": round(float(metrics.get("eval_loss", 0.0)), 4),
            "Val Accuracy": round(float(metrics.get("eval_accuracy", 0.0)), 4),
            "Val F1": round(float(metrics.get("eval_f1", 0.0)), 4),
            "Val Precision": round(float(metrics.get("eval_precision", 0.0)), 4),
            "Val Recall": round(float(metrics.get("eval_recall", 0.0)), 4),
        }
        self.epoch_records.append(record)
        self._write_epoch_files()

        print("\n" + "─" * 78)
        print(f"[EPOCH {record['Epoch']}] {self.model_name} — VALIDATION METRICS")
        print("─" * 78)
        print(pd.DataFrame([record]).to_string(index=False))
        print("Full epoch table so far:")
        print(pd.DataFrame(self.epoch_records).to_string(index=False))
        print("─" * 78 + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        self._write_epoch_files()
        if self.epoch_records:
            print(f"\nSaved epoch history ({self.model_name})")
            print(f"  CSV → {self.csv_path}")
            print(f"  MD  → {self.md_path}")

print("Metrics and helpers defined.")
print(f"  Averaging strategy : {'binary' if num_labels == 2 else 'macro'} (num_labels={num_labels})")
print(f"  Weighted loss      : {USE_WEIGHTED_LOSS}")

# ─────────────────────────────────────────────
# MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────
def train_model(
    model_name, output_dir, raw_datasets, num_labels, id2label, label2id,
    class_weights=None, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
    lr=LEARNING_RATE, max_length=MAX_LENGTH, early_stopping_patience=EARLY_STOPPING_PATIENCE,
):
    n_train = len(raw_datasets["train"])
    n_val = len(raw_datasets["validation"])
    n_test = len(raw_datasets["test"])
    print(f"\n{'─'*60}")
    print(f"Training: {model_name}")
    print(f"  Full splits     : train={n_train:,}  val={n_val:,}  test={n_test:,}")
    print(f"  Max epochs      : {epochs}  |  Early stopping patience: {early_stopping_patience}")
    print(f"  Batch / LR / max_length : {batch_size} / {lr} / {max_length}")
    print(f"{'─'*60}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = tokenize_datasets(tokenizer, raw_datasets, max_length=max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id, use_safetensors=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=SEED,
    )

    epoch_logger = EpochLoggingCallback(model_name=model_name, output_dir=output_dir)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            epoch_logger,
        ],
    )

    trainer.train()

    print(f"\n--- Evaluating on FULL TEST split ({n_test:,} examples): {model_name} ---")
    test_results = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
    print(f"\n=== {model_name} TEST RESULTS ===")
    for k, v in sorted(test_results.items()):
        if isinstance(v, float):
            print(f"  {k:45s}: {v:.4f}")
        else:
            print(f"  {k:45s}: {v}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nModel + tokenizer saved → {output_dir}")

    return trainer, tokenizer, test_results, epoch_logger.epoch_records

# ─────────────────────────────────────────────
# 6.  TRAIN BANGLABERT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — TRAIN BANGLABERT")
print("=" * 60)

bb_out = OUT_DIR / "banglabert-bdshs"
bb_trainer, bb_tokenizer, banglabert_results, bb_epochs = train_model(
    model_name    = "csebuetnlp/banglabert",
    output_dir    = bb_out,
    raw_datasets  = raw_datasets,
    num_labels    = num_labels,
    id2label      = id2label,
    label2id      = label2id,
    class_weights = class_weights if USE_WEIGHTED_LOSS else None,
    epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, max_length=MAX_LENGTH,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
)

# ─────────────────────────────────────────────
# 7.  TRAIN XLM-R
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — TRAIN XLM-R")
print("=" * 60)

xlmr_out = OUT_DIR / "xlmr-bdshs"
xlmr_trainer, xlmr_tokenizer, xlmr_results, xlmr_epochs = train_model(
    model_name    = "xlm-roberta-base",
    output_dir    = xlmr_out,
    raw_datasets  = raw_datasets,
    num_labels    = num_labels,
    id2label      = id2label,
    label2id      = label2id,
    class_weights = class_weights if USE_WEIGHTED_LOSS else None,
    epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, max_length=MAX_LENGTH,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
)

# ─────────────────────────────────────────────
# 8.  FINAL COMPARISON & SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 8 — FINAL COMPARISON & SUMMARY REPORT")
print("=" * 60)

print("\n--- BanglaBERT test results ---")
print(banglabert_results)
print("\n--- XLM-R test results ---")
print(xlmr_results)

metric_keys = ["test_accuracy", "test_f1", "test_precision", "test_recall"]
comparison  = {}
for name, res in [("BanglaBERT", banglabert_results), ("XLM-R", xlmr_results)]:
    comparison[name] = {k.replace("test_", ""): round(res.get(k, float("nan")), 4)
                        for k in metric_keys}

comp_df = pd.DataFrame(comparison).T
print("\n" + comp_df.to_string())

winner = comp_df["f1"].idxmax()
print(f"\n🏆 Best model by F1 : {winner}  (F1 = {comp_df.loc[winner, 'f1']:.4f})")

# Persist all results
results_dict = {
    "dataset_stats": {
        "train_size"    : int(train_df.shape[0]),
        "val_size"      : int(val_df.shape[0]),
        "test_size"     : int(test_df.shape[0]),
        "num_labels"    : num_labels,
        "label2id"      : {str(k): int(v) for k, v in label2id.items()},
        "class_dist_train": {
            str(cls): {"count": int(cnt), "pct": round(cnt / total * 100, 2)}
            for cls, cnt in class_counts.items()
        },
        "sentence_len_p50" : float(wlen.quantile(0.50)),
        "sentence_len_p95" : float(wlen.quantile(0.95)),
        "sentence_len_p99" : float(wlen.quantile(0.99)),
        "sentence_len_max" : int(wlen.max()),
    },
    "fragmentation": frag_results,
    "epoch_history": {
        "BanglaBERT": bb_epochs,
        "XLM_R"     : xlmr_epochs,
    },
    "test_results": {
        "BanglaBERT": {k: float(v) if isinstance(v, float) else v
                       for k, v in banglabert_results.items()},
        "XLM_R"     : {k: float(v) if isinstance(v, float) else v
                       for k, v in xlmr_results.items()},
    },
    "hyperparameters": {
        "epochs"           : NUM_EPOCHS,
        "batch_size"       : BATCH_SIZE,
        "lr"               : LEARNING_RATE,
        "max_length"       : MAX_LENGTH,
        "weight_decay"     : 0.01,
        "fp16"             : bool(torch.cuda.is_available()),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "use_weighted_loss": USE_WEIGHTED_LOSS,
        "class_weights"    : class_weights.tolist() if USE_WEIGHTED_LOSS else None,
        "trained_on_full_dataset": True,
    },
    "best_model": winner,
}

out_json = OUT_DIR / "all_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=2, ensure_ascii=False)

# Markdown report generation
report_path = OUT_DIR / "training_summary_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# BD-SHS Training & Evaluation Summary Report\n\n")
    f.write(f"**Best Model**: `{winner}` (Test F1: {comp_df.loc[winner, 'f1']:.4f})\n\n")
    f.write("## Dataset (full splits)\n\n")
    f.write(f"- Train: {int(train_df.shape[0]):,}\n")
    f.write(f"- Val: {int(val_df.shape[0]):,}\n")
    f.write(f"- Test: {int(test_df.shape[0]):,}\n")
    f.write(f"- Max epochs: {NUM_EPOCHS}\n")
    f.write(f"- Early stopping patience: {EARLY_STOPPING_PATIENCE} (metric: val F1)\n")
    f.write(f"- Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}  |  max_length: {MAX_LENGTH}\n\n")
    f.write("## 1. Test Set Comparison\n\n")
    f.write(df_to_markdown_str(comp_df, include_index=True) + "\n\n")
    f.write("## 2. Epoch-by-Epoch Validation Progression\n\n")
    f.write("### BanglaBERT (`csebuetnlp/banglabert`)\n\n")
    f.write(df_to_markdown_str(pd.DataFrame(bb_epochs), include_index=False) + "\n\n")
    f.write("### XLM-RoBERTa (`xlm-roberta-base`)\n\n")
    f.write(df_to_markdown_str(pd.DataFrame(xlmr_epochs), include_index=False) + "\n\n")

print(f"\nAll results saved     → {out_json}")
print(f"Markdown report saved → {report_path}")
print(f"Run log saved         → {LOG_FILE}")
print("\n" + "=" * 60)
print("Pipeline complete.")
print("=" * 60)

_log_fh.close()
