"""
Phase 10 — Multi-Seed Training
==============================
Trains both models with multiple random seeds (42, 43, 44) to assess
training robustness and enable seed-level statistical analysis.

Run:  python training/multi_seed_training.py
Outputs: ./outputs/training/seeds/
"""

import os, sys, io, json, random
from pathlib import Path

# Keep Hugging Face + temp caches on E: so C: is not filled.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_ROOT = _PROJECT_ROOT / ".cache"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_ROOT / "huggingface" / "transformers"))
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TMP", str(_CACHE_ROOT / "tmp"))
os.environ.setdefault("TEMP", str(_CACHE_ROOT / "tmp"))

# Disable TF/Keras integration
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import torch

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

# Configuration from experiment config
SEEDS = [42, 43, 44]
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
METRIC_FOR_BEST_MODEL = "f1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("PHASE 10 — MULTI-SEED TRAINING")
print("=" * 60)
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
    print("VRAM            :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("PyTorch version :", torch.__version__)
print(f"Seeds to train  : {SEEDS}")
print(f"Max epochs      : {NUM_EPOCHS}")
print(f"Early stopping  : patience={EARLY_STOPPING_PATIENCE} (val F1)")
print("=" * 60)

# ─────────────────────────────────────────────
# PATHS & RUN LOGGER
# ─────────────────────────────────────────────
BASE_DIR = _PROJECT_ROOT
OUT_DIR = BASE_DIR / "outputs" / "training" / "seeds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-seed logging
class SeedLogger:
    def __init__(self, seed, output_dir):
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.log_file = self.output_dir / f"seed_{seed}_log.txt"
        self.fh = open(self.log_file, "w", encoding="utf-8")

    def write(self, data):
        self.fh.write(data)
        self.fh.flush()

    def close(self):
        self.fh.close()

# ─────────────────────────────────────────────
# 1.  LOAD & PREPARE DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1 — DATA PREPARATION")
print("=" * 60)

train_df = pd.read_csv(BASE_DIR / "train.csv")
val_df   = pd.read_csv(BASE_DIR / "val.csv")
test_df  = pd.read_csv(BASE_DIR / "test.csv")

TEXT_COL  = "sentence"
LABEL_COL = "hate speech"

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

label_values = sorted(train_df[LABEL_COL].unique().tolist())
raw_to_id    = {lbl: idx for idx, lbl in enumerate(label_values)}
label2id     = {str(lbl): int(idx) for idx, lbl in enumerate(label_values)}
id2label     = {int(idx): str(lbl) for idx, lbl in enumerate(label_values)}
num_labels   = len(label2id)

for df in [train_df, val_df, test_df]:
    df["labels"] = df[LABEL_COL].map(raw_to_id).astype(int)

# ─────────────────────────────────────────────
# 2.  HUGGING FACE DATASETS
# ─────────────────────────────────────────────
def to_hf_dataset(df):
    return Dataset.from_pandas(df[[TEXT_COL, "labels"]].reset_index(drop=True))

raw_datasets = DatasetDict({
    "train"     : to_hf_dataset(train_df),
    "validation": to_hf_dataset(val_df),
    "test"      : to_hf_dataset(test_df),
})

# ─────────────────────────────────────────────
# 3.  METRICS & HELPERS
# ─────────────────────────────────────────────
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

class SeedEpochLoggingCallback(TrainerCallback):
    """Print and persist validation metrics after every training epoch for specific seed."""
    def __init__(self, model_name, seed, output_dir):
        self.model_name = model_name
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.epoch_records = []
        self.last_train_loss = None
        self.csv_path = self.output_dir / f"seed_{seed}_epoch_metrics.csv"
        self.md_path = self.output_dir / f"seed_{seed}_epoch_metrics.md"

    def on_log(self, args, state, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def _write_epoch_files(self):
        if not self.epoch_records:
            return
        df_all = pd.DataFrame(self.epoch_records)
        df_all.to_csv(self.csv_path, index=False)
        with open(self.md_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Epoch-by-Epoch Validation — {self.model_name} (Seed {self.seed})\n\n")
            fh.write(f"Max epochs: {NUM_EPOCHS}  |  Early stopping patience: {EARLY_STOPPING_PATIENCE}\n\n")
            fh.write(df_all.to_markdown(index=False) + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
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
            "Seed": self.seed,
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

    def on_train_end(self, args, state, control, **kwargs):
        self._write_epoch_files()

# ─────────────────────────────────────────────
# 4.  TRAIN FUNCTION FOR SINGLE SEED
# ─────────────────────────────────────────────
def train_model_with_seed(model_name, model_hf_name, seed, output_base_dir):
    """Train a single model with a specific seed."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name} with seed {seed}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create seed-specific output directory
    seed_output_dir = output_base_dir / model_name.lower().replace("-", "_") / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = SeedLogger(seed, seed_output_dir)
    logger.write(f"Starting training for {model_name} with seed {seed}\n")
    logger.write(f"Output directory: {seed_output_dir}\n")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_hf_name)
    tokenized_datasets = tokenize_datasets(tokenizer, raw_datasets, max_length=MAX_LENGTH)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_hf_name, num_labels=num_labels, id2label=id2label, label2id=label2id, use_safetensors=True,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(seed_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=seed,
    )

    epoch_logger = SeedEpochLoggingCallback(model_name=model_name, seed=seed, output_dir=seed_output_dir)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=None,  # Add class weights if needed
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
            epoch_logger,
        ],
    )

    # Train
    logger.write("Starting training...\n")
    trainer.train()

    # Evaluate on test set
    logger.write("Evaluating on test set...\n")
    test_results = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")

    # Save results
    results = {
        "model": model_name,
        "seed": seed,
        "test_accuracy": float(test_results.get("test_accuracy", 0.0)),
        "test_f1": float(test_results.get("test_f1", 0.0)),
        "test_precision": float(test_results.get("test_precision", 0.0)),
        "test_recall": float(test_results.get("test_recall", 0.0)),
        "test_loss": float(test_results.get("test_loss", 0.0)),
        "best_checkpoint": str(trainer.state.best_model_checkpoint),
        "best_epoch": float(trainer.state.epoch),
    }

    logger.write(f"Test results: {json.dumps(results, indent=2)}\n")
    logger.close()

    # Save results to JSON
    results_file = seed_output_dir / f"seed_{seed}_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")
    return results

# ─────────────────────────────────────────────
# 5.  RUN MULTI-SEED TRAINING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — MULTI-SEED TRAINING")
print("=" * 60)

MODEL_CONFIGS = {
    "BanglaBERT": "csebuetnlp/banglabert",
    "XLM-R": "xlm-roberta-base",
}

all_results = []

for model_name, model_hf_name in MODEL_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    for seed in SEEDS:
        try:
            results = train_model_with_seed(model_name, model_hf_name, seed, OUT_DIR)
            all_results.append(results)
        except Exception as e:
            print(f"[ERROR] Training failed for {model_name} seed {seed}: {e}")
            import traceback
            traceback.print_exc()

# ─────────────────────────────────────────────
# 6.  GENERATE SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MULTI-SEED TRAINING SUMMARY")
print("=" * 60)

# Create summary DataFrame
summary_df = pd.DataFrame(all_results)
print("\nSeed Training Results:")
print(summary_df.to_string(index=False))

# Calculate statistics across seeds
print("\nStatistics across seeds:")
for model_name in MODEL_CONFIGS.keys():
    model_results = summary_df[summary_df["model"] == model_name]
    if len(model_results) > 0:
        print(f"\n{model_name}:")
        print(f"  Test F1:     {model_results['test_f1'].mean():.4f} ± {model_results['test_f1'].std():.4f}")
        print(f"  Test Acc:    {model_results['test_accuracy'].mean():.4f} ± {model_results['test_accuracy'].std():.4f}")
        print(f"  Best Epoch:  {model_results['best_epoch'].mean():.1f} ± {model_results['best_epoch'].std():.1f}")

# Save summary
summary_file = OUT_DIR / "multi_seed_summary.json"
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump({
        "seeds": SEEDS,
        "results": all_results,
        "summary": summary_df.to_dict(orient="records")
    }, f, indent=2)

# Save summary markdown
summary_md = OUT_DIR / "multi_seed_summary.md"
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Multi-Seed Training Summary\n\n")
    f.write(f"**Seeds**: {SEEDS}\n\n")
    f.write("## Results by Model and Seed\n\n")
    f.write(summary_df.to_markdown(index=False) + "\n\n")
    f.write("## Statistics Across Seeds\n\n")
    for model_name in MODEL_CONFIGS.keys():
        model_results = summary_df[summary_df["model"] == model_name]
        if len(model_results) > 0:
            f.write(f"### {model_name}\n\n")
            f.write(f"- Test F1: {model_results['test_f1'].mean():.4f} ± {model_results['test_f1'].std():.4f}\n")
            f.write(f"- Test Accuracy: {model_results['test_accuracy'].mean():.4f} ± {model_results['test_accuracy'].std():.4f}\n")
            f.write(f"- Best Epoch: {model_results['best_epoch'].mean():.1f} ± {model_results['best_epoch'].std():.1f}\n\n")

print(f"\nSummary saved to: {summary_file}")
print(f"Markdown summary saved to: {summary_md}")

print("\n" + "=" * 60)
print("Multi-seed training complete.")
print("=" * 60)
