"""
Phase 7 — Multi-Seed Training (Seeds 43, 44)
=============================================
Trains BanglaBERT and XLM-R with seeds 43 and 44 using EXACTLY the same
protocol as seed 42 (bdshs_pipeline.py). Seed 42 results are loaded from
the existing checkpoints and do NOT need to be re-run.

For each model × seed:
  1. Set all random seeds
  2. Train with same hyperparameters (epochs=15, early-stop on val F1)
  3. Select best checkpoint by val F1 (explicit criterion)
  4. Evaluate on test set
  5. Save predictions CSV

Outputs:
  results/classification/seed_{43,44}/banglabert/
  results/classification/seed_{43,44}/xlmr/
  results/classification/seed_robustness.csv
  results/classification/multiseed_train.log

Run:
  .venv\Scripts\python.exe -u training/train_multiseed.py

To train only one model, set TRAIN_MODELS env var:
  TRAIN_MODELS=BanglaBERT .venv\Scripts\python.exe -u training/train_multiseed.py

NOTE: Seed 42 data is loaded from existing checkpoints without re-training.
"""

import os, sys, io, json, random, logging, time, warnings
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"]              = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Keep caches on E:
_CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache"
os.environ.setdefault("HF_HOME",              str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE",str(_CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE",   str(_CACHE_ROOT / "huggingface" / "transformers"))
os.environ.setdefault("HF_HUB_CACHE",         str(_CACHE_ROOT / "huggingface" / "hub"))

import numpy as np
import pandas as pd
import torch
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "classification"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "multiseed_train.log"
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Configuration (frozen — matches experiment_config.yaml)
# ─────────────────────────────────────────────────────────────
TEXT_COL  = "sentence"
LABEL_COL = "hate speech"

NUM_EPOCHS             = 15
EARLY_STOPPING_PATIENCE = 3
BATCH_SIZE             = 16
LEARNING_RATE          = 2e-5
MAX_LENGTH             = 128
WEIGHT_DECAY           = 0.01

SEEDS_TO_TRAIN = [43, 44]   # seed 42 already exists
SEED_42_CHECKPOINTS = {
    "BanglaBERT": BASE_DIR / "outputs" / "banglabert-bdshs" / "checkpoint-5028",
    "XLM-R":      BASE_DIR / "outputs" / "xlmr-bdshs"       / "checkpoint-12570",
}
SEED_42_EPOCHS = {
    "BanglaBERT": 2,
    "XLM-R":      5,
}
SEED_42_VAL_F1 = {
    "BanglaBERT": 0.9037,
    "XLM-R":      0.9146,
}

MODEL_CONFIGS = {
    "BanglaBERT": "csebuetnlp/banglabert",
    "XLM-R":      "xlm-roberta-base",
}

TRAIN_MODELS = os.environ.get("TRAIN_MODELS", "BanglaBERT,XLM-R").split(",")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
logger.info(f"Seeds to train: {SEEDS_TO_TRAIN}")
logger.info(f"Models to train: {TRAIN_MODELS}")


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
def load_data():
    train_df = pd.read_csv(BASE_DIR / "train.csv")
    val_df   = pd.read_csv(BASE_DIR / "val.csv")
    test_df  = pd.read_csv(BASE_DIR / "test.csv")

    def clean(df):
        df = df.copy().dropna(subset=[TEXT_COL, LABEL_COL])
        df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
        df = df[df[TEXT_COL].str.len() > 0].drop_duplicates(subset=[TEXT_COL])
        return df

    train_df = clean(train_df)
    val_df   = clean(val_df)
    test_df  = clean(test_df)

    label_values = sorted(train_df[LABEL_COL].unique().tolist())
    raw_to_id = {lbl: idx for idx, lbl in enumerate(label_values)}

    for df in [train_df, val_df, test_df]:
        df["labels"] = df[LABEL_COL].map(raw_to_id).astype(int)

    logger.info(f"Data loaded — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────
# Train one model at one seed
# ─────────────────────────────────────────────────────────────
def train_model(model_name: str, hf_id: str, seed: int,
                train_df, val_df, test_df) -> dict:
    set_all_seeds(seed)
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}  seed={seed}  hf_id={hf_id}")

    out_dir = RESULTS_DIR / f"seed_{seed}" / model_name.lower().replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    def tokenize(df):
        enc = tokenizer(
            df[TEXT_COL].tolist(),
            truncation=True, max_length=MAX_LENGTH, padding=False,
        )
        return Dataset.from_dict({
            **{k: v for k, v in enc.items()},
            "labels": df["labels"].tolist(),
        })

    ds_train = tokenize(train_df)
    ds_val   = tokenize(val_df)
    ds_test  = tokenize(test_df)

    collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir                  = str(out_dir / "checkpoints"),
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE * 2,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        seed                        = seed,
        data_seed                   = seed,
        fp16                        = torch.cuda.is_available(),
        logging_steps               = 500,
        report_to                   = "none",
        save_total_limit            = 3,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = ds_train,
        eval_dataset    = ds_val,
        tokenizer       = tokenizer,
        data_collator   = collator,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )],
    )

    trainer.train()

    # ── Identify best epoch by val F1 ──────────────────────
    log_history = trainer.state.log_history
    best_val_f1 = -1
    best_epoch  = -1
    epoch_history = []

    for entry in log_history:
        if "eval_f1" in entry:
            epoch = int(entry.get("epoch", -1))
            f1    = entry["eval_f1"]
            epoch_history.append({
                "epoch":         epoch,
                "eval_f1":       f1,
                "eval_accuracy": entry.get("eval_accuracy"),
                "eval_loss":     entry.get("eval_loss"),
            })
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_epoch  = epoch

    logger.info(f"  Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Evaluate on test set ────────────────────────────────
    test_out  = trainer.predict(ds_test)
    test_preds = np.argmax(test_out.predictions, axis=-1)
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()
    test_labels = ds_test["labels"]

    test_metrics = {
        "test_accuracy":  float(accuracy_score(test_labels, test_preds)),
        "test_f1":        float(f1_score(test_labels, test_preds, average="macro", zero_division=0)),
        "test_precision": float(precision_score(test_labels, test_preds, average="macro", zero_division=0)),
        "test_recall":    float(recall_score(test_labels, test_preds, average="macro", zero_division=0)),
    }
    logger.info(f"  Test: acc={test_metrics['test_accuracy']:.4f}  "
                f"F1={test_metrics['test_f1']:.4f}")

    # ── Save test predictions CSV ───────────────────────────
    pred_df = pd.DataFrame({
        "example_id":       range(len(test_labels)),
        "true_label":       test_labels,
        "predicted_label":  test_preds,
        "prob_class0":      test_probs[:, 0],
        "prob_class1":      test_probs[:, 1],
    })
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8")

    # ── Save model ──────────────────────────────────────────
    model_save_dir = out_dir / "best_model"
    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))
    logger.info(f"  Model saved → {model_save_dir}")

    # ── Save epoch metrics ──────────────────────────────────
    epoch_df = pd.DataFrame(epoch_history)
    epoch_df.to_csv(out_dir / "epoch_metrics.csv", index=False, encoding="utf-8")

    result = {
        "model":          model_name,
        "hf_id":          hf_id,
        "seed":           seed,
        "best_epoch":     best_epoch,
        "best_val_f1":    best_val_f1,
        "checkpoint_dir": str(model_save_dir),
        **test_metrics,
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


# ─────────────────────────────────────────────────────────────
# Evaluate seed 42 from existing checkpoints (no re-training)
# ─────────────────────────────────────────────────────────────
def load_seed42_results(test_df) -> list[dict]:
    logger.info("\nLoading seed 42 from existing checkpoints …")
    rows = []

    for model_name, ckpt_path in SEED_42_CHECKPOINTS.items():
        if model_name not in TRAIN_MODELS:
            continue
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        model     = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path))
        model     = model.to(DEVICE).eval()

        texts  = test_df[TEXT_COL].tolist()
        labels = test_df["labels"].tolist()
        enc    = tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                           padding=True, return_tensors="pt")
        ds = Dataset.from_dict({k: v.tolist() for k, v in enc.items()} | {"labels": labels})

        # Batch inference
        all_preds, all_probs = [], []
        batch_size = 32
        for i in range(0, len(ds), batch_size):
            batch = {k: torch.tensor(v[i:i+batch_size]).to(DEVICE)
                     for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**batch).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(np.argmax(probs, axis=-1).tolist())
            all_probs.extend(probs.tolist())

        all_probs_arr = np.array(all_probs)
        test_metrics = {
            "test_accuracy":  float(accuracy_score(labels, all_preds)),
            "test_f1":        float(f1_score(labels, all_preds, average="macro", zero_division=0)),
            "test_precision": float(precision_score(labels, all_preds, average="macro", zero_division=0)),
            "test_recall":    float(recall_score(labels, all_preds, average="macro", zero_division=0)),
        }

        # Save predictions
        s42_dir = RESULTS_DIR / "seed_42" / model_name.lower().replace("-", "_")
        s42_dir.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame({
            "example_id":       range(len(labels)),
            "true_label":       labels,
            "predicted_label":  all_preds,
            "prob_class0":      all_probs_arr[:, 0],
            "prob_class1":      all_probs_arr[:, 1],
        })
        pred_df.to_csv(s42_dir / "test_predictions.csv", index=False, encoding="utf-8")

        result = {
            "model":          model_name,
            "seed":           42,
            "best_epoch":     SEED_42_EPOCHS[model_name],
            "best_val_f1":    SEED_42_VAL_F1[model_name],
            "checkpoint_dir": str(ckpt_path),
            **test_metrics,
        }
        with open(s42_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        rows.append(result)
        logger.info(f"  {model_name} seed 42: "
                    f"acc={test_metrics['test_accuracy']:.4f}  "
                    f"F1={test_metrics['test_f1']:.4f}")

        del model
        torch.cuda.empty_cache()

    return rows


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    train_df, val_df, test_df = load_data()

    all_results = []

    # Seed 42 — from existing checkpoints
    s42_results = load_seed42_results(test_df)
    all_results.extend(s42_results)

    # Seeds 43, 44 — train fresh
    for seed in SEEDS_TO_TRAIN:
        for model_name, hf_id in MODEL_CONFIGS.items():
            if model_name not in TRAIN_MODELS:
                logger.info(f"Skipping {model_name} (not in TRAIN_MODELS)")
                continue
            result = train_model(
                model_name=model_name,
                hf_id=hf_id,
                seed=seed,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )
            all_results.append(result)
            torch.cuda.empty_cache()

    # ── Robustness summary table ────────────────────────────
    robustness_df = pd.DataFrame(all_results)
    robustness_df = robustness_df.sort_values(["model", "seed"])
    robustness_path = RESULTS_DIR / "seed_robustness.csv"
    robustness_df.to_csv(robustness_path, index=False, encoding="utf-8")
    logger.info(f"\nSeed robustness table → {robustness_path}")
    logger.info(f"\n{robustness_df.to_string(index=False)}")

    # ── Markdown report ────────────────────────────────────
    report_path = RESULTS_DIR / "seed_robustness_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Seed Robustness Report — Classification\n\n")
        f.write("Same hyperparameters, different random seeds.\n\n")
        f.write("## Table A — Classification Robustness\n\n")
        display_cols = ["model", "seed", "best_epoch", "best_val_f1",
                        "test_accuracy", "test_f1", "test_precision", "test_recall"]
        display = robustness_df[[c for c in display_cols if c in robustness_df.columns]]
        try:
            f.write(display.to_markdown(index=False))
        except Exception:
            f.write(display.to_string(index=False))

        f.write("\n\n## Summary Statistics (by model)\n\n")
        for model_name, grp in robustness_df.groupby("model"):
            f.write(f"### {model_name}\n\n")
            for col in ["test_f1", "test_accuracy"]:
                if col in grp.columns:
                    vals = grp[col].values
                    f.write(f"- **{col}**: mean={np.mean(vals):.4f}  "
                            f"std={np.std(vals, ddof=1):.4f}  "
                            f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]\n")
            f.write("\n")

    logger.info(f"Report → {report_path}")
    logger.info(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled error in train_multiseed.py:")
        sys.exit(1)
