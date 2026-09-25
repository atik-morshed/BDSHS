"""Compute word-level IG/SHAP Spearman agreement for the full test outputs."""

from pathlib import Path
import json
import logging
import math
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError as exc:
    raise ImportError(
        "scipy is required. Install it with: "
        "E:\\BD SHS\\.venv\\Scripts\\python.exe -m pip install scipy"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "ig_shap_spearman.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_records(path):
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}")
    return records


def build_map(records, model_name, method_name):
    result = {}
    for index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str) or not text:
            logger.warning("%s %s record %d has no usable text", model_name, method_name, index)
            continue
        if text in result:
            raise ValueError(f"Duplicate text in {model_name} {method_name} records")
        result[text] = record
    return result


def aligned_scores(ig_record, shap_record, field):
    ig_words = ig_record.get("word_attribution")
    shap_words = shap_record.get("word_attribution")
    if not isinstance(ig_words, list) or not isinstance(shap_words, list):
        return None, "missing word_attribution"

    ig_names = [entry.get("word") for entry in ig_words]
    shap_names = [entry.get("word") for entry in shap_words]
    if ig_names != shap_names:
        return None, "word sequence mismatch"

    ig_scores = []
    shap_scores = []
    for position, (ig_entry, shap_entry) in enumerate(zip(ig_words, shap_words)):
        if field not in ig_entry or field not in shap_entry:
            return None, f"missing {field} at word {position}"
        try:
            ig_scores.append(float(ig_entry[field]))
            shap_scores.append(float(shap_entry[field]))
        except (TypeError, ValueError):
            return None, f"non-numeric {field} at word {position}"

    if len(ig_scores) < 2:
        return None, "fewer than two aligned words"
    if not np.isfinite(ig_scores).all() or not np.isfinite(shap_scores).all():
        return None, "non-finite score"
    if np.allclose(ig_scores, ig_scores[0]) or np.allclose(shap_scores, shap_scores[0]):
        return None, "constant score vector"
    return (ig_scores, shap_scores), None


def compute_model(model_name, ig_path, shap_path):
    ig_map = build_map(load_records(ig_path), model_name, "IG")
    shap_map = build_map(load_records(shap_path), model_name, "SHAP")
    texts = sorted(set(ig_map).intersection(shap_map))
    logger.info("%s: IG=%d, SHAP=%d, paired texts=%d", model_name, len(ig_map), len(shap_map), len(texts))

    rows = []
    skipped = []
    for text in texts:
        for field in ("sum", "mean", "max"):
            aligned, reason = aligned_scores(ig_map[text], shap_map[text], field)
            if aligned is None:
                skipped.append({"model": model_name, "field": field, "reason": reason})
                continue
            ig_scores, shap_scores = aligned
            statistic = spearmanr(ig_scores, shap_scores)
            rho = float(statistic.statistic)
            p_value = float(statistic.pvalue)
            rows.append({
                "model": model_name,
                "text": text,
                "field": field,
                "n_words": len(ig_scores),
                "spearman_rho": rho,
                "p_value": p_value,
            })

    if skipped:
        logger.warning("%s: skipped %d model/field pairs", model_name, len(skipped))
        for reason, count in pd.Series(item["reason"] for item in skipped).value_counts().items():
            logger.warning("%s: %d skipped because %s", model_name, count, reason)
    return rows, skipped


def make_summary(details):
    summary = []
    for (model, field), group in details.groupby(["model", "field"], sort=True):
        values = group["spearman_rho"].to_numpy(dtype=float)
        summary.append({
            "model": model,
            "field": field,
            "n_paired_examples": int(len(values)),
            "mean_rho": float(np.mean(values)),
            "median_rho": float(np.median(values)),
            "std_rho": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min_rho": float(np.min(values)),
            "max_rho": float(np.max(values)),
            "pct_positive": float(100.0 * np.mean(values > 0)),
            "mean_p_value": float(group["p_value"].mean()),
        })
    return pd.DataFrame(summary)


def main():
    configs = {
        "BanglaBERT": (
            OUT_DIR / "ig_banglabert_token.json",
            OUT_DIR / "shap_banglabert_token.json",
        ),
        "XLM-R": (
            OUT_DIR / "ig_xlm_r_token.json",
            OUT_DIR / "shap_xlm_r_token.json",
        ),
    }
    all_rows = []
    skipped_total = []
    for model, (ig_path, shap_path) in configs.items():
        for path in (ig_path, shap_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing required attribution file: {path}")
        rows, skipped = compute_model(model, ig_path, shap_path)
        all_rows.extend(rows)
        skipped_total.extend(skipped)

    if not all_rows:
        raise RuntimeError("No aligned IG/SHAP word-level examples were available")

    details = pd.DataFrame(all_rows)
    summary = make_summary(details)
    details_path = OUT_DIR / "ig_shap_spearman_by_example.csv"
    summary_path = OUT_DIR / "ig_shap_spearman_summary.csv"
    details.to_csv(details_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    report_path = OUT_DIR / "ig_shap_spearman_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# IG–SHAP Word-Level Spearman Agreement\n\n")
        handle.write(
            "This report compares Integrated Gradients and SHAP attribution "
            "rankings for the same saved word sequence in each example. "
            "Correlations are computed separately for `sum`, `mean`, and `max` "
            "word attribution scores.\n\n"
        )
        handle.write("## Summary\n\n")
        try:
            summary_table = summary.to_markdown(index=False)
        except (ImportError, ModuleNotFoundError):
            summary_table = summary.to_string(index=False)
        handle.write(summary_table + "\n\n")
        handle.write("## Interpretation\n\n")
        handle.write(
            "- `mean_rho` and `median_rho` summarize cross-method ranking agreement.\n"
            "- Positive values indicate aligned attribution rankings; values near "
            "zero indicate weak agreement.\n"
            "- `n_paired_examples` excludes examples with missing, mismatched, "
            "constant, or non-numeric word scores.\n"
            "- Correlation is word-level; token-level correlation is not reported "
            "because IG and SHAP tokenizations can differ between models/methods.\n\n"
        )
        handle.write(f"## Pairing diagnostics\n\n")
        handle.write(f"- Detail rows: {len(details):,}\n")
        handle.write(f"- Skipped model/field pairs: {len(skipped_total):,}\n")
        handle.write(f"- Detail CSV: `{details_path.name}`\n")
        handle.write(f"- Summary CSV: `{summary_path.name}`\n")
        handle.write(f"- Log: `{LOG_FILE.name}`\n")

    logger.info("Saved detail CSV -> %s", details_path)
    logger.info("Saved summary CSV -> %s", summary_path)
    logger.info("Saved Markdown report -> %s", report_path)
    logger.info("Completed: %d detail correlations", len(details))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Spearman correlation failed")
        sys.exit(1)
