"""
Phase 9 — Seed Stability Analysis + Final Tables + Figures
===========================================================
Reads all per-example faithfulness results (from all seeds) and produces:

  1. Seed stability report (mean ± std across seeds for main metrics)
  2. Final paper tables (Table A: classification, Table B: faithfulness,
     Table C: statistical significance)
  3. Figure 1: Δ SuffEff with 95% CI across budgets
  4. Figure 2: Fragmentation vs relative advantage scatter
  5. Runtime measurements table

Outputs:
  results/analysis/seed_stability_report.md
  results/analysis/final_tables.md
  results/analysis/figures/fig1_suффeff_diff.png
  results/analysis/figures/fig2_fragmentation.png
  results/analysis/analysis.log

Run:
  .venv\Scripts\python.exe -u analysis/seed_stability.py
"""

import os, sys, io, json, warnings, logging
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_DIR = BASE_DIR / "results" / "analysis"
FIGURES_DIR  = ANALYSIS_DIR / "figures"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ANALYSIS_DIR / "analysis.log"
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_OK = True
except ImportError:
    logger.warning("matplotlib not installed — figures will be skipped. pip install matplotlib")
    MATPLOTLIB_OK = False


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────

def load_faithfulness_all_seeds():
    """
    Loads faithfulness CSV for all seeds.
    Seed 42 is at results/faithfulness/per_example_faithfulness.csv.
    Seeds 43, 44 are at results/seeds/seed_{N}/faithfulness/per_example_faithfulness.csv.
    Returns combined DataFrame with 'seed' column.
    """
    frames = []

    # Seed 42
    path_42 = RESULTS_DIR / "faithfulness" / "per_example_faithfulness.csv"
    if path_42.exists():
        df42 = pd.read_csv(path_42)
        df42["seed"] = 42
        frames.append(df42)
        logger.info(f"Loaded seed 42: {len(df42):,} rows")
    else:
        logger.warning(f"Seed 42 faithfulness not found: {path_42}")

    # Seeds 43, 44
    for seed in [43, 44]:
        path = RESULTS_DIR / "seeds" / f"seed_{seed}" / "faithfulness" / "per_example_faithfulness.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["seed"] = seed
            frames.append(df)
            logger.info(f"Loaded seed {seed}: {len(df):,} rows")
        else:
            logger.info(f"Seed {seed} faithfulness not found (Phase 8 may not be done yet)")

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_bootstrap_results():
    path = RESULTS_DIR / "statistics" / "bootstrap_results.csv"
    if path.exists():
        logger.info(f"Loaded bootstrap results: {path}")
        return pd.read_csv(path)
    logger.warning(f"Bootstrap results not found: {path}")
    return None


def load_classification_robustness():
    path = RESULTS_DIR / "classification" / "seed_robustness.csv"
    if path.exists():
        return pd.read_csv(path)
    logger.warning(f"Seed robustness CSV not found: {path}")
    return None


def load_fragmentation():
    path = BASE_DIR / "outputs" / "fragmentation_results.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────
# Table generation
# ─────────────────────────────────────────────────────────────

def table_a_classification(robustness_df) -> str:
    """Table A: Classification robustness across seeds."""
    if robustness_df is None:
        return "*(seed_robustness.csv not found — run training/train_multiseed.py)*"

    cols = ["model", "seed", "best_epoch", "best_val_f1",
            "test_accuracy", "test_f1", "test_precision", "test_recall"]
    display = robustness_df[[c for c in cols if c in robustness_df.columns]].copy()
    for col in ["best_val_f1", "test_accuracy", "test_f1", "test_precision", "test_recall"]:
        if col in display.columns:
            display[col] = display[col].round(4)

    md_lines = ["## Table A — Classification Robustness\n"]
    try:
        md_lines.append(display.to_markdown(index=False))
    except Exception:
        md_lines.append(display.to_string(index=False))
    return "\n".join(md_lines)


def table_b_faithfulness(faith_df) -> str:
    """Table B: Main faithfulness results (seed 42, IG)."""
    if faith_df is None:
        return "*(per_example_faithfulness.csv not found — run evaluation/faithfulness_per_example.py)*"

    subset = faith_df[(faith_df["seed"] == 42) & (faith_df["attribution"] == "IG")]
    if subset.empty:
        return "*(No IG seed-42 rows in faithfulness data)*"

    agg = (subset
           .groupby(["model", "strategy", "budget"])
           [["comp", "suff", "coverage", "comp_eff", "suff_eff"]]
           .mean()
           .round(4)
           .reset_index())

    # Pivot to wide format for paper-style table
    tables = []
    for budget in sorted(agg["budget"].unique()):
        b_df = agg[agg["budget"] == budget][
            ["model", "strategy", "comp", "suff", "coverage", "comp_eff", "suff_eff"]
        ].copy()
        b_df.columns = ["Model", "Strategy", f"Comp@{budget}%", f"Suff@{budget}%",
                        f"Cov@{budget}%", f"CompEff@{budget}%", f"SuffEff@{budget}%"]
        tables.append(b_df)

    md_lines = ["## Table B — Faithfulness Results (Seed 42, IG)\n"]
    for t in tables:
        try:
            md_lines.append(t.to_markdown(index=False))
        except Exception:
            md_lines.append(t.to_string(index=False))
        md_lines.append("")

    return "\n".join(md_lines)


def table_c_statistics(boot_df) -> str:
    """Table C: Primary statistical comparisons."""
    if boot_df is None:
        return "*(bootstrap_results.csv not found — run statistics/bootstrap.py)*"

    primary = boot_df[
        (boot_df["is_primary"] == True) &
        (boot_df["attribution"] == "IG")
    ].copy()

    if primary.empty:
        return "*(No primary IG rows in bootstrap results)*"

    display_cols = ["model", "budget", "baseline", "n",
                    "observed_diff", "ci_low", "ci_high",
                    "p_value_perm", "p_holm", "holm_significant",
                    "effect_size_dz", "favorable"]
    display = primary[[c for c in display_cols if c in primary.columns]].copy()
    for col in ["observed_diff", "ci_low", "ci_high", "p_value_perm", "p_holm", "effect_size_dz"]:
        if col in display.columns:
            display[col] = display[col].round(4)

    md_lines = [
        "## Table C — Primary Statistical Comparisons (Hierarchical vs Baseline, SuffEff, IG)\n",
        "> Negative observed_diff favors Hierarchical (lower sufficiency = better faithfulness).\n",
    ]
    try:
        md_lines.append(display.to_markdown(index=False))
    except Exception:
        md_lines.append(display.to_string(index=False))

    return "\n".join(md_lines)


# ─────────────────────────────────────────────────────────────
# Seed stability analysis
# ─────────────────────────────────────────────────────────────

def seed_stability_report(faith_df) -> str:
    if faith_df is None:
        return "*(faithfulness data not available)*"

    seeds_present = sorted(faith_df["seed"].unique())
    logger.info(f"Seeds present in faithfulness data: {seeds_present}")

    # Main metric: SuffEff at 10% budget, IG, Hierarchical
    target = faith_df[
        (faith_df["attribution"] == "IG") &
        (faith_df["strategy"]    == "Hierarchical") &
        (faith_df["budget"]      == 10)
    ]

    lines = [
        "# Seed Stability Report\n",
        f"Seeds present: {seeds_present}\n",
        "## Primary Metric: SuffEff @10% — IG — Hierarchical\n",
    ]

    for model_name, grp in target.groupby("model"):
        seed_means = (grp
                      .groupby("seed")["suff_eff"]
                      .mean()
                      .round(4))
        vals = seed_means.values
        lines.append(f"### {model_name}\n")
        lines.append(seed_means.to_string())
        if len(vals) > 1:
            lines.append(f"\nMean: {np.mean(vals):.4f}  "
                         f"Std: {np.std(vals, ddof=1):.4f}  "
                         f"Range: [{np.min(vals):.4f}, {np.max(vals):.4f}]\n")
        lines.append("")

    # Compare Hierarchical vs Sum across seeds
    lines.append("## Hierarchical vs Sum — SuffEff @10% by Seed\n")
    for seed in seeds_present:
        s_df = faith_df[
            (faith_df["seed"]        == seed) &
            (faith_df["attribution"] == "IG") &
            (faith_df["budget"]      == 10) &
            (faith_df["strategy"].isin(["Hierarchical", "Sum"]))
        ]
        if s_df.empty: continue
        pivot = s_df.groupby(["model", "strategy"])["suff_eff"].mean().round(4).unstack()
        lines.append(f"**Seed {seed}:**\n")
        lines.append(pivot.to_string())
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────

def figure1_suффeff_diff(boot_df):
    """
    Figure 1: Δ SuffEff (H − Baseline) with 95% CI, by budget.
    Two panels: BanglaBERT, XLM-R.
    """
    if not MATPLOTLIB_OK or boot_df is None:
        return

    primary = boot_df[
        (boot_df["metric"]      == "suff_eff") &
        (boot_df["attribution"] == "IG") &
        (boot_df["is_primary"]  == True)
    ].copy()

    if primary.empty:
        logger.warning("No primary suff_eff rows for Figure 1")
        return

    models     = sorted(primary["model"].unique())
    baselines  = ["Sum", "Mean", "Max"]
    budgets    = sorted(primary["budget"].unique())
    colors     = {"Sum": "#E63946", "Mean": "#457B9D", "Max": "#2A9D8F"}
    markers    = {"Sum": "o", "Mean": "s", "Max": "^"}

    fig, axes = plt.subplots(1, len(models), figsize=(12, 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        for baseline in baselines:
            sub = primary[
                (primary["model"]    == model_name) &
                (primary["baseline"] == baseline)
            ].sort_values("budget")
            if sub.empty: continue

            x    = sub["budget"].values
            y    = sub["observed_diff"].values
            lo   = sub["ci_low"].values
            hi   = sub["ci_high"].values

            ax.errorbar(
                x, y,
                yerr=[y - lo, hi - y],
                fmt=f"{markers[baseline]}-",
                color=colors[baseline],
                label=f"H − {baseline}",
                capsize=4, linewidth=1.8, markersize=6,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Budget (%)", fontsize=11)
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b}%" for b in budgets])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("Δ SuffEff (Hierarchical − Baseline)", fontsize=11)
    axes[0].legend(fontsize=10)

    note = ("Negative Δ favors Hierarchical  |  "
            "Error bars = 95% paired bootstrap CI  |  "
            "Attribution: IG")
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, style="italic")
    fig.suptitle("Figure 1: Difference in Sufficiency Efficiency (H − Baseline)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = FIGURES_DIR / "fig1_suff_eff_diff.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 1 saved → {out}")


def figure2_fragmentation(frag_results, boot_df):
    """
    Figure 2: Fragmentation (subwords/word) vs relative SuffEff advantage.
    Only two points (BanglaBERT, XLM-R) — present descriptively, not as a trend line.
    """
    if not MATPLOTLIB_OK or boot_df is None or frag_results is None:
        return

    frag = {
        "BanglaBERT": frag_results.get("BanglaBERT_mean", 1.25),
        "XLM-R":      frag_results.get("XLM_R_mean",     1.91),
    }

    primary = boot_df[
        (boot_df["metric"]      == "suff_eff") &
        (boot_df["attribution"] == "IG") &
        (boot_df["baseline"]    == "Sum") &
        (boot_df["budget"]      == 10)
    ]

    model_colors = {"BanglaBERT": "#E63946", "XLM-R": "#457B9D"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in primary.iterrows():
        model_name = row["model"]
        if model_name not in frag: continue
        x    = frag[model_name]
        y    = row["observed_diff"]
        lo   = row["ci_low"]
        hi   = row["ci_high"]
        color = model_colors.get(model_name, "gray")

        ax.errorbar(
            x, y,
            yerr=[[y - lo], [hi - y]],
            fmt="o", color=color, markersize=10,
            capsize=5, linewidth=2, label=model_name,
        )
        ax.annotate(model_name, (x, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=10, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Mean subwords/word (fragmentation)", fontsize=11)
    ax.set_ylabel("Δ SuffEff H − Sum @ 10%  (negative = H wins)", fontsize=11)
    ax.set_title("Figure 2: Tokenizer Fragmentation vs Hierarchical Advantage",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.05,
            "Note: N=2 models — descriptive, not a general trend.",
            transform=ax.transAxes, fontsize=8, style="italic", alpha=0.6)
    fig.tight_layout()

    out = FIGURES_DIR / "fig2_fragmentation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 2 saved → {out}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 9 — Seed Stability Analysis + Final Tables + Figures")
    logger.info("=" * 60)

    faith_df   = load_faithfulness_all_seeds()
    boot_df    = load_bootstrap_results()
    robust_df  = load_classification_robustness()
    frag_data  = load_fragmentation()

    # ── Seed stability report ──────────────────────────────
    stability_text = seed_stability_report(faith_df)
    stability_path = ANALYSIS_DIR / "seed_stability_report.md"
    with open(stability_path, "w", encoding="utf-8") as f:
        f.write(stability_text)
    logger.info(f"Seed stability report → {stability_path}")

    # ── Final tables ───────────────────────────────────────
    table_lines = [
        "# Final Paper Tables\n",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        "---\n",
        table_a_classification(robust_df),
        "\n---\n",
        table_b_faithfulness(faith_df),
        "\n---\n",
        table_c_statistics(boot_df),
    ]

    tables_path = ANALYSIS_DIR / "final_tables.md"
    with open(tables_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))
    logger.info(f"Final tables → {tables_path}")

    # ── Figures ────────────────────────────────────────────
    figure1_suффeff_diff(boot_df)
    figure2_fragmentation(frag_data, boot_df)

    # ── Cross-seed faithfulness summary ────────────────────
    if faith_df is not None and len(faith_df["seed"].unique()) > 1:
        cross_seed = (faith_df[
            (faith_df["attribution"] == "IG") &
            (faith_df["budget"]      == 10)
        ]
        .groupby(["model", "strategy", "seed"])["suff_eff"]
        .mean()
        .round(4)
        .reset_index())

        cs_path = ANALYSIS_DIR / "cross_seed_suff_eff_10.csv"
        cross_seed.to_csv(cs_path, index=False, encoding="utf-8")
        logger.info(f"Cross-seed summary → {cs_path}")
        logger.info(f"\n{cross_seed.to_string(index=False)}")

    logger.info(f"\nOutputs in: {ANALYSIS_DIR}")
    logger.info("Phase 9 complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled error in seed_stability.py:")
        sys.exit(1)
