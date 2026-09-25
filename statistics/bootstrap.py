"""
Phase 3 + 4 + 5 — Bootstrap CIs, Permutation Tests, and Effect Sizes
=======================================================================
Reads per_example_faithfulness.csv and computes:

  Phase 3: Paired bootstrap 95% CI (10,000 resamples)
  Phase 4: Paired permutation test (10,000 permutations)
  Phase 5: Standardized paired effect size dz = mean(D) / std(D)

Primary comparisons (18):
  Hierarchical vs {Sum, Mean, Max}
  × {BanglaBERT, XLM-R}
  × {10%, 20%, 30%}
  for metric: suff_eff

All 72 comparisons (4 metrics × 18) also computed and saved.

Outputs:
  results/statistics/bootstrap_results.csv
  results/statistics/primary_comparisons.csv
  results/statistics/statistics_run.log
  results/statistics/bootstrap_distribution_plots.png  (optional)

Run:
  .venv\Scripts\python.exe -u statistics/bootstrap.py

Important note on sufficiency direction:
  Suff = P(y|T) − P(y|T_top_k)
  Lower suff = better (retained units preserve the original prediction).
  D = H − Baseline being NEGATIVE means hierarchical wins.
  The 'favorable' column reflects this.
"""

import os, sys, io, json, logging, warnings, time
from pathlib import Path
from itertools import product

os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent.parent
FAITH_CSV   = BASE_DIR / "results" / "faithfulness" / "per_example_faithfulness.csv"
STATS_DIR   = BASE_DIR / "results" / "statistics"
STATS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = STATS_DIR / "statistics_run.log"
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

N_BOOTSTRAP   = 10_000
N_PERMUTATION = 10_000
CONFIDENCE    = 0.95
ALPHA         = 1 - CONFIDENCE
RNG_SEED      = 42

rng = np.random.default_rng(RNG_SEED)

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
logger.info(f"Loading faithfulness data from {FAITH_CSV}")
if not FAITH_CSV.exists():
    logger.error(
        f"File not found: {FAITH_CSV}\n"
        "Run evaluation/faithfulness_per_example.py first."
    )
    sys.exit(1)

df = pd.read_csv(FAITH_CSV)
logger.info(f"Loaded {len(df):,} rows")
logger.info(f"Columns: {df.columns.tolist()}")
logger.info(f"Models: {df['model'].unique().tolist()}")
logger.info(f"Attributions: {df['attribution'].unique().tolist()}")
logger.info(f"Strategies: {df['strategy'].unique().tolist()}")
logger.info(f"Budgets: {sorted(df['budget'].unique().tolist())}")


# ─────────────────────────────────────────────────────────────
# Core statistical functions
# ─────────────────────────────────────────────────────────────

def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray,
                         n_resamples: int = N_BOOTSTRAP,
                         confidence: float = CONFIDENCE) -> dict:
    """
    Compute paired bootstrap CI for mean(a) - mean(b).

    Parameters
    ----------
    a, b : matched arrays of per-example values (same length, same ordering)
    n_resamples : number of bootstrap resamples
    confidence  : e.g. 0.95 for 95% CI

    Returns
    -------
    dict with keys: observed_diff, ci_low, ci_high, se, significant
    """
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    n = len(a)
    observed_diff = float(np.mean(a) - np.mean(b))

    boot_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx         = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(a[idx]) - np.mean(b[idx])

    alpha_half = (1 - confidence) / 2
    ci_low  = float(np.percentile(boot_diffs, 100 * alpha_half))
    ci_high = float(np.percentile(boot_diffs, 100 * (1 - alpha_half)))
    se      = float(np.std(boot_diffs, ddof=1))

    # Significant if CI excludes 0
    significant = not (ci_low <= 0 <= ci_high)

    return {
        "observed_diff": observed_diff,
        "ci_low":        ci_low,
        "ci_high":       ci_high,
        "se":            se,
        "significant":   significant,
        "n":             n,
    }


def paired_permutation_test(a: np.ndarray, b: np.ndarray,
                             n_permutations: int = N_PERMUTATION) -> dict:
    """
    Paired sign-flip permutation test for H0: mean(a - b) = 0.

    Returns p-value (two-tailed).
    """
    assert len(a) == len(b)
    n = len(a)
    D = a - b
    observed = float(np.mean(D))

    count_extreme = 0
    for _ in range(n_permutations):
        signs   = rng.choice([-1.0, 1.0], size=n)
        perm_D  = D * signs
        if abs(np.mean(perm_D)) >= abs(observed):
            count_extreme += 1

    p_value = count_extreme / n_permutations
    return {
        "observed_mean_diff": observed,
        "p_value":            p_value,
        "n_permutations":     n_permutations,
    }


def effect_size_dz(a: np.ndarray, b: np.ndarray) -> float:
    """
    Standardized paired effect size: dz = mean(D) / std(D)
    where D = a - b.
    """
    D  = a - b
    sd = np.std(D, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(D) / sd)


def holm_correction(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni correction. Returns corrected p-values in the same order.
    """
    m     = len(p_values)
    order = np.argsort(p_values)
    corrected = np.array(p_values, dtype=float)

    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, p_values[idx] * (m - rank))

    # Enforce monotonicity: corrected[i] >= corrected[i-1]
    sorted_corrected = corrected[order]
    for i in range(1, m):
        if sorted_corrected[i] < sorted_corrected[i - 1]:
            sorted_corrected[i] = sorted_corrected[i - 1]

    # Map back
    result = np.empty(m)
    for rank, idx in enumerate(order):
        result[idx] = sorted_corrected[rank]
    return result.tolist()


# ─────────────────────────────────────────────────────────────
# Build per-example pivot tables for fast lookup
# ─────────────────────────────────────────────────────────────

def get_aligned_arrays(df, model, attribution, budget, strategy_a, strategy_b, metric):
    """
    Returns (a, b) numpy arrays of per-example metric values for the two strategies.
    Only examples present in BOTH strategies are included.
    """
    def get_vals(strategy):
        mask = (
            (df["model"]       == model)      &
            (df["attribution"] == attribution) &
            (df["budget"]      == budget)      &
            (df["strategy"]    == strategy)
        )
        sub = df[mask][["example_id", metric]].dropna(subset=[metric])
        return sub.set_index("example_id")[metric]

    s_a = get_vals(strategy_a)
    s_b = get_vals(strategy_b)

    common = s_a.index.intersection(s_b.index)
    if len(common) == 0:
        return None, None

    return s_a.loc[common].values, s_b.loc[common].values


# ─────────────────────────────────────────────────────────────
# Run all comparisons
# ─────────────────────────────────────────────────────────────

MODELS       = sorted(df["model"].unique().tolist())
ATTRIBUTIONS = sorted(df["attribution"].unique().tolist())
BUDGETS      = sorted(df["budget"].unique().tolist())
BASELINES    = ["Sum", "Mean", "Max"]
METRICS      = ["comp", "suff", "comp_eff", "suff_eff"]
PRIMARY_METRIC = "suff_eff"

all_rows = []

total_combs = (len(MODELS) * len(ATTRIBUTIONS) * len(BUDGETS)
               * len(BASELINES) * len(METRICS))
logger.info(f"\nRunning {total_combs:,} pairwise comparisons …")
t0 = time.time()
done = 0

for model, attribution, budget, baseline, metric in product(
        MODELS, ATTRIBUTIONS, BUDGETS, BASELINES, METRICS):

    h_vals, b_vals = get_aligned_arrays(
        df, model, attribution, budget, "Hierarchical", baseline, metric
    )

    if h_vals is None or len(h_vals) < 30:
        logger.debug(f"  Skipping {model}/{attribution}/{budget}/{baseline}/{metric} "
                     f"(n={0 if h_vals is None else len(h_vals)})")
        done += 1
        continue

    # Bootstrap
    boot = paired_bootstrap_ci(h_vals, b_vals)

    # Permutation
    perm = paired_permutation_test(h_vals, b_vals)

    # Effect size
    dz = effect_size_dz(h_vals, b_vals)

    # Is this favorable? For suff/suff_eff, negative diff = hierarchical wins.
    # For comp/comp_eff, positive diff = hierarchical wins.
    if metric in ("suff", "suff_eff"):
        favorable = boot["observed_diff"] < 0
    else:
        favorable = boot["observed_diff"] > 0

    is_primary = (metric == PRIMARY_METRIC)

    all_rows.append({
        "model":          model,
        "attribution":    attribution,
        "budget":         budget,
        "baseline":       baseline,
        "metric":         metric,
        "is_primary":     is_primary,
        "n":              boot["n"],
        "observed_diff":  boot["observed_diff"],
        "ci_low":         boot["ci_low"],
        "ci_high":        boot["ci_high"],
        "se":             boot["se"],
        "boot_significant": boot["significant"],
        "p_value_perm":   perm["p_value"],
        "effect_size_dz": dz,
        "favorable":      favorable,
    })

    done += 1
    if done % 20 == 0:
        logger.info(f"  Progress: {done}/{total_combs}  "
                    f"rows so far: {len(all_rows)}  "
                    f"elapsed: {(time.time()-t0)/60:.1f} min")

logger.info(f"Comparisons done in {(time.time()-t0)/60:.1f} min")

# ─────────────────────────────────────────────────────────────
# Holm correction on primary comparisons
# ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_rows)

primary_mask  = results_df["is_primary"] == True
primary_df    = results_df[primary_mask].copy()
primary_pvals = primary_df["p_value_perm"].tolist()

if primary_pvals:
    corrected = holm_correction(primary_pvals)
    primary_df = primary_df.copy()
    primary_df["p_holm"] = corrected
    primary_df["holm_significant"] = primary_df["p_holm"] < ALPHA
else:
    primary_df["p_holm"]         = np.nan
    primary_df["holm_significant"] = False

# Merge corrected p-values back
results_df = results_df.merge(
    primary_df[["model", "attribution", "budget", "baseline", "metric",
                "p_holm", "holm_significant"]],
    on=["model", "attribution", "budget", "baseline", "metric"],
    how="left",
)

# ─────────────────────────────────────────────────────────────
# Save full results
# ─────────────────────────────────────────────────────────────
full_path = STATS_DIR / "bootstrap_results.csv"
results_df.to_csv(full_path, index=False, encoding="utf-8")
logger.info(f"\nFull results CSV → {full_path}  ({len(results_df):,} rows)")

# Save primary comparisons separately
primary_path = STATS_DIR / "primary_comparisons.csv"
primary_df.sort_values(["model", "budget", "baseline"]).to_csv(
    primary_path, index=False, encoding="utf-8"
)
logger.info(f"Primary comparisons CSV → {primary_path}  ({len(primary_df):,} rows)")

# ─────────────────────────────────────────────────────────────
# Human-readable report
# ─────────────────────────────────────────────────────────────
report_lines = [
    "# BD-SHS Statistical Analysis Report",
    f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    f"\nConfig: n_bootstrap={N_BOOTSTRAP}, n_permutations={N_PERMUTATION}, "
    f"confidence={CONFIDENCE}, seed={RNG_SEED}",
    "\n---\n",
    "## Primary Comparisons (Hierarchical vs Baseline — SuffEff)\n",
    "> **Note**: For sufficiency metrics, NEGATIVE observed_diff favors hierarchical.",
    "> (Lower sufficiency = retained units better preserve model prediction.)\n",
]

display_cols = [
    "model", "attribution", "budget", "baseline",
    "n", "observed_diff", "ci_low", "ci_high",
    "p_value_perm", "p_holm", "holm_significant",
    "effect_size_dz", "favorable",
]
primary_display = primary_df[
    [c for c in display_cols if c in primary_df.columns]
].round(4)

try:
    report_lines.append(primary_display.to_markdown(index=False))
except Exception:
    report_lines.append(primary_display.to_string(index=False))

# Secondary: all metrics for IG
report_lines += [
    "\n\n---\n",
    "## All Comparisons (IG, both models)\n",
]
secondary_display = results_df[
    (results_df["attribution"] == "IG")
][display_cols].round(4)
try:
    report_lines.append(secondary_display.to_markdown(index=False))
except Exception:
    report_lines.append(secondary_display.to_string(index=False))

# Summary: how many primary comparisons favor hierarchical and are significant?
report_lines += [
    "\n\n---\n",
    "## Summary",
    f"\nTotal primary comparisons: {len(primary_df)}",
]
if not primary_df.empty:
    n_favor = int(primary_df["favorable"].sum())
    n_sig_perm = int((primary_df["p_value_perm"] < ALPHA).sum())
    n_sig_holm = int(primary_df.get("holm_significant", pd.Series(dtype=bool)).sum())
    report_lines += [
        f"Favorable to hierarchical: {n_favor}/{len(primary_df)}",
        f"Significant (raw permutation p < {ALPHA}): {n_sig_perm}/{len(primary_df)}",
        f"Significant (Holm-corrected p < {ALPHA}): {n_sig_holm}/{len(primary_df)}",
    ]

report_path = STATS_DIR / "statistics_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
logger.info(f"Report saved → {report_path}")

# ─────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 60)
print(f"Full results : {full_path}")
print(f"Primary CIs  : {primary_path}")
print(f"Report       : {report_path}")
if not primary_df.empty:
    print(f"\nPrimary comparisons ({len(primary_df)}):")
    print(f"  Favorable to Hierarchical: {n_favor}/{len(primary_df)}")
    print(f"  Significant (raw perm p<{ALPHA}): {n_sig_perm}/{len(primary_df)}")
    print(f"  Significant (Holm): {n_sig_holm}/{len(primary_df)}")

    print(f"\nTop-level results (IG / 10% budget):")
    sub = primary_df[
        (primary_df["attribution"] == "IG") &
        (primary_df["budget"]      == 10)
    ]
    if not sub.empty:
        for _, row in sub.iterrows():
            direction = "✓" if row.get("favorable") else "✗"
            print(f"  {row['model']:12s} vs {row['baseline']:6s}: "
                  f"Δ={row['observed_diff']:+.4f}  "
                  f"95%CI=[{row['ci_low']:.4f},{row['ci_high']:.4f}]  "
                  f"dz={row['effect_size_dz']:.3f}  {direction}")
