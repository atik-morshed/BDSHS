"""
Phase 7-9 — Bootstrap and Permutation Statistical Analysis
==========================================================
Implements paired bootstrap confidence intervals and paired permutation tests
for comparing hierarchical vs flat aggregation strategies.

Run:  python statistics/bootstrap_permutation.py
Outputs: ./results/statistics/
"""

import os, sys, io, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "results" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 7-9 — BOOTSTRAP & PERMUTATION STATISTICAL ANALYSIS")
print("=" * 60)

# Configuration from experiment config
BOOTSTRAP_CONFIG = {
    "n_resamples": 10000,
    "confidence_level": 0.95,
    "random_state": 42
}

PERMUTATION_CONFIG = {
    "n_permutations": 10000,
    "random_state": 42
}

class PairedBootstrap:
    """
    Paired bootstrap analysis for comparing two conditions on the same examples.
    """

    def __init__(self, n_resamples: int = 10000, confidence_level: float = 0.95, random_state: int = 42):
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def run_paired_bootstrap(self, condition_a: np.ndarray, condition_b: np.ndarray) -> Dict:
        """
        Run paired bootstrap on two conditions.

        Args:
            condition_a: Values for condition A (e.g., hierarchical)
            condition_b: Values for condition B (e.g., sum)

        Returns:
            Dictionary with bootstrap results including CI and statistics
        """
        if len(condition_a) != len(condition_b):
            raise ValueError(f"Condition lengths must match: {len(condition_a)} vs {len(condition_b)}")

        n = len(condition_a)
        observed_diff = np.mean(condition_a) - np.mean(condition_b)

        # Paired bootstrap: resample the differences
        differences = condition_a - condition_b
        bootstrap_diffs = np.zeros(self.n_resamples)

        for i in range(self.n_resamples):
            indices = self.rng.choice(n, size=n, replace=True)
            bootstrap_diffs[i] = np.mean(differences[indices])

        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        ci_low = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

        # Calculate standard error
        bootstrap_se = np.std(bootstrap_diffs, ddof=1)

        return {
            "observed_difference": float(observed_diff),
            "bootstrap_mean": float(np.mean(bootstrap_diffs)),
            "bootstrap_std": float(np.std(bootstrap_diffs, ddof=1)),
            "bootstrap_se": float(bootstrap_se),
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
            "ci_width": float(ci_high - ci_low),
            "n_resamples": self.n_resamples,
            "confidence_level": self.confidence_level
        }

    def calculate_effect_size(self, condition_a: np.ndarray, condition_b: np.ndarray) -> Dict:
        """
        Calculate paired effect size (Cohen's d_z).
        """
        differences = condition_a - condition_b
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        if std_diff == 0:
            return {"cohens_d_z": 0.0, "interpretation": "no variation"}

        cohens_d_z = mean_diff / std_diff

        # Interpret effect size
        abs_d = abs(cohens_d_z)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            "cohens_d_z": float(cohens_d_z),
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "interpretation": interpretation
        }


class PairedPermutationTest:
    """
    Paired permutation test for comparing two conditions.
    """

    def __init__(self, n_permutations: int = 10000, random_state: int = 42):
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def run_paired_permutation_test(self, condition_a: np.ndarray, condition_b: np.ndarray) -> Dict:
        """
        Run paired permutation test.

        Args:
            condition_a: Values for condition A
            condition_b: Values for condition B

        Returns:
            Dictionary with permutation test results
        """
        if len(condition_a) != len(condition_b):
            raise ValueError(f"Condition lengths must match: {len(condition_a)} vs {len(condition_b)}")

        n = len(condition_a)
        observed_diff = np.mean(condition_a) - np.mean(condition_b)
        differences = condition_a - condition_b

        # Permutation test: randomly flip signs of differences
        permuted_diffs = np.zeros(self.n_permutations)

        for i in range(self.n_permutations):
            signs = self.rng.choice([-1, 1], size=n)
            permuted_diffs[i] = np.mean(differences * signs)

        # Calculate p-value (two-tailed)
        # Count how many permuted differences are as extreme or more extreme than observed
        extreme_count = np.sum(np.abs(permuted_diffs) >= np.abs(observed_diff))
        p_value = (extreme_count + 1) / (self.n_permutations + 1)  # +1 for observed

        return {
            "observed_difference": float(observed_diff),
            "permutation_mean": float(np.mean(permuted_diffs)),
            "permutation_std": float(np.std(permuted_diffs, ddof=1)),
            "extreme_count": int(extreme_count),
            "p_value": float(p_value),
            "n_permutations": self.n_permutations,
            "significant": p_value < 0.05
        }


def run_statistical_analysis(hierarchical_values: np.ndarray, baseline_values: np.ndarray,
                            comparison_name: str, metric_name: str) -> Dict:
    """
    Run complete statistical analysis for a single comparison.

    Args:
        hierarchical_values: Values for hierarchical aggregation
        baseline_values: Values for baseline aggregation (sum/mean/max)
        comparison_name: Name of the comparison (e.g., "Hierarchical vs Sum")
        metric_name: Name of the metric (e.g., "Sufficiency-Efficiency@10")

    Returns:
        Dictionary with complete statistical results
    """
    print(f"\nAnalyzing: {comparison_name} - {metric_name}")

    # Convert to numpy arrays
    hierarchical_values = np.array(hierarchical_values)
    baseline_values = np.array(baseline_values)

    # Remove NaN values
    valid_mask = ~(np.isnan(hierarchical_values) | np.isnan(baseline_values))
    hierarchical_values = hierarchical_values[valid_mask]
    baseline_values = baseline_values[valid_mask]

    if len(hierarchical_values) == 0:
        print(f"[WARNING] No valid data for {comparison_name} - {metric_name}")
        return None

    print(f"  Valid examples: {len(hierarchical_values)}")

    # Basic statistics
    hierarchical_stats = {
        "mean": float(np.mean(hierarchical_values)),
        "std": float(np.std(hierarchical_values, ddof=1)),
        "median": float(np.median(hierarchical_values)),
        "n": len(hierarchical_values)
    }

    baseline_stats = {
        "mean": float(np.mean(baseline_values)),
        "std": float(np.std(baseline_values, ddof=1)),
        "median": float(np.median(baseline_values)),
        "n": len(baseline_values)
    }

    # Bootstrap analysis
    bootstrap = PairedBootstrap(**BOOTSTRAP_CONFIG)
    bootstrap_results = bootstrap.run_paired_bootstrap(hierarchical_values, baseline_values)

    # Effect size
    effect_size = bootstrap.calculate_effect_size(hierarchical_values, baseline_values)

    # Permutation test
    permutation = PairedPermutationTest(**PERMUTATION_CONFIG)
    permutation_results = permutation.run_paired_permutation_test(hierarchical_values, baseline_values)

    return {
        "comparison_name": comparison_name,
        "metric_name": metric_name,
        "hierarchical_stats": hierarchical_stats,
        "baseline_stats": baseline_stats,
        "bootstrap_results": bootstrap_results,
        "effect_size": effect_size,
        "permutation_results": permutation_results,
        "direction": "hierarchical_better" if bootstrap_results["observed_difference"] < 0 else "baseline_better"
        # Note: For sufficiency, lower is better, so negative difference means hierarchical is better
    }


def main():
    """
    Main function to run statistical analysis on faithfulness results.
    Loads per-example results from Phase 5 and performs bootstrap/permutation analysis.
    """
    print("\n" + "=" * 60)
    print("BOOTSTRAP & PERMUTATION STATISTICAL ANALYSIS")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Bootstrap resamples: {BOOTSTRAP_CONFIG['n_resamples']}")
    print(f"  Bootstrap confidence level: {BOOTSTRAP_CONFIG['confidence_level']}")
    print(f"  Permutation tests: {PERMUTATION_CONFIG['n_permutations']}")
    print(f"  Random state: {BOOTSTRAP_CONFIG['random_state']}")

    # Load per-example results from Phase 5
    csv_path = BASE_DIR / "results" / "evaluation" / "per_example" / "per_example_faithfulness_results.csv"
    
    if not csv_path.exists():
        print(f"\n[ERROR] Per-example results not found: {csv_path}")
        print("Please run Phase 4/5 first to generate per-example faithfulness results.")
        return

    print(f"\nLoading per-example results from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Define comparisons
    strategies = ['sum', 'mean', 'max']
    budgets = [10, 20, 30]
    metrics = ['comprehensiveness', 'sufficiency', 'comp_efficiency', 'suff_efficiency']

    all_results = []

    print("\n" + "=" * 60)
    print("RUNNING STATISTICAL ANALYSIS")
    print("=" * 60)

    for strategy in strategies:
        for budget in budgets:
            for metric in metrics:
                print(f"\n{'─'*60}")
                print(f"Comparing: Hierarchical vs {strategy}")
                print(f"Budget: {budget}%, Metric: {metric}")
                print(f"{'─'*60}")

                # Extract hierarchical and baseline values
                hierarchical_mask = (df['strategy'] == 'hierarchical') & (df['budget_pct'] == budget)
                baseline_mask = (df['strategy'] == strategy) & (df['budget_pct'] == budget)

                hierarchical_values = df[hierarchical_mask][metric].values
                baseline_values = df[baseline_mask][metric].values

                # Align by example_id
                hierarchical_df = df[hierarchical_mask][['example_id', metric]].set_index('example_id')
                baseline_df = df[baseline_mask][['example_id', metric]].set_index('example_id')

                # Get common example IDs
                common_ids = hierarchical_df.index.intersection(baseline_df.index)
                
                if len(common_ids) == 0:
                    print(f"[WARNING] No common examples for {strategy} @ {budget}%")
                    continue

                hierarchical_aligned = hierarchical_df.loc[common_ids][metric].values
                baseline_aligned = baseline_df.loc[common_ids][metric].values

                # Run comparison
                result = run_statistical_analysis(
                    hierarchical_aligned,
                    baseline_aligned,
                    f"hierarchical_vs_{strategy}",
                    metric
                )

                if result:
                    result['budget'] = budget
                    result['baseline_strategy'] = strategy
                    all_results.append(result)

    # Save results (convert numpy types to Python types for JSON serialization)
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(all_results)
    results_path = OUT_DIR / "statistical_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Statistical results saved to: {results_path}")
    print(f"Total comparisons: {len(all_results)}")
    print(f"{'='*60}")

    # Generate summary table
    generate_summary_table(all_results, OUT_DIR)

    # Save configuration
    config_path = OUT_DIR / "statistical_config.json"
    config = {
        "bootstrap": BOOTSTRAP_CONFIG,
        "permutation": PERMUTATION_CONFIG,
        "description": "Statistical analysis configuration for Phase 7-9"
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfiguration saved to: {config_path}")

    print("\n" + "=" * 60)
    print("Statistical analysis complete.")
    print("=" * 60)


def generate_summary_table(results, output_dir):
    """
    Generate a summary table of statistical results.
    """
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY TABLE")
    print("=" * 60)

    summary_data = []
    for r in results:
        bootstrap = r['bootstrap_results']
        permutation = r['permutation_results']
        effect = r['effect_size']
        
        summary_data.append({
            'Budget': f"{r['budget']}%",
            'Baseline': r['baseline_strategy'],
            'Metric': r['metric_name'],
            'Δ (H-B)': f"{bootstrap['observed_difference']:.4f}",
            '95% CI': f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]",
            'Permutation p': f"{permutation['p_value']:.4f}",
            'Effect Size': f"{effect.get('cohens_dz', effect.get('effect_size', 0)):.4f}",
            'Direction': r['direction']
        })

    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = output_dir / "statistical_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")

    # Save as simple text table (no tabulate dependency)
    md_path = output_dir / "statistical_summary.txt"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Statistical Analysis Summary\n\n")
        f.write(summary_df.to_string(index=False))
    print(f"Summary text saved to: {md_path}")


if __name__ == "__main__":
    main()
