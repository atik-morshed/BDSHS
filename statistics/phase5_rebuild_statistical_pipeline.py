"""
Phase 5 — Rebuild Statistical Pipeline with Corrected Effect Sizes
=========================================================================

This script rebuilds the complete statistical analysis pipeline incorporating:
- Corrected effect size calculation (from Phase 1)
- Verified data integrity (from Phase 2)
- Verified attribution alignment (from Phase 4)

Pipeline:
per_example.csv → statistical_analysis → corrected_results.json → master_table.csv

Run: python statistics/phase5_rebuild_statistical_pipeline.py
Outputs: results/statistics/rebuilt_statistical_results.json
         results/statistics/master_statistics_table.csv
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

LOG_FILE = OUT_DIR / "phase5_rebuild_pipeline_log.txt"
log_fh = open(LOG_FILE, "w", encoding="utf-8")

def log(message):
    """Write to both stdout and log file."""
    print(message)
    log_fh.write(message + "\n")
    log_fh.flush()

print("=" * 80)
log("=" * 80)
log("PHASE 5 — REBUILD STATISTICAL PIPELINE WITH CORRECTED EFFECT SIZES")
log("=" * 80)

# Configuration
BOOTSTRAP_CONFIG = {
    "n_resamples": 10000,
    "confidence_level": 0.95,
    "random_state": 42
}

PERMUTATION_CONFIG = {
    "n_permutations": 10000,
    "random_state": 42
}

log("\nConfiguration:")
log(f"  Bootstrap resamples: {BOOTSTRAP_CONFIG['n_resamples']}")
log(f"  Bootstrap confidence level: {BOOTSTRAP_CONFIG['confidence_level']}")
log(f"  Permutation tests: {PERMUTATION_CONFIG['n_permutations']}")
log(f"  Random state: {BOOTSTRAP_CONFIG['random_state']}")

# Load verified per-example results
csv_path = BASE_DIR / "results" / "evaluation" / "per_example" / "per_example_faithfulness_results.csv"

if not csv_path.exists():
    log(f"\n[ERROR] Per-example results not found: {csv_path}")
    sys.exit(1)

log(f"\nLoading verified per-example results from: {csv_path}")
df = pd.read_csv(csv_path)
log(f"Loaded {len(df)} rows")

# Define comparisons
strategies = ['sum', 'mean', 'max']
budgets = [10, 20, 30]
metrics = ['comprehensiveness', 'sufficiency', 'comp_efficiency', 'suff_efficiency']

log("\n" + "=" * 80)
log("REBUILDING STATISTICAL ANALYSIS WITH CORRECTED EFFECT SIZES")
log("=" * 80)

class PairedBootstrap:
    """Paired bootstrap analysis for comparing two conditions on the same examples."""
    
    def __init__(self, n_resamples: int = 10000, confidence_level: float = 0.95, random_state: int = 42):
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def run_paired_bootstrap(self, hierarchical_values: np.ndarray, baseline_values: np.ndarray) -> Dict:
        """Run paired bootstrap analysis."""
        n = len(hierarchical_values)
        differences = hierarchical_values - baseline_values
        observed_diff = np.mean(differences)
        
        # Paired bootstrap
        bootstrap_diffs = np.zeros(self.n_resamples)
        for i in range(self.n_resamples):
            indices = self.rng.choice(n, size=n, replace=True)
            bootstrap_diffs[i] = np.mean(differences[indices])
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_low = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        return {
            "observed_difference": float(observed_diff),
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
            "bootstrap_diffs": bootstrap_diffs
        }
    
    def calculate_effect_size(self, hierarchical_values: np.ndarray, baseline_values: np.ndarray) -> Dict:
        """Calculate Cohen's d_z for paired observations."""
        differences = hierarchical_values - baseline_values
        mean_diff = np.mean(differences)
        sd_diff = np.std(differences, ddof=1)
        
        if sd_diff == 0 or not np.isfinite(sd_diff):
            cohens_dz = 0.0
        else:
            cohens_dz = mean_diff / sd_diff
        
        return {
            "cohens_dz": float(cohens_dz),
            "mean_difference": float(mean_diff),
            "std_difference": float(sd_diff)
        }

class PairedPermutationTest:
    """Paired permutation test using sign-flipping."""
    
    def __init__(self, n_permutations: int = 10000, random_state: int = 42):
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def run_paired_permutation_test(self, hierarchical_values: np.ndarray, baseline_values: np.ndarray) -> Dict:
        """Run paired permutation test using sign-flipping."""
        differences = hierarchical_values - baseline_values
        observed_diff = np.mean(differences)
        
        # Paired sign-flipping permutation
        permuted_stats = np.zeros(self.n_permutations)
        for i in range(self.n_permutations):
            signs = self.rng.choice([-1, 1], size=len(differences))
            permuted = differences * signs
            permuted_stats[i] = np.mean(permuted)
        
        # Calculate p-value with +1 correction for finite Monte Carlo
        p_value = (np.sum(np.abs(permuted_stats) >= np.abs(observed_diff)) + 1) / (self.n_permutations + 1)
        
        return {
            "observed_difference": float(observed_diff),
            "p_value": float(p_value),
            "permuted_stats": permuted_stats
        }

def run_statistical_analysis(hierarchical_values: np.ndarray, baseline_values: np.ndarray,
                            comparison_name: str, metric_name: str) -> Dict:
    """Run complete statistical analysis for one comparison."""
    log(f"\nAnalyzing: {comparison_name} - {metric_name}")
    
    # Convert to numpy arrays
    hierarchical_values = np.array(hierarchical_values)
    baseline_values = np.array(baseline_values)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(hierarchical_values) | np.isnan(baseline_values))
    hierarchical_values = hierarchical_values[valid_mask]
    baseline_values = baseline_values[valid_mask]
    
    if len(hierarchical_values) == 0:
        log(f"[WARNING] No valid data for {comparison_name} - {metric_name}")
        return None
    
    log(f"  Valid examples: {len(hierarchical_values)}")
    
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
    }

all_results = []

for strategy in strategies:
    for budget in budgets:
        for metric in metrics:
            log(f"\n{'─'*80}")
            log(f"Comparing: Hierarchical vs {strategy}")
            log(f"Budget: {budget}%, Metric: {metric}")
            log(f"{'─'*80}")
            
            # Extract hierarchical and baseline values
            hierarchical_mask = (df['strategy'] == 'hierarchical') & (df['budget_pct'] == budget)
            baseline_mask = (df['strategy'] == strategy) & (df['budget_pct'] == budget)
            
            hierarchical_df = df[hierarchical_mask][['example_id', metric]].set_index('example_id')
            baseline_df = df[baseline_mask][['example_id', metric]].set_index('example_id')
            
            # Get common example IDs
            common_ids = hierarchical_df.index.intersection(baseline_df.index)
            
            if len(common_ids) == 0:
                log(f"[WARNING] No common examples for {strategy} @ {budget}%")
                continue
            
            hierarchical_aligned = hierarchical_df.loc[list(common_ids)][metric].values
            baseline_aligned = baseline_df.loc[list(common_ids)][metric].values
            
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

# Save results
def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
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
results_path = OUT_DIR / "rebuilt_statistical_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(serializable_results, f, indent=2)

log(f"\n{'='*80}")
log(f"Rebuilt statistical results saved to: {results_path}")
log(f"Total comparisons: {len(all_results)}")
log(f"{'='*80}")

# Generate master statistics table
log("\n" + "=" * 80)
log("GENERATING MASTER STATISTICS TABLE")
log("=" * 80)

master_data = []
for r in all_results:
    bootstrap = r['bootstrap_results']
    permutation = r['permutation_results']
    effect = r['effect_size']
    
    master_data.append({
        'metric': r['metric_name'],
        'budget': r['budget'],
        'baseline_strategy': r['baseline_strategy'],
        'n_examples': r['hierarchical_stats']['n'],
        'mean_hierarchical': r['hierarchical_stats']['mean'],
        'mean_baseline': r['baseline_stats']['mean'],
        'delta': bootstrap['observed_difference'],
        'ci_low': bootstrap['ci_lower'],
        'ci_high': bootstrap['ci_upper'],
        'p_value': permutation['p_value'],
        'cohens_dz': effect['cohens_dz'],
        'direction': r['direction']
    })

master_df = pd.DataFrame(master_data)

# Save master table
csv_path = OUT_DIR / "master_statistics_table.csv"
master_df.to_csv(csv_path, index=False)
log(f"Master statistics table saved to: {csv_path}")

# Save as text
text_path = OUT_DIR / "master_statistics_table.txt"
with open(text_path, "w", encoding="utf-8") as f:
    f.write("# Master Statistics Table\n\n")
    f.write(master_df.to_string(index=False))
log(f"Master statistics text saved to: {text_path}")

# Save configuration
config_path = OUT_DIR / "rebuild_pipeline_config.json"
config = {
    "bootstrap": BOOTSTRAP_CONFIG,
    "permutation": PERMUTATION_CONFIG,
    "description": "Rebuilt statistical pipeline with corrected effect sizes (Phase 5)"
}
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

log(f"\nConfiguration saved to: {config_path}")

log_fh.close()

print("\n" + "=" * 80)
print("Phase 5 complete. Statistical pipeline rebuilt with corrected effect sizes.")
print("=" * 80)
