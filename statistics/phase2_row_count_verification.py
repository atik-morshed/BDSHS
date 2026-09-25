"""
Phase 2 — Fix Row Count Discrepancy
======================================

Issue: Report says 60,348 rows but CSV shows 60,349. 
Expected: 5,029 × 4 × 3 = 60,348.

Investigation:
- Check if extra row is header
- Verify expected combinations (15087 per strategy)
- Check for duplicates or missing examples
- Verify each strategy-budget combination has exactly 5,029 examples

Run: python statistics/phase2_row_count_verification.py
Outputs: results/evaluation/per_example/row_count_verification.md
"""

import os, sys, io
from pathlib import Path
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "evaluation" / "per_example" / "per_example_faithfulness_results.csv"
OUTPUT_DIR = BASE_DIR / "results" / "evaluation" / "per_example"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "phase2_row_count_log.txt"
log_fh = open(LOG_FILE, "w", encoding="utf-8")

def log(message):
    """Write to both stdout and log file."""
    print(message)
    log_fh.write(message + "\n")
    log_fh.flush()

print("=" * 80)
log("=" * 80)
log("PHASE 2 — ROW COUNT DISCREPANCY INVESTIGATION")
log("=" * 80)

# Expected calculation
expected_total = 5029 * 4 * 3
log(f"\nExpected total rows: {expected_total} (5,029 × 4 strategies × 3 budgets)")

# Load CSV
log(f"\nLoading CSV from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
log(f"DataFrame shape: {df.shape}")
log(f"Data rows (pandas): {len(df)}")

# Check physical line count
log(f"\nChecking physical line count in file...")
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    physical_lines = sum(1 for _ in f)
log(f"Physical lines in file: {physical_lines}")

if physical_lines == len(df) + 1:
    log("✓ Extra row is the CSV header (expected)")
elif physical_lines == len(df):
    log("✓ No header line (unusual but possible)")
else:
    log(f"⚠ Unexpected line count: {physical_lines} vs {len(df)} data rows")

# Verify strategy counts
log(f"\n" + "=" * 80)
log("STRATEGY COUNT VERIFICATION")
log("=" * 80)

strategy_counts = df['strategy'].value_counts().sort_index()
log(f"\nStrategy counts:")
for strategy, count in strategy_counts.items():
    expected = 5029 * 3  # 3 budgets
    status = "✓" if count == expected else "✗"
    log(f"  {status} {strategy:12s}: {count:6d} (expected: {expected})")

# Verify budget counts
log(f"\n" + "=" * 80)
log("BUDGET COUNT VERIFICATION")
log("=" * 80)

budget_counts = df['budget_pct'].value_counts().sort_index()
log(f"\nBudget counts:")
for budget, count in budget_counts.items():
    expected = 5029 * 4  # 4 strategies
    status = "✓" if count == expected else "✗"
    log(f"  {status} Budget {budget:3d}%: {count:6d} (expected: {expected})")

# Verify strategy-budget combinations
log(f"\n" + "=" * 80)
log("STRATEGY-BUDGET COMBINATION VERIFICATION")
log("=" * 80)

combination_counts = df.groupby(['strategy', 'budget_pct']).size()
log(f"\nEach strategy-budget combination should have exactly 5,029 examples:")
log(f"Total combinations: {len(combination_counts)}")

all_correct = True
for (strategy, budget), count in combination_counts.items():
    status = "✓" if count == 5029 else "✗"
    if count != 5029:
        all_correct = False
    log(f"  {status} {strategy:12s} @ {budget:3d}%: {count:6d} (expected: 5029)")

# Check for duplicate example IDs
log(f"\n" + "=" * 80)
log("DUPLICATE EXAMPLE ID CHECK")
log("=" * 80)

total_ids = len(df)
unique_ids = df['example_id'].nunique()
log(f"\nTotal example_id entries: {total_ids}")
log(f"Unique example_ids: {unique_ids}")
log(f"Duplicates: {total_ids - unique_ids}")

if total_ids != unique_ids:
    log("⚠ WARNING: Duplicate example_ids detected!")
    log("This is EXPECTED because each example appears under multiple conditions")
    log(f"Expected: Each of {unique_ids} examples appears 12 times (4 strategies × 3 budgets)")
    log(f"Expected total: {unique_ids * 12}")
    if total_ids == unique_ids * 12:
        log("✓ Duplicate count matches expected structure")
    else:
        log(f"✗ Unexpected duplicate count: {total_ids} vs {unique_ids * 12}")
else:
    log("✓ No duplicate example_ids")

# Check for missing example IDs
log(f"\n" + "=" * 80)
log("MISSING EXAMPLE ID CHECK")
log("=" * 80)

expected_ids = set(range(5029))
actual_ids = set(df['example_id'].unique())
missing_ids = expected_ids - actual_ids
extra_ids = actual_ids - expected_ids

log(f"\nExpected ID range: 0 to 5028")
log(f"Missing IDs: {len(missing_ids)}")
log(f"Extra IDs: {len(extra_ids)}")

if missing_ids:
    log(f"Sample missing IDs: {sorted(list(missing_ids))[:10]}")
if extra_ids:
    log(f"Sample extra IDs: {sorted(list(extra_ids))[:10]}")

if not missing_ids and not extra_ids:
    log("✓ All example IDs in expected range")

# Summary
log(f"\n" + "=" * 80)
log("SUMMARY")
log("=" * 80)

log(f"\nMathematical expectation: {expected_total} rows")
log(f"Pandas DataFrame rows: {len(df)}")
log(f"Physical file lines: {physical_lines}")

if physical_lines == len(df) + 1:
    log(f"\n✓ DISCREPANCY RESOLVED: Extra row is CSV header")
    log(f"  Data rows: {len(df)}")
    log(f"  Header: 1")
    log(f"  Total: {len(df) + 1}")
elif len(df) == expected_total:
    log(f"\n✓ Data rows match expectation: {len(df)}")
else:
    log(f"\n⚠ UNEXPECTED: Data rows {len(df)} != expected {expected_total}")

if all_correct and total_ids == unique_ids and not missing_ids and not extra_ids:
    log(f"\n✓ ALL VERIFICATIONS PASSED")
else:
    log(f"\n⚠ SOME VERIFICATIONS FAILED")

log_fh.close()

# Generate markdown report
report_path = OUTPUT_DIR / "row_count_verification.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Phase 2 Report: Row Count Discrepancy\n\n")
    f.write("**Status**: ✅ COMPLETE\n\n")
    f.write("## Investigation Results\n\n")
    f.write(f"**Mathematical expectation**: {expected_total} rows (5,029 × 4 × 3)\n\n")
    f.write(f"**Pandas DataFrame rows**: {len(df)}\n\n")
    f.write(f"**Physical file lines**: {physical_lines}\n\n")
    f.write("## Resolution\n\n")
    if physical_lines == len(df) + 1:
        f.write("✓ **Discrepancy resolved**: The extra row is the CSV header.\n\n")
        f.write(f"- Data rows: {len(df)}\n")
        f.write(f"- Header: 1\n")
        f.write(f"- Total physical lines: {len(df) + 1}\n\n")
    f.write("## Verifications\n\n")
    f.write(f"- Strategy counts: {'✓ PASS' if all(c == 5029*3 for c in strategy_counts.values.tolist()) else '✗ FAIL'}\n")
    f.write(f"- Budget counts: {'✓ PASS' if all(c == 5029*4 for c in budget_counts.values.tolist()) else '✗ FAIL'}\n")
    f.write(f"- Strategy-budget combinations: {'✓ PASS' if all_correct else '✗ FAIL'}\n")
    expected_duplicates = unique_ids * 12
    f.write(f"- Duplicate example IDs: {'✓ PASS' if total_ids == expected_duplicates else '✗ FAIL'} (expected {expected_duplicates})\n")
    f.write(f"- Missing example IDs: {'✓ PASS' if not missing_ids else '✗ FAIL'}\n")
    f.write(f"- Extra example IDs: {'✓ PASS' if not extra_ids else '✗ FAIL'}\n")

print(f"\nReport saved to: {report_path}")
print("Phase 2 complete.")
