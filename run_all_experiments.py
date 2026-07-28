#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run all experiments for:
"Local Differential Privacy Personal Data Pricing Based on Data Sensitivity"

This script sequentially executes all Python scripts in the repository to reproduce
all tables and figures reported in the paper.

Usage:
    python run_all_experiments.py

Outputs:
    - Table 2: Printed to console (LaTeX format)
    - Fig. 3: privacy_utility_tradeoff-1.png
    - Fig. 8: Fig.8.png
    - Table S2: Printed to console (formatted table)
    - Table S3: Printed to console (formatted table)
"""

import subprocess
import sys
import os
import time


def run_script(script_name, description):
    """Run a Python script and print its output."""
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print("=" * 80)

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode == 0:
        print(f"\n✅ {description} completed successfully in {elapsed:.2f} seconds.")
    else:
        print(f"\n❌ {description} failed with return code {result.returncode}.")

    return result.returncode


def main():
    """Execute all experiment scripts in order."""
    print("=" * 80)
    print("LOCAL DIFFERENTIAL PRIVACY PERSONAL DATA PRICING")
    print("Reproducing all experimental results")
    print("=" * 80)

    scripts = [
        ("2026-3-7-Utility comparison-3.py", "Table 2: Utility comparison (ε=1.0)"),
        ("2026-7-23-Local Differential Privacy Personal Data-privacy utility tradeoff.py", "Fig. 3: Privacy-Utility tradeoff"),
        ("2026-7-23-Local Differential Privacy Personal Data-Figure of MAE comparison around distinctive privacy budgets.py", "Fig. 8: MAE across ε budgets"),
        ("2026-7-24-Local Differential Privacy Personal Data-Table of MAE comparison around distinctive privacy budgets.py", "Table S2: MAE across ε budgets with stats"),
        ("2026-7-25-Table of 95%confidence intervals on important performance metrics (averaged across 5 independent runs).py", "Table S3: 95% confidence intervals")
    ]

    failed_scripts = []

    for script_name, description in scripts:
        if not os.path.exists(script_name):
            print(f"⚠️  Warning: {script_name} not found. Skipping.")
            continue

        return_code = run_script(script_name, description)
        if return_code != 0:
            failed_scripts.append(script_name)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if failed_scripts:
        print(f"❌ The following scripts failed: {failed_scripts}")
    else:
        print("✅ All experiments completed successfully!")
        print("\nExpected outputs:")
        print("  - Table 2: Printed in console (LaTeX format)")
        print("  - Fig. 3: privacy_utility_tradeoff-1.png")
        print("  - Fig. 8: Fig.8.png")
        print("  - Table S2: Printed in console (formatted table)")
        print("  - Table S3: Printed in console (formatted table)")
        print("\nAll results are reproducible with fixed random seeds.")


if __name__ == "__main__":
    main()