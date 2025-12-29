#!/usr/bin/env python3
"""
Acceptance Verification for Prompt 5: PDR Unified Peak Detection

This script verifies that:
1. Dataset and inline paths use the same step detector
2. Both paths use detect_steps_peak_detector() implementing Eqs. 6.46-6.47
3. Parameters are tuned for typical walking
4. Sampling rate is automatically detected
5. Step timestamps are consistent

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_unified_detector():
    """Check that both paths use detect_steps_peak_detector."""
    print("\n[1/5] Checking unified detector usage...")
    
    # Read example_pdr.py
    pdr_file = Path(__file__).parent.parent / "ch6_dead_reckoning" / "example_pdr.py"
    content = pdr_file.read_text()
    
    # Check run_pdr_from_dataset uses detect_steps_peak_detector
    dataset_start = content.find("def run_pdr_from_dataset(")
    dataset_end = content.find("def run_with_dataset(")
    dataset_func = content[dataset_start:dataset_end]
    
    if "detect_steps_peak_detector" not in dataset_func:
        print("  [FAIL] run_pdr_from_dataset doesn't use detect_steps_peak_detector")
        return False
    
    # Check it doesn't use threshold crossing
    bad_patterns = [
        "last_a_mag < 11.0 and a_mag >= 11.0",
        "is_step = (last_a_mag",
        "threshold crossing"
    ]
    
    for pattern in bad_patterns:
        if pattern in dataset_func:
            print(f"  [FAIL] run_pdr_from_dataset still uses threshold crossing: {pattern}")
            return False
    
    # Check run_pdr_gyro_heading uses detect_steps_peak_detector
    gyro_start = content.find("def run_pdr_gyro_heading(")
    gyro_end = content.find("def run_pdr_mag_heading(")
    gyro_func = content[gyro_start:gyro_end]
    
    if "detect_steps_peak_detector" not in gyro_func:
        print("  [FAIL] run_pdr_gyro_heading doesn't use detect_steps_peak_detector")
        return False
    
    # Check run_pdr_mag_heading uses detect_steps_peak_detector
    mag_start = content.find("def run_pdr_mag_heading(")
    mag_end = content.find("def plot_results(")
    mag_func = content[mag_start:mag_end]
    
    if "detect_steps_peak_detector" not in mag_func:
        print("  [FAIL] run_pdr_mag_heading doesn't use detect_steps_peak_detector")
        return False
    
    print("  [OK] All three functions use detect_steps_peak_detector")
    return True


def check_book_equations():
    """Check that implementation references Eqs. 6.46-6.47."""
    print("\n[2/5] Checking book equation references...")
    
    pdr_file = Path(__file__).parent.parent / "ch6_dead_reckoning" / "example_pdr.py"
    content = pdr_file.read_text()
    
    # Check run_pdr_from_dataset mentions Eqs. 6.46-6.47
    dataset_start = content.find("def run_pdr_from_dataset(")
    dataset_end = content.find("def run_with_dataset(")
    dataset_func = content[dataset_start:dataset_end]
    
    checks = []
    
    if "(Eqs. 6.46-6.47)" in dataset_func or "(6.46-6.47)" in dataset_func:
        checks.append(("run_pdr_from_dataset cites Eqs. 6.46-6.47", True))
    else:
        checks.append(("run_pdr_from_dataset cites Eqs. 6.46-6.47", False))
    
    if "Compute total acceleration magnitude" in dataset_func or "(6.46)" in dataset_func:
        checks.append(("Mentions Eq. 6.46 (total accel magnitude)", True))
    else:
        checks.append(("Mentions Eq. 6.46", False))
    
    if "Subtract gravity" in dataset_func or "(6.47)" in dataset_func:
        checks.append(("Mentions Eq. 6.47 (subtract gravity)", True))
    else:
        checks.append(("Mentions Eq. 6.47", False))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def check_tuned_parameters():
    """Check that parameters are tuned and documented."""
    print("\n[3/5] Checking tuned parameters...")
    
    pdr_file = Path(__file__).parent.parent / "ch6_dead_reckoning" / "example_pdr.py"
    content = pdr_file.read_text()
    
    # Extract parameter values from run_pdr_from_dataset
    dataset_start = content.find("def run_pdr_from_dataset(")
    dataset_end = content.find("def run_with_dataset(")
    dataset_func = content[dataset_start:dataset_end]
    
    checks = []
    
    # Check min_peak_height
    if "min_peak_height=1.0" in dataset_func:
        checks.append(("min_peak_height=1.0 (documented)", True))
    else:
        checks.append(("min_peak_height=1.0", False))
    
    # Check min_peak_distance
    if "min_peak_distance=0.3" in dataset_func:
        checks.append(("min_peak_distance=0.3 (documented)", True))
    else:
        checks.append(("min_peak_distance=0.3", False))
    
    # Check lowpass_cutoff
    if "lowpass_cutoff=5.0" in dataset_func:
        checks.append(("lowpass_cutoff=5.0 (documented)", True))
    else:
        checks.append(("lowpass_cutoff=5.0", False))
    
    # Check for documentation/comments
    if "# m/s" in dataset_func or "# seconds" in dataset_func or "# Hz" in dataset_func:
        checks.append(("Parameters have comments", True))
    else:
        checks.append(("Parameters documented", False))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def check_sampling_rate_awareness():
    """Check that sampling rate is automatically detected."""
    print("\n[4/5] Checking sampling rate awareness...")
    
    pdr_file = Path(__file__).parent.parent / "ch6_dead_reckoning" / "example_pdr.py"
    content = pdr_file.read_text()
    
    dataset_start = content.find("def run_pdr_from_dataset(")
    dataset_end = content.find("def run_with_dataset(")
    dataset_func = content[dataset_start:dataset_end]
    
    checks = []
    
    # Check dt calculation
    if "dt = t[1] - t[0]" in dataset_func:
        checks.append(("dt calculated from time array", True))
    else:
        checks.append(("dt calculated from time array", False))
    
    # Check fs calculation
    if "fs = 1.0 / dt" in dataset_func or "fs = 1/dt" in dataset_func:
        checks.append(("fs calculated from dt", True))
    else:
        checks.append(("fs calculated", False))
    
    # Check dt passed to detector
    if "dt=dt" in dataset_func:
        checks.append(("dt passed to detect_steps_peak_detector", True))
    else:
        checks.append(("dt passed to detector", False))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def check_step_consistency():
    """Check that step detection is consistent between gyro and mag paths."""
    print("\n[5/5] Checking step consistency...")
    
    pdr_file = Path(__file__).parent.parent / "ch6_dead_reckoning" / "example_pdr.py"
    content = pdr_file.read_text()
    
    dataset_start = content.find("def run_pdr_from_dataset(")
    dataset_end = content.find("def run_with_dataset(")
    dataset_func = content[dataset_start:dataset_end]
    
    checks = []
    
    # Check that detect_steps_peak_detector is called once, not twice
    detector_count = dataset_func.count("detect_steps_peak_detector")
    if detector_count == 1:
        checks.append(("Step detection called once (shared)", True))
    else:
        checks.append((f"Step detection called {detector_count} times (should be 1)", False))
    
    # Check that both gyro and mag use step_indices
    if "if k in step_indices:" in dataset_func:
        step_check_count = dataset_func.count("if k in step_indices:")
        if step_check_count >= 2:
            checks.append((f"Both paths use step_indices ({step_check_count} occurrences)", True))
        else:
            checks.append((f"Only {step_check_count} path uses step_indices", False))
    else:
        checks.append(("step_indices used for step events", False))
    
    # Check that step_count uses len(step_indices)
    if "len(step_indices)" in dataset_func:
        checks.append(("step_count computed from step_indices", True))
    else:
        checks.append(("step_count consistency", False))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def main():
    """Run all acceptance checks."""
    print("="*70)
    print("Prompt 5 Acceptance Verification: PDR Unified Peak Detection")
    print("="*70)
    
    checks = [
        ("Unified detector usage", check_unified_detector),
        ("Book equations referenced", check_book_equations),
        ("Tuned parameters", check_tuned_parameters),
        ("Sampling rate awareness", check_sampling_rate_awareness),
        ("Step consistency", check_step_consistency),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("ACCEPTANCE SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status:8} {name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL CHECKS PASSED -> Prompt 5 acceptance criteria met!")
        print("="*70)
        return 0
    else:
        print("SOME CHECKS FAILED -> Please review failures above")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())






