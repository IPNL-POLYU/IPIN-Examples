#!/usr/bin/env python3
"""
Acceptance Verification for Prompt 4: Allan Deviation Equation References

This script verifies that:
1. Equation numbering in code matches the book's numbering (6.56-6.58)
2. Book's Eq. (6.58) is implemented: σ_ω = ARW × √Δt
3. ARW extraction cites correct equations
4. Units are consistent with the book
5. No functions incorrectly cite Eqs. 6.56-6.58 as computational steps

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sensors.calibration import (
    allan_variance,
    identify_random_walk,
    arw_to_noise_std,
    noise_std_to_arw,
)


def check_equation_658_implementation():
    """Check that Eq. (6.58) is correctly implemented: sigma_omega = ARW * sqrt(dt)."""
    print("\n[1/6] Checking Eq. (6.58) implementation: sigma_omega = ARW * sqrt(dt)...")
    
    # Test case 1: Basic conversion
    arw = 0.01  # rad/sqrt(s)
    dt = 0.01  # 100 Hz
    
    sigma = arw_to_noise_std(arw, dt)
    expected = arw * np.sqrt(dt)
    
    if not np.isclose(sigma, expected):
        print(f"  [FAIL] arw_to_noise_std result mismatch")
        print(f"  [FAIL] Expected: {expected}, Got: {sigma}")
        return False
    
    # Test case 2: Inverse conversion
    sigma2 = 0.001  # rad/s
    dt2 = 0.01
    
    arw2 = noise_std_to_arw(sigma2, dt2)
    expected2 = sigma2 / np.sqrt(dt2)
    
    if not np.isclose(arw2, expected2):
        print(f"  [FAIL] noise_std_to_arw result mismatch")
        print(f"  [FAIL] Expected: {expected2}, Got: {arw2}")
        return False
    
    # Test case 3: Roundtrip consistency
    arw_roundtrip = noise_std_to_arw(sigma, dt)
    if not np.isclose(arw_roundtrip, arw):
        print(f"  [FAIL] Roundtrip not consistent")
        print(f"  [FAIL] Original ARW: {arw}, Recovered: {arw_roundtrip}")
        return False
    
    print("  [OK] Eq. (6.58) correctly implemented: sigma_omega = ARW * sqrt(dt)")
    return True


def check_equation_656_extraction():
    """Check that identify_random_walk implements Eq. (6.56): sigma(tau) = ARW/sqrt(tau)."""
    print("\n[2/6] Checking Eq. (6.56) ARW extraction: sigma(tau) = ARW/sqrt(tau)...")
    
    # Create synthetic Allan deviation curve with -1/2 slope
    taus = np.logspace(-2, 1, 100)
    N_true = 0.01  # true random walk coefficient
    adev = N_true / np.sqrt(taus)  # perfect -1/2 slope
    
    # Extract ARW at τ=1s
    arw = identify_random_walk(taus, adev, tau_target=1.0)
    
    # Should extract correct coefficient (at τ=1s, σ(1) = ARW)
    if not np.isclose(arw, N_true, rtol=0.1):
        print(f"  [FAIL] ARW extraction incorrect")
        print(f"  [FAIL] Expected: {N_true}, Got: {arw}")
        return False
    
    print(f"  [OK] Eq. (6.56) correctly extracts ARW: {arw:.5f} rad/sqrt(s)")
    return True


def check_documentation_references():
    """Check that documentation correctly references Eqs. 6.56-6.58."""
    print("\n[3/6] Checking documentation equation references...")
    
    # Read calibration.py
    calib_file = Path(__file__).parent.parent / "core" / "sensors" / "calibration.py"
    content = calib_file.read_text()
    
    checks = []
    
    # Check 1: Module docstring should reference correct equations
    if "Eq. (6.56): ARW extraction" in content or "σ(τ) = ARW/√τ" in content:
        checks.append(("Module docstring mentions Eq. 6.56 (ARW extraction)", True))
    else:
        checks.append(("Module docstring mentions Eq. 6.56", False))
    
    if "Eq. (6.58): ARW to per-sample noise" in content or "σ_ω = ARW × √Δt" in content:
        checks.append(("Module docstring mentions Eq. 6.58 (conversion)", True))
    else:
        checks.append(("Module docstring mentions Eq. 6.58", False))
    
    # Check 2: Allan variance function should NOT claim to implement Eqs. 6.56-6.58
    # (it implements IEEE Std 952-1997 algorithm, not these equations)
    allan_var_start = content.find("def allan_variance(")
    allan_var_end = content.find("def identify_bias_instability(")
    allan_var_docstring = content[allan_var_start:allan_var_end]
    
    if "IEEE Std 952-1997" in allan_var_docstring:
        checks.append(("allan_variance cites IEEE standard", True))
    else:
        checks.append(("allan_variance cites IEEE standard", False))
    
    # Should not claim to implement Eqs. 6.56-6.58 as computational steps
    bad_refs = ["Implements Eqs. (6.56)-(6.58)", "Eq. (6.56): Cluster", "Eq. (6.57): Allan variance definition"]
    has_bad_refs = any(bad_ref in allan_var_docstring for bad_ref in bad_refs)
    if not has_bad_refs:
        checks.append(("allan_variance doesn't incorrectly cite 6.56-6.58", True))
    else:
        checks.append(("allan_variance doesn't incorrectly cite 6.56-6.58", False))
    
    # Check 3: identify_random_walk should reference Eq. 6.56
    random_walk_start = content.find("def identify_random_walk(")
    random_walk_end = content.find("def identify_rate_random_walk(")
    random_walk_docstring = content[random_walk_start:random_walk_end]
    
    if "Eq. (6.56)" in random_walk_docstring or "σ(τ) = ARW" in random_walk_docstring:
        checks.append(("identify_random_walk cites Eq. 6.56", True))
    else:
        checks.append(("identify_random_walk cites Eq. 6.56", False))
    
    # Check 4: arw_to_noise_std should reference Eq. 6.58
    arw_to_noise_start = content.find("def arw_to_noise_std(")
    arw_to_noise_end = content.find("def noise_std_to_arw(")
    arw_to_noise_docstring = content[arw_to_noise_start:arw_to_noise_end]
    
    if "Eq. (6.58)" in arw_to_noise_docstring:
        checks.append(("arw_to_noise_std implements Eq. 6.58", True))
    else:
        checks.append(("arw_to_noise_std implements Eq. 6.58", False))
    
    # Print results
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def check_units_consistency():
    """Check that ARW units are consistent with the book."""
    print("\n[4/6] Checking units consistency...")
    
    # Book units:
    # - ARW: rad/√s (gyro) or m/s^(3/2) (accel)
    # - σ_ω: rad/s (gyro) or m/s² (accel)
    # - Δt: seconds
    
    # Gyro example
    arw_gyro = 0.01  # rad/√s (typical consumer IMU)
    dt = 0.01  # s (100 Hz)
    
    sigma_gyro = arw_to_noise_std(arw_gyro, dt)
    # Expected: rad/s
    if sigma_gyro < 0 or sigma_gyro > 1.0:
        print(f"  [FAIL] Gyro noise units seem wrong: {sigma_gyro} rad/s")
        return False
    
    # Accel example
    vrw_accel = 0.1  # m/s^(3/2) (typical consumer IMU)
    sigma_accel = arw_to_noise_std(vrw_accel, dt)
    # Expected: m/s²
    if sigma_accel < 0 or sigma_accel > 10.0:
        print(f"  [FAIL] Accel noise units seem wrong: {sigma_accel} m/s²")
        return False
    
    print(f"  [OK] Gyro: ARW={arw_gyro} rad/sqrt(s) -> sigma_omega={sigma_gyro:.5f} rad/s")
    print(f"  [OK] Accel: VRW={vrw_accel} m/s^(3/2) -> sigma_a={sigma_accel:.5f} m/s^2")
    return True


def check_function_signatures():
    """Check that new functions have correct signatures."""
    print("\n[5/6] Checking function signatures...")
    
    import inspect
    
    # Check arw_to_noise_std
    sig1 = inspect.signature(arw_to_noise_std)
    params1 = list(sig1.parameters.keys())
    if params1 != ['arw', 'dt']:
        print(f"  [FAIL] arw_to_noise_std signature wrong: {params1}")
        return False
    
    # Check noise_std_to_arw
    sig2 = inspect.signature(noise_std_to_arw)
    params2 = list(sig2.parameters.keys())
    if params2 != ['sigma', 'dt']:
        print(f"  [FAIL] noise_std_to_arw signature wrong: {params2}")
        return False
    
    print("  [OK] arw_to_noise_std(arw, dt) signature correct")
    print("  [OK] noise_std_to_arw(sigma, dt) signature correct")
    return True


def check_tests_pass():
    """Run tests and check they all pass."""
    print("\n[6/6] Running unit tests...")
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", 
         "tests/core/sensors/test_sensors_calibration.py::TestArwNoiseConversion",
         "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print("  [FAIL] Some tests failed:")
        print(result.stdout[-500:])  # Last 500 chars
        return False
    
    # Count passed tests
    if "passed" in result.stdout:
        # Extract number of passed tests
        import re
        match = re.search(r'(\d+) passed', result.stdout)
        if match:
            n_passed = int(match.group(1))
            print(f"  [OK] All {n_passed} ARW conversion tests passed")
            return True
    
    print("  [FAIL] Could not verify test results")
    return False


def main():
    """Run all acceptance checks."""
    print("="*70)
    print("Prompt 4 Acceptance Verification: Allan Deviation Equation References")
    print("="*70)
    
    checks = [
        ("Eq. (6.58) implementation", check_equation_658_implementation),
        ("Eq. (6.56) ARW extraction", check_equation_656_extraction),
        ("Documentation references", check_documentation_references),
        ("Units consistency", check_units_consistency),
        ("Function signatures", check_function_signatures),
        ("Tests coverage", check_tests_pass),
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
        print("ALL CHECKS PASSED -> Prompt 4 acceptance criteria met!")
        print("="*70)
        return 0
    else:
        print("SOME CHECKS FAILED -> Please review failures above")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

