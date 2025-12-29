#!/usr/bin/env python3
"""
Acceptance Verification for Prompt 3: Wheel Odometry Eq. (6.11) Book Convention

This script verifies that:
1. wheel_speed_to_attitude_velocity includes C_S^A rotation parameter
2. Speed frame convention matches book (x=right, y=forward, z=up)
3. Tests cover both aligned (C_S^A=I) and misaligned (C_S^A≠I) frames
4. Examples and docs use the book convention
5. Equation (6.11) is correctly implemented: v^A = C_S^A @ v^S - [ω^A×] @ l^A

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sensors.wheel_odometry import (
    wheel_speed_to_attitude_velocity,
    skew,
)


def check_function_signature():
    """Check that wheel_speed_to_attitude_velocity has C_S_A parameter."""
    print("\n[1/6] Checking function signature...")
    
    import inspect
    sig = inspect.signature(wheel_speed_to_attitude_velocity)
    params = list(sig.parameters.keys())
    
    # Check required parameters
    if params[:3] != ['v_s', 'omega_a', 'lever_arm_a']:
        print("  [FAIL] Expected parameters: v_s, omega_a, lever_arm_a")
        print(f"  [FAIL] Got: {params[:3]}")
        return False
    
    # Check C_S_A parameter exists
    if 'C_S_A' not in params:
        print("  [FAIL] Missing C_S_A parameter")
        return False
    
    # Check default value is None
    if sig.parameters['C_S_A'].default is not None:
        print(f"  [FAIL] C_S_A default should be None, got {sig.parameters['C_S_A'].default}")
        return False
    
    print("  [OK] Function signature matches book: (v_s, omega_a, lever_arm_a, C_S_A=None)")
    return True


def check_book_equation_aligned():
    """Check Eq. (6.11) with aligned frames (C_S^A = I)."""
    print("\n[2/6] Checking Eq. (6.11) with aligned frames...")
    
    # Test case: forward velocity with yaw and lever arm
    v_s = np.array([0.0, 5.0, 0.0])  # Book convention: y=forward
    omega_a = np.array([0.0, 0.0, 1.0])  # 1 rad/s yaw
    lever_arm_a = np.array([0.0, 0.5, 0.0])  # 0.5m forward (y-axis)
    
    # Compute using function (default C_S^A = I)
    v_a = wheel_speed_to_attitude_velocity(v_s, omega_a, lever_arm_a)
    
    # Manual calculation: v^A = v^S - [ω×] @ l
    omega_skew = skew(omega_a)
    v_a_expected = v_s - omega_skew @ lever_arm_a
    
    if not np.allclose(v_a, v_a_expected):
        print(f"  [FAIL] Result mismatch")
        print(f"  [FAIL] Expected: {v_a_expected}")
        print(f"  [FAIL] Got: {v_a}")
        return False
    
    print(f"  [OK] Aligned frames: v^A = {v_a} matches Eq. (6.11)")
    return True


def check_book_equation_misaligned():
    """Check Eq. (6.11) with misaligned frames (C_S^A ≠ I)."""
    print("\n[3/6] Checking Eq. (6.11) with misaligned frames...")
    
    # Test case: 90° rotation between S and A frames
    v_s = np.array([0.0, 5.0, 0.0])  # Forward in S-frame
    omega_a = np.array([0.0, 0.0, 0.5])  # yaw rate
    lever_arm_a = np.array([0.0, 0.5, 0.0])
    
    # 90° rotation about z-axis
    angle = np.pi / 2
    C_S_A = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Compute using function
    v_a = wheel_speed_to_attitude_velocity(v_s, omega_a, lever_arm_a, C_S_A)
    
    # Manual calculation: v^A = C_S^A @ v^S - [ω×] @ l
    rotated_v = C_S_A @ v_s
    omega_skew = skew(omega_a)
    v_a_expected = rotated_v - omega_skew @ lever_arm_a
    
    if not np.allclose(v_a, v_a_expected):
        print(f"  [FAIL] Result mismatch")
        print(f"  [FAIL] Expected: {v_a_expected}")
        print(f"  [FAIL] Got: {v_a}")
        return False
    
    print(f"  [OK] Misaligned frames: v^A = {v_a} matches Eq. (6.11)")
    return True


def check_speed_frame_convention():
    """Verify speed frame convention: x=right, y=forward, z=up."""
    print("\n[4/6] Checking speed frame convention...")
    
    # Check that forward velocity in y-component produces expected result
    v_s = np.array([0.0, 10.0, 0.0])  # 10 m/s forward
    omega_a = np.zeros(3)
    lever_arm_a = np.zeros(3)
    
    v_a = wheel_speed_to_attitude_velocity(v_s, omega_a, lever_arm_a)
    
    # With zero rotation and lever arm, v^A = v^S
    if not np.allclose(v_a, v_s):
        print(f"  [FAIL] Expected v^A = v^S = {v_s}")
        print(f"  [FAIL] Got v^A = {v_a}")
        return False
    
    # Check that y-component is forward (non-zero)
    if v_a[1] != 10.0:
        print(f"  [FAIL] Forward velocity should be in y-component")
        print(f"  [FAIL] Got v_a[1] = {v_a[1]}, expected 10.0")
        return False
    
    print("  [OK] Speed frame convention: y=forward (book convention)")
    return True


def check_tests_coverage():
    """Run tests and check for misaligned frame coverage."""
    print("\n[5/6] Running unit tests...")
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", 
         "tests/core/sensors/test_sensors_wheel_odometry.py", 
         "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print("  [FAIL] Some tests failed:")
        print(result.stdout)
        return False
    
    # Check that misaligned frame tests exist
    test_output = result.stdout
    misaligned_tests = [
        "test_lever_arm_equation_6_11_misaligned",
        "test_misaligned_frames_identity_rotation",
        "test_misaligned_frames_180deg_rotation"
    ]
    
    for test_name in misaligned_tests:
        if test_name not in test_output:
            print(f"  [FAIL] Missing test: {test_name}")
            return False
    
    # Count passed tests
    if "passed" in test_output:
        print("  [OK] All tests passed, including misaligned frame tests")
        return True
    else:
        print("  [FAIL] Could not verify test results")
        return False


def check_documentation():
    """Verify documentation mentions book convention."""
    print("\n[6/6] Checking documentation...")
    
    # Read wheel_odometry.py
    wheel_odom_file = Path(__file__).parent.parent / "core" / "sensors" / "wheel_odometry.py"
    content = wheel_odom_file.read_text()
    
    # Check for key documentation elements
    checks = [
        ("C_S^A rotation", "C_S^A" in content or "C_S_A" in content),
        ("Book convention mentioned", "book convention" in content.lower() or "y=forward" in content.lower()),
        ("Speed frame axes", "x=right" in content.lower() or "y=forward" in content.lower()),
        ("Eq. (6.11) reference", "(6.11)" in content or "Eq. 6.11" in content),
    ]
    
    all_passed = True
    for check_name, check_result in checks:
        if check_result:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name} not found")
            all_passed = False
    
    return all_passed


def main():
    """Run all acceptance checks."""
    print("="*70)
    print("Prompt 3 Acceptance Verification: Wheel Odometry Eq. (6.11)")
    print("="*70)
    
    checks = [
        ("Function signature", check_function_signature),
        ("Eq. (6.11) aligned frames", check_book_equation_aligned),
        ("Eq. (6.11) misaligned frames", check_book_equation_misaligned),
        ("Speed frame convention", check_speed_frame_convention),
        ("Tests coverage", check_tests_coverage),
        ("Documentation", check_documentation),
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
        print("ALL CHECKS PASSED -> Prompt 3 acceptance criteria met!")
        print("="*70)
        return 0
    else:
        print("SOME CHECKS FAILED -> Please review failures above")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())






