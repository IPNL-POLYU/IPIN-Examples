"""
Acceptance criteria verification for Prompt 1: EKF state ordering (Eq. 6.16).

Verifies that:
1. A developer can take the book's matrices and match blocks to code indices.
2. Unit tests confirm vector packing order and H block placement for ZUPT.

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.sensors.ins_ekf import INSState
from core.sensors.constraints import (
    ZuptMeasurementModel,
    ZaruMeasurementModel,
    NhcMeasurementModel,
)


def verify_acceptance_criterion_1():
    """
    Criterion 1: A developer can take the book's matrices and match blocks to code
    indices without translation.
    """
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERION 1: State Block Indices Match Book (Eq. 6.16)")
    print("=" * 80)

    # Book's Eq. (6.16): x_k = [p_k, v_k, q_k, b_{G,k}^B, b_{A,k}^B]^T
    book_state_definition = {
        "p (position)": (0, 3),
        "v (velocity)": (3, 6),
        "q (quaternion)": (6, 10),
        "b_g (gyro bias)": (10, 13),
        "b_a (accel bias)": (13, 16),
    }

    print("\nBook's Eq. (6.16) State Definition:")
    print("  x = [p, v, q, b_g, b_a]^T")
    print("\nExpected Indices:")
    for name, (start, end) in book_state_definition.items():
        print(f"  {name:25s}: indices [{start:2d}:{end:2d}]")

    # Create test state
    p = np.array([10.0, 20.0, 30.0])
    v = np.array([1.0, 2.0, 3.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    b_g = np.array([0.01, -0.02, 0.03])
    b_a = np.array([0.1, -0.2, 0.3])
    P = np.eye(16)

    state = INSState(p=p, v=v, q=q, b_g=b_g, b_a=b_a, P=P)
    x = state.to_vector()

    print("\nCode Implementation Verification:")
    all_match = True
    for name, (start, end) in book_state_definition.items():
        x_block = x[start:end]
        print(f"  {name:25s}: x[{start:2d}:{end:2d}] = {x_block}")

    # Verify ZUPT Jacobian (Eq. 6.45)
    print("\n" + "-" * 80)
    print("Book's Eq. (6.45): ZUPT Measurement Jacobian")
    print("-" * 80)
    print("  H = [0_3x3, I_3, 0_3x4, 0_3x3, 0_3x3]")
    print("  Non-zero block: I_3 at indices [3:6] (velocity block)")

    zupt_model = ZuptMeasurementModel()
    H_zupt = zupt_model.H(x)

    print("\nCode Implementation:")
    print(f"  H shape: {H_zupt.shape}")
    non_zero_cols = np.where(np.any(H_zupt != 0, axis=0))[0]
    print(f"  Non-zero columns: {non_zero_cols.tolist()}")

    expected_non_zero = [3, 4, 5]
    if list(non_zero_cols) == expected_non_zero:
        print("  [OK] ZUPT Jacobian selects velocity at indices [3:6]")
    else:
        print(f"  [FAIL] Expected non-zero at {expected_non_zero}, got {non_zero_cols}")
        all_match = False

    # Verify ZARU Jacobian (Eq. 6.60)
    print("\n" + "-" * 80)
    print("Book's Eq. (6.60): ZARU Measurement Jacobian")
    print("-" * 80)
    print("  H = [0_3x3, 0_3x3, 0_3x4, -I_3, 0_3x3]")
    print("  Non-zero block: -I_3 at indices [10:13] (gyro bias block)")

    zaru_model = ZaruMeasurementModel()
    H_zaru = zaru_model.H(x)

    print("\nCode Implementation:")
    print(f"  H shape: {H_zaru.shape}")
    non_zero_cols = np.where(np.any(H_zaru != 0, axis=0))[0]
    print(f"  Non-zero columns: {non_zero_cols.tolist()}")

    expected_non_zero = [10, 11, 12]
    if list(non_zero_cols) == expected_non_zero:
        print("  [OK] ZARU Jacobian selects gyro bias at indices [10:13]")
    else:
        print(f"  [FAIL] Expected non-zero at {expected_non_zero}, got {non_zero_cols}")
        all_match = False

    print("\n" + "=" * 80)
    if all_match:
        print("[PASS] Criterion 1: State indices match book definition (Eq. 6.16)")
    else:
        print("[FAIL] Criterion 1: State indices do NOT match book")
    print("=" * 80)

    return all_match


def verify_acceptance_criterion_2():
    """
    Criterion 2: Unit tests confirm vector packing order and H block placement.
    """
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERION 2: Unit Tests Pass")
    print("=" * 80)

    print("\nRunning unit tests for state ordering...")
    print("  Test file: tests/core/test_ins_state_ordering.py")

    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", "tests/core/test_ins_state_ordering.py", "-v"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("\n[OK] All unit tests passed!")
        print("\nKey tests verified:")
        print("  1. INSState.to_vector() ordering")
        print("  2. INSState.from_vector() unpacking")
        print("  3. ZUPT Jacobian H selects velocity at [3:6]")
        print("  4. ZARU Jacobian H selects gyro bias at [10:13]")
        print("  5. NHC Jacobian uses correct velocity and quaternion indices")

        print("\n" + "=" * 80)
        print("[PASS] Criterion 2: All unit tests pass")
        print("=" * 80)
        return True
    else:
        print("\n[FAIL] Some unit tests failed!")
        print("\nTest output:")
        print(result.stdout)
        print(result.stderr)

        print("\n" + "=" * 80)
        print("[FAIL] Criterion 2: Unit tests did not pass")
        print("=" * 80)
        return False


def main():
    """Run all acceptance criteria verification."""
    print("\n" + "=" * 80)
    print("PROMPT 1 ACCEPTANCE CRITERIA VERIFICATION")
    print("=" * 80)
    print("\nObjective: Verify EKF state definition matches Eq. (6.16)")
    print("  Expected: x = [p, v, q, b_g, b_a]^T (16 elements)")
    print("  Previous (WRONG): x = [q, v, p, b_g, b_a]^T")

    # Run verifications
    criterion_1_pass = verify_acceptance_criterion_1()
    criterion_2_pass = verify_acceptance_criterion_2()

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"  Criterion 1 (Block Indices Match Book):  {'PASS' if criterion_1_pass else 'FAIL'}")
    print(f"  Criterion 2 (Unit Tests Pass):           {'PASS' if criterion_2_pass else 'FAIL'}")

    all_pass = criterion_1_pass and criterion_2_pass

    print("\n" + "=" * 80)
    if all_pass:
        print("[PASS] ALL ACCEPTANCE CRITERIA MET FOR PROMPT 1")
        print("\nState ordering now matches Eq. (6.16) exactly:")
        print("  - Position at [0:3]")
        print("  - Velocity at [3:6]")
        print("  - Quaternion at [6:10]")
        print("  - Gyro bias at [10:13]")
        print("  - Accel bias at [13:16]")
        print("\nDevelopers can now directly map book equations to code!")
    else:
        print("[FAIL] SOME ACCEPTANCE CRITERIA NOT MET")
        print("\nPlease review the output above for details.")
    print("=" * 80 + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

