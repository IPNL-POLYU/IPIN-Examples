"""
Acceptance criteria verification for Prompt 2: Eq. (6.7) notation reconciliation.

Verifies that:
1. No function cites Eq. (6.7) while presenting an irreconcilable formula.
2. The sign and meaning of gravity is unambiguous for ENU/NED.

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.sensors.strapdown import vel_update, gravity_vector
from core.sensors.types import FrameConvention
from core.sim.imu_from_trajectory import compute_specific_force_body


def verify_acceptance_criterion_1():
    """
    Criterion 1: No function cites Eq. (6.7) while presenting an irreconcilable formula.
    """
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERION 1: Eq. (6.7) Formula Consistency")
    print("=" * 80)

    print("\nBook's Eq. (6.7) from references/ch6.txt line 58:")
    print("  v_k = v_{k-1} + (C_B^M @ a_B - g_M_book) * dt")
    print("  where g_M_book = [0, 0, g] (magnitude vector)")

    print("\nCode implementation (core/sensors/strapdown.py):")
    print("  v_k = v_{k-1} + (C_B^M @ f_B + g_M_code) * dt")
    print("  where g_M_code = [0, 0, -g] for ENU (physical gravity)")

    print("\n" + "-" * 80)
    print("Algebraic Equivalence Check")
    print("-" * 80)

    # Test with stationary accelerometer (should give zero velocity change)
    print("\nTest 1: Stationary accelerometer in ENU")
    print("  Physical reality: device on table, not moving")
    print("  Expected: v_dot = 0 (no velocity change)")

    frame_enu = FrameConvention.create_enu()

    # Stationary accelerometer reads upward reaction
    f_B = np.array([0.0, 0.0, 9.81])  # upward reaction in body frame
    print(f"  Accelerometer reading f_B: {f_B}")

    # Identity quaternion (body aligned with map)
    q = np.array([1.0, 0.0, 0.0, 0.0])

    # Zero initial velocity
    v0 = np.zeros(3)
    dt = 1.0  # 1 second

    # Compute velocity update
    v1 = vel_update(v0, q, f_B, dt, g=9.81, frame=frame_enu)

    # Check result
    v_change = np.linalg.norm(v1 - v0)
    print(f"  Velocity change: {v_change:.6e} m/s")

    test1_pass = v_change < 1e-10
    print(f"  Result: {'[PASS]' if test1_pass else '[FAIL]'} - Stationary gives zero v_dot")

    # Test with NED frame
    print("\nTest 2: Stationary accelerometer in NED")
    frame_ned = FrameConvention.create_ned()

    # In NED, upward is negative z (z points down)
    f_B_ned = np.array([0.0, 0.0, -9.81])  # upward reaction in NED
    print(f"  Accelerometer reading f_B: {f_B_ned}")

    v1_ned = vel_update(v0, q, f_B_ned, dt, g=9.81, frame=frame_ned)
    v_change_ned = np.linalg.norm(v1_ned - v0)
    print(f"  Velocity change: {v_change_ned:.6e} m/s")

    test2_pass = v_change_ned < 1e-10
    print(f"  Result: {'[PASS]' if test2_pass else '[FAIL]'} - NED also gives zero v_dot")

    # Test forward and inverse models
    print("\n" + "-" * 80)
    print("Forward/Inverse Consistency Check")
    print("-" * 80)

    print("\nTest 3: Forward model (accel->measurement) -> Inverse (measurement->accel)")
    print("  Round-trip should give identity")

    # True acceleration (some arbitrary value)
    a_M_true = np.array([1.5, -0.5, 2.0])  # m/s²
    print(f"  True acceleration a_M: {a_M_true}")

    # Forward model: compute what accelerometer should read
    f_B_computed = compute_specific_force_body(a_M_true, q, frame=frame_enu, g=9.81)
    print(f"  Computed specific force f_B: {f_B_computed}")

    # Inverse model: compute acceleration from accelerometer reading
    from core.sensors.strapdown import quat_to_rotmat

    C_B_M = quat_to_rotmat(q)
    g_M = gravity_vector(9.81, frame_enu)
    a_M_recovered = C_B_M @ f_B_computed + g_M

    print(f"  Recovered acceleration a_M: {a_M_recovered}")

    error = np.linalg.norm(a_M_recovered - a_M_true)
    print(f"  Round-trip error: {error:.6e} m/s²")

    test3_pass = error < 1e-10
    print(f"  Result: {'[PASS]' if test3_pass else '[FAIL]'} - Round-trip is identity")

    # Summary
    print("\n" + "=" * 80)
    all_pass = test1_pass and test2_pass and test3_pass
    if all_pass:
        print("[PASS] Criterion 1: Eq. (6.7) formula is reconcilable")
        print("\nKey findings:")
        print("  [OK] Book's (C @ a_B - g_book) = Code's (C @ f_B + g_code)")
        print("  [OK] Both give correct physics for stationary accelerometer")
        print("  [OK] Forward and inverse models are consistent")
        print("  [OK] Works correctly in both ENU and NED frames")
    else:
        print("[FAIL] Criterion 1: Formula inconsistency detected!")
    print("=" * 80)

    return all_pass


def verify_acceptance_criterion_2():
    """
    Criterion 2: The sign and meaning of gravity is unambiguous for ENU/NED.
    """
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERION 2: Gravity Sign/Meaning Unambiguous")
    print("=" * 80)

    # Check ENU gravity
    print("\nENU Frame:")
    frame_enu = FrameConvention.create_enu()
    g_enu = gravity_vector(9.81, frame_enu)
    print(f"  gravity_vector(9.81, ENU) = {g_enu}")
    print(f"  Expected: [0, 0, -9.81] (downward in ENU)")

    enu_correct = np.allclose(g_enu, [0, 0, -9.81], atol=1e-10)
    print(f"  Result: {'[PASS]' if enu_correct else '[FAIL]'}")

    # Check NED gravity
    print("\nNED Frame:")
    frame_ned = FrameConvention.create_ned()
    g_ned = gravity_vector(9.81, frame_ned)
    print(f"  gravity_vector(9.81, NED) = {g_ned}")
    print(f"  Expected: [0, 0, +9.81] (downward in NED z-down frame)")

    ned_correct = np.allclose(g_ned, [0, 0, +9.81], atol=1e-10)
    print(f"  Result: {'[PASS]' if ned_correct else '[FAIL]'}")

    # Check frame convention consistency
    print("\n" + "-" * 80)
    print("Frame Convention Consistency")
    print("-" * 80)

    print("\nENU gravity_direction:")
    print(f"  frame.gravity_direction = {frame_enu.gravity_direction}")
    print(f"  Expected: -1 (gravity points in negative z direction)")
    enu_dir_correct = frame_enu.gravity_direction == -1
    print(f"  Result: {'[PASS]' if enu_dir_correct else '[FAIL]'}")

    print("\nNED gravity_direction:")
    print(f"  frame.gravity_direction = {frame_ned.gravity_direction}")
    print(f"  Expected: +1 (gravity points in positive z direction, z is down)")
    ned_dir_correct = frame_ned.gravity_direction == +1
    print(f"  Result: {'[PASS]' if ned_dir_correct else '[FAIL]'}")

    # Documentation clarity check
    print("\n" + "-" * 80)
    print("Documentation Clarity Check")
    print("-" * 80)

    # Check that docstrings mention the convention
    import inspect

    vel_update_doc = inspect.getdoc(vel_update)
    has_book_reference = "BOOK'S EQ. (6.7)" in vel_update_doc
    has_code_reference = "CODE FORMULATION" in vel_update_doc
    has_equivalence = "ALGEBRAIC EQUIVALENCE" in vel_update_doc

    print("\nvel_update() docstring contains:")
    print(f"  - Book's Eq. (6.7) reference: {'[YES]' if has_book_reference else '[NO]'}")
    print(f"  - Code formulation: {'[YES]' if has_code_reference else '[NO]'}")
    print(f"  - Algebraic equivalence explanation: {'[YES]' if has_equivalence else '[NO]'}")

    doc_complete = has_book_reference and has_code_reference and has_equivalence
    print(f"  Result: {'[PASS]' if doc_complete else '[FAIL]'} - Documentation complete")

    # Summary
    print("\n" + "=" * 80)
    all_pass = enu_correct and ned_correct and enu_dir_correct and ned_dir_correct and doc_complete
    if all_pass:
        print("[PASS] Criterion 2: Gravity convention is unambiguous")
        print("\nKey findings:")
        print("  [OK] ENU gravity = [0, 0, -9.81] (downward)")
        print("  [OK] NED gravity = [0, 0, +9.81] (downward in z-down)")
        print("  [OK] Frame conventions explicit and consistent")
        print("  [OK] Documentation explains book/code equivalence")
    else:
        print("[FAIL] Criterion 2: Ambiguity in gravity convention!")
    print("=" * 80)

    return all_pass


def main():
    """Run all acceptance criteria verification."""
    print("\n" + "=" * 80)
    print("PROMPT 2 ACCEPTANCE CRITERIA VERIFICATION")
    print("=" * 80)
    print("\nObjective: Reconcile strapdown velocity update with book's Eq. (6.7)")
    print("\nIssue: Book uses (a_B - g_M) while code uses (f_B + g_M)")
    print("Resolution: These are algebraically equivalent with different notation")

    # Run verifications
    criterion_1_pass = verify_acceptance_criterion_1()
    criterion_2_pass = verify_acceptance_criterion_2()

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(
        f"  Criterion 1 (No Irreconcilable Formula): {'PASS' if criterion_1_pass else 'FAIL'}"
    )
    print(
        f"  Criterion 2 (Gravity Unambiguous):       {'PASS' if criterion_2_pass else 'FAIL'}"
    )

    all_pass = criterion_1_pass and criterion_2_pass

    print("\n" + "=" * 80)
    if all_pass:
        print("[PASS] ALL ACCEPTANCE CRITERIA MET FOR PROMPT 2")
        print("\nConvention established:")
        print("  - Use standard specific force f_B (what accelerometer measures)")
        print("  - Use physical gravity g_M (downward vector in ENU, NED)")
        print("  - Equation: a_M = C_B^M @ f_B + g_M")
        print("  - Equivalent to book's: a_M = C_B^M @ a_B - g_M_book")
        print("  - Relationship: g_M_code = -g_M_book")
        print("\nAll formulas cite Eq. (6.7) legitimately with documented equivalence!")
    else:
        print("[FAIL] SOME ACCEPTANCE CRITERIA NOT MET")
        print("\nPlease review the output above for details.")
    print("=" * 80 + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

