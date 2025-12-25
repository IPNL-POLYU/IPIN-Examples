"""
Verification script for acceptance criteria from Prompt 2.

Acceptance criteria:
1. For a level planar trajectory with no vertical motion and correct attitude,
   the vertical channel should not "free fall."
2. ZUPT detector should have a chance to trigger because ||a|| should be near g
   during stance.

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
from core.sensors import FrameConvention
from core.sim import generate_imu_from_trajectory

def test_level_planar_no_free_fall():
    """
    Test that a level planar trajectory does not cause free fall in vertical channel.
    """
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERION 1: No Free Fall in Vertical Channel")
    print("="*70)
    
    # Generate a simple level planar trajectory (constant velocity, z=0)
    dt = 0.01
    duration = 10.0
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Constant velocity in x direction
    v_x = 1.0  # m/s
    pos_map = np.zeros((N, 3))
    vel_map = np.zeros((N, 3))
    
    for k in range(N):
        pos_map[k] = np.array([v_x * t[k], 0.0, 0.0])
        vel_map[k] = np.array([v_x, 0.0, 0.0])
    
    # Attitude: yaw = 0 (facing East in ENU), roll = pitch = 0
    quat_b_to_m = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))
    
    # Generate IMU measurements
    frame = FrameConvention.create_enu()
    accel_body, gyro_body = generate_imu_from_trajectory(
        pos_map=pos_map,
        vel_map=vel_map,
        quat_b_to_m=quat_b_to_m,
        dt=dt,
        frame=frame,
        g=9.81
    )
    
    # Check vertical component of accelerometer
    accel_z = accel_body[:, 2]
    mean_accel_z = np.mean(accel_z)
    std_accel_z = np.std(accel_z)
    
    print(f"\nLevel planar trajectory (constant velocity, z=0):")
    print(f"  Duration:           {duration} s")
    print(f"  Velocity:           {v_x} m/s (East)")
    print(f"  Frame:              {frame.map_frame}")
    print(f"\nVertical accelerometer (z-axis):")
    print(f"  Mean:               {mean_accel_z:.4f} m/s²")
    print(f"  Std Dev:            {std_accel_z:.4f} m/s²")
    print(f"  Expected:           +9.81 m/s² (ENU: upward reaction force)")
    
    # For a level planar trajectory with no vertical motion,
    # the accelerometer should read +g in ENU (upward reaction)
    expected_accel_z = 9.81
    error = abs(mean_accel_z - expected_accel_z)
    
    if error < 0.01:
        print(f"\n[PASS] Vertical channel correctly compensated (error = {error:.4f} m/s^2)")
        print("  No free fall detected!")
        return True
    else:
        print(f"\n[FAIL] Vertical channel error too large (error = {error:.4f} m/s^2)")
        return False


def test_zupt_detector_can_trigger():
    """
    Test that ZUPT detector can trigger because ||a|| ≈ g during stance.
    """
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERION 2: ZUPT Detector Can Trigger During Stance")
    print("="*70)
    
    # Generate a trajectory with a stationary phase
    dt = 0.01
    duration = 5.0
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Walk for 2s, then stop for 3s
    pos_map = np.zeros((N, 3))
    vel_map = np.zeros((N, 3))
    
    for k in range(N):
        if t[k] < 2.0:
            # Walking phase
            v_x = 1.0  # m/s
            pos_map[k] = np.array([v_x * t[k], 0.0, 0.0])
            vel_map[k] = np.array([v_x, 0.0, 0.0])
        else:
            # Stationary phase
            pos_map[k] = pos_map[k-1] if k > 0 else np.array([2.0, 0.0, 0.0])
            vel_map[k] = np.array([0.0, 0.0, 0.0])
    
    # Attitude: yaw = 0, roll = pitch = 0
    quat_b_to_m = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))
    
    # Generate IMU measurements
    frame = FrameConvention.create_enu()
    accel_body, gyro_body = generate_imu_from_trajectory(
        pos_map=pos_map,
        vel_map=vel_map,
        quat_b_to_m=quat_b_to_m,
        dt=dt,
        frame=frame,
        g=9.81
    )
    
    # Check accelerometer magnitude during stance phase (t > 2s)
    stance_mask = t >= 2.0
    accel_mag_stance = np.linalg.norm(accel_body[stance_mask], axis=1)
    mean_accel_mag = np.mean(accel_mag_stance)
    std_accel_mag = np.std(accel_mag_stance)
    
    print(f"\nTrajectory with stance phase:")
    print(f"  Duration:           {duration} s")
    print(f"  Walking:            0-2s (1.0 m/s)")
    print(f"  Stationary:         2-5s")
    print(f"  Frame:              {frame.map_frame}")
    print(f"\nAccelerometer magnitude during stance:")
    print(f"  Mean:               {mean_accel_mag:.4f} m/s²")
    print(f"  Std Dev:            {std_accel_mag:.4f} m/s²")
    print(f"  Expected:           ~9.81 m/s² (gravity)")
    
    # For a stationary period, ||a|| should be approximately g
    expected_mag = 9.81
    error = abs(mean_accel_mag - expected_mag)
    
    # ZUPT detector typically uses threshold like ||a - g|| < 0.5 m/s²
    # So if ||a|| ≈ g, the detector can trigger
    zupt_threshold = 0.5  # m/s² (typical value)
    can_trigger = error < zupt_threshold
    
    if can_trigger:
        print(f"\n[PASS] ZUPT detector can trigger (error = {error:.4f} < {zupt_threshold} m/s^2)")
        print("  Accelerometer magnitude is near g during stance!")
        return True
    else:
        print(f"\n[FAIL] ZUPT detector cannot trigger (error = {error:.4f} >= {zupt_threshold} m/s^2)")
        return False


def main():
    """Run all acceptance criterion tests."""
    print("\n" + "="*70)
    print("VERIFYING ACCEPTANCE CRITERIA FOR PROMPT 2")
    print("="*70)
    print("\nThese tests verify that the IMU forward model is correctly implemented.")
    
    test1_pass = test_level_planar_no_free_fall()
    test2_pass = test_zupt_detector_can_trigger()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Criterion 1 (No Free Fall):       {'[PASS]' if test1_pass else '[FAIL]'}")
    print(f"Criterion 2 (ZUPT Can Trigger):   {'[PASS]' if test2_pass else '[FAIL]'}")
    
    if test1_pass and test2_pass:
        print("\n[SUCCESS] ALL ACCEPTANCE CRITERIA MET!")
        print("  The IMU forward model is correctly implemented.")
    else:
        print("\n[ERROR] SOME ACCEPTANCE CRITERIA NOT MET")
        print("  Please review the implementation.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

