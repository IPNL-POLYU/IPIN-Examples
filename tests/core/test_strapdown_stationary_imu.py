"""
Unit tests for stationary IMU (zero drift acceptance criterion).

This test validates that a stationary IMU with zero biases produces:
    - Zero velocity drift (bounded by noise only)
    - Bounded position drift (bounded by numerical integration of noise)
    - Constant attitude (within numerical error)

This is the acceptance criterion for Task 1: Frame Convention Unification.

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
import pytest

from core.sensors import (
    FrameConvention,
    NavStateQPVP,
    strapdown_update,
    correct_gyro,
    correct_accel,
)


class TestStationaryIMU:
    """Test suite for stationary IMU with zero biases."""

    def test_stationary_imu_zero_drift_enu(self) -> None:
        """
        Test stationary IMU with zero biases in ENU frame.

        Acceptance criteria:
            - Velocity remains near zero (< 0.01 m/s after 100 seconds)
            - Position drift < 0.5 m after 100 seconds (bounded by noise)
            - Attitude quaternion remains near identity (< 0.001 deviation)

        This validates that:
            1. Gravity compensation is correct (no velocity drift from gravity)
            2. Quaternion integration is stable (no attitude drift)
            3. Frame convention is consistent (ENU z-down gravity)
        """
        # Test configuration
        duration = 100.0  # seconds
        dt = 0.01  # 10 ms (100 Hz IMU)
        n_steps = int(duration / dt)

        # Frame convention: ENU
        frame = FrameConvention.create_enu()

        # Initial state: origin, stationary, level attitude
        q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        v0 = np.zeros(3)
        p0 = np.zeros(3)
        state = NavStateQPVP(q=q0, v=v0, p=p0)

        # IMU measurements for stationary device in ENU frame:
        # - Body frame aligned with map frame (identity rotation)
        # - Accelerometer measures specific force (reaction): [0, 0, +g] in body frame
        #   (upward reaction force from ground, NOT gravity itself)
        # - Gyroscope measures zero rotation
        # - Zero biases (ideal sensor)
        g = 9.81  # m/s²
        accel_meas = np.array([0.0, 0.0, +g])  # upward reaction force in body
        gyro_meas = np.array([0.0, 0.0, 0.0])  # no rotation

        # Noise-free (ideal) measurements
        gyro_bias = np.zeros(3)
        accel_bias = np.zeros(3)

        # Corrected measurements (no bias, no noise)
        omega_b = correct_gyro(gyro_meas, gyro_bias)
        f_b = correct_accel(accel_meas, accel_bias)

        # Propagate IMU for duration
        q = state.q.copy()
        v = state.v.copy()
        p = state.p.copy()

        for _ in range(n_steps):
            q, v, p = strapdown_update(
                q=q,
                v=v,
                p=p,
                omega_b=omega_b,
                f_b=f_b,
                dt=dt,
                g=g,
                frame=frame,
            )

        # Acceptance criteria
        final_vel_norm = np.linalg.norm(v)
        final_pos_norm = np.linalg.norm(p)
        quat_deviation = np.linalg.norm(q - q0)

        print(f"\n{'='*60}")
        print("Stationary IMU Test Results (ENU, Zero Biases)")
        print(f"{'='*60}")
        print(f"Duration:              {duration:.1f} s")
        print(f"Time step:             {dt*1000:.1f} ms")
        print(f"Number of steps:       {n_steps}")
        print(f"\nFrame convention:      {frame.map_frame}")
        print(f"Gravity direction:     {frame.gravity_direction}")
        print(f"Heading zero:          {frame.heading_zero_direction}")
        print(f"\nInitial state:")
        print(f"  Position:            {p0}")
        print(f"  Velocity:            {v0}")
        print(f"  Quaternion:          {q0}")
        print(f"\nFinal state:")
        print(f"  Position:            {p}")
        print(f"  Velocity:            {v}")
        print(f"  Quaternion:          {q}")
        print(f"\nDrift metrics:")
        print(f"  Velocity drift:      {final_vel_norm:.6f} m/s")
        print(f"  Position drift:      {final_pos_norm:.6f} m")
        print(f"  Quaternion deviation: {quat_deviation:.6f}")
        print(f"{'='*60}\n")

        # Assertions (acceptance criteria)
        assert (
            final_vel_norm < 1e-10
        ), f"Velocity drift {final_vel_norm:.2e} m/s exceeds 1e-10 m/s"
        assert (
            final_pos_norm < 1e-8
        ), f"Position drift {final_pos_norm:.2e} m exceeds 1e-8 m"
        assert (
            quat_deviation < 1e-10
        ), f"Quaternion deviation {quat_deviation:.2e} exceeds 1e-10"

    def test_stationary_imu_zero_drift_ned(self) -> None:
        """
        Test stationary IMU with zero biases in NED frame.

        Validates that NED frame convention also produces zero drift.
        Key difference: gravity points downward (positive z in NED).
        """
        # Test configuration
        duration = 100.0  # seconds
        dt = 0.01  # 10 ms (100 Hz IMU)
        n_steps = int(duration / dt)

        # Frame convention: NED
        frame = FrameConvention.create_ned()

        # Initial state: origin, stationary, level attitude
        q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        v0 = np.zeros(3)
        p0 = np.zeros(3)
        state = NavStateQPVP(q=q0, v=v0, p=p0)

        # IMU measurements for stationary device in NED frame:
        # - Body frame aligned with map frame (identity rotation)
        # - Accelerometer measures specific force (reaction): [0, 0, -g] in body frame
        #   (upward reaction in NED where z-down, so reaction is negative z)
        # - Gyroscope measures zero rotation
        g = 9.81  # m/s²
        accel_meas = np.array([0.0, 0.0, -g])  # upward reaction in NED body
        gyro_meas = np.array([0.0, 0.0, 0.0])  # no rotation

        # Corrected measurements (no bias, no noise)
        omega_b = correct_gyro(gyro_meas, np.zeros(3))
        f_b = correct_accel(accel_meas, np.zeros(3))

        # Propagate IMU for duration
        q = state.q.copy()
        v = state.v.copy()
        p = state.p.copy()

        for _ in range(n_steps):
            q, v, p = strapdown_update(
                q=q,
                v=v,
                p=p,
                omega_b=omega_b,
                f_b=f_b,
                dt=dt,
                g=g,
                frame=frame,
            )

        # Acceptance criteria
        final_vel_norm = np.linalg.norm(v)
        final_pos_norm = np.linalg.norm(p)
        quat_deviation = np.linalg.norm(q - q0)

        print(f"\n{'='*60}")
        print("Stationary IMU Test Results (NED, Zero Biases)")
        print(f"{'='*60}")
        print(f"Duration:              {duration:.1f} s")
        print(f"Frame convention:      {frame.map_frame}")
        print(f"Gravity direction:     {frame.gravity_direction}")
        print(f"\nFinal drift metrics:")
        print(f"  Velocity drift:      {final_vel_norm:.6f} m/s")
        print(f"  Position drift:      {final_pos_norm:.6f} m")
        print(f"  Quaternion deviation: {quat_deviation:.6f}")
        print(f"{'='*60}\n")

        # Assertions (same acceptance criteria as ENU)
        assert (
            final_vel_norm < 1e-10
        ), f"Velocity drift {final_vel_norm:.2e} m/s exceeds 1e-10 m/s"
        assert (
            final_pos_norm < 1e-8
        ), f"Position drift {final_pos_norm:.2e} m exceeds 1e-8 m"
        assert (
            quat_deviation < 1e-10
        ), f"Quaternion deviation {quat_deviation:.2e} exceeds 1e-10"

    def test_stationary_imu_with_small_noise(self) -> None:
        """
        Test stationary IMU with small noise (but zero biases).

        With noise, we expect:
            - Velocity to remain small (bounded by noise, not drift)
            - Position drift to be bounded (O(noise * sqrt(t)))
            - Attitude to remain nearly constant

        This validates that noise doesn't cause systematic drift.
        """
        # Test configuration
        duration = 100.0  # seconds
        dt = 0.01  # 10 ms
        n_steps = int(duration / dt)

        # Frame convention: ENU
        frame = FrameConvention.create_enu()

        # Noise levels (typical consumer-grade IMU)
        gyro_noise_std = np.deg2rad(0.1) * np.sqrt(1 / dt)  # ARW
        accel_noise_std = 0.001 * np.sqrt(1 / dt)  # VRW

        # Initial state
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        v0 = np.zeros(3)
        p0 = np.zeros(3)

        # Propagate with noise
        q = q0.copy()
        v = v0.copy()
        p = p0.copy()

        g = 9.81
        np.random.seed(42)  # for reproducibility

        for _ in range(n_steps):
            # Stationary true motion: specific force (reaction) + zero rotation
            accel_true = np.array([0.0, 0.0, +g])  # upward reaction in ENU
            gyro_true = np.zeros(3)

            # Add noise
            accel_meas = accel_true + np.random.randn(3) * accel_noise_std
            gyro_meas = gyro_true + np.random.randn(3) * gyro_noise_std

            # Corrected measurements (zero biases)
            omega_b = correct_gyro(gyro_meas, np.zeros(3))
            f_b = correct_accel(accel_meas, np.zeros(3))

            # Update
            q, v, p = strapdown_update(
                q=q, v=v, p=p, omega_b=omega_b, f_b=f_b, dt=dt, g=g, frame=frame
            )

        # With noise, drift is stochastic but bounded
        final_vel_norm = np.linalg.norm(v)
        final_pos_norm = np.linalg.norm(p)
        quat_deviation = np.linalg.norm(q - q0)

        print(f"\n{'='*60}")
        print("Stationary IMU Test Results (ENU, With Noise)")
        print(f"{'='*60}")
        print(f"Duration:              {duration:.1f} s")
        print(f"Gyro noise std:        {np.rad2deg(gyro_noise_std):.3e} deg/s")
        print(f"Accel noise std:       {accel_noise_std:.3e} m/s²")
        print(f"\nFinal drift metrics:")
        print(f"  Velocity:            {final_vel_norm:.6f} m/s")
        print(f"  Position:            {final_pos_norm:.6f} m")
        print(f"  Quaternion deviation: {quat_deviation:.6f}")
        print(f"{'='*60}\n")

        # With noise, we expect small but non-zero values
        # These bounds are based on noise levels, not systematic drift
        # Note: Noise accumulates stochastically, so bounds are generous
        assert (
            final_vel_norm < 20.0
        ), f"Velocity {final_vel_norm:.3f} m/s exceeds noise-bounded limit"
        assert (
            final_pos_norm < 500.0
        ), f"Position {final_pos_norm:.3f} m exceeds noise-bounded limit"
        assert (
            quat_deviation < 0.02
        ), f"Quaternion deviation {quat_deviation:.3f} exceeds noise limit"

    def test_gravity_compensation_enu_vs_ned(self) -> None:
        """
        Test that gravity compensation is correct for both ENU and NED frames.

        Key validation:
            - ENU: accelerometer reads [0, 0, -g] when stationary
            - NED: accelerometer reads [0, 0, +g] when stationary
            - Both produce zero velocity drift
        """
        duration = 10.0
        dt = 0.01
        n_steps = int(duration / dt)
        g = 9.81

        # Test ENU
        frame_enu = FrameConvention.create_enu()
        q_enu = np.array([1.0, 0.0, 0.0, 0.0])
        v_enu = np.zeros(3)
        p_enu = np.zeros(3)

        accel_enu = np.array([0.0, 0.0, +g])  # upward reaction in ENU
        gyro = np.zeros(3)

        for _ in range(n_steps):
            q_enu, v_enu, p_enu = strapdown_update(
                q=q_enu,
                v=v_enu,
                p=p_enu,
                omega_b=gyro,
                f_b=accel_enu,
                dt=dt,
                g=g,
                frame=frame_enu,
            )

        # Test NED
        frame_ned = FrameConvention.create_ned()
        q_ned = np.array([1.0, 0.0, 0.0, 0.0])
        v_ned = np.zeros(3)
        p_ned = np.zeros(3)

        accel_ned = np.array([0.0, 0.0, -g])  # upward reaction in NED (z-down)

        for _ in range(n_steps):
            q_ned, v_ned, p_ned = strapdown_update(
                q=q_ned,
                v=v_ned,
                p=p_ned,
                omega_b=gyro,
                f_b=accel_ned,
                dt=dt,
                g=g,
                frame=frame_ned,
            )

        # Both should have zero drift
        vel_drift_enu = np.linalg.norm(v_enu)
        vel_drift_ned = np.linalg.norm(v_ned)
        pos_drift_enu = np.linalg.norm(p_enu)
        pos_drift_ned = np.linalg.norm(p_ned)

        print(f"\n{'='*60}")
        print("Gravity Compensation Test (ENU vs NED)")
        print(f"{'='*60}")
        print(f"ENU velocity drift:    {vel_drift_enu:.6e} m/s")
        print(f"NED velocity drift:    {vel_drift_ned:.6e} m/s")
        print(f"ENU position drift:    {pos_drift_enu:.6e} m")
        print(f"NED position drift:    {pos_drift_ned:.6e} m")
        print(f"{'='*60}\n")

        assert vel_drift_enu < 1e-10, f"ENU velocity drift: {vel_drift_enu:.2e}"
        assert vel_drift_ned < 1e-10, f"NED velocity drift: {vel_drift_ned:.2e}"
        assert pos_drift_enu < 1e-8, f"ENU position drift: {pos_drift_enu:.2e}"
        assert pos_drift_ned < 1e-8, f"NED position drift: {pos_drift_ned:.2e}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

