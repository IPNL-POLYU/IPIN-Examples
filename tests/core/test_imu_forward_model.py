"""
Unit tests for IMU forward model (trajectory → accelerometer/gyro measurements).

Validates that the synthetic IMU generation is consistent with Eq. 6.7 and 6.9.

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
import pytest

from core.sim import (
    compute_specific_force_body,
    compute_gyro_body,
    generate_imu_from_trajectory,
)
from core.sensors import FrameConvention, strapdown_update


class TestIMUForwardModel:
    """Test suite for IMU forward model."""

    def test_stationary_specific_force_enu(self) -> None:
        """
        Test that stationary device generates correct specific force in ENU.

        For stationary in ENU with identity rotation:
            - True acceleration: a_M = [0, 0, 0]
            - Specific force: f_b = [0, 0, +9.81] (upward reaction)
        """
        frame = FrameConvention.create_enu()

        # Stationary: zero acceleration
        a_M = np.zeros(3)

        # Identity rotation (body aligned with map)
        q = np.array([1.0, 0.0, 0.0, 0.0])

        # Compute specific force
        f_b = compute_specific_force_body(a_M, q, frame=frame, g=9.81)

        # Expected: upward reaction force
        expected = np.array([0.0, 0.0, +9.81])

        print(f"\nStationary ENU:")
        print(f"  True accel:     {a_M}")
        print(f"  Specific force: {f_b}")
        print(f"  Expected:       {expected}")

        np.testing.assert_allclose(f_b, expected, atol=1e-10)

    def test_stationary_specific_force_ned(self) -> None:
        """
        Test that stationary device generates correct specific force in NED.

        For stationary in NED with identity rotation:
            - True acceleration: a_M = [0, 0, 0]
            - Specific force: f_b = [0, 0, -9.81] (upward reaction in NED)
        """
        frame = FrameConvention.create_ned()

        # Stationary: zero acceleration
        a_M = np.zeros(3)

        # Identity rotation
        q = np.array([1.0, 0.0, 0.0, 0.0])

        # Compute specific force
        f_b = compute_specific_force_body(a_M, q, frame=frame, g=9.81)

        # Expected: upward reaction (negative z in NED)
        expected = np.array([0.0, 0.0, -9.81])

        print(f"\nStationary NED:")
        print(f"  True accel:     {a_M}")
        print(f"  Specific force: {f_b}")
        print(f"  Expected:       {expected}")

        np.testing.assert_allclose(f_b, expected, atol=1e-10)

    def test_constant_acceleration_horizontal(self) -> None:
        """
        Test horizontal acceleration (no gravity component).

        For horizontal acceleration in ENU:
            - a_M = [1, 0, 0] (1 m/s² eastward)
            - With identity rotation: f_b = [1, 0, +9.81]
        """
        frame = FrameConvention.create_enu()

        # Horizontal acceleration eastward
        a_M = np.array([1.0, 0.0, 0.0])

        # Identity rotation
        q = np.array([1.0, 0.0, 0.0, 0.0])

        # Compute specific force
        f_b = compute_specific_force_body(a_M, q, frame=frame, g=9.81)

        # Expected: horizontal accel + vertical reaction
        expected = np.array([1.0, 0.0, +9.81])

        print(f"\nHorizontal accel ENU:")
        print(f"  True accel:     {a_M}")
        print(f"  Specific force: {f_b}")
        print(f"  Expected:       {expected}")

        np.testing.assert_allclose(f_b, expected, atol=1e-10)

    def test_free_fall(self) -> None:
        """
        Test free fall (accelerating downward at g).

        For free fall in ENU:
            - a_M = [0, 0, -9.81] (falling downward)
            - Specific force: f_b = [0, 0, 0] (weightless!)
        """
        frame = FrameConvention.create_enu()

        # Free fall: accelerating downward at g
        a_M = np.array([0.0, 0.0, -9.81])

        # Identity rotation
        q = np.array([1.0, 0.0, 0.0, 0.0])

        # Compute specific force
        f_b = compute_specific_force_body(a_M, q, frame=frame, g=9.81)

        # Expected: zero (weightless)
        expected = np.zeros(3)

        print(f"\nFree fall ENU:")
        print(f"  True accel:     {a_M}")
        print(f"  Specific force: {f_b}")
        print(f"  Expected:       {expected}")

        np.testing.assert_allclose(f_b, expected, atol=1e-10)

    def test_round_trip_consistency(self) -> None:
        """
        Test round-trip: trajectory → IMU → strapdown → trajectory.

        This validates that the forward model is the inverse of strapdown integration.
        """
        frame = FrameConvention.create_enu()
        dt = 0.01
        N = 100

        # Simple trajectory: constant velocity eastward
        t = np.linspace(0, 1, N)
        vel_const = 1.0  # m/s eastward
        pos_map = np.column_stack([vel_const * t, np.zeros(N), np.zeros(N)])
        vel_map = np.column_stack([vel_const * np.ones(N), np.zeros(N), np.zeros(N)])
        quat_b_to_m = np.tile([1, 0, 0, 0], (N, 1))  # identity

        # Generate IMU measurements
        accel_b, gyro_b = generate_imu_from_trajectory(
            pos_map, vel_map, quat_b_to_m, dt, frame=frame
        )

        # Integrate back using strapdown
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v = vel_map[0].copy()
        p = pos_map[0].copy()

        for i in range(1, N):
            q, v, p = strapdown_update(
                q=q,
                v=v,
                p=p,
                omega_b=gyro_b[i - 1],
                f_b=accel_b[i - 1],
                dt=dt,
                g=9.81,
                frame=frame,
            )

        # Final state should match ground truth
        vel_error = np.linalg.norm(v - vel_map[-1])
        pos_error = np.linalg.norm(p - pos_map[-1])

        print(f"\nRound-trip test:")
        print(f"  Final velocity error: {vel_error:.6e} m/s")
        print(f"  Final position error: {pos_error:.6e} m")
        print(f"  True final pos:       {pos_map[-1]}")
        print(f"  Integrated final pos: {p}")

        # Should be small (numerical integration errors accumulate)
        # Velocity should be nearly exact (no integration in forward model)
        assert vel_error < 1e-10, f"Velocity error too large: {vel_error:.2e}"
        # Position error accumulates from Euler integration (relaxed tolerance)
        assert pos_error < 0.02, f"Position error too large: {pos_error:.2f} m"

    def test_gyro_from_constant_yaw_rate(self) -> None:
        """
        Test gyro computation from constant yaw rate trajectory.

        For planar motion with constant yaw rate:
            - ω_z should be constant
            - ω_x and ω_y should be zero
        """
        dt = 0.01
        N = 100
        yaw_rate = 0.1  # rad/s

        # Generate quaternion trajectory with constant yaw rate
        t = np.linspace(0, 1, N)
        yaw = yaw_rate * t
        quat = np.column_stack(
            [
                np.cos(yaw / 2),
                np.zeros(N),
                np.zeros(N),
                np.sin(yaw / 2),
            ]
        )

        # Compute gyro
        omega_b = compute_gyro_body(quat, dt)

        # Expected: constant z-rate (after initial transient)
        omega_z_mean = np.mean(omega_b[10:, 2])  # skip first few samples
        omega_xy_mean = np.mean(np.abs(omega_b[10:, :2]))

        print(f"\nConstant yaw rate:")
        print(f"  Expected omega_z: {yaw_rate:.6f} rad/s")
        print(f"  Computed omega_z: {omega_z_mean:.6f} rad/s")
        print(f"  Mean |omega_xy|:  {omega_xy_mean:.6e} rad/s")

        # Should match expected yaw rate
        assert np.abs(omega_z_mean - yaw_rate) < 0.01, f"Yaw rate error too large"
        assert omega_xy_mean < 1e-3, f"Spurious x/y rates: {omega_xy_mean:.2e}"

    def test_time_series_specific_force(self) -> None:
        """
        Test specific force computation for time series (multiple samples).
        """
        frame = FrameConvention.create_enu()
        N = 10

        # Time-varying acceleration (sinusoidal)
        t = np.linspace(0, 1, N)
        accel_map = np.column_stack(
            [
                np.sin(2 * np.pi * t),  # varying x
                np.zeros(N),  # constant y
                np.zeros(N),  # constant z
            ]
        )

        # Constant identity rotation
        quat = np.tile([1, 0, 0, 0], (N, 1))

        # Compute specific force
        f_b = compute_specific_force_body(accel_map, quat, frame=frame)

        # Expected: accel + vertical reaction
        expected = accel_map.copy()
        expected[:, 2] = 9.81  # add vertical reaction

        print(f"\nTime series:")
        print(f"  First sample accel: {accel_map[0]}")
        print(f"  First sample f_b:   {f_b[0]}")
        print(f"  Expected:           {expected[0]}")

        np.testing.assert_allclose(f_b, expected, atol=1e-10)

    def test_no_free_fall_planar_motion(self) -> None:
        """
        Test acceptance criterion: planar motion should not show free fall.

        For level planar trajectory (no vertical motion):
            - Vertical acceleration should be zero
            - Vertical specific force should be ≈ +g (not zero)
        """
        frame = FrameConvention.create_enu()
        dt = 0.01
        N = 100

        # Circular planar motion (no z-movement)
        t = np.linspace(0, 2 * np.pi, N)
        radius = 5.0
        omega = 1.0  # rad/s

        pos_map = np.column_stack([radius * np.cos(omega * t), radius * np.sin(omega * t), np.zeros(N)])
        vel_map = np.column_stack(
            [-radius * omega * np.sin(omega * t), radius * omega * np.cos(omega * t), np.zeros(N)]
        )

        # Yaw follows velocity direction
        yaw = np.arctan2(vel_map[:, 1], vel_map[:, 0])
        quat_b_to_m = np.column_stack(
            [np.cos(yaw / 2), np.zeros(N), np.zeros(N), np.sin(yaw / 2)]
        )

        # Generate IMU
        accel_b, gyro_b = generate_imu_from_trajectory(
            pos_map, vel_map, quat_b_to_m, dt, frame=frame
        )

        # Vertical channel should NOT be zero (would indicate free fall)
        accel_z_mean = np.mean(accel_b[:, 2])
        accel_z_std = np.std(accel_b[:, 2])

        print(f"\nPlanar motion (acceptance test):")
        print(f"  Vertical accel mean: {accel_z_mean:.3f} m/s²")
        print(f"  Vertical accel std:  {accel_z_std:.3f} m/s²")
        print(f"  Expected:            ~9.81 m/s² (no free fall)")

        # Should be close to g (not zero!)
        assert (
            accel_z_mean > 5.0
        ), f"Vertical channel too low: {accel_z_mean:.2f} (free fall?)"
        assert accel_z_mean < 15.0, f"Vertical channel too high: {accel_z_mean:.2f}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

