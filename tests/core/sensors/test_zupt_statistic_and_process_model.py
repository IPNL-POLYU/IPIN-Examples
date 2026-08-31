"""Conformance tests for the two Chapter 6 equations the index had misfiled.

`docs/equation_index.yml` used to point Eq. (6.44) at `detect_zupt` -- the
threshold pair, not the windowed test statistic the equation defines -- and
Eqs. (6.17)-(6.32) at `NavStateQPVPBias`, a dataclass whose entire body is
shape validation. Both entries carried a `verified_by`, so `--strict` was
green: the index checks that a test *exists*, not that it exercises the
equation. Re-pointing them at the real implementations left neither with a
conformance test, and this file is that.

Author: Li-Ta Hsu
"""

import numpy as np

from core.sensors import (
    FrameConvention,
    IMUNoiseParams,
    detect_zupt_windowed,
    zupt_test_statistic,
)
from core.sensors.ins_ekf import ZUPT_EKF


class TestZuptTestStatisticEq644:
    """Eq. (6.44): T_k = (1/N) Σ [ |a_l - g â_k| / σ_A + |ω_l|² / σ_G ]."""

    def test_matches_the_closed_form(self) -> None:
        """Written out here so the code and the equation can be read together."""
        rng = np.random.default_rng(3)
        accel = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.05, (8, 3))
        gyro = rng.normal(0, 0.01, (8, 3))
        sigma_a, sigma_g, gravity = 0.02, 0.004, 9.81

        mean_accel = accel.mean(axis=0)
        gravity_direction = gravity * mean_accel / np.linalg.norm(mean_accel)
        expected = float(
            np.mean(
                np.linalg.norm(accel - gravity_direction, axis=1) / sigma_a
                + np.sum(gyro**2, axis=1) / sigma_g
            )
        )

        computed = zupt_test_statistic(
            accel, gyro, g=gravity, sigma_a=sigma_a, sigma_g=sigma_g
        )
        assert np.isclose(computed, expected, rtol=1e-12)

    def test_a_moving_window_scores_higher_than_a_still_one(self) -> None:
        """The property the detector rests on, and it is not automatic.

        The statistic subtracts the window's *own* mean direction, so a window
        that is merely tilted still scores low. What raises it is disagreement
        within the window -- which is what motion is.
        """
        rng = np.random.default_rng(11)
        still_accel = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.02, (10, 3))
        still_gyro = rng.normal(0, 0.005, (10, 3))
        tilted = np.array([3.0, 0.0, 9.34]) + rng.normal(0, 0.02, (10, 3))
        moving_accel = still_accel + rng.normal(0, 2.0, (10, 3))
        moving_gyro = still_gyro + rng.normal(0, 1.0, (10, 3))

        kwargs = {"g": 9.81, "sigma_a": 0.02, "sigma_g": 0.005}
        still = zupt_test_statistic(still_accel, still_gyro, **kwargs)
        tilt = zupt_test_statistic(tilted, still_gyro, **kwargs)
        moving = zupt_test_statistic(moving_accel, moving_gyro, **kwargs)

        assert moving > 10 * still, f"moving {moving:.1f} vs still {still:.1f}"
        assert tilt < 2 * still, "a stationary but tilted window must still score low"

    def test_the_detector_is_the_statistic_against_the_threshold(self) -> None:
        """`detect_zupt_windowed` is T_k < gamma, and nothing else."""
        rng = np.random.default_rng(5)
        accel = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.05, (10, 3))
        gyro = rng.normal(0, 0.01, (10, 3))
        kwargs = {"g": 9.81, "sigma_a": 0.02, "sigma_g": 0.004}

        statistic = zupt_test_statistic(accel, gyro, **kwargs)
        assert detect_zupt_windowed(accel, gyro, gamma=statistic * 1.01, **kwargs)
        assert not detect_zupt_windowed(accel, gyro, gamma=statistic * 0.99, **kwargs)


class TestInsProcessModelEq617:
    """Eqs. (6.17)-(6.32): the EKF process model and covariance propagation."""

    def _filter(self):
        return ZUPT_EKF(
            frame=FrameConvention.create_enu(),
            imu_params=IMUNoiseParams.consumer_grade(),
            sigma_zupt=0.001,
        )

    def test_a_stationary_prediction_does_not_move_the_state(self) -> None:
        """The mechanization half: level, still, correct specific force."""
        ekf = self._filter()
        state = ekf.initialize(
            p0=np.zeros(3), v0=np.zeros(3), q0=np.array([1.0, 0, 0, 0])
        )
        for _ in range(500):
            state = ekf.predict(state, np.zeros(3), np.array([0.0, 0.0, 9.81]), 0.01)

        assert np.abs(state.v).max() < 1e-10, f"velocity drifted to {state.v}"
        assert np.abs(state.p).max() < 1e-8, f"position drifted to {state.p}"

    def test_estimated_biases_are_removed_before_mechanization(self) -> None:
        """Eq. (6.6) / (6.9): predict must subtract b_g and b_a, not ignore them."""
        ekf = self._filter()
        bias = np.array([0.0, 0.0, 0.3])
        state = ekf.initialize(
            p0=np.zeros(3), v0=np.zeros(3), q0=np.array([1.0, 0, 0, 0])
        )
        state.b_a = bias.copy()
        for _ in range(500):
            state = ekf.predict(
                state, np.zeros(3), np.array([0.0, 0.0, 9.81]) + bias, 0.01
            )

        assert (
            np.abs(state.v).max() < 1e-10
        ), f"velocity drifted to {state.v}: the bias state is not being applied"

    def test_covariance_grows_and_stays_symmetric_positive(self) -> None:
        """The covariance half. Simplified (P + Q, no F) -- but it must still grow."""
        ekf = self._filter()
        state = ekf.initialize(
            p0=np.zeros(3), v0=np.zeros(3), q0=np.array([1.0, 0, 0, 0])
        )
        trace_before = float(np.trace(state.P))
        for _ in range(100):
            state = ekf.predict(state, np.zeros(3), np.array([0.0, 0.0, 9.81]), 0.01)

        assert np.trace(state.P) > trace_before
        assert np.allclose(state.P, state.P.T)
        assert np.all(np.diag(state.P) >= 0.0)

    def test_process_noise_scales_with_the_imu_grade(self) -> None:
        """Q comes from IMUNoiseParams, so a better IMU must give a smaller Q."""
        frame = FrameConvention.create_enu()
        consumer = ZUPT_EKF(frame=frame, imu_params=IMUNoiseParams.consumer_grade())
        tactical = ZUPT_EKF(frame=frame, imu_params=IMUNoiseParams.tactical_grade())

        q_consumer = consumer.compute_process_noise(0.01)
        q_tactical = tactical.compute_process_noise(0.01)

        assert np.trace(q_tactical) < np.trace(q_consumer)
        # Velocity block scales as VRW^2, i.e. the square of the ratio.
        ratio = (
            IMUNoiseParams.consumer_grade().accel_vrw_mps_sqrt_s
            / IMUNoiseParams.tactical_grade().accel_vrw_mps_sqrt_s
        )
        assert np.isclose(q_consumer[3, 3] / q_tactical[3, 3], ratio**2, rtol=1e-9)
