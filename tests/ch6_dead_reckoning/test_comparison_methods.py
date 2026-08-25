"""Tests for the Chapter 6 dead-reckoning comparison.

Every method in this comparison is scored against a ground truth that returns
to its own start point, so a method that has silently stopped integrating still
reports a small *final* error and an RMSE equal to the mean distance of the
truth from the origin. Two of the four did exactly that: IMU+ZUPT traced 0.50 m
and PDR 2.33 m against a 100 m walk, while the printed table gave them final
errors of 0.32 m and 1.13 m and the chapter claimed they were tracking.

So these tests pin the thing an error metric cannot see -- that each method
traces a path -- and, underneath it, the trajectory properties the detectors
depend on. Without gait dynamics the ZUPT statistic cannot tell walking from
standing and the step detector has nothing to count, and both failures are
invisible in the error column.

Author: Li-Ta Hsu
References: Chapter 6, Sections 6.1-6.3; Eqs. (6.11), (6.44), (6.46)-(6.50)
"""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch6_dead_reckoning.example_comparison import (
    DEFAULT_SEED,
    LEVER_ARM_A,
    _speed_envelope,
    _turn_profile,
    _wrap_to_pi,
    add_sensor_noise,
    generate_mixed_trajectory,
    plot_comparison,
    run_imu_only,
    run_imu_zupt,
    run_pdr,
    run_wheel_odom,
)
from core.eval import path_length
from core.sensors import (
    FrameConvention,
    IMUNoiseParams,
    NavStateQPVP,
    detect_steps_peak_detector,
    zupt_test_statistic,
)

DT = 0.01
DURATION = 120.0
STEP_FREQ = 1.75

# The rectangular walk the example defines: 30 x 20 m, closing on the start.
TRUTH_PATH_M = 100.0
WAYPOINTS = np.array([[30.0, 0.0], [30.0, 20.0], [0.0, 20.0], [0.0, 0.0]])


class TestMotionProfiles(unittest.TestCase):
    """The shape functions the trajectory is assembled from."""

    def test_speed_envelope_rises_holds_and_falls(self):
        """Zero at both ends, unity in the middle, never outside [0, 1]."""
        duration, ramp, dt = 10.0, 2.0, 0.01
        tau = np.arange(0.0, duration, dt)
        env = _speed_envelope(tau, duration, ramp)

        self.assertAlmostEqual(env[0], 0.0, places=12)
        self.assertAlmostEqual(env[-1], 0.0, delta=1e-3)
        self.assertAlmostEqual(env[len(env) // 2], 1.0, places=12)
        self.assertTrue(np.all((env >= 0.0) & (env <= 1.0)))
        # Monotone on the way up.
        rising = env[: int(ramp / dt)]
        self.assertTrue(np.all(np.diff(rising) >= 0.0))

    def test_speed_envelope_integrates_to_duration_minus_ramp(self):
        """This identity is how the walk phase hits its segment length."""
        duration, ramp, dt = 10.0, 2.0, 0.001
        tau = np.arange(0.0, duration, dt)
        area = float(np.sum(_speed_envelope(tau, duration, ramp)) * dt)
        self.assertAlmostEqual(area, duration - ramp, places=2)

    def test_speed_envelope_edge_cases(self):
        """No ramp is a plain box; an over-long ramp is clipped, not inverted."""
        tau = np.arange(0.0, 5.0, 0.01)
        np.testing.assert_allclose(_speed_envelope(tau, 5.0, 0.0), 1.0)

        clipped = _speed_envelope(tau, 5.0, 99.0)
        self.assertTrue(np.all((clipped >= 0.0) & (clipped <= 1.0)))
        self.assertAlmostEqual(clipped.max(), 1.0, places=6)

    def test_turn_profile_is_a_smoothstep(self):
        """0 to 1, half way at half time, and flat at both ends."""
        duration, dt = 1.5, 0.001
        tau = np.arange(0.0, duration + dt, dt)
        profile = _turn_profile(tau, duration)

        self.assertAlmostEqual(profile[0], 0.0, places=12)
        self.assertAlmostEqual(profile[-1], 1.0, places=6)
        self.assertAlmostEqual(profile[len(profile) // 2], 0.5, places=3)
        self.assertTrue(np.all(np.diff(profile) >= 0.0))
        # Zero rate at both ends is the point: it keeps the gyro continuous.
        rate = np.diff(profile) / dt
        self.assertLess(rate[0], 0.05 * rate.max())
        self.assertLess(rate[-1], 0.05 * rate.max())

    def test_turn_profile_clamps_outside_its_window(self):
        """Sampling past the end must saturate, not run past the target."""
        np.testing.assert_allclose(
            _turn_profile(np.array([-1.0, 3.0]), 1.5), [0.0, 1.0]
        )

    def test_wrap_to_pi(self):
        """Turn deltas have to take the short way round."""
        self.assertAlmostEqual(_wrap_to_pi(0.5), 0.5)
        # pi -> -pi/2 is a left turn of +pi/2, not a right turn of -3pi/2.
        self.assertAlmostEqual(_wrap_to_pi(-np.pi / 2 - np.pi), np.pi / 2, places=12)
        self.assertAlmostEqual(_wrap_to_pi(3 * np.pi), np.pi, places=12)
        self.assertLessEqual(abs(_wrap_to_pi(100.0)), np.pi)


class TestMixedTrajectory(unittest.TestCase):
    """The shared trajectory has to carry a signal for every method."""

    @classmethod
    def setUpClass(cls):
        (
            cls.t,
            cls.pos_true,
            cls.vel_true,
            cls.accel_body,
            cls.gyro_body,
            cls.heading_true,
            cls.mag_body,
            cls.stance,
            cls.wheel_true,
        ) = generate_mixed_trajectory(
            DURATION,
            DT,
            FrameConvention.create_enu(),
            step_freq=STEP_FREQ,
            lever_arm_a=LEVER_ARM_A,
        )

    def test_walk_follows_the_waypoints(self):
        """100 m rectangle, each corner reached, closing on the start."""
        self.assertAlmostEqual(
            path_length(self.pos_true[:, :2]), TRUTH_PATH_M, delta=0.5
        )
        for waypoint in WAYPOINTS:
            distance = np.linalg.norm(self.pos_true[:, :2] - waypoint, axis=1)
            self.assertLess(distance.min(), 0.05, f"never reached waypoint {waypoint}")
        np.testing.assert_allclose(self.pos_true[-1, :2], 0.0, atol=0.05)

    def test_walking_carries_gait_dynamics(self):
        """The vertical bob is what PDR counts and what ZUPT keys off.

        Without it the trajectory is piecewise constant velocity: the specific
        force while walking is identical to the specific force while standing,
        and both detectors are blind.
        """
        magnitude = np.linalg.norm(self.accel_body, axis=1)
        walking = ~self.stance

        # Standing: gravity alone.
        np.testing.assert_allclose(magnitude[self.stance], 9.81, atol=0.05)
        # Walking: swings either side of gravity by roughly the bob amplitude.
        self.assertGreater(magnitude[walking].max(), 11.5)
        self.assertLess(magnitude[walking].min(), 8.1)

    def test_accelerations_and_turn_rates_stay_physical(self):
        """Ramped starts and in-place turns instead of instantaneous ones.

        Stepping the velocity between samples synthesised a 120 m/s^2
        accelerometer spike at every start and stop -- the only thing the old
        step detector ever fired on -- and switching heading between samples
        synthesised a 200 rad/s gyro spike that no first-order quaternion
        integrator can absorb.
        """
        self.assertLess(np.linalg.norm(self.accel_body, axis=1).max(), 15.0)
        self.assertLess(np.linalg.norm(self.gyro_body, axis=1).max(), 3.0)

    def test_heading_is_continuous(self):
        """No wrap discontinuity, or the quaternion's scalar part flips sign.

        ``compute_gyro_body`` differences the raw quaternion components, so a
        sign flip halfway round the rectangle reads as an enormous rate.
        """
        self.assertLess(np.abs(np.diff(self.heading_true)).max(), 0.05)
        # Three left turns: the walk ends 270 deg from where it started.
        self.assertAlmostEqual(np.rad2deg(self.heading_true[-1]), 270.0, delta=0.5)

    def test_wheel_speed_is_forward_while_driving(self):
        """Straight driving reduces to the book's v^S = [0, v, 0] convention.

        The lateral channel is non-zero only while turning, where it carries
        the lever-arm term Eq. (6.11) exists to remove.
        """
        speed = np.linalg.norm(self.vel_true[:, :2], axis=1)
        np.testing.assert_allclose(self.wheel_true[:, 1], speed, atol=1e-9)
        np.testing.assert_allclose(self.wheel_true[:, 2], 0.0, atol=1e-9)

        driving = speed > 1e-9
        np.testing.assert_allclose(self.wheel_true[driving, 0], 0.0, atol=1e-9)
        self.assertGreater(np.abs(self.wheel_true[:, 0]).max(), 1.0)

    def test_noise_free_wheel_odometry_reproduces_the_truth(self):
        """Closes the loop on C_S^A and on the sign of the lever-arm term.

        Fed its own noise-free measurements, Eqs. (6.11)-(6.15) have to return
        the trajectory they were generated from. Any frame or sign error shows
        up here immediately, where the noisy run would just look like drift.
        """
        initial = NavStateQPVP(
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            v=self.vel_true[0],
            p=self.pos_true[0],
        )
        estimate = run_wheel_odom(
            self.t, self.wheel_true, self.gyro_body, initial, LEVER_ARM_A
        )
        error = np.linalg.norm(estimate[:, :2] - self.pos_true[:, :2], axis=1)
        # The residual is the first-order quaternion integrator and the
        # one-sample lag between measurement and update, nothing else.
        self.assertLess(error.max(), 0.05)


class TestDetectors(unittest.TestCase):
    """The two detectors that were silently failing."""

    @classmethod
    def setUpClass(cls):
        cls.imu_params = IMUNoiseParams.consumer_grade()
        (
            cls.t,
            _,
            _,
            accel_body,
            gyro_body,
            _,
            mag_body,
            cls.stance,
            wheel_true,
        ) = generate_mixed_trajectory(
            DURATION,
            DT,
            FrameConvention.create_enu(),
            step_freq=STEP_FREQ,
            lever_arm_a=LEVER_ARM_A,
        )
        cls.accel_meas, cls.gyro_meas, cls.mag_meas, _ = add_sensor_noise(
            accel_body,
            gyro_body,
            mag_body,
            wheel_true,
            DT,
            cls.imu_params,
            seed=DEFAULT_SEED,
        )

    def test_zupt_statistic_separates_standing_from_walking(self):
        """Eq. (6.44) has to discriminate before any threshold can work.

        On the old trajectory T_k had a median of 20.83 while walking against
        20.85 while standing, so no gamma existed that could tell them apart --
        which is why the example's gamma=1e6 accepted every sample, zeroed the
        velocity at every step, and pinned the solution to its start point.
        """
        sigma_a = self.imu_params.accel_vrw_mps_sqrt_s * np.sqrt(1 / DT)
        sigma_g = self.imu_params.gyro_arw_rad_sqrt_s * np.sqrt(1 / DT)

        statistic = np.full(len(self.t), np.nan)
        for k in range(1, len(self.t), 20):  # every 20th sample is plenty
            start, end = max(0, k - 5), min(len(self.t), k + 6)
            statistic[k] = zupt_test_statistic(
                self.accel_meas[start:end],
                self.gyro_meas[start:end],
                sigma_a,
                sigma_g,
            )

        standing = np.nanmedian(statistic[self.stance])
        walking = np.nanmedian(statistic[~self.stance])
        self.assertGreater(
            walking,
            10 * standing,
            f"no separation: {standing:.1f} standing vs {walking:.1f} walking",
        )
        # The example's default threshold has to land inside that gap.
        self.assertLess(standing, 100.0)
        self.assertGreater(walking, 100.0)

    def test_step_detector_counts_the_simulated_gait(self):
        """Eq. (6.46)-(6.47) peaks, not a bare threshold on the raw magnitude.

        The threshold version tested |a| >= 11.0 against a signal whose mean is
        9.81 and found three "steps" in a 100 m walk.
        """
        step_indices, _ = detect_steps_peak_detector(
            self.accel_meas,
            dt=DT,
            g=9.81,
            min_peak_height=1.0,
            min_peak_distance=0.3,
            lowpass_cutoff=5.0,
        )
        expected = np.sum(~self.stance) * DT * STEP_FREQ
        self.assertAlmostEqual(len(step_indices), expected, delta=0.1 * expected)
        self.assertEqual(
            np.sum(self.stance[step_indices]),
            0,
            "steps detected while the walker was standing still",
        )


class TestMethodsActuallyTrack(unittest.TestCase):
    """The headline pin: every method has to trace a path, not sit still."""

    @classmethod
    def setUpClass(cls):
        frame = FrameConvention.create_enu()
        imu_params = IMUNoiseParams.consumer_grade()

        (
            cls.t,
            cls.pos_true,
            vel_true,
            accel_body,
            gyro_body,
            _,
            mag_body,
            _,
            wheel_true,
        ) = generate_mixed_trajectory(
            DURATION,
            DT,
            frame,
            step_freq=STEP_FREQ,
            lever_arm_a=LEVER_ARM_A,
        )
        accel_meas, gyro_meas, mag_meas, wheel_meas = add_sensor_noise(
            accel_body,
            gyro_body,
            mag_body,
            wheel_true,
            DT,
            imu_params,
            seed=DEFAULT_SEED,
        )
        initial = NavStateQPVP(
            q=np.array([1.0, 0.0, 0.0, 0.0]), v=vel_true[0], p=cls.pos_true[0]
        )

        zupt_pos, _ = run_imu_zupt(
            cls.t, accel_meas, gyro_meas, initial, frame, imu_params
        )
        pdr_pos, cls.step_count = run_pdr(cls.t, accel_meas, mag_meas, 1.75)
        cls.results = {
            "IMU Only": run_imu_only(cls.t, accel_meas, gyro_meas, initial, frame),
            "IMU + ZUPT": zupt_pos,
            "Wheel Odom": run_wheel_odom(
                cls.t, wheel_meas, gyro_meas, initial, LEVER_ARM_A
            ),
            "PDR (Mag)": pdr_pos,
        }

    def test_every_method_traces_a_comparable_path(self):
        """A frozen method traced 0.5% of the truth; a tracking one traces ~1x.

        The band is deliberately wide in both directions -- this is a check
        that a method is integrating at all, not a grade on its accuracy.
        Unaided strapdown is *supposed* to wander further than the truth
        (170 m here), and a ZUPT solution whose velocity keeps being zeroed
        traces less (85 m). The frozen versions traced 0.17 m and 0.00 m, so
        the floor has two orders of magnitude of room.
        """
        for name, pos in self.results.items():
            with self.subTest(method=name):
                path = path_length(pos[:, :2])
                self.assertGreater(
                    path,
                    0.5 * TRUTH_PATH_M,
                    f"{name} traced only {path:.2f} m of a "
                    f"{TRUTH_PATH_M:.0f} m walk",
                )
                self.assertLess(path, 3.0 * TRUTH_PATH_M)

    def test_every_method_covers_the_extent_of_the_walk(self):
        """Path length alone could be spent jittering in place.

        The complement to ``motion_ratio``, which test_methods_actually_move.py
        owns: a track can accumulate the right total distance without ever
        going anywhere. The truth reaches 36 m from the origin; the two frozen
        methods reached 0.33 m and 1.18 m.
        """
        truth_reach = np.linalg.norm(self.pos_true[:, :2], axis=1).max()
        for name, pos in self.results.items():
            with self.subTest(method=name):
                reach = np.linalg.norm(pos[:, :2], axis=1).max()
                self.assertGreater(
                    reach,
                    0.5 * truth_reach,
                    f"{name} never got further than {reach:.2f} m from the "
                    f"origin ({truth_reach:.1f} m expected)",
                )

    def test_wheel_odometry_is_not_rotated(self):
        """C_S^A must map the speed frame's y=forward onto the body x-axis.

        Left at its identity default the forward speed lands on the body
        y-axis and the whole track comes out rotated 90 deg: x spanned
        [-52.11, 0.01] and y [0.00, 41.92] where the truth is x [0, 30],
        y [0, 20].
        """
        pos = self.results["Wheel Odom"]
        self.assertGreater(pos[:, 0].max(), 25.0)  # walks East first
        self.assertGreater(pos[:, 1].max(), 15.0)  # then North
        self.assertGreater(pos[:, 0].min(), -2.0)  # never West of the start
        self.assertLess(pos[:, 1].max(), 25.0)

    def test_pdr_detects_a_plausible_number_of_steps(self):
        """Roughly one step per 0.6 m, per Eq. (6.49) at this gait."""
        self.assertGreater(self.step_count, 100)
        self.assertLess(self.step_count, 250)

    def test_corrections_beat_unaided_strapdown(self):
        """The chapter's whole claim, now that every method actually runs."""
        rmse = {
            name: float(
                np.sqrt(
                    np.mean(np.sum((pos[:, :2] - self.pos_true[:, :2]) ** 2, axis=1))
                )
            )
            for name, pos in self.results.items()
        }
        for corrected in ["IMU + ZUPT", "Wheel Odom", "PDR (Mag)"]:
            with self.subTest(method=corrected):
                self.assertLess(rmse[corrected], 0.5 * rmse["IMU Only"])
        # Bounded methods should stay within a few percent of the distance.
        self.assertLess(rmse["Wheel Odom"], 0.05 * TRUTH_PATH_M)
        self.assertLess(rmse["PDR (Mag)"], 0.05 * TRUTH_PATH_M)

    def test_plot_comparison_reports_path_and_writes_every_figure(self):
        """The path metric is what makes a frozen method visible in the table."""
        with tempfile.TemporaryDirectory() as tmp:
            metrics = plot_comparison(self.t, self.pos_true, self.results, Path(tmp))
            for name in [
                "comparison_trajectories",
                "comparison_error_time",
                "comparison_error_cdf",
            ]:
                for suffix in ["svg", "pdf", "png"]:
                    self.assertTrue(
                        (Path(tmp) / f"{name}.{suffix}").exists(),
                        f"{name}.{suffix} was not written",
                    )

        for name, pos in self.results.items():
            with self.subTest(method=name):
                self.assertAlmostEqual(
                    metrics[name]["path"],
                    path_length(pos[:, :2]),
                    places=6,
                )


if __name__ == "__main__":
    unittest.main()
