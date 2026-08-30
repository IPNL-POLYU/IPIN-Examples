"""Two defects in the wheel odometry example, both silent, both measurable.

First, the speed frame. core.sensors.wheel_odom_update documents its input as
v_s = [0, v_forward, 0] -- the book's y-forward speed frame -- and defaults
C_S_A, the rotation from that frame into the attitude frame, to the identity.
The attitude frame is not y-forward: its quaternion is a yaw about the ENU
heading, so x points forward. Taking the default therefore rotates the whole
track by 90 degrees. The example reported 30.00 m of final error on a 270 m
square, 11.1% of distance, directly above the line "KEY INSIGHT: Wheel odometry
drift is BOUNDED! Errors ~1-5% of distance". Passing the correct C_S_A gives
2.32 m, or 0.9%. Chapter 6's comparison example already carried that constant.

Second, the slip. The example advertises "sensitivity to wheel slip" and
injected 30% slip over four windows labelled "during turns". This trajectory
turns in place, so the forward speed through every one of those windows is
exactly 0.000 m/s and the slip multiplied nothing: the no-slip and slip runs
printed identical errors to three significant figures. Moved onto the
straights, the slip adds 12.3 m of phantom travel and separates the tracks by
about 4 m.

Author: Li-Ta Hsu
References: Chapter 6, Section 6.2, Eqs. (6.11)-(6.15)
"""

import contextlib
import io
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch6_dead_reckoning.example_wheel_odometry import (
    C_SPEED_TO_BODY,
    add_wheel_noise,
    generate_vehicle_trajectory,
    run_wheel_odometry,
)
from core.eval import path_length
from core.sensors import NavStateQPVP

LEVER_ARM = np.array([1.5, 0.0, -0.3])
SLIP_WINDOWS = [(2, 4), (8, 10), (14, 16), (20, 22)]


def _setup():
    """The example's own square scenario, quietly."""
    with contextlib.redirect_stdout(io.StringIO()):
        return generate_vehicle_trajectory(shape="square")


def _track(t, wheel, gyro, truth, vel, quat):
    """Run the odometry and return the estimated track."""
    initial = NavStateQPVP(q=quat[0].copy(), v=vel[0].copy(), p=truth[0].copy())
    with contextlib.redirect_stdout(io.StringIO()):
        return np.asarray(run_wheel_odometry(t, wheel, gyro, initial, LEVER_ARM))


class TestSpeedFrameConvention(unittest.TestCase):
    """The rotation between the speed frame and the attitude frame."""

    @classmethod
    def setUpClass(cls):
        cls.t, cls.truth, cls.vel, cls.quat, cls.wheel, cls.gyro = _setup()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.wheel_meas, cls.gyro_meas = add_wheel_noise(cls.wheel, cls.gyro)

    def test_the_speed_frame_is_y_forward(self):
        """The premise, taken from the data rather than assumed."""
        means = np.abs(self.wheel.mean(axis=0))

        self.assertEqual(int(np.argmax(means)), 1)

    def test_the_rotation_is_ninety_degrees_about_z(self):
        """C_S_A must map speed-frame y onto attitude-frame x.

        Pinned as a matrix identity rather than a number, so that a future
        change to either convention fails here with the reason visible.
        """
        forward_in_speed_frame = np.array([0.0, 1.0, 0.0])

        np.testing.assert_allclose(
            C_SPEED_TO_BODY @ forward_in_speed_frame,
            np.array([1.0, 0.0, 0.0]),
            atol=1e-12,
        )

    def test_the_example_stays_within_the_accuracy_it_claims(self):
        """0.9% of distance, against the printed claim of 1-5%.

        With C_S_A left at its identity default this was 11.1%, printed
        directly above a line asserting bounded drift of 1-5%.
        """
        track = _track(
            self.t, self.wheel_meas, self.gyro_meas, self.truth, self.vel, self.quat
        )
        distance = float(np.sum(np.linalg.norm(np.diff(self.truth, axis=0), axis=1)))
        final_error = float(np.linalg.norm(track[-1, :2] - self.truth[-1, :2]))

        self.assertLess(final_error / distance, 0.05)


class TestSlipActuallySlips(unittest.TestCase):
    """The demonstration the module docstring advertises."""

    @classmethod
    def setUpClass(cls):
        cls.t, cls.truth, cls.vel, cls.quat, cls.wheel, cls.gyro = _setup()

    def test_the_turns_are_stationary(self):
        """Why the original slip windows injected nothing.

        Guards the fix at its cause: if the trajectory ever gains rolling
        turns, windows placed on the straights stop being the only option and
        this test says so.
        """
        forward = self.wheel[:, 1]
        for low, high in [(4, 6), (10, 12), (16, 18), (22, 24)]:
            with self.subTest(window=(low, high)):
                during = forward[(self.t >= low) & (self.t < high)]
                self.assertLess(float(np.abs(during).max()), 1e-9)

    def test_slip_windows_land_where_the_vehicle_is_moving(self):
        """The windows the example now uses are on the straights."""
        forward = self.wheel[:, 1]
        for low, high in SLIP_WINDOWS:
            with self.subTest(window=(low, high)):
                during = forward[(self.t >= low) & (self.t < high)]
                self.assertGreater(float(during.mean()), 1.0)

    def test_slip_changes_the_track(self):
        """It must cost something, and roughly what was injected.

        30% over four 2 s windows at 5 m/s is 12 m of phantom travel; the
        estimated path should lengthen by about that. Bracketed loosely
        because noise and the lever arm contribute too.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            clean_w, clean_g = add_wheel_noise(self.wheel, self.gyro)
            slip_w, slip_g = add_wheel_noise(
                self.wheel, self.gyro, add_slip=True, slip_intervals=SLIP_WINDOWS
            )
        clean = _track(self.t, clean_w, clean_g, self.truth, self.vel, self.quat)
        slipped = _track(self.t, slip_w, slip_g, self.truth, self.vel, self.quat)

        extra = path_length(slipped) - path_length(clean)
        self.assertGreater(extra, 6.0)
        self.assertLess(extra, 20.0)
        self.assertGreater(np.linalg.norm(slipped - clean, axis=1).max(), 1.0)


if __name__ == "__main__":
    unittest.main()
