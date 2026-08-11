"""The magnetometer figure claims bounded heading except under disturbance.

`environment_mag_heading.png` exists to show one thing: a magnetometer gives an
absolute heading that does not drift, and indoor disturbances are what break
it. Two shaded windows mark the disturbances, and a 10 degree threshold line
marks "good".

The committed figure showed an error sawtoothing between 0 and 180 degrees
across the *whole* run, disturbance or not. The shaded windows were
indistinguishable from the baseline and the threshold line was crossed
constantly. The claim was not merely unsupported, it was contradicted, and no
test looked because every test here checks that files were written.

The cause was the same sign convention that `generate_ch6_env_sensors_dataset`
already had fixed: the example built its field as
`R_body_to_map.T @ [1, 0, 0]`, for which `atan2(m_y, m_x)` returns exactly
minus the heading. The example generates its own data inline, so fixing the
dataset generator had not touched it.

These tests assert the claim rather than the rendering: quiet stretches must be
accurate, disturbed ones must be visibly worse, and the difference must be
large enough that the figure's shading means something.

Author: Li-Ta Hsu
References: Chapter 6, Eqs. (6.51)-(6.53)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch6_dead_reckoning.example_environment import (
    DEFAULT_SEED,
    add_env_sensor_noise,
    generate_building_walk,
    run_mag_heading,
)

# The example shades these windows as "disturbance zones".
DISTURBANCE_WINDOWS = ((30.0, 50.0), (100.0, 120.0))


def _heading_error_deg():
    rng = np.random.default_rng(DEFAULT_SEED)
    t, _, att_true, mag_true, pressure_true, _ = generate_building_walk(rng=rng)
    mag_meas, _ = add_env_sensor_noise(mag_true, pressure_true, t, dt=0.1, rng=rng)
    heading_est = run_mag_heading(t, mag_meas, att_true)

    truth = att_true[:, 2]
    diff = heading_est - truth
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return t, np.abs(np.rad2deg(wrapped))


def _disturbed_mask(t):
    mask = np.zeros(len(t), dtype=bool)
    for lo, hi in DISTURBANCE_WINDOWS:
        mask |= (t >= lo) & (t < hi)
    return mask


class TestMagHeadingIsBounded(unittest.TestCase):

    def test_heading_is_accurate_when_undisturbed(self):
        """The absolute-reference claim, on the stretches that carry it.

        The old sign error put this median near 90 degrees.
        """
        t, error = _heading_error_deg()
        quiet = error[~_disturbed_mask(t)]

        self.assertLess(float(np.median(quiet)), 10.0)
        self.assertLess(float(np.percentile(quiet, 95)), 25.0)

    def test_heading_does_not_drift(self):
        """Absolute means the second half is no worse than the first.

        A gyro would fail this; that contrast is the point of the section.
        """
        t, error = _heading_error_deg()
        quiet = ~_disturbed_mask(t)
        first = error[quiet & (t < t[-1] / 2)]
        second = error[quiet & (t >= t[-1] / 2)]

        self.assertLess(float(np.median(second)), float(np.median(first)) + 5.0)

    def test_disturbances_are_visibly_worse_than_the_baseline(self):
        """Otherwise the shaded windows in the figure mean nothing."""
        t, error = _heading_error_deg()
        disturbed = _disturbed_mask(t)

        quiet_med = float(np.median(error[~disturbed]))
        loud_med = float(np.median(error[disturbed]))

        self.assertGreater(loud_med, 3.0 * quiet_med)


if __name__ == "__main__":
    unittest.main()
