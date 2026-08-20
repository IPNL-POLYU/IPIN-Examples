"""The observability demo's numbers must match the theory it teaches.

Section 8.2's claim is structural: translation is unobservable from odometry
alone, so an initial position offset survives to the end of the run untouched.
That is falsifiable -- the final error should equal the offset magnitude, give
or take the drift odometry accumulates anyway -- and the demo now prints the
predicted and measured values side by side instead of printing the theoretical
one formatted as though it were the result.

The second claim is the one most easily lost: absolute fixes buy observability,
not precision. Here they leave a larger error than offset-free odometry does,
because each fix carries more noise than the odometry drifts over a run this
short. Stating only "corrected by absolute measurements" invites the opposite
conclusion.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.2 (Observability), Eq. (8.3)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch8_sensor_fusion.example_observability import (
    generate_absolute_fixes,
    generate_odometry_measurements,
    generate_trajectory,
    run_odometry_only_fusion,
    run_odometry_with_fixes_fusion,
)

TRANSLATION_OFFSET = np.array([3.0, 2.0])


def _final_error(result, trajectory):
    """Distance between the final estimate and the final truth."""
    est = np.asarray(result["x_est"])[-1, :2]
    return float(np.linalg.norm(est - trajectory["p_xy"][-1]))


class TestObservabilityClaims(unittest.TestCase):
    """Theory against measurement, for both directions of the claim."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.trajectory = generate_trajectory()
        cls.odometry = generate_odometry_measurements(cls.trajectory)
        cls.fixes = generate_absolute_fixes(cls.trajectory, fix_rate=1.0, noise_std=0.5)

        cls.no_offset = run_odometry_only_fusion(
            cls.trajectory, cls.odometry, translation_offset=np.zeros(2)
        )
        cls.with_offset = run_odometry_only_fusion(
            cls.trajectory, cls.odometry, translation_offset=TRANSLATION_OFFSET
        )
        cls.with_fixes = run_odometry_with_fixes_fusion(
            cls.trajectory,
            cls.odometry,
            cls.fixes,
            translation_offset=TRANSLATION_OFFSET,
        )

    def test_offset_survives_because_translation_is_unobservable(self):
        """The measured error must match the offset magnitude, within drift.

        This is the quantitative form of "unobservable". An estimator that
        somehow corrected the offset, or one that diverged, both fail here --
        and either would falsify the section's claim rather than illustrate it.
        """
        predicted = float(np.linalg.norm(TRANSLATION_OFFSET))
        measured = _final_error(self.with_offset, self.trajectory)
        drift = _final_error(self.no_offset, self.trajectory)

        self.assertAlmostEqual(measured, predicted, delta=max(2.0 * drift, 0.5))

    def test_odometry_alone_barely_drifts(self):
        """The drift term above has to be small, or the test proves nothing.

        If odometry drifted metres, "the offset survived" would be
        indistinguishable from "the estimate wandered by coincidence".
        """
        self.assertLess(_final_error(self.no_offset, self.trajectory), 1.0)

    def test_absolute_fixes_remove_the_offset(self):
        """What observability buys: the unobservable direction is corrected."""
        with_offset = _final_error(self.with_offset, self.trajectory)
        with_fixes = _final_error(self.with_fixes, self.trajectory)

        self.assertLess(with_fixes, 0.25 * with_offset)

    def test_absolute_fixes_do_not_buy_precision(self):
        """What it does not buy, pinned so the caveat cannot quietly vanish.

        Each fix carries 0.5 m of noise, more than odometry drifts over this
        run, so offset-free odometry ends up closer to the truth than the
        fixed solution does. The demo says so; if a future change makes the
        fixes cleaner this fails and the wording has to be revisited.
        """
        self.assertLess(
            _final_error(self.no_offset, self.trajectory),
            _final_error(self.with_fixes, self.trajectory),
        )


if __name__ == "__main__":
    unittest.main()
