"""AOA position error is linear in angular noise, and the solver can lie.

The example used to solve once per noise level and tabulate the result. Its
own numbers then contradicted the physics: 1.74 m at 5 deg and 2.04 m at
10 deg, an apparent saturation, when doubling the angular noise should double
the position error. Single draws, tabulated as if they were a trend.

Averaging fixed the trend and exposed something the single draw had hidden: at
10 deg a few solves diverge to 1e14 m *while reporting convergence*, enough to
move an RMS by eight orders of magnitude. Hence a median, plus explicit counts.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.64), (4.66)-(4.70)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from core.rf import AOAPositioner, aoa_angle_vector

ANCHORS = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
TRUE_POS = np.array([6.0, 9.0])
SEED = 42


def _errors(noise_deg, trials=200):
    """Position errors and the solver's convergence flags over many draws."""
    aoa_true = aoa_angle_vector(ANCHORS, TRUE_POS, include_elevation=False)
    rng = np.random.default_rng(SEED)
    errors, converged = [], []
    for _ in range(trials):
        noisy = aoa_true + rng.standard_normal(len(aoa_true)) * np.deg2rad(noise_deg)
        est, info = AOAPositioner(ANCHORS).solve(
            noisy, initial_guess=np.array([7.5, 7.5])
        )
        errors.append(float(np.linalg.norm(est - TRUE_POS)))
        converged.append(bool(info["converged"]))
    return np.asarray(errors), np.asarray(converged)


class TestAoaErrorScalesLinearly(unittest.TestCase):
    """The law the table is supposed to show."""

    def test_median_error_is_proportional_to_angular_noise(self):
        """Doubling the noise doubles the error, in the median.

        Uses the median because the mean does not survive the divergences
        below -- which is the whole reason the example reports one.
        """
        slopes = [np.median(_errors(deg)[0]) / deg for deg in (1.0, 5.0, 10.0)]

        for slope in slopes:
            self.assertAlmostEqual(slope / slopes[0], 1.0, delta=0.35)

    def test_a_single_draw_can_contradict_that_law(self):
        """Why one solve per noise level was not enough.

        The spread at a given noise level overlaps the level above it, so a
        table built from single draws can show error falling as noise rises.
        """
        five = _errors(5.0)[0]
        ten = _errors(10.0)[0]

        self.assertGreater(np.percentile(five, 90), np.percentile(ten, 10))

    def test_the_converged_flag_is_not_sufficient_at_high_noise(self):
        """Some solves report success and land absurdly far away.

        Pinned because it is a trap for anyone using this solver: filtering on
        info["converged"] alone still admits garbage. If a future fix makes the
        flag trustworthy, this fails and the example's warning should go.
        """
        errors, converged = _errors(10.0)
        lying = errors[converged] > 100.0

        self.assertGreater(int(np.sum(lying)), 0)


if __name__ == "__main__":
    unittest.main()
