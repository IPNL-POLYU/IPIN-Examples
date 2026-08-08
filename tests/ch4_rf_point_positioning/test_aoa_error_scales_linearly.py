"""AOA position error is linear in angular noise, and the solver can lie.

The example used to solve once per noise level and tabulate the result. Its
own numbers then contradicted the physics: 1.74 m at 5 deg and 2.04 m at
10 deg, an apparent saturation, when doubling the angular noise should double
the position error. Single draws, tabulated as if they were a trend.

Averaging fixed the trend and exposed something the single draw had hidden: at
10 deg a few solves diverged to 1e14 m *while reporting convergence*, enough to
move an RMS by eight orders of magnitude. Hence a median, plus explicit counts.

Those divergences were the tan parameterisation, not the noise, and the solver
now forms residuals in wrapped angle space -- see
`test_aoa_initialisation_basin.py`. The convergence flag is trustworthy again,
which the last test here pins. The median is still the right summary: it is
what makes the linear law visible without a handful of tail draws dominating.

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

    def test_the_converged_flag_is_trustworthy_at_high_noise(self):
        """No solve reports success while landing absurdly far away.

        This used to assert the opposite, and warned that a future fix would
        break it. The fix is the wrapped-angle residual: with no attractor at
        infinity, a runaway keeps a large residual instead of a shrinking one,
        so it can no longer converge to nowhere.
        """
        errors, converged = _errors(10.0)
        lying = errors[converged] > 100.0

        self.assertEqual(int(np.sum(lying)), 0)

    def test_every_solve_converges_at_high_noise(self):
        """10 deg is severe, but severity should show up as error, not failure."""
        _, converged = _errors(10.0)

        self.assertTrue(converged.all())


if __name__ == "__main__":
    unittest.main()
