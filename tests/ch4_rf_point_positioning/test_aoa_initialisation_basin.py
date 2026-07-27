"""AOA Gauss-Newton can converge to a wrong solution and report success.

The Chapter 4 comparison table reported AOA at 5.3e9 m RMSE with *zero*
angular noise. Not a noise result: from the anchor centroid as an initial
guess, 8 of 39 converged solves land over a metre away -- the worst by 3e10 m
-- while the other 31 are exact to 1e-8 m. Started near the truth, all 50
converge cleanly.

So the azimuth residual has a basin-of-attraction problem, and the convergence
flag does not detect it. The comparison now reports a median with a separate
count of gross failures, which is the only way both facts survive in one table.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.64), (4.66)-(4.70)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from core.rf import AOAPositioner, aoa_azimuth

ANCHORS = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
SEED = 0


def _solve_from(guess_fn, n=50):
    """Errors and convergence flags solving noiseless AOA from a given guess."""
    rng = np.random.default_rng(SEED)
    points = rng.uniform(1, 9, size=(n, 2))
    errors, converged = [], []
    for p in points:
        angles = np.array([aoa_azimuth(a, p) for a in ANCHORS])
        est, info = AOAPositioner(ANCHORS).solve(angles, initial_guess=guess_fn(p))
        errors.append(float(np.linalg.norm(est - p)))
        converged.append(bool(info["converged"]))
    return np.asarray(errors), np.asarray(converged)


class TestAoaInitialisationBasin(unittest.TestCase):
    """Zero noise, so any error here is the solver and not the measurements."""

    def test_a_good_initial_guess_solves_exactly(self):
        """The measurements are sufficient; nothing is wrong with the geometry.

        Without this the failures below could be blamed on the configuration
        rather than on where the iteration starts.
        """
        errors, converged = _solve_from(lambda p: p + 1.0)

        self.assertTrue(converged.all())
        self.assertLess(errors.max(), 1e-3)

    def test_the_centroid_guess_produces_gross_failures(self):
        """And they are not rare -- roughly a fifth of the converged solves."""
        errors, converged = _solve_from(lambda p: ANCHORS.mean(axis=0))
        gross = errors[converged] > 1.0

        self.assertGreater(int(np.sum(gross)), 0)
        self.assertGreater(errors.max(), 100.0)

    def test_the_good_solves_are_still_exact(self):
        """It fails completely or not at all, which is why a median works.

        A method degrading gracefully would need a different summary; this one
        is bimodal, so the median reports the working mode and the failure
        count reports the rest.
        """
        errors, converged = _solve_from(lambda p: ANCHORS.mean(axis=0))

        self.assertLess(float(np.median(errors[converged])), 1e-3)


if __name__ == "__main__":
    unittest.main()
