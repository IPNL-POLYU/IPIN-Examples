"""AOA Gauss-Newton converges from a cold start, and used not to.

The Chapter 4 comparison table reported AOA at 5.3e9 m RMSE with *zero*
angular noise. Not a noise result: from the anchor centroid as an initial
guess, 8 of 39 converged solves landed over a metre away -- the worst by 3e10
m -- while the other 31 were exact to 1e-8 m. Started near the truth, all 50
converged cleanly. So the residual had a basin-of-attraction problem and the
convergence flag did not detect it.

The cause was the parameterisation, not the initialisation. Solving on
z = tan(psi) (the book's Eq. 4.64 written literally) has two defects that no
starting point repairs:

  - tan has period pi, so an anchor ahead and an anchor behind give the same
    measurement, and the residual cannot tell the two apart;
  - as the estimate runs to infinity every anchor tends to the same bearing,
    so the tan residuals *shrink*. Infinity is a spurious attractor, and the
    iteration reaches it reporting success. A traced failure walked
    (5,5) -> (-4,-4.6) -> (-23,-27) -> (-364,-470) -> 1e10 and set
    converged=True.

`AOAPositioner.solve` now forms residuals as wrap(psi_measured - atan2(dE, dN))
by default. Same measurement model, inverted without discarding the quadrant:
bounded, wrappable, and with no attractor at infinity.

These tests run at zero noise, so any error is the solver and not the
measurements. The old behaviour is still reachable as residual="tan" and is
pinned below, because it is the evidence for why the default is what it is.

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


def _solve_from(guess_fn, n=50, residual="angle"):
    """Errors and convergence flags solving noiseless AOA from a given guess."""
    rng = np.random.default_rng(SEED)
    points = rng.uniform(1, 9, size=(n, 2))
    errors, converged = [], []
    for p in points:
        angles = np.array([aoa_azimuth(a, p) for a in ANCHORS])
        est, info = AOAPositioner(ANCHORS).solve(
            angles, initial_guess=guess_fn(p), residual=residual
        )
        errors.append(float(np.linalg.norm(est - p)))
        converged.append(bool(info["converged"]))
    return np.asarray(errors), np.asarray(converged)


class TestAoaInitialisationBasin(unittest.TestCase):
    """Zero noise, so any error here is the solver and not the measurements."""

    def test_a_good_initial_guess_solves_exactly(self):
        """The measurements are sufficient; nothing is wrong with the geometry."""
        errors, converged = _solve_from(lambda p: p + 1.0)

        self.assertTrue(converged.all())
        self.assertLess(errors.max(), 1e-3)

    def test_the_centroid_guess_now_solves_exactly_too(self):
        """The whole point: a cold start is no longer a different regime.

        This is the test that used to assert the opposite -- that a fifth of
        the converged solves were gross failures and the worst exceeded 100 m.
        """
        errors, converged = _solve_from(lambda p: ANCHORS.mean(axis=0))

        self.assertTrue(converged.all())
        self.assertLess(errors.max(), 1e-3)

    def test_the_convergence_flag_is_now_trustworthy(self):
        """No solve reports success while sitting far from the answer."""
        errors, converged = _solve_from(lambda p: ANCHORS.mean(axis=0))

        self.assertEqual(int(np.sum(errors[converged] > 1.0)), 0)

    def test_the_tan_parameterisation_still_fails_from_the_centroid(self):
        """Kept as the evidence for why the default changed.

        If this ever stops failing, the tan form has been repaired too and the
        `residual` switch has lost its reason to exist.
        """
        errors, converged = _solve_from(
            lambda p: ANCHORS.mean(axis=0), residual="tan"
        )
        gross = errors[converged] > 1.0

        self.assertGreater(int(np.sum(gross)), 0)
        self.assertGreater(errors.max(), 100.0)

    def test_tan_and_angle_agree_when_tan_succeeds(self):
        """The fix changes which solutions are reached, not what is correct.

        Both parameterise the same measurement model, so where the tan form
        does converge it must land on the same point.
        """
        angle_errors, _ = _solve_from(lambda p: p + 1.0, residual="angle")
        tan_errors, tan_converged = _solve_from(lambda p: p + 1.0, residual="tan")

        agreed = tan_converged & (tan_errors < 1e-3)
        self.assertGreater(int(np.sum(agreed)), 0)
        np.testing.assert_allclose(angle_errors[agreed], tan_errors[agreed], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
