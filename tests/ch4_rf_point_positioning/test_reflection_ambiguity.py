"""What the reflection-ambiguity figure claims, asserted.

The figure argues that a collinear array leaves an exact mirror twin of every position,
that TOA and TDOA solve INTO it as often as into the truth, and that AOA is not exposed to
the ambiguity at all. These tests pin the counts a fresh run of the sweep produces (see
each test's docstring for the number and what it means), so a change to the solvers, the
shipped dataset, or the seed lattice that moves any of them fails loudly here rather than
being noticed on the committed figure.

The sweeps are the expensive part (~1681 solves per arm, three arms), so they are computed
once and shared -- same pattern as test_initial_guess_basin.py's `_CACHE`.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.1)-(4.3), (4.27)-(4.33), (4.63)-(4.66). Companion to
            test_initial_guess_basin.py, which pins the square-room basin this dataset's
            corridor array is contrasted against.
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch4_rf_point_positioning.example_reflection_ambiguity import (
    ARMS,
    CENTROID_SEED,
    FAIL_S,
    LINE_Y,
    MIRROR,
    MIRROR_S,
    OFFLINE_SEED,
    TRUTH,
    TRUTH_S,
    measurements,
    state_at,
    sweep,
)

_CACHE = {}


def sweeps():
    """All three arms, computed once per session; returns {key: result}."""
    if not _CACHE:
        for arm in ARMS:
            result = sweep(*arm, verbose=False)
            _CACHE[result["key"]] = result
    return _CACHE


class TestReflectionAmbiguity(unittest.TestCase):
    """Zero measurement noise, so every FAIL counted here is the solver or the geometry,
    never the data."""

    def test_toa_and_tdoa_split_almost_evenly_between_truth_and_mirror(self):
        """The headline the figure states: 315 truth / 315 mirror / 1051 fail for TOA."""
        r = sweeps()

        self.assertEqual(r["toa"]["counts"][TRUTH_S], 315)
        self.assertEqual(r["toa"]["counts"][MIRROR_S], 315)
        self.assertEqual(r["toa"]["counts"][FAIL_S], 1051)

    def test_tdoa_lands_within_one_seed_of_toa_in_each_converged_class(self):
        """TDOA differences the same ranges TOA solves, so its split should track TOA's,
        not equal it exactly (TDOA drops one equation, TOA carries the clock state)."""
        r = sweeps()

        self.assertEqual(r["tdoa"]["counts"][TRUTH_S], 314)
        self.assertEqual(r["tdoa"]["counts"][MIRROR_S], 314)
        self.assertLessEqual(
            abs(r["tdoa"]["counts"][TRUTH_S] - r["toa"]["counts"][TRUTH_S]), 1
        )
        self.assertLessEqual(
            abs(r["tdoa"]["counts"][MIRROR_S] - r["toa"]["counts"][MIRROR_S]), 1
        )

    def test_every_mirror_solve_reports_converged_true(self):
        """The sneaky part: a solve that lands on the mirror is not flagged by the
        solver's own convergence report -- it is indistinguishable from a correct solve
        without knowing the geometry is symmetric."""
        r = sweeps()

        self.assertEqual(r["toa"]["n_mirror_claimed"], r["toa"]["counts"][MIRROR_S])
        self.assertGreater(r["toa"]["counts"][MIRROR_S], 0)

    def test_aoa_has_zero_mirror_solves(self):
        """Bearings break the symmetry: reflecting a position flips every azimuth, so
        nothing converges to the mirror under AOA."""
        r = sweeps()

        self.assertEqual(r["aoa"]["counts"][MIRROR_S], 0)
        self.assertEqual(r["aoa"]["counts"][TRUTH_S], 461)
        self.assertEqual(r["aoa"]["counts"][FAIL_S], 1220)

    def test_toa_and_tdoa_stall_at_every_seed_on_the_beacon_line_and_nowhere_else(self):
        """The degenerate starting point: on the line of symmetry the range Jacobian has
        no across-line column, so a seed there cannot move -- for TOA and TDOA only."""
        r = sweeps()

        self.assertEqual(r["toa"]["n_stalled"], r["toa"]["stalled_on_line"])
        self.assertEqual(r["toa"]["stalled_on_line"], r["toa"]["n_seeds_on_line"])
        self.assertEqual(r["tdoa"]["n_stalled"], r["tdoa"]["stalled_on_line"])
        self.assertEqual(r["tdoa"]["stalled_on_line"], r["tdoa"]["n_seeds_on_line"])
        self.assertEqual(r["aoa"]["n_stalled"], 0)

    def test_centroid_seed_fails_toa_and_tdoa_but_not_aoa(self):
        """The beacon centroid sits exactly on the line of symmetry."""
        r = sweeps()
        names = {TRUTH_S: "TRUTH", MIRROR_S: "MIRROR", FAIL_S: "FAIL"}

        self.assertEqual(names[state_at(r["toa"], CENTROID_SEED)], "FAIL")
        self.assertEqual(names[state_at(r["tdoa"], CENTROID_SEED)], "FAIL")
        self.assertNotEqual(names[state_at(r["aoa"], CENTROID_SEED)], "FAIL")

    def test_offline_seed_reaches_the_truth_under_every_arm(self):
        """A seed off the line of symmetry is not degenerate for any of the three arms."""
        r = sweeps()

        for key in ("toa", "tdoa", "aoa"):
            self.assertEqual(state_at(r[key], OFFLINE_SEED), TRUTH_S)

    def test_the_ambiguity_is_exact_in_range_but_not_in_bearing(self):
        """What makes TOA/TDOA blind and AOA not: measured directly from the forward
        models, not inferred from solver behaviour."""
        m_truth = measurements(TRUTH)
        m_mirror = measurements(MIRROR)

        self.assertLess(np.max(np.abs(m_truth["toa"] - m_mirror["toa"])), 1e-6)
        self.assertLess(np.max(np.abs(m_truth["tdoa"] - m_mirror["tdoa"])), 1e-6)
        self.assertGreater(
            np.degrees(np.max(np.abs(m_truth["aoa"] - m_mirror["aoa"]))), 90.0
        )

    def test_truth_and_mirror_are_reflections_about_the_beacon_line(self):
        self.assertAlmostEqual(TRUTH[0], MIRROR[0])
        self.assertAlmostEqual((TRUTH[1] + MIRROR[1]) / 2.0, LINE_Y)


if __name__ == "__main__":
    unittest.main()
