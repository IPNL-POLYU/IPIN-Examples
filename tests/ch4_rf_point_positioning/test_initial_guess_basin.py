"""What the initial-guess basin figure claims, asserted.

The figure argues one thing and it is easy to overstate: that AOA's cold-start failures are
a property of the residual's parameterisation rather than of the starting point. The example
was in fact written expecting the wrapped-angle form to remove the basin outright, and the
sweep said otherwise -- 341 of 1681 seeds still fail. What it removes is the QUIET class.

So these tests pin the narrow claim, in both directions:

  * the quiet failures -- stalled at the seed, or stopped somewhere plausible but wrong --
    are what changing the residual removes;
  * the loud ones survive, and the convergence flag is still not a check, so the figure must
    not be re-captioned as "fixed".

`test_the_ratio_is_not_one` is the "a demonstration that does not demonstrate" guard: if the
two parameterisations ever perform the same, this example has stopped demonstrating anything
and should be deleted rather than left to argue from a caption.

The sweeps are ~13 s each, so they are computed once and shared -- see the Cost note in
.cursor/rules/030-figures-and-claims.mdc.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.63)-(4.65). Companion to
            test_aoa_initialisation_basin.py, which pins the same behaviour from a single
            cold start; this one sweeps the whole floor.
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch4_rf_point_positioning.example_initial_guess_basin import (
    ANCHORS,
    DIVERGED,
    METHOD_CLOCK_BIAS_M,
    SOLVED,
    STALLED,
    TRUTH,
    WRONG,
    ClockStateSolver,
    measurements,
    sweep,
    sweep_arm,
    trace_worst,
)
from core.rf import AOAPositioner, TDOAPositioner, toa_range

_CACHE = {}


def sweeps():
    """Both sweeps, computed once per session."""
    if not _CACHE:
        for residual in ("tan", "angle"):
            _CACHE[residual] = sweep(residual, verbose=False)
    return _CACHE["tan"], _CACHE["angle"]


def method_results():
    """The three `sweep_arm` method arms (`--compare method`), computed once per session.

    Same lattice, same target as the residual sweep above; only the measurement type
    (TOA with the clock state, TDOA, AOA) changes.
    """
    if "method" not in _CACHE:
        ranges = np.array([toa_range(a, TRUTH) for a in ANCHORS]) + METHOD_CLOCK_BIAS_M
        tdoa = ranges[1:] - ranges[:1]
        _CACHE["method"] = {
            "toa": sweep_arm(ClockStateSolver(ANCHORS), ranges, "TOA + clock state"),
            "tdoa": sweep_arm(
                TDOAPositioner(ANCHORS, reference_anchor_index=0), tdoa, "TDOA"
            ),
            "aoa": sweep_arm(
                AOAPositioner(ANCHORS), measurements(), "AOA", residual="angle"
            ),
        }
    return _CACHE["method"]


def quiet(result):
    """Failures that look like answers: never moved, or stopped somewhere plausible."""
    return result["counts"][STALLED] + result["counts"][WRONG]


class TestInitialGuessBasin(unittest.TestCase):
    """Zero measurement noise, so every failure counted here is the solver."""

    def test_the_quiet_failures_are_what_the_residual_fix_removes(self):
        """The claim the figure is actually allowed to make."""
        tan, angle = sweeps()

        self.assertGreater(quiet(tan), 100)
        self.assertEqual(quiet(angle), 0)

    def test_the_wrapped_form_never_stops_somewhere_plausible(self):
        """No `converged, wrong place` outcome: the sneaky class goes to zero.

        A near-miss that reports success is the one failure mode a reader cannot detect
        downstream, which is why it gets its own assertion.
        """
        _, angle = sweeps()

        self.assertEqual(angle["counts"][WRONG], 0)

    def test_far_seeds_still_diverge_under_both(self):
        """The honest half of the result, pinned so the caption cannot drift.

        If this ever fails, the wrapped-angle form has become globally convergent on this
        geometry and the example's "honest, not safe" paragraph is out of date.
        """
        tan, angle = sweeps()

        self.assertGreater(tan["counts"][DIVERGED], 0)
        self.assertGreater(angle["counts"][DIVERGED], 0)

    def test_the_convergence_flag_is_not_a_check_under_either(self):
        """Failures that set converged=True exist in both sweeps.

        This is why `solve_batch`'s four conditions are not optional: fixing the residual
        does not turn the flag into a test.
        """
        tan, angle = sweeps()

        self.assertGreater(tan["silent"], 0)
        self.assertGreater(angle["silent"], 0)

    def test_the_ratio_is_not_one(self):
        """A demonstration that does not demonstrate is a failing test."""
        tan, angle = sweeps()
        failed_tan = tan["n"] - tan["counts"][SOLVED]
        failed_angle = angle["n"] - angle["counts"][SOLVED]

        self.assertGreater(failed_tan / max(failed_angle, 1), 1.5)

    def test_seeds_inside_the_room_all_solve_with_the_wrapped_form(self):
        """The practically relevant statement: a seed anywhere in the room is fine."""
        _, angle = sweeps()
        inside = (
            (angle["xx"] >= ANCHORS[:, 0].min())
            & (angle["xx"] <= ANCHORS[:, 0].max())
            & (angle["yy"] >= ANCHORS[:, 1].min())
            & (angle["yy"] <= ANCHORS[:, 1].max())
        )

        self.assertGreater(int(np.sum(inside)), 100)
        self.assertTrue(np.all(angle["codes"][inside] == SOLVED))

    def test_the_traced_run_is_a_silent_divergence(self):
        """The fourth panel must show a lie, not an honest failure.

        The largest error in the tan sweep reports converged=False, which is correct
        behaviour and not worth a panel; `trace_worst` deliberately picks the furthest run
        that still set the flag.
        """
        tan, _ = sweeps()
        seed, history, converged = trace_worst(tan)

        self.assertTrue(converged)
        self.assertGreater(np.linalg.norm(history[-1] - TRUTH), 1e6)
        self.assertLess(np.linalg.norm(history[0] - seed), 1e-9)

    def test_the_measurements_are_sufficient(self):
        """Nothing is wrong with the data: seeded at the answer, tan solves too.

        Without this the figure could be read as a geometry or an observability problem.
        """
        from ch4_rf_point_positioning.example_initial_guess_basin import measurements
        from core.rf import AOAPositioner

        for residual in ("tan", "angle"):
            est, info = AOAPositioner(ANCHORS).solve(
                measurements(), initial_guess=TRUTH + 0.5, residual=residual
            )
            self.assertTrue(info["converged"], residual)
            self.assertLess(float(np.linalg.norm(est - TRUTH)), 1e-3, residual)


class TestSweepArmMatchesSweep(unittest.TestCase):
    """`sweep_arm` (added for `--compare method`) must not silently diverge from `sweep`.

    `sweep("angle")` is `AOAPositioner(ANCHORS)` and `residual="angle"` hard-coded;
    `sweep_arm` is the same body with both lifted into arguments. This proves the lift
    changed nothing by running the AOA arm both ways and requiring bit-identical output.
    """

    def test_aoa_angle_arm_is_bit_identical_to_the_residual_sweep(self):
        _, angle = sweeps()
        ported = sweep_arm(
            AOAPositioner(ANCHORS), measurements(), "AOA", residual="angle"
        )

        self.assertTrue(np.array_equal(ported["codes"], angle["codes"]))
        self.assertTrue(np.allclose(ported["errors"], angle["errors"], equal_nan=True))
        self.assertEqual(ported["counts"], angle["counts"])


class TestMethodComparison(unittest.TestCase):
    """`--compare method`: does the MEASUREMENT TYPE change the basin?

    Same 1681 seeds and the same target as the residual sweep, zero measurement noise;
    only TOA (with the clock state, Eqs. 4.24-4.26)/TDOA/AOA changes. Headline counts
    measured fresh this session (`python -m ch4_rf_point_positioning.example_initial_guess_basin
    --compare method`) and pinned here so a repo change that moves them fails loudly.
    """

    def test_headline_fail_counts_out_of_1681(self):
        m = method_results()
        fail = {k: v["n"] - v["counts"][SOLVED] for k, v in m.items()}

        self.assertEqual(fail["toa"], 905)
        self.assertEqual(fail["tdoa"], 964)
        self.assertEqual(fail["aoa"], 341)

    def test_the_aoa_arm_agrees_with_the_residual_sweeps_own_angle_count(self):
        """The AOA arm inside method-mode is the same solve the residual sweep pins."""
        m = method_results()
        _, angle = sweeps()

        self.assertEqual(
            m["aoa"]["n"] - m["aoa"]["counts"][SOLVED],
            angle["n"] - angle["counts"][SOLVED],
        )

    def test_toa_and_tdoa_fail_more_often_than_aoa_on_this_lattice(self):
        """The story `--compare method` exists to tell: ranges fail far more than
        bearings on a lattice that extends well outside the room, on this square
        geometry -- unlike the collinear array, where TOA/TDOA fail from the beacon
        centroid specifically (see example_comparison.py's geometry comparison)."""
        m = method_results()
        fail = {k: v["n"] - v["counts"][SOLVED] for k, v in m.items()}

        self.assertGreater(fail["toa"], fail["aoa"])
        self.assertGreater(fail["tdoa"], fail["aoa"])


if __name__ == "__main__":
    unittest.main()
