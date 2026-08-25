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
    SOLVED,
    STALLED,
    TRUTH,
    WRONG,
    sweep,
    trace_worst,
)

_CACHE = {}


def sweeps():
    """Both sweeps, computed once per session."""
    if not _CACHE:
        for residual in ("tan", "angle"):
            _CACHE[residual] = sweep(residual, verbose=False)
    return _CACHE["tan"], _CACHE["angle"]


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


if __name__ == "__main__":
    unittest.main()
