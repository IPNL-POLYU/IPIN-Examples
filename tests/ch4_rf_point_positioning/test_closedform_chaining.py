"""What the closed-form chaining figure claims, asserted.

Two stories, one per dataset, measured fresh here rather than assumed:

- Square (control): Fang's closed-form TOA solve (no initial guess anywhere) already
  lands within 0.011 m of the fully-refined iterative answer, and CHAINING it into the
  iterative refiner reproduces that answer exactly.
- Corridor (collinear): the naive centroid seed is degenerate for TOA/TDOA (100/100
  fail), and Fang/Chan's own linear system has an exactly zero y-column -- every closed
  -form y-estimate lands at 0.0, never near the true y. Chaining that (x, 0.0) seed into
  the iterative refiner still converges, but the corridor's mirror ambiguity survives it:
  a converged fix lands on the truth or on its reflection depending only on which side of
  the beacon line the true point started on.

The sweeps run every row of two 100-point datasets through up to six solvers, so they are
computed once per session and shared -- same `_CACHE` pattern as the sibling ch4 tests.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.43)-(4.49) Fang, (4.50)-(4.62) Chan, (4.14)-(4.23)
            iterative TOA/I-WLS. Companion to test_reflection_ambiguity.py, whose
            TRUTH/MIRROR/FAIL convention `mirror_split` reuses.
"""

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: no display during tests

from ch4_rf_point_positioning.example_closedform_chaining import (
    DATASETS,
    load_dataset,
    mirror_split,
    run_corridor,
    run_square,
)

_CACHE = {}


def square_results():
    if "square" not in _CACHE:
        beacons, truth, toa, tdoa, room_diagonal_m = load_dataset(
            DATASETS["square (control)"]
        )
        _CACHE["square"] = run_square(beacons, truth, toa, tdoa, room_diagonal_m)
        _CACHE["square_truth"] = truth
    return _CACHE["square"]


def corridor_results():
    if "corridor" not in _CACHE:
        beacons, truth, toa, tdoa, room_diagonal_m = load_dataset(
            DATASETS["corridor (collinear)"]
        )
        _CACHE["corridor"] = run_corridor(beacons, truth, toa, tdoa, room_diagonal_m)
        _CACHE["corridor_truth"] = truth
    return _CACHE["corridor"]


class TestSquareControl(unittest.TestCase):
    """The chicken-and-egg is broken on the well-conditioned geometry: closed-form needs
    no seed and chaining it in matches the good-seed iterative answer exactly."""

    def test_headline_medians(self):
        r = square_results()

        self.assertAlmostEqual(r["iter_toa"]["median_m"], 0.088, places=3)
        self.assertAlmostEqual(r["fang"]["median_m"], 0.099, places=3)
        self.assertAlmostEqual(r["fang_to_iter"]["median_m"], 0.088, places=3)
        self.assertAlmostEqual(r["chan_to_iter"]["median_m"], 0.092, places=3)

    def test_nothing_fails(self):
        """Every arm on the square dataset solves all 100 rows."""
        r = square_results()
        for key in (
            "iter_toa",
            "iter_tdoa",
            "fang",
            "chan",
            "fang_to_iter",
            "chan_to_iter",
        ):
            self.assertEqual(r[key]["fails"], 0, key)

    def test_chaining_reproduces_the_good_seed_iterative_answer_exactly(self):
        """The claim the figure exists to make: chaining needs no seed and loses nothing."""
        r = square_results()

        self.assertAlmostEqual(
            r["fang_to_iter"]["median_m"], r["iter_toa"]["median_m"], places=3
        )

    def test_fang_alone_is_close_to_but_not_better_than_the_chained_answer(self):
        """Closed-form is good, not free: chaining still buys something (0.011 m here)."""
        r = square_results()

        gap = r["fang"]["median_m"] - r["fang_to_iter"]["median_m"]
        self.assertGreater(gap, 0.0)
        self.assertLess(gap, 0.02)

    def test_fangs_linear_system_is_well_conditioned_here(self):
        """The square geometry gives Fang's H_a full rank -- unlike the corridor."""
        r = square_results()

        self.assertTrue(np.isfinite(r["fang"]["cond_median"]))
        self.assertFalse(r["fang"]["cond_is_ever_infinite"])


class TestCorridorCollinear(unittest.TestCase):
    """The degenerate geometry: a naive seed fails outright, closed-form's own y-column
    is exactly singular, and chaining converges but inherits the mirror ambiguity."""

    def test_the_naive_centroid_seed_fails_every_row(self):
        """The beacon centroid sits on the line of symmetry: the motivating failure for
        needing ANY other way to seed the solver."""
        r = corridor_results()

        self.assertEqual(r["iter_toa"]["fails"], 100)
        self.assertEqual(r["iter_tdoa"]["fails"], 100)

    def test_fangs_y_column_is_exactly_singular(self):
        """Every beacon shares y = 10, so Eq. (4.47)'s h_n^i is exactly zero for every
        row: cond(H_a^T H_a) is not merely large, it is infinite."""
        r = corridor_results()

        self.assertFalse(np.isfinite(r["fang"]["cond_median"]))
        self.assertTrue(r["fang"]["cond_is_ever_infinite"])

    def test_fangs_closed_form_y_estimate_is_always_exactly_zero(self):
        """The closed form does not merely do badly on y -- it recovers none of it: the
        minimum-norm solution of a zero column is zero, on every one of the 100 rows."""
        r = corridor_results()

        self.assertTrue(np.all(r["fang"]["estimates"][:, 1] == 0.0))

    def test_fang_chain_converges_every_row_and_splits_evenly(self):
        """Chaining the degenerate (x, 0.0) seed into the iterative refiner still
        converges on every row -- but the mirror ambiguity survives the chaining."""
        r = corridor_results()
        truth = _CACHE["corridor_truth"]
        split = mirror_split(
            r["fang_to_iter"], truth, "Fang -> iterative TOA", verbose=False
        )

        self.assertEqual(split["fail"], 0)
        self.assertEqual(split["truth_side"], 50)
        self.assertEqual(split["mirror_side"], 50)
        self.assertEqual(
            split["truth_side"] + split["mirror_side"] + split["fail"], split["n"]
        )

    def test_chan_chain_has_a_nonzero_fail_rate_and_still_splits(self):
        """Chan's chain is measurably worse than Fang's here: it fails 15/100 rows, and
        of the rows that DO converge the split still leans toward the mirror."""
        r = corridor_results()
        truth = _CACHE["corridor_truth"]
        split = mirror_split(
            r["chan_to_iter"], truth, "Chan -> iterative TDOA", verbose=False
        )

        self.assertEqual(split["fail"], 15)
        self.assertEqual(split["truth_side"], 40)
        self.assertEqual(split["mirror_side"], 45)
        self.assertEqual(
            split["truth_side"] + split["mirror_side"] + split["fail"], split["n"]
        )

    def test_the_tri_state_split_always_partitions_every_row_exactly_once(self):
        """The invariant `mirror_split` itself asserts -- pinned again here so a change
        that weakens that internal assertion is still caught from outside the function.
        """
        r = corridor_results()
        truth = _CACHE["corridor_truth"]
        for key in ("fang_to_iter", "chan_to_iter"):
            split = mirror_split(r[key], truth, key, verbose=False)
            self.assertEqual(
                split["truth_side"] + split["mirror_side"] + split["fail"], split["n"]
            )


if __name__ == "__main__":
    unittest.main()
