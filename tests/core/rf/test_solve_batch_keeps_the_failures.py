"""`solve_batch` counts all four ways a fix fails, not just the reported one.

Chapter 4's geometry comparison aggregated each method as an RMSE over the
solves that reported `converged`, and that single choice produced three
separate wrong answers at once:

  - AOA on the collinear dataset reported 2.2e10 m, because three of its 95
    "converged" fixes had walked to 1e11 m and one such fix sets an RMSE.
  - TOA and TDOA on that dataset reported *nothing*, because none of their 100
    converged, so the method silently left the table.
  - The error-vs-GDOP scatter paired `errors[:n]` against `gdop[:n]` with the
    errors already compacted to the successes, so point i's error sat beside
    some other point's GDOP whenever anything failed.

The policy this file pins was not invented for the fix. It is what
`scripts/generate_ch4_rf_2d_positioning_dataset.py` already used to compute the
`failed_count` in every ch4 `config.json`; `solve_batch` is that loop extracted
so the example and the generator cannot disagree. The extraction was checked by
regenerating all four ch4 datasets and diffing every byte, `config.json`
included.

**Each test below was written by mutating the thing it claims to catch**, which
is the only way to know an assertion is load-bearing rather than merely true.
A solver that reports success from 1e11 m away passes any check that trusts
`info["converged"]`, and a solver that never moves passes it twice.

Author: Li-Ta Hsu
References: Chapter 4, Section 4.5
"""

import numpy as np
import pytest

from core.rf import DIVERGENCE_M, solve_batch

TRUTH = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
GUESS = np.array([0.0, 0.0])


class ScriptedSolver:
    """Returns a scripted (position, info) per call, in order.

    A stub rather than a real positioner: the four failure modes are hard to
    provoke on demand from a real geometry, and a test that has to construct a
    degenerate array to check its bookkeeping is testing the array.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def solve(self, measurement, initial_guess, **kwargs):
        position, converged = self.script[self.calls]
        self.calls += 1
        if position is None:
            raise RuntimeError("solver blew up")
        return np.asarray(position, dtype=float), {"converged": converged}


def _outcome(script, **kwargs):
    solver = ScriptedSolver(script)
    measurements = np.zeros((len(TRUTH), 3))
    return solve_batch(solver, measurements, GUESS, TRUTH, **kwargs)


def test_a_converged_fix_that_landed_far_away_is_a_failure():
    """The AOA case: `converged=True` at 1e11 m is not a measurement."""
    out = _outcome([
        ([1.0, 1.0], True),
        ([1e11, 1e11], True),          # converged, and absurd
        ([3.0, 3.0], True),
        ([4.0, 4.0], True),
    ])
    assert out.n_failed == 1
    assert not out.solved[1]
    assert out.max_solved_m < DIVERGENCE_M, (
        "the diverged fix leaked into the solved statistics, which is exactly "
        "the 2.2e10 m RMSE this helper exists to prevent"
    )


def test_a_fix_that_never_left_the_initial_guess_is_a_failure():
    """The TOA/TDOA case: a zero step on a rank-deficient Jacobian.

    Gauss-Newton reports convergence for standing still, and the error then
    scores as the distance from the seed to the truth -- which is why all three
    methods returned an identical 6.77 m on the collinear beacons.
    """
    out = _outcome([
        (GUESS, True),                 # stalled, but says it converged
        ([2.0, 2.0], True),
        ([3.0, 3.0], True),
        ([4.0, 4.0], True),
    ])
    assert out.stalled[0]
    assert not out.solved[0]
    assert out.n_failed == 1


def test_a_raise_is_a_failure_and_does_not_become_a_zero_error():
    """NaN, not 0.0 -- a dropped fix must not read as a perfect one."""
    out = _outcome([
        (None, True),                  # raises
        ([2.0, 2.0], True),
        ([3.0, 3.0], True),
        ([4.0, 4.0], True),
    ])
    assert np.isnan(out.errors[0])
    assert not out.solved[0]
    assert out.n_failed == 1
    assert out.median_m == pytest.approx(0.0), (
        "a raised fix must be excluded from the median rather than counted as "
        "zero error"
    )


def test_errors_stay_aligned_with_the_measurements():
    """Full length, so `gdop[solved]` pairs with `errors[solved]`.

    This is the assertion that discriminates: compacting the errors to the
    successes gives an array that is still a valid array, still plottable, and
    silently paired with the wrong per-position quantity.
    """
    out = _outcome([
        ([1.0, 1.0], True),
        (None, True),
        ([3.0, 3.0], True),
        ([1e11, 1e11], True),
    ])
    gdop = np.array([10.0, 20.0, 30.0, 40.0])

    assert len(out.errors) == len(TRUTH) == len(gdop)
    assert np.array_equal(gdop[out.solved], [10.0, 30.0]), (
        "the surviving GDOP values are not the ones belonging to the surviving "
        "fixes, so the error-vs-GDOP pairing is off"
    )


def test_the_median_includes_failures_and_the_mean_does_not():
    """Two different questions, and reporting only one of them hides a stall.

    `median_m` over every fix that returned a number is what makes a stalled
    solver visible at all -- over the successes alone it would report the
    accuracy of the fixes that happened to work. `mean_solved_m` is the other
    question and needs the failures gone.
    """
    out = _outcome([
        (GUESS, True),                 # stalled at 0,0: error |(1,1)| = sqrt(2)
        (GUESS, True),                 # stalled at 0,0: error |(2,2)| = 2 sqrt(2)
        ([3.0, 3.0], True),            # exact
        ([4.0, 4.0], True),            # exact
    ])
    # Two stalls and two exact fixes, so the median sits between them and the
    # mean over the successes is 0. One stall would not discriminate: the
    # median of [sqrt(2), 0, 0, 0] is 0, and the assertion would hold whether
    # or not the failure was counted.
    assert out.median_m == pytest.approx(np.sqrt(2) / 2), (
        "a stalled fix vanished from the median, which is what makes a solver "
        "that never moves look like one that solved"
    )
    assert out.mean_solved_m == pytest.approx(0.0)
    assert out.summary() == {
        "median_m": pytest.approx(np.sqrt(2) / 2),
        "mean_solved_m": pytest.approx(0.0),
        "max_solved_m": pytest.approx(0.0),
        "failed_count": 2,
        "n_positions": 4,
    }


def test_nothing_solved_reports_nan_rather_than_zero():
    """A method that failed everywhere must not plot as a perfect score.

    The original bar chart drew 0 for a method with no converged solves, which
    reads as the best result in the figure rather than the absence of one.
    """
    out = _outcome([(GUESS, False)] * 4)
    assert out.n_failed == 4
    assert np.isnan(out.mean_solved_m)
    assert np.isnan(out.max_solved_m)
    assert np.isfinite(out.median_m), (
        "the fixes still returned positions, so the median is still a number -- "
        "it is just a number about the seed"
    )


def test_the_divergence_threshold_is_configurable_and_defaults_to_100_m():
    """The generator applies the magnitude check downstream, so it passes inf."""
    script = [([1e11, 1e11], True)] + [(list(p), True) for p in TRUTH[1:]]
    assert _outcome(script).n_failed == 1
    assert _outcome(script, divergence_m=np.inf).n_failed == 0
