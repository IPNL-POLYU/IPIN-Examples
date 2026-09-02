"""What estimating the clock costs, measured against a beacon that answers back.

Chapter 4's inline comparison carries a one-way TOA series whose state is
`(x, y, c*dt)` -- Eqs. (4.24)-(4.26), because a one-way pseudorange contains an
unknown receiver clock offset -- beside a two-way TOA (RTT) series whose state
is `(x, y)`, because Eq. (4.7)'s `d = c(t_rtt - t_proc)/2` is timed on one
clock and cancels the offset by construction.

**The two rows are matched so that the gap between them means something.** Same
anchors, same 50 test points, same seed, same sigma ladder, and the same
realisation drawn from it -- `run_inline_comparison` calls `draw_range_noise`
once per level and hands the array to both. Only the state vector differs. The
repo has been bitten by the unpaired version of exactly this: `aoa_bearings`
exists because the weighted and unweighted AOA columns used to draw their own
noise, so the ratio between them carried a full realisation on each side.

The RTT row is deliberately *not* noised through processing time and clock
drift, though `simulate_rtt_measurement` offers that and
`example_toa_positioning`'s Example 5 uses it. That would vary the error budget
as well as the state vector, and this row exists to vary one thing.

**The size of the effect is small, and that is the finding these tests are
shaped around.** Adding the clock column to H inflates the position covariance
by a factor set by the geometry alone -- it does not grow with sigma. For this
square array and these 50 points the analytic inflation is a median 1.026 over
per-point ratios (1.033 as a ratio of medians), and a 50-point median has a
sampling spread of roughly +/-7%. So "the RTT median is below the TOA median"
is **not** a law that a single noise level can demonstrate: over 40 repeats at
each sigma the ratio of medians ranged 0.87 to 1.27. It happens to hold at
every noisy level of the shipped seed, and that is pinned below as a property
of the shipped realisation rather than as physics. The physics is pinned twice
over instead -- once analytically, where it is exact, and once over a sample
large enough to resolve 2.6%.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.6)-(4.9), (4.14)-(4.23), (4.24)-(4.26)
"""

import contextlib
import functools
import io
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch4_rf_point_positioning.example_comparison import (
    INLINE_SEED,
    draw_range_noise,
    generate_scenario,
    rtt_positioning_test,
    run_inline_comparison,
    toa_positioning_test,
)
from core.rf import STALL_M, TOAPositioner, toa_solve_with_clock_bias

#: The clock bias `run_inline_comparison` injects into the one-way ranges.
CLOCK_BIAS_M = 1.5

#: Points per repeat in the statistical arm. The effect is 2.6% and a
#: 50-point median carries ~7% of sampling noise, so the shipped sweep's own
#: sample cannot resolve it -- measuring it needs its own, larger one.
STAT_POINTS = 1200
STAT_SEEDS = (0, 1, 2, 3)
STAT_SIGMA_M = 0.2


@functools.lru_cache(maxsize=1)
def _sweep():
    """The inline sweep, run once and shared.

    Six seconds a call, and four tests want it. `.cursor/rules/030` asks for
    exactly this (`tests/ch7_slam/slam_example_runner.py` is the pattern).
    No assertion lives in here: `lru_cache` does not cache exceptions, so a
    memoised helper that asserts re-runs its whole body for every caller once
    anything goes wrong.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return run_inline_comparison()


def _paired_medians(sigma_m, n_points, seed, rtt_gets_the_clock_state=False):
    """Median one-way and two-way error over `n_points`, on one noise draw.

    Both estimators see the identical ranges up to the injected bias, which is
    the whole design. `rtt_gets_the_clock_state` is the mutation this file's
    tolerance was justified against -- see
    `test_the_measured_cost_is_the_state_vector_and_not_the_noise`.
    """
    anchors, _ = generate_scenario(seed=42)
    rng = np.random.default_rng(seed)
    points = rng.uniform(1, 9, size=(n_points, 2))
    true_ranges = np.linalg.norm(
        points[:, None, :] - anchors[None, :, :], axis=2
    )
    noise = rng.standard_normal(true_ranges.shape) * sigma_m

    seed_3 = np.concatenate([INLINE_SEED, [0.0]])
    one_way, two_way = [], []
    for i, point in enumerate(points):
        est, _bias, _info = toa_solve_with_clock_bias(
            anchors, true_ranges[i] + CLOCK_BIAS_M + noise[i], seed_3
        )
        one_way.append(float(np.linalg.norm(est - point)))

        if rtt_gets_the_clock_state:
            est, _bias, _info = toa_solve_with_clock_bias(
                anchors, true_ranges[i] + noise[i], seed_3
            )
        else:
            est, _info = TOAPositioner(anchors, method="iterative_ls").solve(
                true_ranges[i] + noise[i], INLINE_SEED
            )
        two_way.append(float(np.linalg.norm(est - point)))

    return float(np.median(one_way)), float(np.median(two_way))


def _analytic_inflation():
    """Per-point ratio of 3-state to 2-state position DOP, over the 50 points.

    Sigma factors out of both sides, which is the point: the cost of the clock
    column is a property of the geometry and not of the noise.
    """
    anchors, truth = generate_scenario(seed=42)
    two_state, three_state = [], []
    for point in truth:
        offsets = point - anchors
        unit = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
        h_2 = unit
        # A one-way pseudorange is d + b, so the clock column is +1.
        h_3 = np.hstack([unit, np.ones((len(anchors), 1))])
        two_state.append(np.sqrt(np.trace(np.linalg.inv(h_2.T @ h_2))))
        three_state.append(
            np.sqrt(np.trace(np.linalg.inv(h_3.T @ h_3)[:2, :2]))
        )
    return np.array(three_state) / np.array(two_state)


class RttSeriesIsPresentAndMatched(unittest.TestCase):
    """The series exists, and the things held equal are actually equal."""

    def test_the_sweep_returns_a_two_way_series(self):
        schedules, results = _sweep()
        self.assertIn("RTT", results)

        names = [name for name, _, _ in schedules]
        self.assertIn("RTT", names)
        self.assertEqual(len(results["RTT"]), len(results["TOA"]))

    def test_rtt_shares_toa_s_noise_ladder(self):
        """A matched pair whose sigma schedules differ is not matched.

        Pinned rather than trusted because the two schedules are separate
        lists in the source, and a plausible future edit widens one of them.
        """
        schedules, _ = _sweep()
        ladders = {name: schedule for name, schedule, _ in schedules}
        units = {name: unit for name, _, unit in schedules}

        self.assertEqual(ladders["RTT"], ladders["TOA"])
        self.assertEqual(units["RTT"], "m")

    def test_the_two_range_series_solve_the_same_noise_draw(self):
        """Unpaired draws would make the gap a comparison of realisations.

        Checked by handing both functions a noise array of our own: if either
        one drew its own instead, the errors below would not reproduce.
        """
        anchors, truth = generate_scenario(seed=42)
        np.random.seed(7)
        noise = draw_range_noise(len(truth), len(anchors), 0.2)

        toa_a = toa_positioning_test(
            anchors, truth, 0.2, clock_bias_m=CLOCK_BIAS_M, range_noise=noise
        )
        rtt_a = rtt_positioning_test(anchors, truth, 0.2, range_noise=noise)

        # Same arrays again, from a global RNG left somewhere else entirely.
        np.random.seed(999)
        toa_b = toa_positioning_test(
            anchors, truth, 0.2, clock_bias_m=CLOCK_BIAS_M, range_noise=noise
        )
        rtt_b = rtt_positioning_test(anchors, truth, 0.2, range_noise=noise)

        np.testing.assert_array_equal(toa_a.errors, toa_b.errors)
        np.testing.assert_array_equal(rtt_a.errors, rtt_b.errors)

    def test_a_zero_sigma_level_draws_nothing_at_all(self):
        """The noiseless level must not consume the global stream.

        This is what keeps the four pre-existing series byte-stable: the
        in-line `if range_noise_std_m > 0` this replaced did not draw either,
        so every draw later in the sweep still lands where it did.
        """
        self.assertIsNone(draw_range_noise(50, 4, 0.0))

        np.random.seed(3)
        before = np.random.randn(5).copy()
        np.random.seed(3)
        self.assertIsNone(draw_range_noise(50, 4, 0.0))
        np.testing.assert_array_equal(before, np.random.randn(5))


class TheClockStateIsWhatCostsAccuracy(unittest.TestCase):
    """The physics, pinned where it is exact and where it is measurable."""

    def test_the_clock_column_inflates_position_dop_at_every_test_point(self):
        """Exact, tolerance-free, and the reason the RTT row is worth having.

        Estimating a fourth quantity from the same four ranges leaves one
        degree of freedom instead of two. The position block of
        `(H3' H3)^-1` is the Schur complement of the clock column, so it
        cannot be smaller than `(H2' H2)^-1` -- at any geometry, at any sigma.
        """
        inflation = _analytic_inflation()

        self.assertTrue(np.all(inflation > 1.0))
        # Recorded so a geometry change that flattens the effect is visible
        # rather than merely still-passing: this array is 4 anchors on a 10 m
        # square with the agent inside, and the cost is a few percent.
        self.assertAlmostEqual(float(np.median(inflation)), 1.026, places=2)

    def test_the_measured_cost_is_the_state_vector_and_not_the_noise(self):
        """Measured over a sample big enough to resolve 2.6%, then mutated.

        **Both halves of the bound were measured.** Against the noise: over ten
        seeds at these settings the ratio ran 1.0115 to 1.0376, mean 1.0257,
        std 0.0070, so a floor of 1.010 on the mean of four seeds sits about
        four standard errors below it. Against the defect: giving the two-way
        arm the clock state as well -- the one mutation that makes this test
        meaningless -- returns **exactly** 1.0000, because a common bias is
        absorbed whole into `c*dt` and the position estimates become
        bit-identical. The ceiling catches the opposite error, a two-way arm
        accidentally handed less noise than the one-way arm.
        """
        ratios = [
            (lambda pair: pair[0] / pair[1])(
                _paired_medians(STAT_SIGMA_M, STAT_POINTS, seed)
            )
            for seed in STAT_SEEDS
        ]
        measured = float(np.mean(ratios))

        self.assertGreater(measured, 1.010)
        self.assertLess(measured, 1.045)

        one_way, two_way = _paired_medians(
            STAT_SIGMA_M, 300, STAT_SEEDS[0], rtt_gets_the_clock_state=True
        )
        self.assertAlmostEqual(one_way / two_way, 1.0, places=9)

    def test_the_bias_is_unobservable_to_position_in_the_three_state_solve(self):
        """Why the one-way row can carry a 1.5 m bias and still be comparable.

        The bias goes entirely into `c*dt`, so the one-way *position* is the
        same whether or not it was injected, and the clock estimate moves by
        exactly the injected amount. That is the claim the comparison rests
        on: the two rows differ in their state vector and in nothing else, and
        the 1.5 m is not a handicap loaded onto TOA.

        **What is exact here is the difference, not the estimate.** A first
        draft asserted the recovered bias was 1.5 m on noisy data and read
        1.4642 -- the solve absorbs the injected bias *and* whatever common
        part the range noise happens to have, which at sigma = 0.2 m over four
        anchors is a few centimetres. The example's own docstring already said
        "+1.500 m on **noiseless** data" and it means it; both cases are
        separated below rather than sharing one tolerance.
        """
        anchors, truth = generate_scenario(seed=42)
        rng = np.random.default_rng(5)
        clean = np.linalg.norm(truth[:, None, :] - anchors[None, :, :], axis=2)
        noisy = clean + rng.standard_normal(clean.shape) * 0.2
        seed_3 = np.concatenate([INLINE_SEED, [0.0]])

        # Noiseless: the bias comes back as itself.
        for row, point in zip(clean, truth, strict=True):
            est, bias, _ = toa_solve_with_clock_bias(
                anchors, row + CLOCK_BIAS_M, seed_3
            )
            self.assertAlmostEqual(bias, CLOCK_BIAS_M, places=6)
            self.assertLess(float(np.linalg.norm(est - point)), 1e-6)

        # Noisy: the position does not move and the clock absorbs the whole
        # of the shift, both exactly.
        worst_position, worst_bias = 0.0, 0.0
        for row in noisy:
            with_bias, bias_hi, _ = toa_solve_with_clock_bias(
                anchors, row + CLOCK_BIAS_M, seed_3
            )
            without, bias_lo, _ = toa_solve_with_clock_bias(anchors, row, seed_3)
            worst_position = max(
                worst_position, float(np.linalg.norm(with_bias - without))
            )
            worst_bias = max(
                worst_bias, abs((bias_hi - bias_lo) - CLOCK_BIAS_M)
            )

        self.assertLess(worst_position, 1e-9)
        self.assertLess(worst_bias, 1e-9)


class WhatTheShippedSweepActuallyReports(unittest.TestCase):
    """Pins on the committed table and figure, labelled as realisation pins."""

    def test_rtt_is_at_or_below_toa_at_every_noisy_level_of_this_seed(self):
        """True at seed 42; **not** a law, and the docstring is the point.

        The effect is 2.6% and a 50-point median carries roughly 7% of
        sampling noise, so this ordering is not something one noise level can
        establish -- over 40 repeats per sigma the ratio of medians ranged
        0.87 to 1.27. It holds at every noisy level here, which is what the
        README table and the figure show, so it is pinned as a regression
        guard on the shipped numbers.
        `test_the_measured_cost_is_the_state_vector_and_not_the_noise` is
        where the underlying claim is actually established.
        """
        schedules, results = _sweep()
        ladder = {name: schedule for name, schedule, _ in schedules}["RTT"]

        for level, sigma in enumerate(ladder):
            if sigma == 0:
                continue
            with self.subTest(level=level + 1, sigma=sigma):
                self.assertLessEqual(
                    results["RTT"][level].median_m,
                    results["TOA"][level].median_m,
                )

    def test_both_range_series_are_exact_on_noiseless_measurements(self):
        """The one row with a known right answer, and neither wins it.

        At sigma = 0 both medians are roundoff -- 3.5e-10 m and 9.9e-10 m --
        so the two-way series is in fact the *larger* of the two, by 0.6
        nanometres. Nothing is measured there, which is why the example prints
        no ratio for level 1 and why the ordering test above skips it. An
        earlier draft of that print guarded on `median > 0` and duly reported
        "0.35x" as the cost of the clock state.
        """
        _, results = _sweep()

        self.assertLess(results["TOA"][0].median_m, STALL_M)
        self.assertLess(results["RTT"][0].median_m, STALL_M)

    def test_failures_are_accounted_by_solve_batch(self):
        """Median over every fix, failure count beside it -- not an RMSE.

        `solve_batch`'s four conditions: raised, reported `converged=False`,
        never left the seed, or landed over 100 m away. The two-way series has
        to go through it like the other four, so that a level where it stops
        solving reports as a failure rather than as an accuracy over the
        survivors.
        """
        _, results = _sweep()

        for level, outcome in enumerate(results["RTT"]):
            with self.subTest(level=level + 1):
                self.assertEqual(outcome.n, 50)
                self.assertEqual(len(outcome.errors), 50)
                self.assertEqual(outcome.solved.shape, (50,))
                self.assertEqual(
                    outcome.n_failed, int((~outcome.solved).sum())
                )
                self.assertGreaterEqual(outcome.n - outcome.n_failed, 45)

        # The one failure in the sweep is a step-tolerance stop, not a
        # divergence: its error is an ordinary sub-metre fix. `converged` here
        # means "finished inside max_iters", which is why the failure count
        # and the median are reported separately rather than one filtering the
        # other.
        for outcome in results["RTT"]:
            unsolved = outcome.errors[~outcome.solved]
            self.assertTrue(np.all(np.isfinite(unsolved)))
            self.assertTrue(np.all(unsolved < 100.0))

    def test_running_the_sweep_twice_gives_the_same_numbers(self):
        """An unseeded series cannot regenerate its own committed figure."""
        schedules_a, results_a = _sweep()
        with contextlib.redirect_stdout(io.StringIO()):
            schedules_b, results_b = run_inline_comparison()

        self.assertEqual(schedules_a, schedules_b)
        for method in results_a:
            for level, (a, b) in enumerate(
                zip(results_a[method], results_b[method], strict=True)
            ):
                with self.subTest(method=method, level=level + 1):
                    np.testing.assert_array_equal(a.errors, b.errors)
                    np.testing.assert_array_equal(a.solved, b.solved)


if __name__ == "__main__":
    unittest.main()
