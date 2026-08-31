"""The tuning demo's results are claims, so they are tested as ones.

Two separate stories live in this demo and both used to be wrong.

**The robust losses did nothing.** ``huber_R_scale`` and ``cauchy_R_scale``
take a residual normalized by sigma, and the demo handed them an innovation in
metres with a threshold of 1.5. Against innovations whose median was 0.256 m,
Huber's dead zone covered everything: it fired **once in 2271 updates** and its
largest inflation was 1.020. The results table read "Huber 0.722" beside
"Baseline 0.722" and invited the reader to conclude that robustness did not
help. Nothing had happened at all -- the arithmetic signature CLAUDE.md calls
out, a stage whose output exactly equals its input.

**Chi-square gating was blamed twice, and both times wrongly.** The first
explanation was that R is set from line-of-sight noise while half the NLOS
ranges carry a bias, so the gate rejects good data. That was replaced by a
second: gating "collapses" to 24 m on the *clean* dataset too, through a
starvation feedback loop over heavy-tailed innovations. The 24 m was real and
the second explanation was wrong in the same way as the first -- it looked for
the cause inside the gate.

The innovations were heavy-tailed because the shipped accelerometer was
map-frame where every filter here integrates it as body-frame, so the filter
was fighting a double rotation. The gate is the only strategy in this demo
that checks its input against a distributional assumption, which made it the
first thing to break and therefore the thing that looked broken. Corrected, it
accepts 95% of a clean run -- exactly its nominal confidence -- and is the
best-performing method in the table.

What is asserted here:

  1. The losses actually fire, and by an amount that varies.
  2. Sporadic outliers -- an inlier majority, which is what an M-estimator
     assumes -- are repaired by a clear margin, and by the *median*, which is
     the statistic that survives a change of outlier draw.
  3. Robustness is close to free on clean data. What pins the threshold from
     above is the persistent-NLOS run, where an aggressive delta actively
     costs; the textbook 1.345 is ruled out by that, not by the clean case.
  4. Persistent NLOS is not repaired by reweighting, and the per-anchor bias
     says why -- but it *is* repaired by rejection.
  5. Gating does what it is specified to do in all three scenarios.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.3; Eqs. (8.6), (8.7), (8.9)
"""

import functools
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch8_sensor_fusion.example_robust_tuning import (
    CAUCHY_C_SIGMA,
    HUBER_DELTA_SIGMA,
    make_sporadic_outlier_dataset,
    per_anchor_range_bias,
    position_errors,
    run_fusion_with_strategy,
)
from core.eval import compute_rmse
from core.fusion import load_fusion_dataset

LOS_DATASET = "data/sim/ch8_fusion_2d_imu_uwb"
NLOS_DATASET = "data/sim/ch8_fusion_2d_imu_uwb_nlos"

# The demo's assumed range noise, from run_fusion_with_strategy.
ASSUMED_RANGE_NOISE_STD = 0.05

# Median of the chi-square distribution with one degree of freedom, and the
# 95% critical value the gate uses.
CHI2_1DOF_MEDIAN = 0.4549
CHI2_1DOF_95 = 3.841


@functools.cache
def _dataset(name: str, seed: int = 1):
    """One loaded (and possibly corrupted) dataset per name, shared."""
    if name == "los":
        return load_fusion_dataset(LOS_DATASET)
    if name == "nlos":
        return load_fusion_dataset(NLOS_DATASET)
    if name == "sporadic":
        return make_sporadic_outlier_dataset(_dataset("los"), seed=seed)
    raise ValueError(name)


@functools.cache
def _run(scenario: str, strategy: str, threshold=None, seed: int = 1):
    """One fusion run per distinct configuration.

    The whole file is about a dozen runs at a few seconds each, and several
    tests want the same ones. Memoised on the argument tuple, in the spirit of
    tests/example_runner.py -- note that functools.cache does not cache exceptions,
    so nothing here may assert.
    """
    return run_fusion_with_strategy(
        _dataset(scenario, seed) if scenario == "sporadic" else _dataset(scenario),
        strategy=strategy,
        use_gating=(strategy == "gating"),
        robust_threshold=threshold,
    )


def _errors(scenario: str, strategy: str, threshold=None, seed: int = 1):
    dataset = _dataset(scenario, seed) if scenario == "sporadic" else _dataset(scenario)
    return position_errors(_run(scenario, strategy, threshold, seed), dataset["truth"])


class TestTheRobustLossesActuallyFire(unittest.TestCase):
    """Link 1: the stage does something, before anyone asks whether it helps."""

    def test_huber_inflates_on_the_scenario_it_is_for(self):
        """A constant w_R is the signature of a loss that never engaged.

        This is the assertion the old code could not have passed: with the
        innovation handed over in metres it returned 1.0 on 2270 of 2271
        updates, so the set of distinct scale factors had size two and its
        maximum was 1.020.
        """
        scales = np.asarray(_run("sporadic", "huber")["robust_scales"])

        self.assertGreater(len(np.unique(scales)), 10)
        self.assertGreater(scales.max(), 2.0)
        # ...and it is genuinely selective rather than always on: Huber has a
        # dead zone, so most measurements must come through untouched.
        self.assertGreater(float(np.mean(scales == 1.0)), 0.8)

    def test_the_residual_handed_to_the_loss_is_normalized(self):
        """The units, asserted directly rather than inferred from behaviour.

        A normalized residual on clean data has median near 1; an innovation
        in metres against a 0.05 m sigma has median near 0.05. Two orders of
        magnitude apart, so this cannot pass for the wrong quantity.
        """
        r = np.asarray(_run("los", "baseline")["normalized_residuals"])
        innovations = np.asarray(_run("los", "baseline")["innovations"])

        self.assertAlmostEqual(
            float(np.median(r)),
            float(np.median(innovations)) / ASSUMED_RANGE_NOISE_STD,
            places=6,
        )
        self.assertGreater(float(np.median(r)), 0.3)
        self.assertLess(float(np.median(r)), 3.0)


class TestSporadicOutliersAreRepaired(unittest.TestCase):
    """Link 2: the case M-estimation is built for, and it must work."""

    def test_both_losses_beat_baseline_by_a_clear_margin(self):
        baseline = _errors("sporadic", "baseline")
        for strategy in ("huber", "cauchy"):
            with self.subTest(strategy=strategy):
                improved = _errors("sporadic", strategy)
                self.assertLess(
                    float(np.median(improved)), 0.6 * float(np.median(baseline))
                )
                self.assertLess(compute_rmse(improved), 0.85 * compute_rmse(baseline))

    def test_the_median_gain_survives_a_different_outlier_draw(self):
        """The reason the demo reports the median and not only the RMSE.

        RMSE here is dominated by the transient after the 57 deg/s turn at
        t = 52-54 s, where a manoeuvre the process model does not predict
        looks exactly like an outlier and the losses inflate R too. Measured
        over five draws the median gain is 57-62% every time while the RMSE
        gain ranges from +38% to -44%, so an RMSE-only claim here would be a
        property of seed 1 rather than of the method. Seed 0 is deliberately
        included: it is the draw where the RMSE claim fails.
        """
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                baseline = np.median(_errors("sporadic", "baseline", seed=seed))
                huber = np.median(_errors("sporadic", "huber", seed=seed))
                self.assertLess(float(huber), 0.6 * float(baseline))

    def test_the_scenario_really_does_have_an_inlier_majority(self):
        """Without this the test above could be passing on a different problem.

        Per anchor the mean range residual stays far below the +3 m bias,
        because only a twentieth of the ranges carry it -- which is what
        distinguishes this scenario from the NLOS one.
        """
        bias = per_anchor_range_bias(_dataset("sporadic"))

        self.assertTrue(np.all(np.abs(bias) < 0.5))
        self.assertGreater(_dataset("sporadic")["n_outliers"], 50)


class TestRobustnessIsCheapWhenItIsNotNeeded(unittest.TestCase):
    """Link 3: the constraint that actually pins the thresholds."""

    def test_huber_is_neutral_on_clean_data(self):
        """delta sits above the clean-data residual maximum, so nothing fires."""
        baseline = compute_rmse(_errors("los", "baseline"))
        huber = compute_rmse(_errors("los", "huber"))

        self.assertLess(huber, 1.01 * baseline)
        self.assertEqual(
            float(np.max(_run("los", "huber")["robust_scales"])),
            1.0,
        )

    def test_cauchy_always_fires_and_still_costs_nothing_measurable(self):
        """w_R = 1 + (r/c)^2 has no dead zone, so firing is structural.

        Firing is not the same as costing. At c = 20 the inflation on clean
        data is at most a couple of percent, and the run comes out at 0.998x
        baseline -- fractionally *better*, because mildly down-weighting the
        largest clean residuals is a small real improvement. The old form of
        this test asserted Cauchy must be strictly worse than baseline, which
        confused "always active" with "always a cost".
        """
        baseline = compute_rmse(_errors("los", "baseline"))
        cauchy = compute_rmse(_errors("los", "cauchy"))
        scales = np.asarray(_run("los", "cauchy")["robust_scales"])

        self.assertLess(abs(cauchy - baseline), 0.05 * baseline)
        self.assertTrue(np.all(scales >= 1.0))
        self.assertGreater(float(np.mean(scales > 1.0)), 0.99)

    def test_the_textbook_threshold_costs_where_the_loss_cannot_help(self):
        """Why delta is 10 sigma and not 1.345, pinned so it cannot drift back.

        The reason has moved, and the old reason is worth recording. It used
        to be that 1.345 *destroyed* the clean case -- five times baseline --
        because the clean tail ran to 17 sigma. That tail was the map-frame
        accelerometer, not the sensor: corrected, the clean maximum is 3.76
        sigma and the textbook threshold is fractionally **better** than
        baseline on clean data (0.963x).

        What rules it out now is the third scenario. At 1.345 the persistent
        NLOS run costs 16% (1.162x), because with no inlier majority the loss
        down-weights honest and biased links alike -- so the more aggressive
        the threshold the more good information it discards.
        """
        r = np.asarray(_run("los", "baseline")["normalized_residuals"])
        self.assertGreater(float(np.mean(r > 1.345)), 0.15)
        # The dead zone must still cover every clean residual.
        self.assertLess(float(np.max(r)), HUBER_DELTA_SIGMA)

        # Clean data no longer punishes the textbook value ...
        clean_base = compute_rmse(_errors("los", "baseline"))
        clean_textbook = compute_rmse(_errors("los", "huber", threshold=1.345))
        self.assertLess(clean_textbook, 1.05 * clean_base)

        # ... but the NLOS run does, which is what fixes the ceiling.
        nlos_base = compute_rmse(_errors("nlos", "baseline"))
        nlos_textbook = compute_rmse(_errors("nlos", "huber", threshold=1.345))
        self.assertGreater(nlos_textbook, 1.10 * nlos_base)

    def test_cauchy_scale_is_chosen_against_the_measured_clean_tail(self):
        """c = 20 keeps the inflation at the clean 99th percentile under 10%."""
        r = np.asarray(_run("los", "baseline")["normalized_residuals"])
        p99 = float(np.percentile(r, 99))

        self.assertLess(1.0 + (p99 / CAUCHY_C_SIGMA) ** 2, 1.10)


class TestPersistentNlosIsNotAnOutlierProblem(unittest.TestCase):
    """Link 4: the honest negative result, and the measurement behind it."""

    def test_half_the_anchors_are_biased_for_the_whole_run(self):
        """The premise. An M-estimator needs a minority to down-weight."""
        bias = per_anchor_range_bias(_dataset("nlos"))

        biased = np.abs(bias) > 10 * ASSUMED_RANGE_NOISE_STD
        self.assertEqual(int(np.sum(biased)), len(bias) // 2)

    def test_neither_loss_recovers_much_of_it(self):
        """Asserted as a *bound*, so a real repair fails here and is noticed.

        If someone teaches this demo to fix persistent bias -- by augmenting
        the state with a per-anchor bias, which is the right answer -- this
        test must go red rather than silently keep passing.
        """
        baseline = compute_rmse(_errors("nlos", "baseline"))
        for strategy in ("huber", "cauchy"):
            with self.subTest(strategy=strategy):
                self.assertGreater(
                    compute_rmse(_errors("nlos", strategy)), 0.9 * baseline
                )

    def test_the_filter_still_tracks_despite_the_bias(self):
        """A caveat worth pinning: the position is fine, the covariance is not."""
        self.assertLess(compute_rmse(_errors("nlos", "baseline")), 2.0)


class TestGatingDoesWhatItIsSpecifiedToDo(unittest.TestCase):
    """Link 5: the correction to a correction.

    This class used to be called ``TestGatingCollapsesEverywhereNotJustUnderNlos``
    and pinned the claim that a chi-square gate scores 24-26 m in every
    scenario, accepting under half of a *clean* run through a starvation
    feedback loop. The numbers were real and the attribution was wrong: the
    innovations were heavy-tailed because the shipped accelerometer was
    map-frame where every filter here integrates it as body-frame.

    The gate was the only strategy in the demo that tests its input against a
    distributional assumption, so it was the one that failed loudly -- and the
    conclusion drawn was that the gate was fragile. With the frame corrected
    it is the best-performing method in the table. **The component that breaks
    first under a bad input is not usually the broken one.**
    """

    def test_it_accepts_its_nominal_confidence_on_clean_data(self):
        """A 95% gate on consistent innovations accepts about 95%."""
        history = _run("los", "gating")
        total = history["n_uwb_accepted"] + history["n_uwb_rejected"]
        accepted = history["n_uwb_accepted"] / total

        self.assertGreater(accepted, 0.90)
        self.assertLess(accepted, 0.99)

    def test_it_costs_almost_nothing_on_clean_data(self):
        """Measured 1.074x baseline -- a gate is not free, but it is cheap."""
        baseline = compute_rmse(_errors("los", "baseline"))
        self.assertLess(compute_rmse(_errors("los", "gating")), 1.3 * baseline)

    def test_the_clean_filter_is_consistent(self):
        """Which is why the gate behaves: its assumption now holds.

        NIS median 0.467 against 0.455 for a consistent 1-DOF filter, and 95%
        of samples inside the 3.84 gate.
        """
        nis = np.asarray(_run("los", "baseline")["nis"])

        self.assertLess(abs(float(np.median(nis)) - CHI2_1DOF_MEDIAN), 0.15)
        self.assertGreater(float(np.mean(nis < CHI2_1DOF_95)), 0.90)

    def test_it_rejects_the_biased_half_under_persistent_nlos(self):
        """Two of four anchors are biased, and it accepts about half."""
        history = _run("nlos", "gating")
        total = history["n_uwb_accepted"] + history["n_uwb_rejected"]
        accepted = history["n_uwb_accepted"] / total

        self.assertGreater(accepted, 0.35)
        self.assertLess(accepted, 0.65)

    def test_the_nlos_filter_really_is_overconfident(self):
        """Still true, and still worth keeping: NLOS makes it much worse."""
        nis = np.asarray(_run("nlos", "baseline")["nis"])

        self.assertGreater(np.median(nis), 20 * CHI2_1DOF_MEDIAN)
        self.assertLess(float(np.mean(nis < CHI2_1DOF_95)), 0.5)

    def test_rejecting_a_persistent_bias_beats_reweighting_it(self):
        """The contrast the figure exists to draw, with the sign it really has.

        On persistent NLOS the gate removes the biased anchors outright and
        reaches 0.048x baseline. Neither M-estimator moves the number at all
        (0.997x and 1.004x), because a reweighting cannot separate honest from
        biased links when there is no inlier majority.
        """
        baseline = compute_rmse(_errors("nlos", "baseline"))

        self.assertLess(compute_rmse(_errors("nlos", "gating")), 0.2 * baseline)
        for method in ("huber", "cauchy"):
            with self.subTest(method=method):
                self.assertGreater(
                    compute_rmse(_errors("nlos", method)), 0.9 * baseline
                )


class TestTheDatasetsAreWhatTheClaimsAssume(unittest.TestCase):
    """Guard the guard: these tests read shipped bytes, so check them."""

    def test_nlos_ranges_really_are_corrupted(self):
        dataset = _dataset("nlos")
        truth = dataset["truth"]
        anchors = np.asarray(dataset["uwb_anchors"])
        idx = np.searchsorted(truth["t"], dataset["uwb"]["t"]).clip(
            0, len(truth["t"]) - 1
        )
        true_range = np.linalg.norm(
            truth["p_xy"][idx][:, None, :] - anchors[None, :, :2], axis=2
        )
        error = np.abs(np.asarray(dataset["uwb"]["ranges"]) - true_range)
        error = error[np.isfinite(error)]

        fraction_biased = float(np.mean(error > 10 * ASSUMED_RANGE_NOISE_STD))
        self.assertGreater(fraction_biased, 0.2)

    def test_the_los_dataset_carries_no_bias_at_all(self):
        """Otherwise "robustness is free on clean data" is measured on dirt."""
        self.assertTrue(np.all(np.abs(per_anchor_range_bias(_dataset("los"))) < 0.01))


if __name__ == "__main__":
    unittest.main()
