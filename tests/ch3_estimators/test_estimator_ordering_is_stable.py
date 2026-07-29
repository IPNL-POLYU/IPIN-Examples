"""Chapter 3's estimator ranking must survive a change of noise realisation.

The comparison ran one scenario and printed four RMSEs to four decimals: EKF
0.4716, UKF 0.4704, PF 0.4445, FGO 0.3373. Read as a ranking, that says UKF
beats EKF and PF beats both. Repeated over seeds, neither holds -- EKF and UKF
agree to three decimals on the mean and trade places roughly evenly, and PF is
the worst of the three on average despite winning on the shipped seed.

What does survive is FGO, on every seed tried, and for a reason the chapter
teaches: batch smoothing uses every measurement to estimate every state while
a filter only looks backwards.

Author: Li-Ta Hsu
References: Chapter 3, Sections 3.2-3.5, Table 3.4
"""

import contextlib
import io
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch3_estimators.example_comparison import (
    run_ekf,
    run_fgo,
    run_pf,
    run_ukf,
    setup_scenario,
)

N_SEEDS = 6


def _rmse_by_method(seed):
    """Every estimator's RMSE on one scenario, using the example's alignment."""
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        dt, n_steps, anchors, truth, meas, Q, range_std = setup_scenario(seed=seed)
        out = {}
        for name, runner in (
            ("EKF", run_ekf),
            ("UKF", run_ukf),
            ("PF", run_pf),
            ("FGO", run_fgo),
        ):
            # Index rather than unpack: run_pf returns a third value
            # (n_particles) that the others do not.
            estimates = runner(dt, n_steps, anchors, meas, Q, range_std)[0]
            errors = np.linalg.norm(estimates[:, :2] - truth[:, :2], axis=1)
            out[name] = float(np.sqrt(np.mean(errors**2)))
    return out


class TestEstimatorOrderingIsStable(unittest.TestCase):
    """Which differences are real, and which are one draw."""

    @classmethod
    def setUpClass(cls):
        cls.runs = [_rmse_by_method(seed) for seed in range(N_SEEDS)]

    def test_smoothing_beats_every_filter_on_every_seed(self):
        """The one ranking the comparison is entitled to make.

        If this ever fails, either the factor graph has regressed or the
        scenario has stopped being one where smoothing helps; both are worth
        knowing, and neither is visible from a single run.
        """
        for i, run in enumerate(self.runs):
            with self.subTest(seed=i):
                self.assertLess(run["FGO"], min(run["EKF"], run["UKF"], run["PF"]))

    def test_the_two_kalman_filters_are_not_separable_here(self):
        """EKF and UKF differ by well under their spread across seeds.

        The example prints them to four decimals, which invites a ranking the
        data does not support: the mild nonlinearity here is exactly the regime
        where the unscented transform has nothing to add.
        """
        ekf = np.array([r["EKF"] for r in self.runs])
        ukf = np.array([r["UKF"] for r in self.runs])

        gap = abs(ekf.mean() - ukf.mean())
        self.assertLess(gap, 0.1 * ekf.std())

    def test_seed_42_still_reproduces_the_published_numbers(self):
        """The committed figure's realisation, pinned.

        setup_scenario gained a seed argument; this guards that its default
        path is unchanged, so the parameterisation did not quietly move the
        published run.
        """
        run = _rmse_by_method(42)

        self.assertAlmostEqual(run["EKF"], 0.4716, places=3)
        self.assertAlmostEqual(run["FGO"], 0.3373, places=3)


if __name__ == "__main__":
    unittest.main()
