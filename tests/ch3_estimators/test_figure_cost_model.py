"""Tests for the computational cost the Chapter 3 comparison figure plots.

That panel used to plot measured seconds, which cannot be committed to a tracked
figure: the number moves on every run, so the figure churned whenever it was
regenerated. It plots model evaluations instead -- one state vector pushed
through one process or measurement model.

The appeal of the count is that each estimator's ratio to the EKF is its own
defining parameter, so the figure explains itself. Those ratios are the claims,
and they are what gets tested here.

Author: Li-Ta Hsu
References: Chapter 3, Sections 3.3-3.5, Eqs. (3.32)-(3.34)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

from ch3_estimators.example_comparison import model_evaluation_counts

N_STEPS = 30
N_STATES = 4
N_PARTICLES = 300
FGO_ITERATIONS = 10


class TestModelEvaluationCounts(unittest.TestCase):
    """Check the cost ratios the figure asserts."""

    def setUp(self):
        self.counts = model_evaluation_counts(
            n_steps=N_STEPS,
            n_states=N_STATES,
            n_particles=N_PARTICLES,
            fgo_iterations=FGO_ITERATIONS,
        )

    def test_counts_are_plain_integers(self):
        """A float here would mean a measurement leaked back in."""
        for method, count in self.counts.items():
            with self.subTest(method=method):
                self.assertIsInstance(count, int, f"{method} is not an int")

    def test_repeated_calls_agree(self):
        """The whole point: the number cannot move between runs."""
        again = model_evaluation_counts(
            n_steps=N_STEPS,
            n_states=N_STATES,
            n_particles=N_PARTICLES,
            fgo_iterations=FGO_ITERATIONS,
        )
        self.assertEqual(again, self.counts)

    def test_ekf_is_two_evaluations_per_step(self):
        """One mean propagation, one measurement prediction."""
        self.assertEqual(self.counts["EKF"], 2 * N_STEPS)

    def test_ukf_costs_its_sigma_points(self):
        """The UKF/EKF ratio is exactly 2n+1 (Section 3.4).

        This is the figure's whole explanation of the UKF: it does what the EKF
        does, once per sigma point.
        """
        n_sigma = 2 * N_STATES + 1

        self.assertEqual(self.counts["UKF"], self.counts["EKF"] * n_sigma)

    def test_particle_filter_costs_its_particles(self):
        """The PF/EKF ratio is exactly the particle count.

        This is why the panel needs a log axis: 300 particles put the PF two and
        a half decades above everything else.
        """
        self.assertEqual(self.counts["PF"], self.counts["EKF"] * N_PARTICLES)

    def test_fgo_costs_the_iterations_it_actually_took(self):
        """FGO scales with real iterations, not the limit it was given.

        run_fgo reads the iteration count off the cost history that optimize()
        returns, so an early-converging solve is reported as cheaper rather than
        being charged for all ten.
        """
        self.assertEqual(self.counts["FGO"], self.counts["EKF"] * FGO_ITERATIONS)

        half = model_evaluation_counts(
            n_steps=N_STEPS,
            n_states=N_STATES,
            n_particles=N_PARTICLES,
            fgo_iterations=FGO_ITERATIONS // 2,
        )
        self.assertEqual(half["FGO"] * 2, self.counts["FGO"])

    def test_batch_smoother_is_not_the_expensive_one(self):
        """FGO revisits every step ten times and still costs less than the PF.

        Worth pinning: the intuition that a batch smoother must be the most
        expensive option is wrong here, and the figure says so.
        """
        self.assertLess(self.counts["FGO"], self.counts["PF"])
        self.assertGreater(self.counts["FGO"], self.counts["EKF"])


if __name__ == "__main__":
    unittest.main()
