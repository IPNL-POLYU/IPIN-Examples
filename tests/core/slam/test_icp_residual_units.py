"""`icp_point_to_point` reports alignment error in metres, not as a sum.

It used to return Eq. (7.10)'s objective directly -- a sum of squared errors
over the matched pairs -- while every caller gated it with a threshold named
and documented in metres:

    core/slam/frontend_2d.py     max_icp_residual=1.0   "Maximum ICP residual"
    core/slam/loop_closure_2d.py max_icp_residual=0.2   "Maximum ICP residual"

A sum cannot be a distance, because it grows with the number of points matched.
Matching a 360-point scan against a submap voxelised at 0.2 m costs about
360 * 0.058^2 = 1.2 from quantisation alone, so the ch7 front-end's threshold of
1.5 was really demanding 0.065 m RMS against a 0.058 m floor. It rejected every
alignment it was ever given and silently returned the odometry prediction
instead, for 145 consecutive steps.

`test_residual_does_not_grow_with_the_number_of_points` is the test that would
have caught it, and it is the shape worth reaching for whenever a scalar is
compared against a threshold: feed the same *quality* of input at two different
sizes and require the number to stay put. Checking one size only tells you
nothing about what the units are.

Author: Li-Ta Hsu
References: Chapter 7, Section 7.3.1 (ICP), Eq. (7.10), Eq. (7.11)
"""

import unittest

import numpy as np

from core.slam.scan_matching import compute_icp_residual, icp_point_to_point


def _ring(n, radius=5.0, seed=0):
    """n points on a circle -- a scan with enough structure to pin a pose."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return pts, rng


class TestResidualIsAnRmsDistance(unittest.TestCase):

    def test_perfect_alignment_scores_zero(self):
        scan, _ = _ring(120)
        _, _, residual, converged = icp_point_to_point(scan, scan)

        self.assertTrue(converged)
        self.assertLess(residual, 1e-6)

    def test_residual_does_not_grow_with_the_number_of_points(self):
        """The discriminating test: a sum would scale with N, a distance will not.

        Both clouds carry the same per-point noise, so the achievable alignment
        quality is identical and only the point count differs. Under the old
        sum-of-squares return these two differed by roughly 4x.
        """
        sigma = 0.05
        residuals = {}
        for n in (100, 400):
            scan, rng = _ring(n, seed=1)
            target = scan + rng.normal(0, sigma, scan.shape)
            _, _, residual, _ = icp_point_to_point(scan, target, max_iterations=50)
            residuals[n] = residual

        self.assertAlmostEqual(residuals[100], residuals[400], delta=0.2 * residuals[100])

    def test_residual_tracks_the_noise_it_is_measuring(self):
        """It is in metres, so it should read like the displacement it sees."""
        scan, rng = _ring(300, seed=2)
        for sigma in (0.02, 0.10):
            target = scan + rng.normal(0, sigma, scan.shape)
            _, _, residual, _ = icp_point_to_point(scan, target, max_iterations=50)

            self.assertGreater(residual, 0.3 * sigma)
            self.assertLess(residual, 3.0 * sigma)

    def test_the_raw_eq_7_10_objective_is_still_available(self):
        """The book's cost function is unchanged; only ICP's report is normalised."""
        source = np.array([[1.0, 0.0], [0.0, 1.0]])
        target = np.array([[1.3, 0.0], [0.0, 1.4]])

        # 0.3^2 + 0.4^2 = 0.25, a sum -- not divided by the 2 pairs.
        self.assertAlmostEqual(compute_icp_residual(source, target), 0.25, places=12)


class TestCorrespondenceGatingPreventsDivergence(unittest.TestCase):
    """Eq. (7.11)'s d_threshold, which the front-end never used to pass.

    Without it, scan points with no nearby counterpart are still paired with
    whatever is closest, however far away, and the SVD step is dragged toward
    that. On the ch7 square loop this produced residuals of 3.9e3 and 2.1e13 on
    individual steps.
    """

    @staticmethod
    def _partial_overlap():
        """A target that covers only part of the source, plus a distant blob."""
        source, rng = _ring(200, seed=3)
        target = source[:120] + rng.normal(0, 0.01, (120, 2))
        stray = rng.normal(0, 0.5, (40, 2)) + np.array([80.0, 80.0])
        return source, np.vstack([target, stray])

    def test_ungated_icp_can_be_dragged_far_off(self):
        """Documents why the gate is needed, so the next test means something."""
        source, target = self._partial_overlap()
        _, _, ungated, _ = icp_point_to_point(
            source, target, max_iterations=50, max_correspondence_distance=None)
        _, _, gated, _ = icp_point_to_point(
            source, target, max_iterations=50, max_correspondence_distance=1.0)

        self.assertLess(gated, ungated)

    def test_gating_keeps_the_alignment_near_the_truth(self):
        source, target = self._partial_overlap()
        pose, _, residual, _ = icp_point_to_point(
            source, target, max_iterations=50, max_correspondence_distance=1.0)

        self.assertLess(np.linalg.norm(pose[:2]), 1.0)
        self.assertLess(residual, 0.5)


if __name__ == "__main__":
    unittest.main()
