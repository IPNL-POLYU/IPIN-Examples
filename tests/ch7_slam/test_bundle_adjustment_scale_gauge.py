"""A prior on pose 0 alone does not fix monocular bundle adjustment's gauge.

`example_bundle_adjustment.py` used to add a single weak prior on pose 0 and
say, in a comment, that this "prevent[s] gauge freedom". It pins translation
and yaw (3 DOF), but monocular BA has a fourth, unrelated gauge freedom: scale.
Project a 3D point in camera frame through the pinhole model,
`u = fx * X/Z + cx`, and the result depends only on the ratios `X/Z`, `Y/Z` --
so scaling every camera translation and every landmark position by the same
factor `s` about ANY fixed point reprojects identically for every observation,
at every pose, for every `s`. A prior on pose 0 cannot see this: scaling about
pose 0 leaves pose 0 exactly where it was, so its own residual -- and the
whole reprojection cost -- stays flat while the reconstruction itself drifts
arbitrarily far from the truth.

Measured on the noisy initial estimate `example_bundle_adjustment.main()`
optimizes from (n_poses=8, n_landmarks=6, seed=42), with only the pose-0
prior: scaling by s=1.5 about camera 0 leaves `graph.compute_error()` at
4940699.4531553555, identical to s=1.0 to 1 part in 1e16 (float64 epsilon),
while landmark RMSE against ground truth is already nonzero and grows further.
The fix -- also present in this file's test names -- anchors a SECOND pose
(not collocated with pose 0) with the same weak prior. That pose does move
under the scaling, so its residual grows with `s` and the flat direction in
the Hessian is gone.

This is the same "sum of squares is not a distance" family CLAUDE.md already
tracks for this chapter (see test_bundle_adjustment_reports_pixels.py), but
the property under test here is different: not a units error, a genuine
degree of freedom the cost function cannot see, and no amount of `--tol` or
extra iterations would ever find it.

References: Chapter 7, Section 7.4.2 (bundle adjustment), and the "monocular
scale degeneracy" noted around Eq. (7.67) (docs/equation_index.yml).

Author: Li-Ta Hsu
"""

import math
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch7_slam.example_bundle_adjustment import (
    PIXEL_NOISE_STD,
    add_noise_to_estimates,
    generate_camera_trajectory,
    generate_landmarks,
    generate_observations,
    reprojection_residuals_px,
    rms,
)
from core.estimators.factor_graph import FactorGraph
from core.slam import CameraIntrinsics, create_reprojection_factor
from core.slam.factors import create_prior_factor
from tests.example_runner import run_example

# The exact scenario example_bundle_adjustment.main() builds. Duplicated
# rather than imported because main() does not expose graph construction as a
# reusable function -- the same choice test_bundle_adjustment_reports_pixels.py
# makes for its own synthetic graphs.
N_POSES = 8
N_LANDMARKS = 6


def _build_graph(second_pose_id=None):
    """Build the noisy-initial-estimate BA graph, with 1 or 2 gauge priors.

    No optimization is run: the invariance under test is a property of the
    reprojection + prior model itself, true for ANY variable assignment, not
    only a converged one -- verified by checking it holds here, on the raw
    noisy initial estimate, exactly as it does on the optimized solution.

    Args:
        second_pose_id: If given, also add a weak prior on this pose (the
            fix). If None, only pose 0 gets a prior (the bug).

    Returns:
        (graph, n_reprojection, poses_true, landmarks_true)
    """
    np.random.seed(42)
    intrinsics = CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        k1=-0.05,
        k2=0.01,
        p1=0.001,
        p2=0.001,
    )
    poses_true = generate_camera_trajectory(n_poses=N_POSES, radius=5.0)
    landmarks_true = generate_landmarks(n_landmarks=N_LANDMARKS, area_size=4.0)
    observations = generate_observations(
        poses_true,
        landmarks_true,
        intrinsics,
        observation_noise=PIXEL_NOISE_STD,
        min_depth=1.0,
        max_depth=12.0,
    )
    poses_init, landmarks_init = add_noise_to_estimates(
        poses_true, landmarks_true, pose_noise=0.3, landmark_noise=0.5
    )

    graph = FactorGraph()
    for i, pose in enumerate(poses_init):
        graph.add_variable(i, pose)
    for i, landmark in enumerate(landmarks_init):
        graph.add_variable(N_POSES + i, landmark)

    n_reprojection = 0
    pixel_info = np.eye(2) / (PIXEL_NOISE_STD**2)
    for pose_id, obs_list in observations.items():
        for landmark_id, observed_pixel in obs_list:
            graph.add_factor(
                create_reprojection_factor(
                    camera_pose_id=pose_id,
                    landmark_id=N_POSES + landmark_id,
                    observed_pixel=observed_pixel,
                    camera_intrinsics=intrinsics,
                    information=pixel_info,
                )
            )
            n_reprojection += 1

    prior_info = np.diag([10.0, 10.0, 10.0])
    graph.add_factor(create_prior_factor(0, poses_true[0], information=prior_info))
    if second_pose_id is not None:
        graph.add_factor(
            create_prior_factor(
                second_pose_id, poses_true[second_pose_id], information=prior_info
            )
        )

    return graph, n_reprojection, poses_true, landmarks_true


def _scale_about_camera0(variables, s, n_poses=N_POSES, n_landmarks=N_LANDMARKS):
    """Scale every camera x/y and landmark x/y/z by `s` about camera 0.

    Camera 0's own height is 0 (`generate_observations` fixes
    `camera_height = 0.0`), so landmark z is scaled about 0 too. For ANY
    fixed centre, this leaves `landmark - camera` vectors -- and therefore
    every reprojection -- scaled by exactly `s`, which is invariant for the
    pinhole model (see module docstring). Camera 0 itself does not move,
    because `s * (cam0 - cam0) == 0` for any s.
    """
    cam0_xy = variables[0][:2].copy()
    scaled = {}
    for i in range(n_poses):
        pose = variables[i].copy()
        pose[0] = cam0_xy[0] + s * (pose[0] - cam0_xy[0])
        pose[1] = cam0_xy[1] + s * (pose[1] - cam0_xy[1])
        scaled[i] = pose
    for i in range(n_landmarks):
        lm = variables[n_poses + i].copy()
        lm[0] = cam0_xy[0] + s * (lm[0] - cam0_xy[0])
        lm[1] = cam0_xy[1] + s * (lm[1] - cam0_xy[1])
        lm[2] = s * lm[2]
        scaled[n_poses + i] = lm
    return scaled


class TestASinglePosePriorLeavesScaleFree(unittest.TestCase):
    """The bug shape: one prior pins the rigid gauge but not scale."""

    def test_cost_is_invariant_to_scale(self):
        graph, _n_reprojection, _poses_true, _landmarks_true = _build_graph(
            second_pose_id=None
        )
        base_vars = {k: v.copy() for k, v in graph.variables.items()}
        base_cost = graph.compute_error()

        for s in (1.1, 1.5, 2.0):
            graph.variables = _scale_about_camera0(base_vars, s)
            cost = graph.compute_error()
            self.assertTrue(
                math.isclose(cost, base_cost, rel_tol=1e-9),
                f"cost changed under s={s}: {cost} vs {base_cost} "
                f"(the gauge would then not need fixing)",
            )

    def test_reprojection_error_is_also_invariant(self):
        """Reprojection alone is exactly scale-blind, with or without a fix.

        Isolates where the (non-)information comes from: the vision terms
        never distinguish `s`, at any stage of this file.
        """
        graph, n_reprojection, _poses_true, _landmarks_true = _build_graph(
            second_pose_id=None
        )
        base_vars = {k: v.copy() for k, v in graph.variables.items()}
        base_px = rms(reprojection_residuals_px(graph, n_reprojection))

        for s in (1.1, 1.5, 2.0):
            graph.variables = _scale_about_camera0(base_vars, s)
            px = rms(reprojection_residuals_px(graph, n_reprojection))
            self.assertTrue(math.isclose(px, base_px, rel_tol=1e-9))

    def test_the_invariant_solution_is_nevertheless_wrong(self):
        """Same cost, different answer -- why the invariance matters.

        An optimizer minimizing this cost has no way to prefer s=1 over any
        other s: the scaled reconstruction disagrees with ground truth by a
        large, growing amount while remaining exactly as good a fit.
        """
        graph, _n_reprojection, poses_true, landmarks_true = _build_graph(
            second_pose_id=None
        )
        base_vars = {k: v.copy() for k, v in graph.variables.items()}

        scaled = _scale_about_camera0(base_vars, 1.5)
        landmark_positions = np.array(
            [scaled[N_POSES + i] for i in range(N_LANDMARKS)]
        )
        landmark_rmse = np.sqrt(
            np.mean(np.sum((landmark_positions - landmarks_true) ** 2, axis=1))
        )
        pose_rmse = np.sqrt(
            np.mean(
                [
                    np.sum((scaled[i][:2] - poses_true[i][:2]) ** 2)
                    for i in range(N_POSES)
                ]
            )
        )
        self.assertGreater(landmark_rmse, 1.0)
        self.assertGreater(pose_rmse, 1.0)


class TestASecondPosePriorFixesTheScaleGauge(unittest.TestCase):
    """The fix: anchor a second, non-collocated pose with the same weak prior.

    Mirrors `example_bundle_adjustment.main()`'s `scale_ref_pose_id =
    n_poses // 2`.
    """

    def test_cost_increases_with_scale(self):
        graph, _n_reprojection, _poses_true, _landmarks_true = _build_graph(
            second_pose_id=N_POSES // 2
        )
        base_vars = {k: v.copy() for k, v in graph.variables.items()}
        base_cost = graph.compute_error()

        previous = base_cost
        for s in (1.1, 1.5, 2.0):
            graph.variables = _scale_about_camera0(base_vars, s)
            cost = graph.compute_error()
            self.assertGreater(
                cost, previous, f"cost did not rise moving out to s={s}"
            )
            previous = cost

        # base_cost here is ~4.94e6 (dominated by the noisy initial guess's
        # reprojection error, not yet optimized), so a relative check would
        # bury the effect. The floating-point noise floor for this magnitude,
        # measured for the invariant (unfixed) case above, is ~9e-10
        # absolute; the real, prior-driven increase at s=1.5 is ~223. A
        # threshold of 1.0 sits nine orders of magnitude above the floor and
        # two below the real effect -- nowhere near either.
        graph.variables = _scale_about_camera0(base_vars, 1.5)
        self.assertGreater(graph.compute_error() - base_cost, 1.0)

    def test_reprojection_error_is_unaffected_by_the_fix(self):
        """The extra information is entirely in the prior, not the vision.

        Confirms the fix works by adding real (if weak) external information,
        not by changing what the visual measurements alone can say -- which
        cannot change, by the previous class's invariance tests.
        """
        graph, n_reprojection, _poses_true, _landmarks_true = _build_graph(
            second_pose_id=N_POSES // 2
        )
        base_vars = {k: v.copy() for k, v in graph.variables.items()}
        base_px = rms(reprojection_residuals_px(graph, n_reprojection))

        for s in (1.1, 1.5, 2.0):
            graph.variables = _scale_about_camera0(base_vars, s)
            px = rms(reprojection_residuals_px(graph, n_reprojection))
            self.assertTrue(math.isclose(px, base_px, rel_tol=1e-9))


class TestTheShippedExampleUsesTheFix(unittest.TestCase):
    """End to end: the real `main()`, not a parallel reconstruction of it.

    The tests above build their own graph with the underlying primitives, so
    they cannot see a regression in `main()` itself (e.g. someone reverting
    to a single prior, or moving `scale_ref_pose_id` onto a pose collocated
    with pose 0). This test runs the actual example as a subprocess and reads
    its own reported numbers.
    """

    def test_factor_count_and_accuracy(self):
        run = run_example("ch7_slam.example_bundle_adjustment")
        self.assertEqual(run.process.returncode, 0, run.process.stderr)
        stdout = run.process.stdout

        self.assertIn("Factors: 48 (46 reprojection + 2 prior)", stdout)

        landmark_rmse = _extract_metre_value(stdout, "Landmark RMSE (optimized):")
        pose_rmse = _extract_metre_value(stdout, "Pose RMSE (optimized):")

        # A single prior (the bug this file guards against) gives ~0.114 m /
        # ~0.151 m on this exact seeded scenario (see the module docstring's
        # sibling numbers); the fix gives ~0.048 m / ~0.060 m. These
        # thresholds sit clearly between the two, not near either.
        self.assertLess(landmark_rmse, 0.08)
        self.assertLess(pose_rmse, 0.09)


def _extract_metre_value(stdout: str, label: str) -> float:
    """The number in a `"<label> <value> m"` line of an example's stdout."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(label):
            return float(line[len(label) :].split()[0])
    raise AssertionError(f"{label!r} not found in stdout:\n{stdout}")


if __name__ == "__main__":
    unittest.main()
