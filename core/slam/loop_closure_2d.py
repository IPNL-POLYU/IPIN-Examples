"""Observation-Based Loop Closure Detection for 2D SLAM.

This module implements loop closure detection using scan descriptor similarity
as the PRIMARY candidate selection criterion, with optional distance gating as
a SECONDARY filter. This replaces oracle-based position-only detection.

The detection pipeline is:
    1. CANDIDATE GENERATION: Find scans with similar descriptors
    2. GEOMETRIC VERIFICATION: Run ICP to verify loop closure
    3. QUALITY CHECK: Accept only high-quality matches

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .scan_descriptor_2d import (
    compute_descriptor_similarity,
    batch_compute_descriptors,
)
from .scan_matching import compute_icp_covariance, icp_point_to_point
from .se2 import se2_relative

# Floor on the loop-closure covariance's diagonal (m^2 for x/y, rad^2 for
# yaw). compute_icp_covariance scales sigma as sqrt(residual / N), which is
# exactly singular at residual == 0 -- a coincidentally perfect match. That
# is far below anything a genuine match produces: measured 1.6e-6 to 1.8e-6
# on this module's own square-trajectory dataset (147 closures, ICP residual
# 0.024-0.054 m over ~360-point scans). This floor sits ~1000x below that, so
# it only trips on a near-zero residual, not on an ordinarily good match; its
# job is to stop np.linalg.inv (called by callers building the loop-closure
# information matrix, e.g. ch7_slam/example_pose_graph_slam.py) from raising
# on a singular matrix or returning a near-infinite weight for one closure.
_MIN_LOOP_CLOSURE_VARIANCE = 1e-9


@dataclass
class LoopClosureCandidate:
    """Loop closure candidate with similarity score.

    Attributes:
        i: Query scan index.
        j: Match scan index (j < i).
        descriptor_similarity: Descriptor similarity score.
        distance: Optional position distance (if poses provided).
    """

    i: int
    j: int
    descriptor_similarity: float
    distance: Optional[float] = None


@dataclass
class LoopClosure:
    """Verified loop closure with geometric transformation.

    Attributes:
        i: Query scan index.
        j: Match scan index (j < i).
        rel_pose: Relative pose from j to i [dx, dy, dyaw].
        covariance: 3x3 covariance matrix for the constraint.
        descriptor_similarity: Descriptor similarity score.
        icp_residual: ICP alignment residual.
        icp_iterations: Number of ICP iterations.
    """

    i: int
    j: int
    rel_pose: np.ndarray
    covariance: np.ndarray
    descriptor_similarity: float
    icp_residual: float
    icp_iterations: int


class LoopClosureDetector2D:
    """Observation-based loop closure detector for 2D LiDAR SLAM.

    This detector finds loop closures using scan descriptor similarity as the
    primary criterion, with optional distance-based filtering as a secondary
    check. Geometric verification via ICP ensures only valid loop closures
    are returned.

    Attributes:
        n_bins: Number of bins for range histogram descriptor.
        max_range: Maximum range for descriptor (meters).
        min_time_separation: Minimum time steps between query and match.
        min_descriptor_similarity: Minimum descriptor similarity threshold.
        max_candidates: Maximum number of candidates to verify per query.
        max_distance: Optional maximum position distance for candidates.
        max_icp_residual: Maximum ICP residual to accept loop closure.
        icp_max_iterations: Maximum ICP iterations.
        icp_tolerance: ICP convergence tolerance.

    Example:
        >>> detector = LoopClosureDetector2D(min_descriptor_similarity=0.7)
        >>>
        >>> # Detect loop closures
        >>> loop_closures = detector.detect(
        ...     scans=scans,
        ...     poses=poses,  # Optional, for distance gating
        ... )
        >>>
        >>> print(f"Found {len(loop_closures)} loop closures")
    """

    def __init__(
        self,
        n_bins: int = 32,
        max_range: float = 10.0,
        min_time_separation: int = 10,
        min_descriptor_similarity: float = 0.7,
        max_candidates: int = 5,
        max_distance: Optional[float] = None,
        max_icp_residual: float = 0.1,
        icp_max_iterations: int = 50,
        icp_tolerance: float = 1e-4,
    ):
        """Initialize loop closure detector.

        Args:
            n_bins: Number of histogram bins for descriptor.
            max_range: Maximum range for descriptor histogram.
            min_time_separation: Minimum time steps between i and j.
                                Prevents matching with immediate neighbors.
            min_descriptor_similarity: Minimum descriptor similarity to consider
                                      as candidate (primary filter).
            max_candidates: Maximum number of candidates to verify per query.
            max_distance: Optional maximum position distance between candidates
                         (secondary filter). Set to None to disable distance gating.
            max_icp_residual: Maximum alignment error to accept a loop
                closure, as RMS distance per correspondence in metres.
                icp_point_to_point reports RMS; this was 0.2 when it
                still returned a sum of squared errors, which is a
                different quantity and scales with the scan size.
            icp_max_iterations: Maximum ICP iterations for verification.
            icp_tolerance: ICP convergence tolerance.
        """
        self.n_bins = n_bins
        self.max_range = max_range
        self.min_time_separation = min_time_separation
        self.min_descriptor_similarity = min_descriptor_similarity
        self.max_candidates = max_candidates
        self.max_distance = max_distance
        self.max_icp_residual = max_icp_residual
        self.icp_max_iterations = icp_max_iterations
        self.icp_tolerance = icp_tolerance

    def detect(
        self,
        scans: List[np.ndarray],
        poses: Optional[List[np.ndarray]] = None,
    ) -> List[LoopClosure]:
        """Detect loop closures in a sequence of scans.

        Pipeline:
            1. Compute descriptors for all scans
            2. For each query scan i:
                a. Find candidates j < i with high descriptor similarity
                b. Optionally filter by position distance (if poses provided)
                c. Verify with ICP geometric alignment
                d. Accept if ICP converges with low residual

        Args:
            scans: List of N scans, each with shape (M_i, 2) in robot frame.
            poses: Optional list of N poses [x, y, yaw] for distance gating.

        Returns:
            List of verified loop closures, sorted by query index i.

        Example:
            >>> scans = [scan0, scan1, ..., scanN]
            >>> poses = [pose0, pose1, ..., poseN]  # Optional
            >>>
            >>> detector = LoopClosureDetector2D()
            >>> loop_closures = detector.detect(scans, poses)
            >>>
            >>> for lc in loop_closures:
            ...     print(f"Loop: {lc.j} -> {lc.i}, sim={lc.descriptor_similarity:.3f}")
        """
        n_scans = len(scans)

        if n_scans < self.min_time_separation + 1:
            # Not enough scans for loop closure
            return []

        # 1. Compute descriptors for all scans
        descriptors = batch_compute_descriptors(
            scans, n_bins=self.n_bins, max_range=self.max_range
        )

        loop_closures = []

        # 2. For each query scan (starting after min_time_separation)
        for i in range(self.min_time_separation, n_scans):
            # Find candidates using descriptor similarity
            candidates = self._find_candidates(i, descriptors, poses)

            if len(candidates) == 0:
                continue

            # Verify candidates with ICP
            for candidate in candidates:
                j = candidate.j

                # Run ICP to verify geometric consistency
                verified = self._verify_candidate(
                    scans[i],
                    scans[j],
                    poses[i] if poses else None,
                    poses[j] if poses else None,
                )

                if verified is not None:
                    # Accept loop closure
                    rel_pose, covariance, residual, iters = verified

                    loop_closure = LoopClosure(
                        i=i,
                        j=j,
                        rel_pose=rel_pose,
                        covariance=covariance,
                        descriptor_similarity=candidate.descriptor_similarity,
                        icp_residual=residual,
                        icp_iterations=iters,
                    )
                    loop_closures.append(loop_closure)

        return loop_closures

    def _find_candidates(
        self,
        query_idx: int,
        descriptors: np.ndarray,
        poses: Optional[List[np.ndarray]],
    ) -> List[LoopClosureCandidate]:
        """Find loop closure candidates for a query scan.

        Primary filter: Descriptor similarity
        Secondary filter (optional): Position distance

        Args:
            query_idx: Query scan index i.
            descriptors: Array of descriptors, shape (N, n_bins).
            poses: Optional list of poses for distance gating.

        Returns:
            List of candidates, sorted by descriptor similarity (descending).
        """
        query_desc = descriptors[query_idx]

        candidates = []

        # Compute similarity to all previous scans (respecting time separation)
        for j in range(0, query_idx - self.min_time_separation):
            match_desc = descriptors[j]

            # Primary filter: Descriptor similarity
            similarity = compute_descriptor_similarity(
                query_desc, match_desc, method="cosine"
            )

            if similarity < self.min_descriptor_similarity:
                continue

            # Secondary filter: Position distance (optional)
            if self.max_distance is not None and poses is not None:
                distance = np.linalg.norm(poses[query_idx][:2] - poses[j][:2])

                if distance > self.max_distance:
                    continue
            else:
                distance = None

            candidates.append(
                LoopClosureCandidate(
                    i=query_idx,
                    j=j,
                    descriptor_similarity=similarity,
                    distance=distance,
                )
            )

        # Sort by descriptor similarity (descending) and limit to top K
        candidates.sort(key=lambda c: c.descriptor_similarity, reverse=True)
        return candidates[: self.max_candidates]

    def _verify_candidate(
        self,
        scan_i: np.ndarray,
        scan_j: np.ndarray,
        pose_i: Optional[np.ndarray],
        pose_j: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """Verify loop closure candidate with ICP.

        Args:
            scan_i: Query scan (robot frame).
            scan_j: Match scan (robot frame).
            pose_i: Optional query pose [x, y, yaw] for initial guess.
            pose_j: Optional match pose [x, y, yaw] for initial guess.

        Returns:
            Tuple of (rel_pose, covariance, residual, iterations) if verified,
            None if ICP fails or residual too high.
        """
        # Check scan sizes
        if len(scan_i) < 5 or len(scan_j) < 5:
            return None

        # Initial guess for ICP: transform from j (earlier) to i (later)
        # This is se2_relative(pose_j, pose_i) = inv(pose_j) @ pose_i
        if pose_i is not None and pose_j is not None:
            initial_guess = se2_relative(pose_j, pose_i)
        else:
            initial_guess = np.array([0.0, 0.0, 0.0])

        # Run ICP to find transform from j to i
        # ICP(source, target) returns transform that aligns source to target
        # So ICP(scan_j, scan_i) returns transform from frame_j to frame_i
        try:
            rel_pose, iters, residual, converged = icp_point_to_point(
                source_scan=scan_j,  # Earlier scan (match)
                target_scan=scan_i,  # Later scan (query)
                initial_pose=initial_guess,  # Initial guess: j to i
                max_iterations=self.icp_max_iterations,
                tolerance=self.icp_tolerance,
            )
        except Exception:
            return None

        # Check verification criteria
        if not converged:
            return None

        if residual > self.max_icp_residual:
            return None

        # Estimate covariance from how well this specific pair actually
        # matched, so a confident closure (many correspondences, tight
        # residual) outweighs a marginal one in the backend instead of every
        # closure being judged identically regardless of match quality.
        # Same source/target/final_pose convention as the icp_point_to_point
        # call above: source_scan=scan_j (earlier), target_scan=scan_i
        # (later), final_pose=rel_pose (the transform ICP found from j to i).
        covariance = compute_icp_covariance(scan_j, scan_i, rel_pose)

        # See _MIN_LOOP_CLOSURE_VARIANCE above for why this floor exists and
        # what it does and does not guard against.
        covariance = np.maximum(covariance, _MIN_LOOP_CLOSURE_VARIANCE * np.eye(3))

        # This covariance is optimistic, not merely small. Measured against
        # ground truth over the 147 closures of the shipped inline run
        # (python -m ch7_slam.example_pose_graph_slam; true relative poses
        # come from true_poses, which the pipeline never fits to):
        #
        #   actual closure error, RMS      x=0.0045 m  y=0.0028 m  yaw=0.00625 rad
        #   reported sigma, median         x=0.0014 m  y=0.0014 m  yaw=0.00140 rad
        #   overconfidence (error / sigma)     3.2x        2.0x        4.5x
        #   per-closure error/sigma, RMS       2.15        1.31        2.49
        #
        # A consistent covariance would put that last row at ~1.0. It does
        # not, and not by a small margin -- so the sigma this function
        # returns understates the true uncertainty by a factor of 2-4,
        # consistently enough across 147 closures that it is not a few
        # outliers. This is expected, not a bug in compute_icp_covariance:
        # its own docstring calls it "not a rigorous uncertainty estimate,"
        # because it derives sigma from the residual and correspondence
        # count alone. That sees only the noise it can model -- sensor
        # noise averaged over N matched points -- and is structurally blind
        # to the error sources that actually dominate a real scan match:
        # wrong correspondences, scene change between visits, and the
        # scan's own discretisation. A *formal* ICP covariance is optimistic
        # for this reason in general, not because of anything specific to
        # this dataset.
        #
        # Still a large net improvement over what this replaced: the same
        # 147 closures against the old constant diag([0.05, 0.05, 0.01])
        # (sigma 0.2236/0.2236/0.1000) were underconfident by 50x/79x/16x --
        # wrong in the opposite direction, and by an order of magnitude
        # more. Going from "wrong by fifty, and identical for every
        # closure" to "wrong by three, and tracking match quality" is the
        # improvement this function makes. "Wrong by three" is not
        # "correct," which is what this comment is for.
        #
        # No inflation factor is applied to correct the remaining 2-4x. A
        # single scalar cannot: the three axes are optimistic by different
        # amounts (2.15x/1.31x/2.49x in sigma), while compute_icp_covariance
        # ties sigma_x = sigma_y = sigma_yaw together by construction, so no
        # single multiplier fits all three. Checked directly -- the scalar
        # that brings the *pooled* normalised residual closest to 1.0 is
        # about 2.0x in sigma, and at that value the per-axis residuals are
        # 1.07 / 0.66 / 1.24: x lands close to consistent, y overshoots into
        # underconfidence, yaw is still overconfident. That is with a factor
        # fit to this exact trajectory, environment, and 0.02 m sensor noise
        # -- there is no reason to expect it transfers to a different
        # dataset, and baking in a number fit to one 147-closure sample from
        # one trajectory is the kind of unjustified constant this repository
        # has repeatedly found and removed elsewhere (see CLAUDE.md). Where
        # ground truth is available, as it was for this measurement, the
        # right move is to calibrate against it directly and check the
        # normalised residual, not to assume a multiplier. Where it is not,
        # a fixed conservative covariance -- which is what the constant this
        # replaced was attempting, just calibrated badly -- is the honest
        # fallback, not a precisely-fit scalar bolted onto a heuristic that
        # cannot see its own dominant error sources.

        return rel_pose, covariance, residual, iters
