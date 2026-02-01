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
    compute_scan_descriptor,
    compute_descriptor_similarity,
    batch_compute_descriptors,
)
from .scan_matching import icp_point_to_point
from .se2 import se2_relative


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
        max_icp_residual: float = 0.2,
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
            max_icp_residual: Maximum ICP residual to accept loop closure.
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
            candidates = self._find_candidates(
                i, descriptors, poses
            )
            
            if len(candidates) == 0:
                continue
            
            # Verify candidates with ICP
            for candidate in candidates:
                j = candidate.j
                
                # Run ICP to verify geometric consistency
                verified = self._verify_candidate(
                    scans[i], scans[j], poses[i] if poses else None, poses[j] if poses else None
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
        
        # Initial guess for ICP
        if pose_i is not None and pose_j is not None:
            initial_guess = se2_relative(pose_j, pose_i)
        else:
            initial_guess = np.array([0.0, 0.0, 0.0])
        
        # Run ICP
        try:
            rel_pose, iters, residual, converged = icp_point_to_point(
                source_scan=scan_i,
                target_scan=scan_j,
                initial_pose=initial_guess,
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
        
        # Estimate covariance (simple diagonal, could be improved)
        covariance = np.diag([0.05, 0.05, 0.01])
        
        return rel_pose, covariance, residual, iters
