"""2D LiDAR scan generation with proper occlusion handling.

This module provides functions to generate realistic 2D LiDAR scans from
wall-based environments using ray-casting to properly handle occlusions.

The ray-casting approach ensures that only the closest obstacle along each
ray direction is detected, mimicking real LiDAR sensor behavior.

Author: Li-Ta Hsu
Date: December 2025
"""

from typing import List, Optional, Tuple

import numpy as np


def ray_segment_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """Compute intersection between a ray and a line segment.
    
    Uses parametric line equations to find intersection:
        Ray: P = origin + t * direction (t >= 0)
        Segment: Q = start + s * (end - start) (0 <= s <= 1)
    
    Args:
        ray_origin: Ray starting point [x, y].
        ray_direction: Ray direction vector [dx, dy] (should be normalized).
        segment_start: Segment start point [x, y].
        segment_end: Segment end point [x, y].
    
    Returns:
        Tuple of (intersection_point, distance):
            - intersection_point: [x, y] if intersection exists, None otherwise
            - distance: Distance along ray to intersection (inf if no intersection)
    
    Notes:
        - Ray direction should be normalized for distance to be meaningful
        - Returns None if ray doesn't intersect segment or intersection is behind ray
    """
    # Convert to numpy arrays
    o = np.array(ray_origin, dtype=float)
    d = np.array(ray_direction, dtype=float)
    s_start = np.array(segment_start, dtype=float)
    s_end = np.array(segment_end, dtype=float)
    
    # Segment direction
    s_dir = s_end - s_start
    s_len_sq = np.dot(s_dir, s_dir)
    
    # Check for degenerate segment (zero length)
    if s_len_sq < 1e-10:
        return None, float('inf')
    
    # Solve for intersection using Cramer's rule
    # Ray: o + t*d, Segment: s_start + u*s_dir
    # o + t*d = s_start + u*s_dir
    # t*d - u*s_dir = s_start - o
    
    # Matrix form: [d | -s_dir] * [t; u] = s_start - o
    diff = s_start - o
    
    # Determinant
    det = d[0] * (-s_dir[1]) - d[1] * (-s_dir[0])
    det = d[0] * s_dir[1] - d[1] * s_dir[0]
    
    # Check if ray and segment are parallel
    if abs(det) < 1e-10:
        return None, float('inf')
    
    # Solve for t and u
    t = (diff[0] * s_dir[1] - diff[1] * s_dir[0]) / det
    u = (diff[0] * d[1] - diff[1] * d[0]) / det
    
    # Check validity: t >= 0 (ray goes forward), 0 <= u <= 1 (point on segment)
    if t < 0 or u < 0 or u > 1:
        return None, float('inf')
    
    # Compute intersection point
    intersection = o + t * d
    distance = t  # Distance along ray (meaningful if d is normalized)
    
    return intersection, distance


def generate_scan_with_occlusion(
    pose: np.ndarray,
    walls: List[Tuple[np.ndarray, np.ndarray]],
    num_rays: int = 360,
    max_range: float = 10.0,
    noise_std: float = 0.02,
    min_range: float = 0.1,
) -> np.ndarray:
    """Generate 2D LiDAR scan with proper occlusion handling using ray-casting.
    
    Simulates a 2D LiDAR sensor that casts rays in 360 directions around the
    robot. For each ray, finds the CLOSEST wall intersection, properly handling
    occlusions where near objects block far objects.
    
    This corrects the bug in `generate_dense_wall_scan` which included all
    walls within range regardless of occlusion.
    
    Args:
        pose: Robot pose [x, y, yaw] in global frame.
        walls: List of (start_point, end_point) tuples defining wall segments.
        num_rays: Number of rays to cast (angular resolution). Default 360 gives
                 1-degree resolution. Typical LiDAR: 360-720 rays.
        max_range: Maximum sensor range in meters.
        noise_std: Standard deviation of range measurement noise (meters).
        min_range: Minimum sensor range (meters). Points closer than this are
                  filtered out to simulate sensor blind zone.
    
    Returns:
        Point cloud in robot's local frame, shape (M, 2) where M <= num_rays.
        Each point is [x_local, y_local].
    
    Example:
        >>> walls = [
        ...     (np.array([0, 0]), np.array([10, 0])),  # Bottom wall
        ...     (np.array([0, 10]), np.array([10, 10])), # Top wall
        ... ]
        >>> pose = np.array([5.0, 5.0, 0.0])
        >>> scan = generate_scan_with_occlusion(pose, walls, num_rays=360)
        >>> print(f"Generated {len(scan)} points")
    
    Notes:
        - Ray-casting ensures only the CLOSEST hit is recorded per ray direction
        - Properly handles occlusions: near obstacles block far walls
        - Adds Gaussian noise to simulate real LiDAR measurement uncertainty
        - Filters points outside [min_range, max_range] to simulate sensor specs
    """
    x, y, yaw = pose
    scan_points = []
    
    # Cast rays in all directions
    for ray_idx in range(num_rays):
        # Ray angle in global frame
        angle = yaw + (2 * np.pi * ray_idx / num_rays)
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        
        # Find closest intersection with all walls
        min_distance = max_range
        closest_point = None
        
        for wall_start, wall_end in walls:
            intersection, distance = ray_segment_intersection(
                ray_origin=[x, y],
                ray_direction=ray_dir,
                segment_start=wall_start,
                segment_end=wall_end,
            )
            
            if intersection is not None and distance < min_distance:
                min_distance = distance
                closest_point = intersection
        
        # If we hit a wall within range, add the point
        if closest_point is not None and min_distance < max_range:
            # Add measurement noise
            if noise_std > 0:
                noisy_distance = min_distance + np.random.normal(0, noise_std)
                noisy_distance = max(min_range, noisy_distance)  # Clamp to min range
                noisy_point = np.array([x, y]) + noisy_distance * ray_dir
            else:
                noisy_point = closest_point
            
            # Transform to robot's local frame
            diff = noisy_point - np.array([x, y])
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            x_local = cos_yaw * diff[0] + sin_yaw * diff[1]
            y_local = -sin_yaw * diff[0] + cos_yaw * diff[1]
            
            # Filter by range
            range_check = np.sqrt(x_local**2 + y_local**2)
            if min_range <= range_check <= max_range:
                scan_points.append([x_local, y_local])
    
    if not scan_points:
        return np.zeros((0, 2))
    
    return np.array(scan_points)


def generate_dense_wall_scan(
    pose: np.ndarray,
    walls: List[Tuple[np.ndarray, np.ndarray]],
    max_range: float = 8.0,
    noise_std: float = 0.02,
    points_per_wall: int = 50,
) -> np.ndarray:
    """Generate dense LiDAR scan from walls (legacy, without occlusion handling).
    
    [DEPRECATED] This function has a known occlusion bug: it includes points
    from ALL walls within range, even if they are blocked by closer obstacles.
    
    Use `generate_scan_with_occlusion()` instead for physically accurate scans.
    
    This function is kept for backward compatibility with existing code.
    
    Args:
        pose: Robot pose [x, y, yaw].
        walls: List of (start_point, end_point) tuples defining wall segments.
        max_range: Maximum sensor range in meters.
        noise_std: Standard deviation of measurement noise.
        points_per_wall: Number of points to sample per wall segment.
    
    Returns:
        Point cloud in robot local frame, shape (M, 2).
    
    Warning:
        This function does NOT handle occlusions properly. Near objects do not
        block far objects, resulting in non-physical scan data.
    """
    x, y, yaw = pose
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    all_points = []
    
    for wall_start, wall_end in walls:
        # Generate dense points along the wall
        t = np.linspace(0, 1, points_per_wall)
        wall_points = wall_start + np.outer(t, wall_end - wall_start)
        
        # Transform to robot frame
        diff = wall_points - np.array([x, y])
        x_local = cos_yaw * diff[:, 0] + sin_yaw * diff[:, 1]
        y_local = -sin_yaw * diff[:, 0] + cos_yaw * diff[:, 1]
        
        # Filter by range
        ranges = np.sqrt(x_local**2 + y_local**2)
        valid = ranges < max_range
        
        if np.any(valid):
            local_points = np.column_stack([x_local[valid], y_local[valid]])
            all_points.append(local_points)
    
    if not all_points:
        return np.zeros((0, 2))
    
    scan = np.vstack(all_points)
    
    # Add measurement noise
    if noise_std > 0:
        scan += np.random.normal(0, noise_std, scan.shape)
    
    return scan
