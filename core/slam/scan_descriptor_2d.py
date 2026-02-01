"""2D LiDAR Scan Descriptors for Loop Closure Detection.

This module implements simple, fast scan descriptors for loop closure candidate
generation in 2D SLAM. The descriptor is based on range histograms, which are:
    - Fast to compute (O(n) where n = number of points)
    - Rotation-invariant (range doesn't depend on angle)
    - Compact (fixed-length vector)
    - Robust to noise

The descriptor is used for initial loop closure candidate selection based on
observation similarity, replacing position-based oracle selection.

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
from typing import Optional


def compute_scan_descriptor(
    scan_xy: np.ndarray,
    n_bins: int = 32,
    max_range: float = 10.0,
) -> np.ndarray:
    """Compute range histogram descriptor for a 2D LiDAR scan.
    
    This descriptor represents the distribution of ranges in a scan as a
    normalized histogram. It is rotation-invariant and provides a simple
    yet effective signature for place recognition.
    
    Algorithm:
        1. Compute range r = sqrt(x^2 + y^2) for each point
        2. Create histogram of ranges with n_bins over [0, max_range]
        3. Normalize to sum = 1 (probability distribution)
    
    Args:
        scan_xy: Scan points in robot frame, shape (N, 2).
        n_bins: Number of histogram bins. More bins = more discriminative
               but less robust to noise. Typical: 16-64.
        max_range: Maximum range for histogram (meters). Points beyond
                  this range are placed in the last bin. Should match
                  LiDAR max range.
    
    Returns:
        Descriptor vector of shape (n_bins,), normalized to sum = 1.
        Returns zero vector if scan is empty.
    
    Raises:
        ValueError: If scan_xy has invalid shape or n_bins < 1.
    
    Example:
        >>> scan = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        >>> desc = compute_scan_descriptor(scan, n_bins=8, max_range=5.0)
        >>> print(desc.shape)
        (8,)
        >>> print(np.sum(desc))  # Should be 1.0
        1.0
    
    Notes:
        - Rotation-invariant: descriptor(scan) = descriptor(rotate(scan))
        - Not translation-invariant: assumes scan in robot frame
        - Simple but effective for structured environments (corridors, rooms)
    """
    # Input validation
    if scan_xy.ndim != 2 or scan_xy.shape[1] != 2:
        raise ValueError(f"scan_xy must have shape (N, 2), got {scan_xy.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if max_range <= 0:
        raise ValueError(f"max_range must be > 0, got {max_range}")
    
    # Handle empty scan
    if len(scan_xy) == 0:
        return np.zeros(n_bins, dtype=np.float64)
    
    # Compute ranges
    ranges = np.linalg.norm(scan_xy, axis=1)
    
    # Clip ranges to max_range (points beyond go to last bin)
    ranges = np.clip(ranges, 0, max_range - 1e-9)
    
    # Create histogram
    # Bins span [0, max_range] uniformly
    histogram, _ = np.histogram(
        ranges,
        bins=n_bins,
        range=(0, max_range),
        density=False,
    )
    
    # Convert to float and normalize
    descriptor = histogram.astype(np.float64)
    total = np.sum(descriptor)
    
    if total > 0:
        descriptor /= total
    
    return descriptor


def compute_descriptor_similarity(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: str = "cosine",
) -> float:
    """Compute similarity between two scan descriptors.
    
    Args:
        desc1: First descriptor, shape (n_bins,).
        desc2: Second descriptor, shape (n_bins,).
        method: Similarity metric, one of:
               - "cosine": Cosine similarity (range: [-1, 1], higher = more similar)
               - "correlation": Pearson correlation (range: [-1, 1])
               - "l2": Negative L2 distance (range: (-inf, 0], higher = more similar)
    
    Returns:
        Similarity score. Higher values indicate more similar descriptors.
        For "cosine" and "correlation": range is [-1, 1], with 1 = identical.
        For "l2": range is (-inf, 0], with 0 = identical.
    
    Raises:
        ValueError: If descriptors have different shapes or invalid method.
    
    Example:
        >>> desc1 = np.array([0.5, 0.3, 0.2])
        >>> desc2 = np.array([0.5, 0.3, 0.2])
        >>> sim = compute_descriptor_similarity(desc1, desc2, method="cosine")
        >>> print(sim)  # Should be 1.0 (identical)
        1.0
    """
    # Input validation
    if desc1.shape != desc2.shape:
        raise ValueError(
            f"Descriptors must have same shape, got {desc1.shape} and {desc2.shape}"
        )
    
    if method == "cosine":
        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        
        if norm1 < 1e-9 or norm2 < 1e-9:
            # One or both descriptors are zero vectors
            return 0.0
        
        similarity = np.dot(desc1, desc2) / (norm1 * norm2)
        return float(similarity)
    
    elif method == "correlation":
        # Pearson correlation coefficient
        # Handles case where descriptors have zero variance
        if np.std(desc1) < 1e-9 or np.std(desc2) < 1e-9:
            return 0.0
        
        correlation = np.corrcoef(desc1, desc2)[0, 1]
        return float(correlation)
    
    elif method == "l2":
        # Negative L2 distance (so higher = more similar)
        distance = np.linalg.norm(desc1 - desc2)
        return float(-distance)
    
    else:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: cosine, correlation, l2"
        )


def batch_compute_descriptors(
    scans: list,
    n_bins: int = 32,
    max_range: float = 10.0,
) -> np.ndarray:
    """Compute descriptors for a batch of scans.
    
    Args:
        scans: List of N scans, each with shape (M_i, 2).
        n_bins: Number of histogram bins.
        max_range: Maximum range for histogram.
    
    Returns:
        Array of descriptors with shape (N, n_bins).
    
    Example:
        >>> scans = [
        ...     np.array([[1.0, 0.0], [2.0, 0.0]]),
        ...     np.array([[3.0, 0.0], [4.0, 0.0]]),
        ... ]
        >>> descriptors = batch_compute_descriptors(scans)
        >>> print(descriptors.shape)
        (2, 32)
    """
    descriptors = []
    for scan in scans:
        desc = compute_scan_descriptor(scan, n_bins=n_bins, max_range=max_range)
        descriptors.append(desc)
    
    return np.array(descriptors)
