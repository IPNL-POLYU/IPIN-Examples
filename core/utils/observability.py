"""
Observability analysis for state estimation.

Provides tools to check if a system's state is observable from measurements,
based on observability matrix rank analysis.

Key concepts:
- A system is observable if the state can be uniquely determined from measurements
- Observability matrix: O = [H; H*F; H*F^2; ...; H*F^(n-1)] where n is state dimension
- System is observable if rank(O) = n (full rank)

References:
    Chapter 8, Observability Analysis
"""

import numpy as np
from typing import Tuple, Optional, Callable
import warnings


def compute_observability_matrix(
    F: np.ndarray,
    H: np.ndarray,
    n_steps: Optional[int] = None
) -> np.ndarray:
    """
    Compute the observability matrix for a linear system.
    
    For a linear system:
        x_{k+1} = F x_k
        z_k = H x_k
    
    The observability matrix is:
        O = [H]
            [H*F]
            [H*F^2]
            [...] 
            [H*F^(n-1)]
    
    where n is the state dimension.
    
    Args:
        F: State transition matrix, shape (n, n)
        H: Measurement matrix, shape (m, n)
        n_steps: Number of steps to use (default: n, the state dimension)
    
    Returns:
        Observability matrix O, shape (m*n_steps, n)
    
    Example:
        >>> # Position-only observable system
        >>> F = np.array([[1, 1], [0, 1]])  # [pos, vel]
        >>> H = np.array([[1, 0]])  # Observe position only
        >>> O = compute_observability_matrix(F, H)
        >>> np.linalg.matrix_rank(O)  # Should be 2 (full rank)
        2
        
        >>> # Unobservable system (constant offset)
        >>> F = np.array([[1, 0], [0, 1]])  # No dynamics
        >>> H = np.array([[1, 0]])  # Observe x only
        >>> O = compute_observability_matrix(F, H)
        >>> np.linalg.matrix_rank(O)  # Should be 1 (y unobservable)
        1
    
    References:
        Linear system theory, Chapter 8
    """
    F = np.asarray(F)
    H = np.asarray(H)
    
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError(f"F must be square matrix, got shape {F.shape}")
    
    if H.ndim != 2:
        raise ValueError(f"H must be 2D matrix, got shape {H.shape}")
    
    n_states = F.shape[0]
    m_meas = H.shape[0]
    
    if H.shape[1] != n_states:
        raise ValueError(
            f"H shape {H.shape} incompatible with F shape {F.shape}"
        )
    
    if n_steps is None:
        n_steps = n_states
    
    # Build observability matrix
    O = np.zeros((m_meas * n_steps, n_states))
    
    H_Fi = H.copy()
    for i in range(n_steps):
        O[i * m_meas:(i + 1) * m_meas, :] = H_Fi
        H_Fi = H_Fi @ F  # H * F^i
    
    return O


def check_observability(
    F: np.ndarray,
    H: np.ndarray,
    n_steps: Optional[int] = None,
    tolerance: float = 1e-10
) -> Tuple[bool, int, np.ndarray]:
    """
    Check if a linear system is observable.
    
    A system is observable if all states can be determined from measurements.
    This is checked by computing the rank of the observability matrix.
    
    Args:
        F: State transition matrix, shape (n, n)
        H: Measurement matrix, shape (m, n)
        n_steps: Number of steps for observability matrix (default: n)
        tolerance: Numerical tolerance for rank computation
    
    Returns:
        Tuple of (is_observable, rank, singular_values):
            - is_observable: True if system is fully observable
            - rank: Rank of observability matrix
            - singular_values: Singular values of O (for diagnost

ics)
    
    Example:
        >>> # Observable system
        >>> F = np.eye(2)
        >>> H = np.eye(2)
        >>> is_obs, rank, _ = check_observability(F, H)
        >>> is_obs
        True
        >>> rank
        2
        
        >>> # Unobservable system (position bias)
        >>> F = np.eye(2)
        >>> H = np.array([[1, -1]])  # Observe difference only
        >>> is_obs, rank, _ = check_observability(F, H)
        >>> is_obs
        False
        >>> rank
        1
    
    References:
        Chapter 8, Observability Analysis
    """
    O = compute_observability_matrix(F, H, n_steps)
    
    # Compute rank via SVD
    singular_values = np.linalg.svd(O, compute_uv=False)
    
    # Determine rank with tolerance
    n_states = F.shape[0]
    if len(singular_values) == 0:
        rank = 0
    else:
        rank = np.sum(singular_values > tolerance * singular_values[0])
    
    is_observable = (rank == n_states)
    
    return is_observable, rank, singular_values


def check_range_only_observability_2d(
    anchors: np.ndarray,
    position: np.ndarray,
    warn: bool = True
) -> Tuple[bool, str]:
    """
    Check observability for 2D range-only positioning.
    
    For range-only measurements in 2D:
    - Need at least 3 non-colinear anchors
    - Position must not be at an anchor (singularity)
    - Better geometry = better observability
    
    Args:
        anchors: Anchor positions, shape (N, 2)
        position: Receiver position, shape (2,)
        warn: If True, issue warnings for poor observability
    
    Returns:
        Tuple of (is_observable, message):
            - is_observable: True if position is observable
            - message: Description of observability issue
    
    Example:
        >>> # Good configuration
        >>> anchors = np.array([[0, 0], [10, 0], [5, 10]])
        >>> position = np.array([5.0, 3.0])
        >>> is_obs, msg = check_range_only_observability_2d(anchors, position)
        >>> is_obs
        True
        
        >>> # Bad: colinear anchors
        >>> anchors = np.array([[0, 0], [5, 0], [10, 0]])
        >>> position = np.array([5.0, 1.0])
        >>> is_obs, msg = check_range_only_observability_2d(anchors, position)
        >>> is_obs
        False
        >>> 'colinear' in msg.lower()
        True
    
    References:
        Range-based positioning (Chapter 4), Observability (Chapter 8)
    """
    from .geometry import check_anchor_geometry
    
    # Check anchor geometry
    is_valid, msg = check_anchor_geometry(anchors, position, warn_degenerate=False)
    if not is_valid:
        if warn:
            warnings.warn(f"Observability check failed: {msg}", RuntimeWarning)
        return False, msg
    
    # Check if position is at or very close to an anchor (singularity)
    distances = np.linalg.norm(anchors - position, axis=1)
    min_distance = np.min(distances)
    
    if min_distance < 1e-6:  # 1 micrometer threshold
        msg = f"Position is at anchor (distance {min_distance:.2e}m). This is a singularity."
        if warn:
            warnings.warn(msg, RuntimeWarning)
        return False, msg
    
    # For range-only in 2D: with 3+ non-colinear anchors, position is observable
    return True, "Position is observable from range measurements"


def estimate_observability_time_constant(
    F: np.ndarray,
    H: np.ndarray,
    dt: float = 1.0
) -> float:
    """
    Estimate time constant for observability to manifest.
    
    For systems where observability depends on dynamics (e.g., velocity
    observable through position changes), estimates how long it takes
    for the state to become observable.
    
    Args:
        F: State transition matrix (discrete time)
        H: Measurement matrix
        dt: Time step (seconds)
    
    Returns:
        Estimated time constant (seconds) for observability
        Returns np.inf if unobservable
    
    Example:
        >>> # Constant velocity: velocity observable after some motion
        >>> F = np.array([[1, 0.1], [0, 1]])  # dt=0.1s
        >>> H = np.array([[1, 0]])  # Position only
        >>> tau = estimate_observability_time_constant(F, H, dt=0.1)
        >>> tau > 0  # Need some time for velocity to manifest
        True
    """
    is_obs, rank, singular_values = check_observability(F, H)
    
    if not is_obs:
        return np.inf
    
    # Estimate based on smallest significant singular value
    # Smaller singular values = longer time to observe
    n_states = F.shape[0]
    if len(singular_values) < n_states:
        return np.inf
    
    min_sv = singular_values[n_states - 1]
    
    if min_sv < 1e-10:
        return np.inf
    
    # Rough estimate: time constant inversely proportional to smallest SV
    # Multiply by number of steps needed for full observability
    tau = n_states * dt / min_sv
    
    return tau



