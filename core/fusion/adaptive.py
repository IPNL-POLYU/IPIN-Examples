"""Adaptive gating and covariance management for practical fusion systems.

Implements robustness mechanisms described in Chapter 8 for handling
real-world sensor failures, outliers, and tuning challenges.

Key features:
- Consecutive reject tracking with covariance inflation
- NIS-based consistency monitoring and automatic scaling
- Adaptive gate widening to prevent filter starvation

Author: Li-Ta Hsu
Date: December 2025
References: Chapter 8, Section 8.3 (Innovation Monitoring and Tuning)
"""

from typing import Optional

import numpy as np


class AdaptiveGatingManager:
    """Manages adaptive gating mechanisms for robust sensor fusion.
    
    This class implements practical robustness features to prevent
    chi-square gating from starving the filter:
    
    1. **Consecutive Reject Tracking**: If a sensor stream is rejected
       too many times in a row, apply covariance inflation or widen gate.
    
    2. **NIS Consistency Monitoring**: Track rolling mean of NIS values.
       If NIS >> DOF consistently, the filter is overconfident → scale up R or Q.
    
    3. **Adaptive Recovery**: Automatically adjust parameters to restore
       filter consistency when gating becomes too aggressive.
    
    Usage:
        >>> manager = AdaptiveGatingManager(dof=4, consecutive_reject_limit=5)
        >>> 
        >>> # In fusion loop:
        >>> accept, action = manager.update(nis_value, gated_accept)
        >>> 
        >>> if action == 'inflate_P':
        >>>     P = manager.inflate_covariance(P)
        >>> elif action == 'scale_R':
        >>>     R = manager.get_R_scale() * R
    
    References:
        Chapter 8, Section 8.3.2: Filter Tuning and Consistency Checking
    """
    
    def __init__(
        self,
        dof: int,
        consecutive_reject_limit: int = 5,
        nis_window_size: int = 20,
        nis_scale_threshold: float = 1.5,
        P_inflation_factor: float = 1.5,
        R_scale_factor: float = 1.2,
        min_R_scale: float = 1.0,
        max_R_scale: float = 5.0,
    ):
        """Initialize adaptive gating manager.
        
        Args:
            dof: Degrees of freedom (measurement dimension)
            consecutive_reject_limit: Max consecutive rejects before adaptation
            nis_window_size: Rolling window size for NIS monitoring
            nis_scale_threshold: Ratio of (NIS mean / DOF) triggering scaling
            P_inflation_factor: Factor λ for P <- λP when adapting
            R_scale_factor: Factor to scale R up when filter overconfident
            min_R_scale: Minimum R scale factor
            max_R_scale: Maximum R scale factor (safety limit)
        """
        self.dof = dof
        self.consecutive_reject_limit = consecutive_reject_limit
        self.nis_window_size = nis_window_size
        self.nis_scale_threshold = nis_scale_threshold
        self.P_inflation_factor = P_inflation_factor
        self.R_scale_factor = R_scale_factor
        self.min_R_scale = min_R_scale
        self.max_R_scale = max_R_scale
        
        # State tracking
        self.consecutive_rejects = 0
        self.nis_history = []  # Rolling window of NIS values
        self.current_R_scale = 1.0
        self.total_measurements = 0
        self.total_accepts = 0
        self.total_rejects = 0
        self.total_adaptations = 0
    
    def update(
        self,
        nis_value: float,
        gated_accept: bool
    ) -> tuple[bool, Optional[str]]:
        """Update adaptive gating state and determine if action needed.
        
        Args:
            nis_value: Current Normalized Innovation Squared (NIS)
            gated_accept: Whether measurement was accepted by gate
        
        Returns:
            Tuple of (final_accept, action):
                final_accept: Whether to accept measurement (may override gate)
                action: Recommended action ('inflate_P', 'scale_R', None)
        """
        self.total_measurements += 1
        
        # Track NIS for consistency monitoring
        self.nis_history.append(nis_value)
        if len(self.nis_history) > self.nis_window_size:
            self.nis_history.pop(0)
        
        # Track consecutive rejects
        if gated_accept:
            self.consecutive_rejects = 0
            self.total_accepts += 1
            action = None
        else:
            self.consecutive_rejects += 1
            self.total_rejects += 1
            action = None
        
        # Check for consecutive reject limit
        if self.consecutive_rejects >= self.consecutive_reject_limit:
            # Apply covariance inflation to prevent filter starvation
            action = 'inflate_P'
            self.consecutive_rejects = 0  # Reset after adaptation
            self.total_adaptations += 1
            # Force accept this measurement with inflated uncertainty
            gated_accept = True
        
        # Check NIS consistency (only if we have enough history)
        if len(self.nis_history) >= self.nis_window_size:
            mean_nis = np.mean(self.nis_history)
            expected_nis = self.dof  # E[χ²(m)] = m
            
            # If mean NIS >> expected, filter is overconfident
            if mean_nis > self.nis_scale_threshold * expected_nis:
                if action is None:  # Don't override P inflation
                    action = 'scale_R'
                # Gradually increase R scale
                self.current_R_scale = min(
                    self.current_R_scale * self.R_scale_factor,
                    self.max_R_scale
                )
            elif mean_nis < 0.7 * expected_nis:
                # Filter is too conservative, reduce R scale
                self.current_R_scale = max(
                    self.current_R_scale / self.R_scale_factor,
                    self.min_R_scale
                )
        
        return gated_accept, action
    
    def inflate_covariance(self, P: np.ndarray) -> np.ndarray:
        """Apply covariance inflation: P <- λP.
        
        Used when consecutive rejects suggest filter is overconfident.
        
        Args:
            P: Current state covariance (n x n)
        
        Returns:
            Inflated covariance P_inflated = λ * P
        """
        return self.P_inflation_factor * P
    
    def get_R_scale(self) -> float:
        """Get current R scale factor based on NIS monitoring.
        
        Returns:
            Scale factor w_R >= 1.0 to apply to measurement covariance R
        """
        return self.current_R_scale
    
    def get_stats(self) -> dict:
        """Get diagnostic statistics for logging.
        
        Returns:
            Dictionary with acceptance rate, NIS stats, etc.
        """
        acceptance_rate = (
            self.total_accepts / self.total_measurements
            if self.total_measurements > 0
            else 0.0
        )
        
        mean_nis = np.mean(self.nis_history) if self.nis_history else 0.0
        
        return {
            'total_measurements': self.total_measurements,
            'total_accepts': self.total_accepts,
            'total_rejects': self.total_rejects,
            'acceptance_rate': acceptance_rate,
            'consecutive_rejects': self.consecutive_rejects,
            'mean_nis': mean_nis,
            'expected_nis': self.dof,
            'current_R_scale': self.current_R_scale,
            'total_adaptations': self.total_adaptations,
        }
    
    def reset(self):
        """Reset all tracking state (e.g., for new episode)."""
        self.consecutive_rejects = 0
        self.nis_history = []
        self.current_R_scale = 1.0
        self.total_measurements = 0
        self.total_accepts = 0
        self.total_rejects = 0
        self.total_adaptations = 0


def create_adaptive_manager_for_tc(n_anchors: int = 4, **kwargs) -> AdaptiveGatingManager:
    """Create adaptive gating manager for TC fusion (per-anchor updates).
    
    Args:
        n_anchors: Number of UWB anchors
        **kwargs: Additional parameters for AdaptiveGatingManager
    
    Returns:
        Configured AdaptiveGatingManager instance
    """
    # TC fusion: each UWB range is 1D measurement
    return AdaptiveGatingManager(dof=1, **kwargs)


def create_adaptive_manager_for_lc(**kwargs) -> AdaptiveGatingManager:
    """Create adaptive gating manager for LC fusion (position updates).
    
    Args:
        **kwargs: Additional parameters for AdaptiveGatingManager
    
    Returns:
        Configured AdaptiveGatingManager instance
    """
    # LC fusion: position fix is 2D measurement
    return AdaptiveGatingManager(dof=2, **kwargs)

