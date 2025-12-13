"""Pattern recognition methods for fingerprinting (Chapter 5).

This module implements linear regression-based fingerprinting, treating
localization as a supervised learning regression problem.

Key concept:
    Learn mapping f: z → x where z is RSS fingerprint and x is location.
    Use linear model: x̂ = Wz + b

Author: Navigation Engineer
Date: 2024
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import Fingerprint, FingerprintDatabase, Location


@dataclass
class LinearRegressionLocalizer:
    """
    Linear regression model for fingerprint-based localization.

    This model learns a linear mapping from RSS fingerprint space to
    location space. For each coordinate dimension, it fits:
        x_d = w_d^T z + b_d

    where w_d is the weight vector for dimension d, and b_d is the bias.

    The model can be trained separately per floor or across all floors.
    For multi-floor localization, use separate models per floor.

    Attributes:
        weights: Weight matrix W, shape (d, N) where d is location dimension
                 (2D or 3D) and N is number of features (APs).
        bias: Bias vector b, shape (d,).
        floor_id: Optional floor identifier. If set, model is trained only
                  on data from this floor. If None, trained on all floors.
        n_training_samples: Number of samples used for training.
        meta: Metadata dictionary (e.g., AP IDs, building info).

    Example:
        >>> db = load_fingerprint_database('data/sim/wifi_fingerprint_grid')
        >>> 
        >>> # Train separate model for floor 0
        >>> model_floor0 = LinearRegressionLocalizer.fit(db, floor_id=0)
        >>> 
        >>> # Predict location
        >>> z_query = np.array([-51, -61, -71])
        >>> x_hat = model_floor0.predict(z_query)
        >>> print(f"Predicted location: {x_hat}")

    References:
        Chapter 5: Pattern recognition approach to fingerprinting.
    """

    weights: np.ndarray
    bias: np.ndarray
    floor_id: Optional[int]
    n_training_samples: int
    meta: dict

    def __post_init__(self) -> None:
        """Validate model consistency."""
        # Check that weights and bias have compatible dimensions
        d = self.bias.shape[0]  # Location dimension

        if self.weights.ndim != 2:
            raise ValueError(
                f"weights must be 2D array (d, N), got shape {self.weights.shape}"
            )

        if self.weights.shape[0] != d:
            raise ValueError(
                f"weights shape {self.weights.shape} incompatible with "
                f"bias shape {self.bias.shape}"
            )

        if self.bias.ndim != 1:
            raise ValueError(f"bias must be 1D array (d,), got shape {self.bias.shape}")

        if self.n_training_samples < 1:
            raise ValueError(
                f"n_training_samples must be >= 1, got {self.n_training_samples}"
            )

    @property
    def location_dim(self) -> int:
        """Dimensionality of location space (2D or 3D)."""
        return self.bias.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features N (e.g., number of APs)."""
        return self.weights.shape[1]

    @classmethod
    def fit(
        cls,
        db: FingerprintDatabase,
        floor_id: Optional[int] = None,
        regularization: float = 0.0,
    ) -> "LinearRegressionLocalizer":
        """
        Fit linear regression model from fingerprint database.

        Solves the least squares problem:
            min_{W, b} Σ_i ||x_i - (Wz_i + b)||²_2 + λ||W||²_F

        where λ is the regularization parameter (ridge regression).

        Uses the closed-form solution with bias trick:
            X = [Z, 1] [W^T; b^T]^T
        where Z is (M, N) feature matrix and X is (M, d) location matrix.

        Args:
            db: FingerprintDatabase containing training data.
            floor_id: Optional floor constraint. If provided, trains only on
                      data from this floor. If None, uses all floors.
            regularization: L2 regularization parameter λ. Default is 0.0
                            (no regularization). Use λ > 0 to prevent overfitting.

        Returns:
            Trained LinearRegressionLocalizer model.

        Raises:
            ValueError: If floor_id doesn't exist or if insufficient training data.

        Examples:
            >>> db = load_fingerprint_database('data/sim/wifi_fingerprint_grid')
            >>> 
            >>> # Train on floor 0 only
            >>> model = LinearRegressionLocalizer.fit(db, floor_id=0)
            >>> 
            >>> # Train on all floors with regularization
            >>> model_all = LinearRegressionLocalizer.fit(db, regularization=1.0)

        References:
            Chapter 5: Linear regression for fingerprinting.
            Ridge regression: Hoerl & Kennard (1970).
        """
        # Filter by floor if specified
        if floor_id is not None:
            mask = db.get_floor_mask(floor_id)
            if not np.any(mask):
                raise ValueError(f"Floor {floor_id} not found in database")
            X = db.locations[mask]  # (M', d)
            Z = db.features[mask]  # (M', N)
        else:
            X = db.locations  # (M, d)
            Z = db.features  # (M, N)

        M = X.shape[0]  # Number of training samples
        N = Z.shape[1]  # Number of features
        d = X.shape[1]  # Location dimension

        if M < N:
            raise ValueError(
                f"Insufficient training data: M={M} samples < N={N} features. "
                f"Linear regression requires M >= N. Consider regularization."
            )

        # Add bias column to feature matrix: Z_aug = [Z, 1]
        # Shape: (M, N+1)
        Z_aug = np.column_stack([Z, np.ones(M)])

        # Solve least squares problem
        # min ||Z_aug θ - X||² + λ||W||²
        # where θ = [W^T; b^T]^T, shape (N+1, d)

        if regularization > 0:
            # Ridge regression: solve (Z^T Z + λI) θ = Z^T X
            ZTZ = Z_aug.T @ Z_aug  # (N+1, N+1)
            ZTX = Z_aug.T @ X  # (N+1, d)

            # Add regularization to weight terms only (not bias)
            reg_matrix = np.eye(N + 1) * regularization
            reg_matrix[-1, -1] = 0  # Don't regularize bias
            ZTZ += reg_matrix

            # Solve normal equations: θ = (Z^T Z + λI)^-1 Z^T X
            try:
                theta = np.linalg.solve(ZTZ, ZTX)  # (N+1, d)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "Singular matrix encountered. Try increasing regularization parameter."
                )
        else:
            # Ordinary least squares: use lstsq for robustness
            # Solves: min ||Z_aug θ - X||²
            # lstsq handles rank-deficient matrices gracefully
            theta, residuals, rank, s = np.linalg.lstsq(Z_aug, X, rcond=None)

        # Extract weights and bias
        # θ = [W^T; b^T]^T where W is (d, N) and b is (d,)
        weights = theta[:-1, :].T  # (d, N)
        bias = theta[-1, :]  # (d,)

        return cls(
            weights=weights,
            bias=bias,
            floor_id=floor_id,
            n_training_samples=M,
            meta=db.meta.copy(),
        )

    def predict(self, z: Fingerprint) -> Location:
        """
        Predict location from RSS fingerprint.

        Implements the linear model:
            x̂ = Wz + b

        Args:
            z: Query fingerprint vector, shape (N,).

        Returns:
            Predicted location x̂, shape (d,).

        Raises:
            ValueError: If query dimension doesn't match model.

        Examples:
            >>> model = LinearRegressionLocalizer.fit(db, floor_id=0)
            >>> z_query = np.array([-51, -61, -71])
            >>> x_hat = model.predict(z_query)
            >>> print(f"Predicted location: {x_hat}")

        References:
            Chapter 5: Linear prediction for fingerprinting.
        """
        # Validate query dimension
        if z.shape[0] != self.n_features:
            raise ValueError(
                f"Query has {z.shape[0]} features, but model expects {self.n_features}"
            )

        # Linear prediction: x̂ = Wz + b
        # weights: (d, N), z: (N,), bias: (d,)
        # Result: (d,)
        x_hat = self.weights @ z + self.bias

        return x_hat

    def predict_batch(self, Z: np.ndarray) -> np.ndarray:
        """
        Predict locations for multiple fingerprints.

        Vectorized version of predict() for efficiency.

        Args:
            Z: Query fingerprints matrix, shape (M, N) where M is number
               of queries.

        Returns:
            Predicted locations, shape (M, d).

        Raises:
            ValueError: If query dimension doesn't match model.

        Examples:
            >>> model = LinearRegressionLocalizer.fit(db, floor_id=0)
            >>> Z_queries = np.array([[-51, -61, -71],
            ...                       [-52, -62, -72],
            ...                       [-53, -63, -73]])
            >>> X_hat = model.predict_batch(Z_queries)
            >>> print(f"Predicted locations:\\n{X_hat}")

        References:
            Chapter 5: Batch prediction for efficiency.
        """
        # Validate input
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2D array (M, N), got shape {Z.shape}")

        if Z.shape[1] != self.n_features:
            raise ValueError(
                f"Z has {Z.shape[1]} features, but model expects {self.n_features}"
            )

        # Vectorized prediction: X̂ = ZW^T + b
        # Z: (M, N), W^T: (N, d), bias: (d,) -> broadcast to (M, d)
        # Result: (M, d)
        X_hat = Z @ self.weights.T + self.bias

        return X_hat

    def score(self, db: FingerprintDatabase, floor_id: Optional[int] = None) -> float:
        """
        Compute R² coefficient of determination on test data.

        R² measures the proportion of variance explained by the model:
            R² = 1 - SS_res / SS_tot

        where:
            SS_res = Σ_i ||x_i - x̂_i||²  (residual sum of squares)
            SS_tot = Σ_i ||x_i - x̄||²    (total sum of squares)

        R² = 1 indicates perfect prediction, R² = 0 indicates the model
        performs no better than predicting the mean location.

        Args:
            db: FingerprintDatabase containing test data.
            floor_id: Optional floor constraint. If provided, evaluates only
                      on data from this floor. If None, uses all data (or
                      model's floor if model.floor_id is set).

        Returns:
            R² score (coefficient of determination).

        Raises:
            ValueError: If floor_id doesn't match model's training floor or
                        if floor doesn't exist in database.

        Examples:
            >>> # Train on floor 0
            >>> model = LinearRegressionLocalizer.fit(train_db, floor_id=0)
            >>> 
            >>> # Evaluate on test set (same floor)
            >>> r2 = model.score(test_db, floor_id=0)
            >>> print(f"R² score: {r2:.3f}")

        References:
            R² coefficient: Statistical measure of model fit.
        """
        # Determine which floor to evaluate on
        if floor_id is None:
            floor_id = self.floor_id

        # Filter by floor if specified
        if floor_id is not None:
            mask = db.get_floor_mask(floor_id)
            if not np.any(mask):
                raise ValueError(f"Floor {floor_id} not found in database")
            X_true = db.locations[mask]
            Z = db.features[mask]
        else:
            X_true = db.locations
            Z = db.features

        # Predict locations
        X_pred = self.predict_batch(Z)

        # Compute R² score
        # SS_res = Σ_i ||x_i - x̂_i||²
        ss_res = np.sum((X_true - X_pred) ** 2)

        # SS_tot = Σ_i ||x_i - x̄||²
        X_mean = np.mean(X_true, axis=0)
        ss_tot = np.sum((X_true - X_mean) ** 2)

        # R² = 1 - SS_res / SS_tot
        if ss_tot == 0:
            # All true locations are identical (degenerate case)
            return 1.0 if ss_res == 0 else 0.0

        r2 = 1.0 - (ss_res / ss_tot)

        return r2

    def __repr__(self) -> str:
        """Readable string representation."""
        floor_str = f"floor={self.floor_id}" if self.floor_id is not None else "all_floors"
        return (
            f"LinearRegressionLocalizer("
            f"location_dim={self.location_dim}, "
            f"n_features={self.n_features}, "
            f"n_training_samples={self.n_training_samples}, "
            f"{floor_str})"
        )

