"""Probabilistic fingerprinting methods for Chapter 5.

This module implements Bayesian fingerprinting using Gaussian Naive Bayes
models, as described in Section 5.1.3 (Probabilistic Fingerprinting) of the book.

Key equations:
    - Eq. (5.3): Bayes posterior P(x_i | z) = P(z | x_i) P(x_i) / P(z)
    - Eq. (5.4): MAP estimate i* = argmax_i P(x_i | z)
    - Eq. (5.5): Posterior mean estimate x̂ = Σ P(x_i | z) x_i
    - Eq. (5.6): Gaussian likelihood P(z | x_i) = N(z; μ_i, Σ_i)

Author: Li-Ta Hsu
Date: 2024
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import Fingerprint, FingerprintDatabase, Location


@dataclass
class NaiveBayesFingerprintModel:
    """
    Trained Naive Bayes model for probabilistic fingerprinting.

    This model stores the learned Gaussian distributions for each feature
    (e.g., RSS from each AP) at each reference point. Under the Naive Bayes
    assumption, features are conditionally independent given the location.

    Attributes:
        means: Mean RSS values, shape (M, N) where M is number of RPs,
               N is number of features (APs).
        stds: Standard deviations, shape (M, N). Each std[i, j] represents
              the variability of feature j at location i.
        locations: Reference point locations, shape (M, d).
        floor_ids: Floor identifiers, shape (M,).
        prior_probs: Prior probabilities p(x_i), shape (M,). Defaults to
                     uniform: 1/M for all i.
        meta: Metadata dictionary from training database.

    References:
        Chapter 5, Section 5.1.3: Probabilistic fingerprinting with Gaussian models.
    """

    means: np.ndarray
    stds: np.ndarray
    locations: np.ndarray
    floor_ids: np.ndarray
    prior_probs: np.ndarray
    meta: dict

    def __post_init__(self) -> None:
        """Validate model consistency."""
        M = self.means.shape[0]

        # Check shapes
        if self.stds.shape != self.means.shape:
            raise ValueError(
                f"stds shape {self.stds.shape} must match means shape {self.means.shape}"
            )
        if self.locations.shape[0] != M:
            raise ValueError(
                f"locations has {self.locations.shape[0]} RPs, but means has {M} RPs"
            )
        if self.floor_ids.shape[0] != M:
            raise ValueError(
                f"floor_ids has {self.floor_ids.shape[0]} RPs, but means has {M} RPs"
            )
        if self.prior_probs.shape != (M,):
            raise ValueError(
                f"prior_probs shape {self.prior_probs.shape} must be ({M},)"
            )

        # Check for non-positive standard deviations
        if np.any(self.stds <= 0):
            raise ValueError("All standard deviations must be positive")

        # Check that priors sum to 1
        if not np.isclose(np.sum(self.prior_probs), 1.0):
            raise ValueError(
                f"Prior probabilities must sum to 1, got {np.sum(self.prior_probs)}"
            )

    @property
    def n_reference_points(self) -> int:
        """Number of reference points M."""
        return self.means.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features N (e.g., number of APs)."""
        return self.means.shape[1]

    @property
    def location_dim(self) -> int:
        """Dimensionality of location space (2D or 3D)."""
        return self.locations.shape[1]

    def get_floor_mask(self, floor_id: int) -> np.ndarray:
        """Get boolean mask for reference points on specified floor."""
        return self.floor_ids == floor_id


def fit_gaussian_naive_bayes(
    db: FingerprintDatabase,
    min_std: float = 1.0,
    prior: str = "uniform",
) -> NaiveBayesFingerprintModel:
    """
    Fit a Gaussian Naive Bayes model from fingerprint database.

    For each reference point i and each feature j (AP), computes:
        μ_ij = mean RSS at RP i from AP j
        σ_ij = std RSS at RP i from AP j

    Under Naive Bayes assumption, the likelihood is:
        P(z | x_i) = ∏_j N(z_j; μ_ij, σ_ij²)

    where N(·; μ, σ²) is the Gaussian density.

    **Behavior depends on database format:**
    - Single-sample DB (features shape M×N): Sets σ_ij = min_std everywhere.
    - Multi-sample DB (features shape M×S×N): Computes actual μ and σ from
      the S samples at each RP, then applies min_std as a numerical floor.

    This aligns with the book's assumption (Eq. 5.6) that sufficient survey
    samples are available to estimate P(z|x_i) parameters.

    Args:
        db: FingerprintDatabase containing reference fingerprints.
            Can be single-sample (M, N) or multi-sample (M, S, N) format.
        min_std: Minimum standard deviation floor to prevent numerical issues.
                 Default is 1.0 dBm. Applied even when σ is computed from data.
        prior: Prior distribution type. Currently only 'uniform' is supported.

    Returns:
        NaiveBayesFingerprintModel with learned parameters.

    Raises:
        ValueError: If prior type is not supported.

    Examples:
        >>> # Single-sample database (constant std)
        >>> db = load_fingerprint_database('data/sim/ch5_wifi_fingerprint_grid')
        >>> model = fit_gaussian_naive_bayes(db, min_std=2.0)
        >>> print(f"Trained model with {model.n_reference_points} RPs")
        >>> # All stds will be 2.0 dBm
        
        >>> # Multi-sample database (actual variance estimation)
        >>> db_multi = load_fingerprint_database('data/sim/ch5_wifi_fp_multisamples')
        >>> model = fit_gaussian_naive_bayes(db_multi, min_std=1.0)
        >>> # stds vary by RP and feature, floor at 1.0 dBm

    References:
        Chapter 5, Eq. (5.6): Gaussian likelihood model for P(z | x_i).
        Chapter 5, Section 5.1.3: Probabilistic fingerprinting with statistics.
    """
    M = db.n_reference_points
    N = db.n_features

    # Compute mean and std from database
    # These methods handle both single-sample and multi-sample formats
    means = db.get_mean_features()  # Shape: (M, N)
    stds = db.get_std_features(min_std=min_std)  # Shape: (M, N)

    # Prior probabilities
    if prior == "uniform":
        # Uniform prior: P(x_i) = 1/M for all i
        prior_probs = np.ones(M) / M
    else:
        raise ValueError(f"Unsupported prior type: '{prior}'. Use 'uniform'.")

    return NaiveBayesFingerprintModel(
        means=means,
        stds=stds,
        locations=db.locations.copy(),
        floor_ids=db.floor_ids.copy(),
        prior_probs=prior_probs,
        meta=db.meta.copy(),
    )


def log_likelihood(
    z: Fingerprint,
    model: NaiveBayesFingerprintModel,
    floor_id: Optional[int] = None,
) -> np.ndarray:
    """
    Compute log-likelihood log P(z | x_i) for all reference points.

    This function computes the likelihood term P(z | x_i) that appears in
    Bayes' rule (Eq. 5.3). Under Gaussian Naive Bayes (Eq. 5.6):
        log P(z | x_i) = Σ_j log N(z_j; μ_ij, σ_ij²)
                       = Σ_j [-0.5 log(2π σ_ij²) - 0.5 (z_j - μ_ij)² / σ_ij²]

    where the sum is over all features j = 1, ..., N.
    
    **Missing AP Handling:**
    If z contains NaN values (representing missing AP readings), the sum
    includes only terms for observed (non-NaN) features. If no observed
    features exist for a particular RP, returns -inf for that RP.

    Args:
        z: Query fingerprint vector, shape (N,). May contain NaN for missing APs.
        model: Trained NaiveBayesFingerprintModel.
        floor_id: Optional floor constraint. If provided, returns log-likelihoods
                  only for RPs on that floor (others set to -inf).

    Returns:
        Log-likelihood values, shape (M,), where M is number of RPs.
        Element i is log P(z | x_i). Returns -inf for RPs with no observed features.

    Raises:
        ValueError: If query dimension doesn't match model or floor doesn't exist.

    Examples:
        >>> model = fit_gaussian_naive_bayes(db)
        >>> z_query = np.array([-51, -61, -71])
        >>> log_probs = log_likelihood(z_query, model, floor_id=0)
        >>> print(f"Log-likelihoods: {log_probs}")
        
        >>> # With missing values (NaN)
        >>> z_missing = np.array([-51, np.nan, -71])  # AP2 missing
        >>> log_probs_missing = log_likelihood(z_missing, model, floor_id=0)
        >>> # Likelihood computed only using AP1 and AP3

    References:
        Chapter 5, Eq. (5.6): Gaussian likelihood model.
        Chapter 5, Eq. (5.3): Bayes posterior uses this likelihood term.
        Chapter 5, Section 5.1: Handling missing AP readings (dropout).
    """
    # Validate query dimension
    if z.shape[0] != model.n_features:
        raise ValueError(
            f"Query has {z.shape[0]} features, but model expects {model.n_features}"
        )

    M = model.n_reference_points
    N = model.n_features

    # Compute Gaussian log-likelihood for all RPs
    # log N(z_j; μ_ij, σ_ij²) = -0.5 * log(2π σ_ij²) - 0.5 * (z_j - μ_ij)² / σ_ij²

    # Expand z to (1, N) for broadcasting
    z_expanded = z.reshape(1, -1)  # Shape: (1, N)

    # Compute squared error: (z_j - μ_ij)² for all i, j
    # Shape: (M, N)
    squared_errors = (z_expanded - model.means) ** 2

    # Compute normalized squared error: (z_j - μ_ij)² / σ_ij²
    # Shape: (M, N)
    normalized_sq_errors = squared_errors / (model.stds**2)

    # Compute log normalization constant: -0.5 * log(2π σ_ij²)
    # Shape: (M, N)
    log_norm = -0.5 * np.log(2 * np.pi * model.stds**2)

    # Combine: log N(z_j; μ_ij, σ_ij²) for all i, j
    # Shape: (M, N)
    log_gaussian = log_norm - 0.5 * normalized_sq_errors

    # Handle missing values: identify observed (non-NaN) features
    observed_mask = ~np.isnan(z)  # Shape: (N,)
    n_observed = np.sum(observed_mask)
    
    if n_observed == 0:
        # No observed features: return -inf for all RPs
        log_likelihoods = np.full(M, -np.inf)
    else:
        # Sum only over observed features (axis=1)
        # Use nansum to ignore NaN contributions in log_gaussian
        # Shape: (M,)
        # Note: where z is NaN, log_gaussian will be NaN, and nansum ignores it
        log_likelihoods = np.nansum(log_gaussian, axis=1)

    # Apply floor constraint if specified
    if floor_id is not None:
        mask = model.get_floor_mask(floor_id)
        if not np.any(mask):
            raise ValueError(f"Floor {floor_id} not found in model")
        # Set log-likelihood to -inf for RPs not on target floor
        log_likelihoods[~mask] = -np.inf

    return log_likelihoods


def log_posterior(
    z: Fingerprint,
    model: NaiveBayesFingerprintModel,
    floor_id: Optional[int] = None,
) -> np.ndarray:
    """
    Compute log-posterior log P(x_i | z) for all reference points.

    Implements Eq. (5.3) using Bayes' rule:
        P(x_i | z) = P(z | x_i) P(x_i) / P(z)

    In log space:
        log P(x_i | z) = log P(z | x_i) + log P(x_i) - log P(z)

    Since log P(z) is constant for all i, we compute unnormalized log-posterior:
        log P̃(x_i | z) = log P(z | x_i) + log P(x_i)

    and normalize afterward:
        P(x_i | z) = exp(log P̃(x_i | z)) / Σ_k exp(log P̃(x_k | z))

    Args:
        z: Query fingerprint vector, shape (N,).
        model: Trained NaiveBayesFingerprintModel.
        floor_id: Optional floor constraint.

    Returns:
        Log-posterior probabilities, shape (M,), normalized so that
        Σ_i exp(log_posterior[i]) = 1.

    Raises:
        ValueError: If query dimension doesn't match model or floor doesn't exist.

    Examples:
        >>> model = fit_gaussian_naive_bayes(db)
        >>> z_query = np.array([-51, -61, -71])
        >>> log_post = log_posterior(z_query, model, floor_id=0)
        >>> posteriors = np.exp(log_post)
        >>> print(f"Posterior probabilities: {posteriors}")

    References:
        Chapter 5, Eq. (5.3): Bayes posterior P(x_i | z) = P(z | x_i) P(x_i) / P(z).
        Chapter 5, Eqs. (5.4)-(5.5): Used in MAP and posterior mean estimation.
    """
    # Compute log-likelihood: log P(z | x_i) using Eq. (5.6)
    log_like = log_likelihood(z, model, floor_id=floor_id)

    # Compute unnormalized log-posterior: log P(z | x_i) + log P(x_i)
    # Handle floor constraint: prior_probs already has proper shape (M,)
    if floor_id is not None:
        mask = model.get_floor_mask(floor_id)
        log_prior = np.full(model.n_reference_points, -np.inf)
        log_prior[mask] = np.log(model.prior_probs[mask])
    else:
        log_prior = np.log(model.prior_probs)

    log_unnorm = log_like + log_prior

    # Normalize using log-sum-exp trick for numerical stability
    # log P(z) = log Σ_i P(z | x_i) P(x_i)
    #          = log Σ_i exp(log P(z | x_i) + log P(x_i))
    log_evidence = _log_sum_exp(log_unnorm)

    # Compute normalized log-posterior (Eq. 5.3)
    # log P(x_i | z) = log P̃(x_i | z) - log P(z)
    log_post = log_unnorm - log_evidence

    return log_post


def _log_sum_exp(log_values: np.ndarray) -> float:
    """
    Compute log(Σ exp(log_values)) using numerically stable algorithm.

    This function implements the log-sum-exp trick:
        log Σ_i exp(x_i) = a + log Σ_i exp(x_i - a)
    where a = max(x_i) to prevent overflow/underflow.

    Args:
        log_values: Array of log-space values, shape (M,).

    Returns:
        Scalar log(Σ exp(log_values)).

    References:
        Numerical stability technique for log-space computations.
    """
    # Handle all -inf case (empty valid set)
    if np.all(np.isinf(log_values)):
        return -np.inf

    # Log-sum-exp trick
    max_val = np.max(log_values)
    return max_val + np.log(np.sum(np.exp(log_values - max_val)))


def map_localize(
    z: Fingerprint,
    model: NaiveBayesFingerprintModel,
    floor_id: Optional[int] = None,
) -> Location:
    """
    Maximum A Posteriori (MAP) probabilistic fingerprinting.

    Implements Eq. (5.4) in Chapter 5:
        i* = argmax_{i=1..M} p(x_i | z)
           = argmax_{i=1..M} [p(z | x_i) p(x_i)]

    and returns x̂ = x_{i*}.

    Args:
        z: Query fingerprint vector, shape (N,).
        model: Trained NaiveBayesFingerprintModel.
        floor_id: Optional floor constraint. If None, searches all floors.

    Returns:
        Estimated location x̂, shape (d,), corresponding to RP with
        maximum posterior probability.

    Raises:
        ValueError: If query dimension doesn't match model or floor doesn't exist.

    Examples:
        >>> db = load_fingerprint_database('data/sim/ch5_wifi_fingerprint_grid')
        >>> model = fit_gaussian_naive_bayes(db, min_std=2.0)
        >>> z_query = np.array([-51, -61, -71])
        >>> x_hat = map_localize(z_query, model, floor_id=0)
        >>> print(f"MAP estimate: {x_hat}")

    References:
        Chapter 5, Eq. (5.4): MAP decision rule for probabilistic fingerprinting.
    """
    # Compute log-posterior for all RPs
    # log p(x_i | z) for i = 1, ..., M
    log_post = log_posterior(z, model, floor_id=floor_id)

    # Find RP with maximum posterior
    # Implements: i* = argmax_i p(x_i | z) from Eq. (5.4)
    i_star = np.argmax(log_post)

    # Return location of MAP RP
    # Implements: x̂ = x_{i*} from Eq. (5.4)
    return model.locations[i_star]


def posterior_mean_localize(
    z: Fingerprint,
    model: NaiveBayesFingerprintModel,
    floor_id: Optional[int] = None,
    top_k: Optional[int] = None,
) -> Location:
    """
    Posterior mean (Bayesian) probabilistic fingerprinting.

    Implements Eq. (5.5) in Chapter 5:
        x̂ = Σ_{i=1}^M p(x_i | z) x_i
          = E[x | z]

    where the expectation is with respect to the posterior distribution.

    **Practical optimization (book guidance):** The sum often includes many
    negligible probabilities and only a few dominant ones. Setting `top_k`
    computes the posterior mean using only the k highest posterior candidates,
    which is typically sufficient and more efficient.

    Args:
        z: Query fingerprint vector, shape (N,).
        model: Trained NaiveBayesFingerprintModel.
        floor_id: Optional floor constraint. If None, searches all floors.
        top_k: Optional. If set, compute posterior mean using only the top-k
               candidates with highest posterior probabilities. If None (default),
               uses all RPs. Book guidance: k=10-20 typically sufficient.

    Returns:
        Estimated location x̂, shape (d,), weighted average of RPs
        according to their posterior probabilities (all RPs or top-k subset).

    Raises:
        ValueError: If query dimension doesn't match model, floor doesn't exist,
                    or top_k is invalid.

    Examples:
        >>> db = load_fingerprint_database('data/sim/ch5_wifi_fingerprint_grid')
        >>> model = fit_gaussian_naive_bayes(db, min_std=2.0)
        >>> z_query = np.array([-51, -61, -71])
        
        >>> # Full posterior mean (all RPs)
        >>> x_hat_full = posterior_mean_localize(z_query, model, floor_id=0)
        
        >>> # Top-k posterior mean (faster, typically sufficient)
        >>> x_hat_topk = posterior_mean_localize(z_query, model, floor_id=0, top_k=10)
        >>> # Results are nearly identical but top-k is faster

    References:
        Chapter 5, Eq. (5.5): Posterior mean estimate for probabilistic fingerprinting.
        Chapter 5, Section 5.1.2: "...a calculation based on the top k candidates
                                    is typically sufficient."
    """
    # Compute log-posterior for all RPs
    log_post = log_posterior(z, model, floor_id=floor_id)

    # Convert to probability space
    # P(x_i | z) = exp(log P(x_i | z))
    posteriors = np.exp(log_post)

    # Apply top-k filtering if requested
    if top_k is not None:
        # Validate top_k
        n_valid = np.sum(np.isfinite(posteriors))  # Count valid (non-inf) posteriors
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > n_valid:
            raise ValueError(
                f"top_k={top_k} exceeds number of valid candidates ({n_valid})"
            )

        # Find indices of top-k highest posteriors
        # Use np.argpartition for efficient top-k selection (O(n) vs O(n log n))
        # Note: argpartition doesn't sort, just partitions around k-th element
        top_k_indices = np.argpartition(posteriors, -top_k)[-top_k:]

        # Extract top-k posteriors and locations
        top_k_posteriors = posteriors[top_k_indices]
        top_k_locations = model.locations[top_k_indices]

        # Renormalize probabilities on top-k subset
        # Sum may not be exactly 1.0 due to truncation
        posterior_sum = np.sum(top_k_posteriors)
        if posterior_sum > 0:
            top_k_posteriors = top_k_posteriors / posterior_sum
        else:
            # All top-k posteriors are zero (shouldn't happen in practice)
            raise ValueError("All top-k posterior probabilities are zero")

        # Compute posterior mean using only top-k candidates
        # x̂ = Σ_{i in top-k} P(x_i | z) x_i
        x_hat = np.sum(top_k_posteriors[:, np.newaxis] * top_k_locations, axis=0)
    else:
        # Full posterior mean (all RPs)
        # Implements: x̂ = Σ_i P(x_i | z) x_i from Eq. (5.5)
        # posteriors shape: (M,)
        # model.locations shape: (M, d)
        # Result shape: (d,)
        x_hat = np.sum(posteriors[:, np.newaxis] * model.locations, axis=0)

    return x_hat

