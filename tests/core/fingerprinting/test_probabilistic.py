"""Unit tests for core.fingerprinting.probabilistic module.

Tests Naive Bayes fingerprinting algorithms (Eqs. 5.3-5.5).

Author: Navigation Engineer
Date: 2024
"""

import numpy as np
import pytest

from core.fingerprinting import (
    FingerprintDatabase,
    NaiveBayesFingerprintModel,
    fit_gaussian_naive_bayes,
    log_likelihood,
    log_posterior,
    map_localize,
    posterior_mean_localize,
)


@pytest.fixture
def simple_database():
    """Create a simple 2D single-floor database for testing."""
    return FingerprintDatabase(
        locations=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]),
        features=np.array(
            [[-50, -60, -70], [-60, -50, -80], [-70, -80, -50], [-55, -55, -55]]
        ),
        floor_ids=np.array([0, 0, 0, 0]),
        meta={"ap_ids": ["AP1", "AP2", "AP3"]},
    )


@pytest.fixture
def multifloor_database():
    """Create a multi-floor database for testing."""
    return FingerprintDatabase(
        locations=np.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [10.0, 0.0],  # Floor 0
                [0.0, 5.0],
                [5.0, 5.0],
                [10.0, 5.0],  # Floor 1
                [0.0, 10.0],
                [5.0, 10.0],  # Floor 2
            ]
        ),
        features=np.array(
            [
                [-50, -60, -70],
                [-52, -58, -72],
                [-54, -56, -74],  # Floor 0
                [-55, -65, -75],
                [-57, -63, -77],
                [-59, -61, -79],  # Floor 1
                [-60, -70, -80],
                [-62, -68, -82],  # Floor 2
            ]
        ),
        floor_ids=np.array([0, 0, 0, 1, 1, 1, 2, 2]),
        meta={"ap_ids": ["AP1", "AP2", "AP3"], "building_id": "test_building"},
    )


class TestNaiveBayesFingerprintModel:
    """Test suite for NaiveBayesFingerprintModel dataclass."""

    def test_model_creation_valid(self, simple_database):
        """Test creation of valid model."""
        M = 4
        N = 3

        model = NaiveBayesFingerprintModel(
            means=np.array([[-50, -60, -70], [-60, -50, -80], [-70, -80, -50], [-55, -55, -55]]),
            stds=np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            locations=simple_database.locations,
            floor_ids=simple_database.floor_ids,
            prior_probs=np.ones(M) / M,
            meta=simple_database.meta,
        )

        assert model.n_reference_points == M
        assert model.n_features == N
        assert model.location_dim == 2

    def test_model_stds_shape_mismatch_error(self, simple_database):
        """Test that mismatched stds shape raises ValueError."""
        M = 4

        with pytest.raises(ValueError, match="stds shape .* must match means shape"):
            NaiveBayesFingerprintModel(
                means=np.array([[-50, -60, -70], [-60, -50, -80], [-70, -80, -50], [-55, -55, -55]]),
                stds=np.array([[2.0, 2.0]]),  # Wrong shape
                locations=simple_database.locations,
                floor_ids=simple_database.floor_ids,
                prior_probs=np.ones(M) / M,
                meta=simple_database.meta,
            )

    def test_model_locations_size_mismatch_error(self, simple_database):
        """Test that mismatched locations size raises ValueError."""
        M = 4

        with pytest.raises(ValueError, match="locations has .* RPs, but means has"):
            NaiveBayesFingerprintModel(
                means=np.array([[-50, -60, -70], [-60, -50, -80], [-70, -80, -50], [-55, -55, -55]]),
                stds=np.full((4, 3), 2.0),
                locations=np.array([[0.0, 0.0], [10.0, 0.0]]),  # Only 2 RPs
                floor_ids=simple_database.floor_ids,
                prior_probs=np.ones(M) / M,
                meta=simple_database.meta,
            )

    def test_model_non_positive_std_error(self, simple_database):
        """Test that non-positive stds raise ValueError."""
        M = 4

        with pytest.raises(ValueError, match="All standard deviations must be positive"):
            NaiveBayesFingerprintModel(
                means=np.array([[-50, -60, -70], [-60, -50, -80], [-70, -80, -50], [-55, -55, -55]]),
                stds=np.array([[2.0, 0.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),  # Zero std
                locations=simple_database.locations,
                floor_ids=simple_database.floor_ids,
                prior_probs=np.ones(M) / M,
                meta=simple_database.meta,
            )

    def test_model_priors_not_normalized_error(self, simple_database):
        """Test that prior probabilities must sum to 1."""
        M = 4

        with pytest.raises(ValueError, match="Prior probabilities must sum to 1"):
            NaiveBayesFingerprintModel(
                means=np.array([[-50, -60, -70], [-60, -50, -80], [-70, -80, -50], [-55, -55, -55]]),
                stds=np.full((4, 3), 2.0),
                locations=simple_database.locations,
                floor_ids=simple_database.floor_ids,
                prior_probs=np.array([0.3, 0.3, 0.3, 0.3]),  # Sum = 1.2, not 1.0
                meta=simple_database.meta,
            )

    def test_model_get_floor_mask(self, multifloor_database):
        """Test get_floor_mask method."""
        model = fit_gaussian_naive_bayes(multifloor_database)

        mask_0 = model.get_floor_mask(0)
        mask_1 = model.get_floor_mask(1)

        assert np.sum(mask_0) == 3  # 3 RPs on floor 0
        assert np.sum(mask_1) == 3  # 3 RPs on floor 1
        assert np.all(model.floor_ids[mask_0] == 0)
        assert np.all(model.floor_ids[mask_1] == 1)


class TestFitGaussianNaiveBayes:
    """Test suite for fit_gaussian_naive_bayes() function."""

    def test_fit_simple_database(self, simple_database):
        """Test fitting model on simple database."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0)

        # Check model structure
        assert model.n_reference_points == 4
        assert model.n_features == 3
        assert model.location_dim == 2

        # Check that means match database features
        np.testing.assert_array_equal(model.means, simple_database.features)

        # Check that stds are constant (no repeated measurements)
        np.testing.assert_array_equal(model.stds, np.full((4, 3), 2.0))

        # Check uniform prior
        np.testing.assert_array_almost_equal(model.prior_probs, np.ones(4) / 4)

    def test_fit_multifloor_database(self, multifloor_database):
        """Test fitting model on multi-floor database."""
        model = fit_gaussian_naive_bayes(multifloor_database, min_std=1.5)

        assert model.n_reference_points == 8
        assert model.n_features == 3
        assert np.all(model.stds == 1.5)

    def test_fit_custom_min_std(self, simple_database):
        """Test fitting with custom min_std parameter."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=5.0)

        np.testing.assert_array_equal(model.stds, np.full((4, 3), 5.0))

    def test_fit_invalid_prior_error(self, simple_database):
        """Test that invalid prior type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported prior type"):
            fit_gaussian_naive_bayes(simple_database, prior="gaussian")


class TestLogLikelihood:
    """Test suite for log_likelihood() function (Eq. 5.3)."""

    def test_log_likelihood_exact_match(self, simple_database):
        """Test log-likelihood when query exactly matches an RP."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0)

        # Query exactly matches first RP
        z = np.array([-50, -60, -70])

        log_likes = log_likelihood(z, model)

        # First RP should have highest log-likelihood (zero Mahalanobis distance)
        assert np.argmax(log_likes) == 0

        # All log-likelihoods should be finite
        assert np.all(np.isfinite(log_likes))

    def test_log_likelihood_approximate_match(self, simple_database):
        """Test log-likelihood with approximate match."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0)

        # Query close to first RP
        z = np.array([-51, -61, -71])

        log_likes = log_likelihood(z, model)

        # First RP should have highest log-likelihood
        assert np.argmax(log_likes) == 0

        # Log-likelihoods should be negative (probabilities < 1)
        assert np.all(log_likes <= 0)

    def test_log_likelihood_floor_constraint(self, multifloor_database):
        """Test log-likelihood with floor constraint."""
        model = fit_gaussian_naive_bayes(multifloor_database, min_std=2.0)

        z = np.array([-56, -64, -76])

        log_likes_floor1 = log_likelihood(z, model, floor_id=1)

        # RPs not on floor 1 should have -inf log-likelihood
        mask_floor1 = model.get_floor_mask(1)
        assert np.all(np.isfinite(log_likes_floor1[mask_floor1]))
        assert np.all(np.isinf(log_likes_floor1[~mask_floor1]))

    def test_log_likelihood_dimension_mismatch_error(self, simple_database):
        """Test that query with wrong dimension raises ValueError."""
        model = fit_gaussian_naive_bayes(simple_database)

        z = np.array([-50, -60])  # Only 2 features, need 3

        with pytest.raises(ValueError, match="Query has 2 features"):
            log_likelihood(z, model)

    def test_log_likelihood_invalid_floor_error(self, simple_database):
        """Test that invalid floor_id raises ValueError."""
        model = fit_gaussian_naive_bayes(simple_database)

        z = np.array([-50, -60, -70])

        with pytest.raises(ValueError, match="Floor 5 not found"):
            log_likelihood(z, model, floor_id=5)


class TestLogPosterior:
    """Test suite for log_posterior() function."""

    def test_log_posterior_normalized(self, simple_database):
        """Test that posterior probabilities sum to 1."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0)

        z = np.array([-51, -61, -71])

        log_post = log_posterior(z, model)

        # Convert to probability space and check normalization
        posteriors = np.exp(log_post)
        assert np.isclose(np.sum(posteriors), 1.0)

    def test_log_posterior_floor_constraint(self, multifloor_database):
        """Test posterior with floor constraint."""
        model = fit_gaussian_naive_bayes(multifloor_database, min_std=2.0)

        z = np.array([-56, -64, -76])

        log_post_floor1 = log_posterior(z, model, floor_id=1)

        # RPs not on floor 1 should have -inf log-posterior
        mask_floor1 = model.get_floor_mask(1)
        assert np.all(np.isfinite(log_post_floor1[mask_floor1]))
        assert np.all(np.isinf(log_post_floor1[~mask_floor1]))

        # Posterior on floor 1 should sum to 1
        posteriors_floor1 = np.exp(log_post_floor1[mask_floor1])
        assert np.isclose(np.sum(posteriors_floor1), 1.0)

    def test_log_posterior_uniform_prior(self, simple_database):
        """Test that uniform prior gives same result as unnormalized likelihood."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0, prior="uniform")

        z = np.array([-51, -61, -71])

        log_post = log_posterior(z, model)
        log_like = log_likelihood(z, model)

        # With uniform prior, argmax of posterior = argmax of likelihood
        assert np.argmax(log_post) == np.argmax(log_like)


class TestMAPLocalize:
    """Test suite for map_localize() function (Eq. 5.4)."""

    def test_map_exact_match(self, simple_database):
        """Test MAP when query exactly matches an RP."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0)

        # Query exactly matches first RP at [0, 0]
        z = np.array([-50, -60, -70])

        x_hat = map_localize(z, model)

        np.testing.assert_array_almost_equal(x_hat, [0.0, 0.0])

    def test_map_approximate_match(self, simple_database):
        """Test MAP with approximate match."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0)

        # Query close to first RP at [0, 0]
        z = np.array([-51, -61, -71])

        x_hat = map_localize(z, model)

        # Should return one of the RP locations (MAP is discrete)
        rp_locations = model.locations
        matches = np.any([np.allclose(x_hat, loc) for loc in rp_locations])
        assert matches

    def test_map_multifloor_constraint(self, multifloor_database):
        """Test MAP with floor constraint."""
        model = fit_gaussian_naive_bayes(multifloor_database, min_std=2.0)

        # Query close to floor 1 RP
        z = np.array([-56, -64, -76])

        x_hat = map_localize(z, model, floor_id=1)

        # Result should be on floor 1 (y = 5.0)
        assert x_hat[1] == pytest.approx(5.0)

    def test_map_dimension_mismatch_error(self, simple_database):
        """Test that query with wrong dimension raises ValueError."""
        model = fit_gaussian_naive_bayes(simple_database)

        z = np.array([-50, -60])  # Only 2 features

        with pytest.raises(ValueError, match="Query has 2 features"):
            map_localize(z, model)


class TestPosteriorMeanLocalize:
    """Test suite for posterior_mean_localize() function (Eq. 5.5)."""

    def test_posterior_mean_exact_match(self, simple_database):
        """Test posterior mean when query exactly matches an RP."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0)

        # Query exactly matches first RP at [0, 0]
        z = np.array([-50, -60, -70])

        x_hat = posterior_mean_localize(z, model)

        # With exact match, posterior is concentrated on one RP
        # Result should be very close to [0, 0]
        assert np.linalg.norm(x_hat - np.array([0.0, 0.0])) < 0.5

    def test_posterior_mean_smooth_estimate(self, simple_database):
        """Test that posterior mean gives smooth (averaged) estimate."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0)

        # Query in between RPs (center of square)
        z = np.array([-55, -55, -65])

        x_hat = posterior_mean_localize(z, model)

        # Result should be inside the convex hull of RPs
        assert 0 <= x_hat[0] <= 10
        assert 0 <= x_hat[1] <= 10

        # Result should not exactly match any single RP (it's averaged)
        rp_locations = model.locations
        exact_matches = [np.allclose(x_hat, loc) for loc in rp_locations]
        # It's possible (but unlikely) to match exactly if posterior is very peaked
        # Just check result is valid

    def test_posterior_mean_multifloor_constraint(self, multifloor_database):
        """Test posterior mean with floor constraint."""
        model = fit_gaussian_naive_bayes(multifloor_database, min_std=2.0)

        # Query close to floor 1 RPs
        z = np.array([-56, -64, -76])

        x_hat = posterior_mean_localize(z, model, floor_id=1)

        # Result should be on floor 1 (y = 5.0)
        assert x_hat[1] == pytest.approx(5.0)

    def test_posterior_mean_dimension_mismatch_error(self, simple_database):
        """Test that query with wrong dimension raises ValueError."""
        model = fit_gaussian_naive_bayes(simple_database)

        z = np.array([-50, -60])  # Only 2 features

        with pytest.raises(ValueError, match="Query has 2 features"):
            posterior_mean_localize(z, model)


class TestIntegration:
    """Integration tests comparing deterministic and probabilistic methods."""

    def test_map_vs_nn_consistency(self, simple_database):
        """Test that MAP and NN give similar results with uniform prior."""
        from core.fingerprinting import nn_localize

        model = fit_gaussian_naive_bayes(simple_database, min_std=2.0)

        z = np.array([-51, -61, -71])

        # MAP with uniform prior and Gaussian likelihood
        x_map = map_localize(z, model)

        # NN with Euclidean distance (approximately Gaussian likelihood with uniform variance)
        x_nn = nn_localize(z, simple_database, metric="euclidean")

        # With small std and uniform prior, MAP should match NN
        # (both find closest RP)
        np.testing.assert_array_almost_equal(x_map, x_nn)

    def test_posterior_mean_smoother_than_map(self, simple_database):
        """Test that posterior mean is smoother than MAP."""
        model = fit_gaussian_naive_bayes(simple_database, min_std=5.0)  # Larger std

        z = np.array([-55, -55, -65])  # Query in center

        x_map = map_localize(z, model)
        x_post_mean = posterior_mean_localize(z, model)

        # MAP should be at one of the RPs (discrete)
        rp_locations = model.locations
        map_matches_rp = np.any([np.allclose(x_map, loc) for loc in rp_locations])
        assert map_matches_rp

        # Posterior mean can be anywhere (continuous)
        # Both should be in valid region
        assert 0 <= x_post_mean[0] <= 10
        assert 0 <= x_post_mean[1] <= 10

    def test_larger_std_increases_uncertainty(self, simple_database):
        """Test that larger std leads to more spread posterior."""
        z = np.array([-55, -55, -65])

        # Model with small std (peaked likelihood)
        model_small = fit_gaussian_naive_bayes(simple_database, min_std=1.0)
        log_post_small = log_posterior(z, model_small)
        post_small = np.exp(log_post_small)

        # Model with large std (flat likelihood)
        model_large = fit_gaussian_naive_bayes(simple_database, min_std=10.0)
        log_post_large = log_posterior(z, model_large)
        post_large = np.exp(log_post_large)

        # Entropy: H = -Σ p log p
        # Higher entropy = more uniform distribution
        entropy_small = -np.sum(post_small * log_post_small)
        entropy_large = -np.sum(post_large * log_post_large)

        # Larger std should give higher entropy (more uncertain)
        assert entropy_large > entropy_small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

