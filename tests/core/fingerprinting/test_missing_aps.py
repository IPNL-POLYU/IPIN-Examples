"""Unit tests for missing AP (dropout) handling in fingerprinting.

Tests the NaN-based missing AP support across deterministic and probabilistic
fingerprinting methods. This validates that the system gracefully handles
scenarios where some AP readings are unavailable (signal dropout).

Author: Li-Ta Hsu
Date: December 2024
"""

import numpy as np
import pytest

from core.fingerprinting import (
    FingerprintDatabase,
    distance,
    pairwise_distances,
    nn_localize,
    knn_localize,
    fit_gaussian_naive_bayes,
    log_likelihood,
    map_localize,
    posterior_mean_localize,
)


class TestDistanceWithMissingAPs:
    """Test distance computation with NaN values."""

    def test_distance_no_missing_values(self):
        """Test that distance works normally without missing values."""
        z = np.array([-50.0, -60.0, -70.0])
        f = np.array([-52.0, -58.0, -72.0])
        
        d_eucl = distance(z, f, metric="euclidean")
        d_manh = distance(z, f, metric="manhattan")
        
        # Expected: sqrt(4 + 4 + 4) = 3.46, |2| + |2| + |2| = 6
        assert abs(d_eucl - 3.464) < 0.01
        assert abs(d_manh - 6.0) < 0.01

    def test_distance_partial_missing_query(self):
        """Test distance when query has missing values."""
        z = np.array([-50.0, np.nan, -70.0])
        f = np.array([-52.0, -58.0, -72.0])
        
        # Should compute distance only on AP1 and AP3
        # Expected: sqrt(4 + 4) = 2.828
        d = distance(z, f, metric="euclidean")
        assert abs(d - 2.828) < 0.01
        
        # Manhattan: |2| + |2| = 4
        d_manh = distance(z, f, metric="manhattan")
        assert abs(d_manh - 4.0) < 0.01

    def test_distance_partial_missing_reference(self):
        """Test distance when reference has missing values."""
        z = np.array([-50.0, -60.0, -70.0])
        f = np.array([-52.0, np.nan, -72.0])
        
        # Should compute distance only on AP1 and AP3
        d = distance(z, f, metric="euclidean")
        assert abs(d - 2.828) < 0.01

    def test_distance_partial_missing_both(self):
        """Test distance when both have missing values (overlapping dims)."""
        z = np.array([-50.0, np.nan, -70.0])
        f = np.array([-52.0, -58.0, np.nan])
        
        # Only AP1 is valid in both
        # Expected: |2| = 2
        d = distance(z, f, metric="euclidean")
        assert abs(d - 2.0) < 0.01

    def test_distance_no_overlap(self):
        """Test distance when no overlapping valid dimensions."""
        z = np.array([np.nan, -60.0, np.nan])
        f = np.array([-52.0, np.nan, -72.0])
        
        # No overlap -> should return infinity
        d = distance(z, f, metric="euclidean")
        assert np.isinf(d) and d > 0

    def test_distance_all_nan_query(self):
        """Test distance when query is all NaN."""
        z = np.array([np.nan, np.nan, np.nan])
        f = np.array([-52.0, -58.0, -72.0])
        
        d = distance(z, f, metric="euclidean")
        assert np.isinf(d) and d > 0

    def test_distance_all_nan_reference(self):
        """Test distance when reference is all NaN."""
        z = np.array([-50.0, -60.0, -70.0])
        f = np.array([np.nan, np.nan, np.nan])
        
        d = distance(z, f, metric="euclidean")
        assert np.isinf(d) and d > 0

    def test_distance_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        z = np.array([-50.0, -60.0])
        f = np.array([-52.0, -58.0, -72.0])
        
        with pytest.raises(ValueError, match="must have same shape"):
            distance(z, f)


class TestPairwiseDistancesWithMissingAPs:
    """Test pairwise distance computation with NaN values."""

    def test_pairwise_no_missing(self):
        """Test normal pairwise distances without missing values."""
        z = np.array([-50.0, -60.0, -70.0])
        F = np.array([
            [-52.0, -58.0, -72.0],
            [-48.0, -62.0, -68.0],
            [-55.0, -55.0, -75.0],
        ])
        
        distances = pairwise_distances(z, F, metric="euclidean")
        
        assert distances.shape == (3,)
        # RP0: sqrt(4 + 4 + 4) = 3.464
        assert abs(distances[0] - 3.464) < 0.01
        # RP1: sqrt(4 + 4 + 4) = 3.464
        assert abs(distances[1] - 3.464) < 0.01
        # RP2: sqrt(25 + 25 + 25) = 8.660
        assert abs(distances[2] - 8.660) < 0.01

    def test_pairwise_query_missing(self):
        """Test pairwise distances with missing values in query."""
        z = np.array([-50.0, np.nan, -70.0])
        F = np.array([
            [-52.0, -58.0, -72.0],  # AP1, AP3 valid
            [-48.0, -62.0, -68.0],  # AP1, AP3 valid
            [-55.0, -55.0, np.nan],  # Only AP1 valid
        ])
        
        distances = pairwise_distances(z, F, metric="euclidean")
        
        assert distances.shape == (3,)
        # RP0: sqrt(4 + 4) = 2.828
        assert abs(distances[0] - 2.828) < 0.01
        # RP1: sqrt(4 + 4) = 2.828
        assert abs(distances[1] - 2.828) < 0.01
        # RP2: only AP1: |5| = 5.0
        assert abs(distances[2] - 5.0) < 0.01

    def test_pairwise_reference_missing(self):
        """Test pairwise distances with missing values in references."""
        z = np.array([-50.0, -60.0, -70.0])
        F = np.array([
            [-52.0, np.nan, -72.0],  # AP2 missing
            [np.nan, -62.0, -68.0],  # AP1 missing
            [np.nan, np.nan, np.nan],  # All missing -> inf
        ])
        
        distances = pairwise_distances(z, F, metric="euclidean")
        
        assert distances.shape == (3,)
        # RP0: AP1, AP3: sqrt(4 + 4) = 2.828
        assert abs(distances[0] - 2.828) < 0.01
        # RP1: AP2, AP3: sqrt(4 + 4) = 2.828
        assert abs(distances[1] - 2.828) < 0.01
        # RP2: no overlap -> inf
        assert np.isinf(distances[2]) and distances[2] > 0

    def test_pairwise_manhattan_with_missing(self):
        """Test pairwise Manhattan distance with missing values."""
        z = np.array([-50.0, np.nan, -70.0])
        F = np.array([
            [-52.0, -58.0, -72.0],
            [-48.0, -62.0, -68.0],
        ])
        
        distances = pairwise_distances(z, F, metric="manhattan")
        
        # RP0: |2| + |2| = 4
        assert abs(distances[0] - 4.0) < 0.01
        # RP1: |2| + |2| = 4
        assert abs(distances[1] - 4.0) < 0.01


class TestNNLocalizationWithMissingAPs:
    """Test NN localization with missing AP readings."""

    def test_nn_with_missing_query(self):
        """Test NN localization when query has missing APs."""
        # Create database with 4 RPs
        locations = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
            [-80, -70, -60],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        # Query with missing AP2
        query = np.array([-55.0, np.nan, -75.0])
        
        # Should still localize (using AP1 and AP3 only)
        pos = nn_localize(query, db, floor_id=0)
        
        assert pos.shape == (2,)
        assert not np.any(np.isnan(pos))
        assert not np.any(np.isinf(pos))

    def test_nn_with_missing_database(self):
        """Test NN when some RPs have missing APs."""
        locations = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, np.nan, -80],  # RP1 missing AP2
            [np.nan, -80, -50],  # RP2 missing AP1
        ], dtype=float)
        floor_ids = np.array([0, 0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        query = np.array([-55.0, -65.0, -75.0])
        pos = nn_localize(query, db, floor_id=0)
        
        assert pos.shape == (2,)
        assert not np.any(np.isnan(pos))


class TestKNNLocalizationWithMissingAPs:
    """Test k-NN localization with missing AP readings."""

    def test_knn_with_missing_query(self):
        """Test k-NN localization when query has missing APs."""
        locations = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
            [-80, -70, -60],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        # Query with missing AP2
        query = np.array([-55.0, np.nan, -75.0])
        
        # k-NN with k=3
        pos = knn_localize(query, db, k=3, floor_id=0)
        
        assert pos.shape == (2,)
        assert not np.any(np.isnan(pos))
        assert not np.any(np.isinf(pos))

    def test_knn_with_high_dropout_rate(self):
        """Test k-NN when 20% of query APs are missing (acceptance criterion)."""
        np.random.seed(42)
        n_aps = 10
        n_rps = 20
        
        # Generate synthetic database
        locations = np.random.rand(n_rps, 2) * 50
        features = -50 - np.random.rand(n_rps, n_aps) * 40
        floor_ids = np.zeros(n_rps, dtype=int)
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": [f"AP{i}" for i in range(n_aps)], "unit": "dBm"}
        )
        
        # Generate query with 20% dropout
        query = -50 - np.random.rand(n_aps) * 40
        dropout_indices = np.random.choice(n_aps, size=int(0.2 * n_aps), replace=False)
        query[dropout_indices] = np.nan
        
        # Should not crash and should return valid position
        pos = knn_localize(query, db, k=5, floor_id=0)
        
        assert pos.shape == (2,)
        assert not np.any(np.isnan(pos))
        assert not np.any(np.isinf(pos))


class TestLogLikelihoodWithMissingAPs:
    """Test probabilistic log-likelihood with missing APs."""

    def test_log_likelihood_no_missing(self):
        """Test log-likelihood without missing values (baseline)."""
        # Create simple database
        locations = np.array([[0, 0], [10, 0]], dtype=float)
        features = np.array([[-50, -60], [-60, -50]], dtype=float)
        floor_ids = np.array([0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2"], "unit": "dBm"}
        )
        
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        
        query = np.array([-55.0, -65.0])
        log_lik = log_likelihood(query, model, floor_id=0)
        
        assert log_lik.shape == (2,)
        assert np.all(np.isfinite(log_lik))

    def test_log_likelihood_partial_missing(self):
        """Test log-likelihood when query has partial missing APs."""
        locations = np.array([[0, 0], [10, 0]], dtype=float)
        features = np.array([[-50, -60, -70], [-60, -50, -80]], dtype=float)
        floor_ids = np.array([0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        
        # Query with AP2 missing
        query = np.array([-55.0, np.nan, -75.0])
        log_lik = log_likelihood(query, model, floor_id=0)
        
        assert log_lik.shape == (2,)
        # Should compute likelihood using only AP1 and AP3
        assert np.all(np.isfinite(log_lik))

    def test_log_likelihood_all_missing(self):
        """Test log-likelihood when all query APs are missing."""
        locations = np.array([[0, 0], [10, 0]], dtype=float)
        features = np.array([[-50, -60], [-60, -50]], dtype=float)
        floor_ids = np.array([0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2"], "unit": "dBm"}
        )
        
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        
        # Query with all APs missing
        query = np.array([np.nan, np.nan])
        log_lik = log_likelihood(query, model, floor_id=0)
        
        # Should return -inf for all RPs (no information)
        assert log_lik.shape == (2,)
        assert np.all(np.isinf(log_lik))
        assert np.all(log_lik < 0)

    def test_log_likelihood_high_dropout_rate(self):
        """Test log-likelihood with 20% dropout (acceptance criterion)."""
        np.random.seed(42)
        n_aps = 10
        n_rps = 20
        
        # Generate synthetic database
        locations = np.random.rand(n_rps, 2) * 50
        features = -50 - np.random.rand(n_rps, n_aps) * 40
        floor_ids = np.zeros(n_rps, dtype=int)
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": [f"AP{i}" for i in range(n_aps)], "unit": "dBm"}
        )
        
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        
        # Generate query with 20% dropout
        query = -50 - np.random.rand(n_aps) * 40
        dropout_indices = np.random.choice(n_aps, size=int(0.2 * n_aps), replace=False)
        query[dropout_indices] = np.nan
        
        # Should not crash and should return valid log-likelihoods
        log_lik = log_likelihood(query, model, floor_id=0)
        
        assert log_lik.shape == (n_rps,)
        # Should have finite values (80% of APs still available)
        assert np.all(np.isfinite(log_lik))


class TestMAPLocalizationWithMissingAPs:
    """Test MAP localization with missing APs."""

    def test_map_with_missing_query(self):
        """Test MAP localization when query has missing APs."""
        locations = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
            [-80, -70, -60],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        
        # Query with missing AP2
        query = np.array([-55.0, np.nan, -75.0])
        
        pos = map_localize(query, model, floor_id=0)
        
        assert pos.shape == (2,)
        assert not np.any(np.isnan(pos))
        assert not np.any(np.isinf(pos))


class TestPosteriorMeanLocalizationWithMissingAPs:
    """Test posterior mean localization with missing APs."""

    def test_posterior_mean_with_missing_query(self):
        """Test posterior mean when query has missing APs."""
        locations = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
            [-80, -70, -60],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        
        # Query with missing AP2
        query = np.array([-55.0, np.nan, -75.0])
        
        pos = posterior_mean_localize(query, model, floor_id=0)
        
        assert pos.shape == (2,)
        assert not np.any(np.isnan(pos))
        assert not np.any(np.isinf(pos))

    def test_posterior_mean_with_top_k_and_missing(self):
        """Test posterior mean with top_k when query has missing APs."""
        np.random.seed(42)
        n_aps = 8
        n_rps = 20
        
        # Generate synthetic database
        locations = np.random.rand(n_rps, 2) * 50
        features = -50 - np.random.rand(n_rps, n_aps) * 40
        floor_ids = np.zeros(n_rps, dtype=int)
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": [f"AP{i}" for i in range(n_aps)], "unit": "dBm"}
        )
        
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        
        # Query with 20% dropout
        query = -50 - np.random.rand(n_aps) * 40
        dropout_indices = np.random.choice(n_aps, size=int(0.2 * n_aps), replace=False)
        query[dropout_indices] = np.nan
        
        # Test with top_k
        pos = posterior_mean_localize(query, model, floor_id=0, top_k=10)
        
        assert pos.shape == (2,)
        assert not np.any(np.isnan(pos))
        assert not np.any(np.isinf(pos))


class TestDatabaseWithMissingAPsInSamples:
    """Test database with NaN in multi-sample features."""

    def test_database_creation_with_nan(self):
        """Test that database allows NaN in features."""
        locations = np.array([[0, 0], [10, 0]], dtype=float)
        features = np.array([
            [[-50, -60], [-51, np.nan], [-49, -61]],  # RP0: 3 samples, sample 2 missing AP2
            [[-60, -50], [np.nan, -49], [-59, -51]],  # RP1: 3 samples, sample 2 missing AP1
        ])  # Shape: (2, 3, 2)
        floor_ids = np.array([0, 0])
        
        # Should not raise error
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2"], "unit": "dBm"}
        )
        
        assert db.n_reference_points == 2
        assert db.n_features == 2
        assert db.n_samples_per_rp == 3

    def test_mean_std_computation_with_nan(self):
        """Test that mean/std computation handles NaN properly."""
        locations = np.array([[0, 0]], dtype=float)
        features = np.array([
            [[-50, -60], [-52, np.nan], [-48, -62], [np.nan, -58], [-51, -59]],
        ])  # Shape: (1, 5, 2)
        floor_ids = np.array([0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2"], "unit": "dBm"}
        )
        
        # Compute mean (should use nanmean)
        means = db.get_mean_features()
        
        # AP1: mean of [-50, -52, -48, -51] = -50.25 (ignoring NaN)
        # AP2: mean of [-60, -62, -58, -59] = -59.75 (ignoring NaN)
        assert abs(means[0, 0] - (-50.25)) < 0.01
        assert abs(means[0, 1] - (-59.75)) < 0.01
        
        # Compute std (should use nanstd)
        stds = db.get_std_features(min_std=0.5)
        
        # Should have positive stds (ignoring NaN)
        assert stds[0, 0] > 0.5
        assert stds[0, 1] > 0.5

    def test_fit_model_with_nan_samples(self):
        """Test fitting Gaussian Naive Bayes with NaN in samples."""
        locations = np.array([[0, 0], [10, 0]], dtype=float)
        features = np.array([
            [[-50, -60], [-51, np.nan], [-49, -61]],
            [[-60, -50], [np.nan, -49], [-59, -51]],
        ])
        floor_ids = np.array([0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2"], "unit": "dBm"}
        )
        
        # Should fit successfully
        model = fit_gaussian_naive_bayes(db, min_std=1.0)
        
        assert model.n_reference_points == 2
        assert model.n_features == 2
        assert np.all(model.stds > 0)  # All stds should be positive


class TestAcceptanceCriteria:
    """Test acceptance criteria from P1 requirements."""

    def test_acceptance_20_percent_dropout_no_crash(self):
        """
        Acceptance: If 20% of AP readings are dropped from queries,
        localization still runs and doesn't crash.
        """
        np.random.seed(42)
        n_aps = 10
        n_rps = 50
        
        # Generate synthetic database
        locations = np.random.rand(n_rps, 2) * 100
        features = -50 - np.random.rand(n_rps, n_aps) * 40
        floor_ids = np.zeros(n_rps, dtype=int)
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": [f"AP{i}" for i in range(n_aps)], "unit": "dBm"}
        )
        
        # Fit probabilistic model
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        
        # Test 100 queries with 20% dropout each
        n_queries = 100
        for _ in range(n_queries):
            # Generate query
            query = -50 - np.random.rand(n_aps) * 40
            
            # Apply 20% dropout
            dropout_indices = np.random.choice(n_aps, size=int(0.2 * n_aps), replace=False)
            query[dropout_indices] = np.nan
            
            # Test deterministic methods
            pos_nn = nn_localize(query, db, floor_id=0)
            assert not np.any(np.isnan(pos_nn))
            
            pos_knn = knn_localize(query, db, k=5, floor_id=0)
            assert not np.any(np.isnan(pos_knn))
            
            # Test probabilistic methods
            pos_map = map_localize(query, model, floor_id=0)
            assert not np.any(np.isnan(pos_map))
            
            pos_mean = posterior_mean_localize(query, model, floor_id=0)
            assert not np.any(np.isnan(pos_mean))
        
        # If we got here, all 100 queries succeeded!
        assert True

    def test_acceptance_varying_dropout_rates(self):
        """Test with varying dropout rates (0%, 10%, 30%, 50%)."""
        np.random.seed(42)
        n_aps = 10
        n_rps = 20
        
        locations = np.random.rand(n_rps, 2) * 50
        features = -50 - np.random.rand(n_rps, n_aps) * 40
        floor_ids = np.zeros(n_rps, dtype=int)
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": [f"AP{i}" for i in range(n_aps)], "unit": "dBm"}
        )
        
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        
        dropout_rates = [0.0, 0.1, 0.3, 0.5]
        
        for dropout_rate in dropout_rates:
            query = -50 - np.random.rand(n_aps) * 40
            
            if dropout_rate > 0:
                n_dropout = int(dropout_rate * n_aps)
                dropout_indices = np.random.choice(n_aps, size=n_dropout, replace=False)
                query[dropout_indices] = np.nan
            
            # All methods should work
            pos_nn = nn_localize(query, db, floor_id=0)
            pos_knn = knn_localize(query, db, k=5, floor_id=0)
            pos_map = map_localize(query, model, floor_id=0)
            pos_mean = posterior_mean_localize(query, model, floor_id=0)
            
            # All positions should be valid
            for pos in [pos_nn, pos_knn, pos_map, pos_mean]:
                assert not np.any(np.isnan(pos))
                assert not np.any(np.isinf(pos))

