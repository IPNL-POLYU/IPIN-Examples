"""Unit tests for core.fingerprinting.pattern_recognition module.

Tests linear regression-based fingerprinting.

Author: Navigation Engineer
Date: 2024
"""

import numpy as np
import pytest

from core.fingerprinting import (
    FingerprintDatabase,
    LinearRegressionLocalizer,
)


@pytest.fixture
def simple_database():
    """Create a simple 2D single-floor database for testing."""
    # Create a linear relationship with more variation:
    # x ≈ 0.1 * z1, y ≈ 0.1 * z2
    # z3 has independent variation
    return FingerprintDatabase(
        locations=np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]),
        features=np.array(
            [
                [0, 0, -70],
                [10, 20, -65],
                [20, 40, -75],
                [30, 60, -68],
                [40, 80, -72],
            ]
        ),
        floor_ids=np.array([0, 0, 0, 0, 0]),
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
                [5.0, 10.0],
                [10.0, 10.0],  # Floor 2 (added one more RP)
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
                [-62, -68, -82],
                [-64, -66, -84],  # Floor 2 (added one more RP)
            ]
        ),
        floor_ids=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        meta={"ap_ids": ["AP1", "AP2", "AP3"], "building_id": "test_building"},
    )


class TestLinearRegressionLocalizer:
    """Test suite for LinearRegressionLocalizer class."""

    def test_model_creation_valid(self, simple_database):
        """Test creation of valid model."""
        model = LinearRegressionLocalizer.fit(simple_database)

        assert model.location_dim == 2
        assert model.n_features == 3
        assert model.n_training_samples == 5
        assert model.floor_id is None

    def test_model_weights_shape(self, simple_database):
        """Test that weights have correct shape."""
        model = LinearRegressionLocalizer.fit(simple_database)

        # Weights should be (d, N) = (2, 3)
        assert model.weights.shape == (2, 3)
        # Bias should be (d,) = (2,)
        assert model.bias.shape == (2,)

    def test_model_validation_weights_not_2d_error(self):
        """Test that non-2D weights raise ValueError."""
        with pytest.raises(ValueError, match="weights must be 2D array"):
            LinearRegressionLocalizer(
                weights=np.array([1.0, 2.0, 3.0]),  # 1D, should be 2D
                bias=np.array([0.0, 0.0]),
                floor_id=0,
                n_training_samples=5,
                meta={},
            )

    def test_model_validation_bias_not_1d_error(self):
        """Test that non-1D bias raises ValueError."""
        with pytest.raises(ValueError, match="bias must be 1D array|incompatible"):
            LinearRegressionLocalizer(
                weights=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                bias=np.array([[0.0, 0.0]]),  # 2D, should be 1D
                floor_id=0,
                n_training_samples=5,
                meta={},
            )

    def test_model_validation_incompatible_dimensions_error(self):
        """Test that incompatible weights and bias dimensions raise ValueError."""
        with pytest.raises(ValueError, match="incompatible with bias shape"):
            LinearRegressionLocalizer(
                weights=np.array([[1.0, 2.0, 3.0]]),  # (1, 3)
                bias=np.array([0.0, 0.0]),  # (2,), incompatible
                floor_id=0,
                n_training_samples=5,
                meta={},
            )

    def test_model_validation_zero_samples_error(self):
        """Test that zero training samples raise ValueError."""
        with pytest.raises(ValueError, match="n_training_samples must be >= 1"):
            LinearRegressionLocalizer(
                weights=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                bias=np.array([0.0, 0.0]),
                floor_id=0,
                n_training_samples=0,  # Invalid
                meta={},
            )


class TestFit:
    """Test suite for LinearRegressionLocalizer.fit() method."""

    def test_fit_simple_database(self, simple_database):
        """Test fitting on simple database with linear relationship."""
        model = LinearRegressionLocalizer.fit(simple_database)

        # Check model structure
        assert model.location_dim == 2
        assert model.n_features == 3
        assert model.n_training_samples == 5
        assert model.floor_id is None

        # For the simple linear data, model should learn an approximate relationship
        # x[0] is primarily influenced by z[0], x[1] by z[1]
        # Check that model can fit the training data reasonably
        r2 = model.score(simple_database)
        assert r2 > 0.9  # High R² indicates good fit

    def test_fit_multifloor_database_all_floors(self, multifloor_database):
        """Test fitting on multi-floor database (all floors)."""
        model = LinearRegressionLocalizer.fit(multifloor_database)

        assert model.n_training_samples == 9  # Updated to 9 RPs
        assert model.floor_id is None

    def test_fit_multifloor_database_single_floor(self, multifloor_database):
        """Test fitting on single floor from multi-floor database."""
        model_floor0 = LinearRegressionLocalizer.fit(multifloor_database, floor_id=0)

        assert model_floor0.n_training_samples == 3  # 3 RPs on floor 0
        assert model_floor0.floor_id == 0

        model_floor1 = LinearRegressionLocalizer.fit(multifloor_database, floor_id=1)

        assert model_floor1.n_training_samples == 3  # 3 RPs on floor 1
        assert model_floor1.floor_id == 1

    def test_fit_with_regularization(self, simple_database):
        """Test fitting with L2 regularization."""
        model_noreg = LinearRegressionLocalizer.fit(simple_database, regularization=0.0)
        model_reg = LinearRegressionLocalizer.fit(simple_database, regularization=10.0)

        # With regularization, weights should be smaller (shrunk toward zero)
        weights_norm_noreg = np.linalg.norm(model_noreg.weights)
        weights_norm_reg = np.linalg.norm(model_reg.weights)

        # Regularized model should have smaller weights
        assert weights_norm_reg < weights_norm_noreg

    def test_fit_insufficient_data_error(self):
        """Test that insufficient training data raises ValueError."""
        # Create database with M < N (2 samples, 3 features)
        db_small = FingerprintDatabase(
            locations=np.array([[0.0, 0.0], [1.0, 1.0]]),
            features=np.array([[-50, -60, -70], [-51, -61, -71]]),
            floor_ids=np.array([0, 0]),
            meta={},
        )

        with pytest.raises(ValueError, match="Insufficient training data"):
            LinearRegressionLocalizer.fit(db_small)

    def test_fit_invalid_floor_error(self, simple_database):
        """Test that invalid floor_id raises ValueError."""
        with pytest.raises(ValueError, match="Floor 5 not found"):
            LinearRegressionLocalizer.fit(simple_database, floor_id=5)


class TestPredict:
    """Test suite for LinearRegressionLocalizer.predict() method."""

    def test_predict_simple(self, simple_database):
        """Test prediction on simple linear data."""
        model = LinearRegressionLocalizer.fit(simple_database)

        # Query that matches training point: z = [10, 20, -71] → x = [1, 2]
        z = np.array([10.0, 20.0, -71.0])
        x_hat = model.predict(z)

        # Should predict close to [1, 2]
        np.testing.assert_array_almost_equal(x_hat, [1.0, 2.0], decimal=5)

    def test_predict_interpolation(self, simple_database):
        """Test prediction with interpolation."""
        model = LinearRegressionLocalizer.fit(simple_database)

        # Query between training points: z = [15, 30, -71.5] → x ≈ [1.5, 3.0]
        z = np.array([15.0, 30.0, -71.5])
        x_hat = model.predict(z)

        # Should predict approximately [1.5, 3.0]
        assert x_hat[0] == pytest.approx(1.5, abs=0.1)
        assert x_hat[1] == pytest.approx(3.0, abs=0.1)

    def test_predict_dimension_mismatch_error(self, simple_database):
        """Test that query with wrong dimension raises ValueError."""
        model = LinearRegressionLocalizer.fit(simple_database)

        z = np.array([10.0, 20.0])  # Only 2 features, need 3

        with pytest.raises(ValueError, match="Query has 2 features"):
            model.predict(z)


class TestPredictBatch:
    """Test suite for LinearRegressionLocalizer.predict_batch() method."""

    def test_predict_batch_simple(self, simple_database):
        """Test batch prediction."""
        model = LinearRegressionLocalizer.fit(simple_database)

        # Multiple queries
        Z = np.array([[10.0, 20.0, -71.0], [20.0, 40.0, -72.0], [30.0, 60.0, -73.0]])

        X_hat = model.predict_batch(Z)

        # Should predict close to [[1, 2], [2, 4], [3, 6]]
        expected = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        np.testing.assert_array_almost_equal(X_hat, expected, decimal=5)

    def test_predict_batch_shape(self, simple_database):
        """Test that batch prediction returns correct shape."""
        model = LinearRegressionLocalizer.fit(simple_database)

        M = 10  # Number of queries
        Z = np.random.randn(M, 3) * 10  # Random queries

        X_hat = model.predict_batch(Z)

        assert X_hat.shape == (M, 2)

    def test_predict_batch_not_2d_error(self, simple_database):
        """Test that non-2D input raises ValueError."""
        model = LinearRegressionLocalizer.fit(simple_database)

        Z = np.array([10.0, 20.0, -71.0])  # 1D, should be 2D

        with pytest.raises(ValueError, match="Z must be 2D array"):
            model.predict_batch(Z)

    def test_predict_batch_dimension_mismatch_error(self, simple_database):
        """Test that query with wrong dimension raises ValueError."""
        model = LinearRegressionLocalizer.fit(simple_database)

        Z = np.array([[10.0, 20.0], [20.0, 40.0]])  # Only 2 features, need 3

        with pytest.raises(ValueError, match="Z has 2 features"):
            model.predict_batch(Z)


class TestScore:
    """Test suite for LinearRegressionLocalizer.score() method."""

    def test_score_perfect_fit(self, simple_database):
        """Test R² score on training data (should be high)."""
        model = LinearRegressionLocalizer.fit(simple_database)

        # Score on training data (overfitting test)
        r2 = model.score(simple_database)

        # R² should be close to 1.0 for perfect linear fit
        assert r2 >= 0.99

    def test_score_multifloor_constraint(self, multifloor_database):
        """Test R² score with floor constraint."""
        model_floor0 = LinearRegressionLocalizer.fit(multifloor_database, floor_id=0)

        # Score on floor 0 data
        r2 = model_floor0.score(multifloor_database, floor_id=0)

        # Should be able to fit floor 0 data reasonably
        assert r2 >= 0.0  # At least better than random

    def test_score_invalid_floor_error(self, simple_database):
        """Test that invalid floor_id raises ValueError."""
        model = LinearRegressionLocalizer.fit(simple_database)

        with pytest.raises(ValueError, match="Floor 5 not found"):
            model.score(simple_database, floor_id=5)

    def test_score_range(self, multifloor_database):
        """Test that R² score is in reasonable range."""
        model = LinearRegressionLocalizer.fit(multifloor_database)

        r2 = model.score(multifloor_database)

        # R² can be negative for very bad fits, but should be < 1.0
        assert r2 <= 1.0


class TestRepr:
    """Test suite for __repr__ method."""

    def test_repr_single_floor(self, simple_database):
        """Test string representation for single-floor model."""
        model = LinearRegressionLocalizer.fit(simple_database, floor_id=0)

        repr_str = repr(model)

        assert "LinearRegressionLocalizer" in repr_str
        assert "location_dim=2" in repr_str
        assert "n_features=3" in repr_str
        assert "n_training_samples=5" in repr_str
        assert "floor=0" in repr_str

    def test_repr_all_floors(self, multifloor_database):
        """Test string representation for multi-floor model."""
        model = LinearRegressionLocalizer.fit(multifloor_database)

        repr_str = repr(model)

        assert "LinearRegressionLocalizer" in repr_str
        assert "all_floors" in repr_str


class TestIntegration:
    """Integration tests comparing linear regression with other methods."""

    def test_linear_regression_vs_knn_consistency(self, simple_database):
        """Test that linear regression gives reasonable results vs k-NN."""
        from core.fingerprinting import knn_localize

        model = LinearRegressionLocalizer.fit(simple_database)

        z = np.array([15.0, 30.0, -71.5])

        # Linear regression prediction
        x_lr = model.predict(z)

        # k-NN prediction (k=3, should give interpolated result)
        x_knn = knn_localize(z, simple_database, k=3, weighting="inverse_distance")

        # Both should give reasonable estimates in same region
        # (not necessarily identical, but both valid)
        assert 0 <= x_lr[0] <= 5
        assert 0 <= x_lr[1] <= 10
        assert 0 <= x_knn[0] <= 5
        assert 0 <= x_knn[1] <= 10

    def test_linear_regression_separate_floor_models(self, multifloor_database):
        """Test training separate models per floor."""
        # Train separate models for each floor
        model_floor0 = LinearRegressionLocalizer.fit(multifloor_database, floor_id=0)
        model_floor1 = LinearRegressionLocalizer.fit(multifloor_database, floor_id=1)
        model_floor2 = LinearRegressionLocalizer.fit(multifloor_database, floor_id=2)

        # Each model should have correct training samples
        assert model_floor0.n_training_samples == 3
        assert model_floor1.n_training_samples == 3
        assert model_floor2.n_training_samples == 3  # Updated to 3 RPs

        # Query on floor 0
        z = np.array([-51, -59, -71])

        x_hat0 = model_floor0.predict(z)
        x_hat1 = model_floor1.predict(z)

        # Both predictions should be valid (different models, different results)
        assert x_hat0.shape == (2,)
        assert x_hat1.shape == (2,)

    def test_regularization_prevents_overfitting(self):
        """Test that regularization helps with small datasets."""
        # Create small noisy dataset
        np.random.seed(42)
        n_samples = 10
        n_features = 8  # More features than ideal for 10 samples

        # Generate synthetic data with noise
        X = np.random.randn(n_samples, 2) * 5
        Z = np.random.randn(n_samples, n_features) * 10

        db = FingerprintDatabase(
            locations=X,
            features=Z,
            floor_ids=np.zeros(n_samples, dtype=int),
            meta={},
        )

        # Model without regularization (prone to overfitting)
        model_noreg = LinearRegressionLocalizer.fit(db, regularization=0.0)

        # Model with regularization (should be more stable)
        model_reg = LinearRegressionLocalizer.fit(db, regularization=5.0)

        # Regularized weights should be smaller
        weights_norm_noreg = np.linalg.norm(model_noreg.weights)
        weights_norm_reg = np.linalg.norm(model_reg.weights)

        assert weights_norm_reg < weights_norm_noreg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

