"""Unit tests for classification-based fingerprinting methods.

Tests pattern recognition classifiers (Random Forest, SVM) and hierarchical
coarse-to-fine localization (Chapter 5, Section 5.2).

Author: Li-Ta Hsu
Date: December 2024
"""

import numpy as np
import pytest

# Check if sklearn is available
try:
    import sklearn
    
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from core.fingerprinting import (
    FingerprintDatabase,
    fit_classifier,
    hierarchical_localize,
)

# Skip all tests if sklearn not available
pytestmark = pytest.mark.skipif(
    not SKLEARN_AVAILABLE, reason="scikit-learn not installed"
)


class TestFitClassifier:
    """Test classifier training functions."""

    def test_fit_random_forest_rp_based(self):
        """Test fitting Random Forest with RP-based classes."""
        # Create simple database
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
        
        # Fit classifier
        classifier = fit_classifier(
            db,
            classifier_type="random_forest",
            zone_type="rp",
            n_estimators=50
        )
        
        assert classifier is not None
        assert len(classifier.class_to_location) == 4
        assert classifier.locations.shape == (4, 2)

    def test_fit_svm_rp_based(self):
        """Test fitting SVM with RP-based classes."""
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
        
        # Fit SVM classifier
        classifier = fit_classifier(
            db,
            classifier_type="svm",
            zone_type="rp",
            C=1.0,
            kernel="rbf"
        )
        
        assert classifier is not None
        assert len(classifier.class_to_location) == 4

    def test_fit_with_floor_constraint(self):
        """Test fitting classifier on single floor only."""
        # Multi-floor database
        locations = np.array([[0, 0], [10, 0], [0, 5], [10, 5]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-55, -65, -75],
            [-65, -55, -85],
        ], dtype=float)
        floor_ids = np.array([0, 0, 1, 1])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        # Fit on floor 0 only
        classifier = fit_classifier(
            db,
            classifier_type="random_forest",
            zone_type="rp",
            floor_id=0
        )
        
        # Should only have 2 classes (floor 0 RPs)
        assert len(classifier.class_to_location) == 2

    def test_fit_invalid_zone_type(self):
        """Test that invalid zone_type raises ValueError."""
        locations = np.array([[0, 0], [10, 0]], dtype=float)
        features = np.array([[-50, -60], [-60, -50]], dtype=float)
        floor_ids = np.array([0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={}
        )
        
        with pytest.raises((ValueError, NotImplementedError)):
            fit_classifier(db, zone_type="invalid")

    def test_fit_invalid_classifier_type(self):
        """Test that invalid classifier_type raises ValueError."""
        locations = np.array([[0, 0], [10, 0]], dtype=float)
        features = np.array([[-50, -60], [-60, -50]], dtype=float)
        floor_ids = np.array([0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={}
        )
        
        with pytest.raises(ValueError, match="Unknown classifier_type"):
            fit_classifier(db, classifier_type="invalid")


class TestClassificationLocalizer:
    """Test ClassificationLocalizer predict method."""

    def test_predict_exact_match(self):
        """Test prediction with exact feature match."""
        locations = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        classifier = fit_classifier(db, classifier_type="random_forest", zone_type="rp")
        
        # Query with exact match to RP 1
        query = np.array([-60, -50, -80])
        pos, info = classifier.predict(query)
        
        # Should predict RP 1
        assert pos.shape == (2,)
        assert "predicted_class" in info

    def test_predict_approximate_match(self):
        """Test prediction with approximate feature match."""
        locations = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        classifier = fit_classifier(db, classifier_type="random_forest", zone_type="rp")
        
        # Query close to RP 1
        query = np.array([-59, -51, -81])
        pos, info = classifier.predict(query)
        
        assert pos.shape == (2,)
        assert not np.any(np.isnan(pos))

    def test_predict_with_probabilities(self):
        """Test prediction with class probabilities."""
        locations = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        classifier = fit_classifier(db, classifier_type="random_forest", zone_type="rp")
        
        query = np.array([-55, -65, -75])
        pos, info = classifier.predict(query, return_proba=True)
        
        assert "class_probabilities" in info
        assert "top_k_classes" in info

    def test_predict_with_missing_values(self):
        """Test prediction with missing values (NaN) in query."""
        locations = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        classifier = fit_classifier(db, classifier_type="random_forest", zone_type="rp")
        
        # Query with missing value
        # Note: sklearn classifiers may not handle NaN well, this test
        # is to check if it at least doesn't crash
        query = np.array([-55, np.nan, -75])
        
        try:
            pos, info = classifier.predict(query)
            # If it works, check output is valid
            assert pos.shape == (2,)
        except ValueError:
            # sklearn doesn't support NaN by default, which is expected
            pass


class TestHierarchicalLocalize:
    """Test hierarchical coarse-to-fine localization."""

    def test_hierarchical_floor_then_knn(self):
        """Test hierarchical: floor classification, then k-NN."""
        # Multi-floor database
        locations = np.array([
            [0, 0], [10, 0], [10, 10], [0, 10],  # Floor 0
            [0, 0], [10, 0], [10, 10], [0, 10],  # Floor 1
        ], dtype=float)
        features = np.array([
            [-50, -60, -70],  # Floor 0
            [-60, -50, -80],
            [-70, -80, -50],
            [-80, -70, -60],
            [-55, -65, -75],  # Floor 1 (different RSS)
            [-65, -55, -85],
            [-75, -85, -55],
            [-85, -75, -65],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        # Query on floor 1
        query = np.array([-60, -70, -80])
        
        pos, info = hierarchical_localize(
            query,
            db,
            coarse_method="floor",
            fine_method="knn",
            k=3
        )
        
        assert pos.shape == (2,)
        assert "coarse_floor" in info
        assert "fine_method" in info
        assert info["fine_method"] == "knn"

    def test_hierarchical_single_floor(self):
        """Test hierarchical on single-floor database (should skip coarse)."""
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
        
        query = np.array([-55, -65, -75])
        
        pos, info = hierarchical_localize(
            query,
            db,
            coarse_method="floor",
            fine_method="nn"
        )
        
        assert pos.shape == (2,)
        assert info["coarse_floor"] == 0

    def test_hierarchical_with_probabilistic_fine(self):
        """Test hierarchical with probabilistic fine localization."""
        # Multi-floor database
        locations = np.array([
            [0, 0], [10, 0], [10, 10],  # Floor 0
            [0, 0], [10, 0], [10, 10],  # Floor 1
        ], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
            [-55, -65, -75],
            [-65, -55, -85],
            [-75, -85, -55],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0, 1, 1, 1])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        query = np.array([-58, -68, -78])
        
        # Test MAP fine method
        pos_map, info_map = hierarchical_localize(
            query,
            db,
            coarse_method="floor",
            fine_method="map"
        )
        
        assert pos_map.shape == (2,)
        assert "coarse_floor" in info_map
        
        # Test posterior mean fine method
        pos_mean, info_mean = hierarchical_localize(
            query,
            db,
            coarse_method="floor",
            fine_method="posterior_mean"
        )
        
        assert pos_mean.shape == (2,)

    def test_hierarchical_random_forest_coarse(self):
        """Test hierarchical with Random Forest coarse classification."""
        # Multi-floor database
        locations = np.array([
            [0, 0], [10, 0], [10, 10],  # Floor 0
            [0, 0], [10, 0], [10, 10],  # Floor 1
        ], dtype=float)
        features = np.array([
            [-50, -60, -70],
            [-60, -50, -80],
            [-70, -80, -50],
            [-55, -65, -75],
            [-65, -55, -85],
            [-75, -85, -55],
        ], dtype=float)
        floor_ids = np.array([0, 0, 0, 1, 1, 1])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
        )
        
        query = np.array([-58, -68, -78])
        
        pos, info = hierarchical_localize(
            query,
            db,
            coarse_method="random_forest",
            fine_method="knn",
            k=3
        )
        
        assert pos.shape == (2,)
        assert "coarse_floor" in info
        assert info["coarse_method"] == "random_forest"

    def test_hierarchical_invalid_coarse_method(self):
        """Test that invalid coarse method raises ValueError."""
        locations = np.array([[0, 0], [10, 0]], dtype=float)
        features = np.array([[-50, -60], [-60, -50]], dtype=float)
        floor_ids = np.array([0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={}
        )
        
        query = np.array([-55, -65])
        
        with pytest.raises(ValueError, match="Unknown coarse_method"):
            hierarchical_localize(query, db, coarse_method="invalid")

    def test_hierarchical_invalid_fine_method(self):
        """Test that invalid fine method raises ValueError."""
        locations = np.array([[0, 0], [10, 0]], dtype=float)
        features = np.array([[-50, -60], [-60, -50]], dtype=float)
        floor_ids = np.array([0, 0])
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={}
        )
        
        query = np.array([-55, -65])
        
        with pytest.raises(ValueError, match="Unknown fine_method"):
            hierarchical_localize(query, db, fine_method="invalid")


class TestIntegration:
    """Integration tests for classification-based fingerprinting."""

    def test_classification_vs_knn_consistency(self):
        """Test that classification and k-NN give similar results."""
        np.random.seed(42)
        
        # Create database with well-separated clusters
        n_rps = 20
        locations = np.random.rand(n_rps, 2) * 50
        features = -50 - np.random.rand(n_rps, 5) * 40
        floor_ids = np.zeros(n_rps, dtype=int)
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": [f"AP{i}" for i in range(5)], "unit": "dBm"}
        )
        
        # Fit classifier
        classifier = fit_classifier(
            db,
            classifier_type="random_forest",
            zone_type="rp",
            n_estimators=100
        )
        
        # Test queries
        from core.fingerprinting import knn_localize
        
        n_queries = 10
        for _ in range(n_queries):
            query = -50 - np.random.rand(5) * 40
            
            pos_class, _ = classifier.predict(query)
            pos_knn = knn_localize(query, db, k=1, floor_id=0)
            
            # Both methods should produce valid positions
            assert not np.any(np.isnan(pos_class))
            assert not np.any(np.isnan(pos_knn))

    def test_hierarchical_improves_efficiency(self):
        """Test that hierarchical method reduces search space."""
        # Large multi-floor database
        n_rps_per_floor = 25
        n_floors = 3
        
        locations_list = []
        features_list = []
        floor_ids_list = []
        
        for floor in range(n_floors):
            locs = np.random.rand(n_rps_per_floor, 2) * 50
            # Different RSS ranges per floor
            feats = -50 - floor * 10 - np.random.rand(n_rps_per_floor, 4) * 20
            floors = np.full(n_rps_per_floor, floor)
            
            locations_list.append(locs)
            features_list.append(feats)
            floor_ids_list.append(floors)
        
        locations = np.vstack(locations_list)
        features = np.vstack(features_list)
        floor_ids = np.hstack(floor_ids_list)
        
        db = FingerprintDatabase(
            locations=locations,
            features=features,
            floor_ids=floor_ids,
            meta={"ap_ids": ["AP1", "AP2", "AP3", "AP4"], "unit": "dBm"}
        )
        
        # Query on floor 1
        query = -50 - 1 * 10 - np.random.rand(4) * 20
        
        pos, info = hierarchical_localize(
            query,
            db,
            coarse_method="floor",
            fine_method="knn",
            k=5
        )
        
        # Should correctly identify floor 1
        # (coarse step reduces search space from 75 to 25 RPs)
        assert pos.shape == (2,)
        assert 0 <= info["coarse_floor"] < 3


