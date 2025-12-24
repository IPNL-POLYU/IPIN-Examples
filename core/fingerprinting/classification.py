"""Classification-based fingerprinting methods for Chapter 5.

This module implements pattern recognition classifiers as described in Section 5.2
of the book. Unlike deterministic (NN, k-NN) or probabilistic (Bayesian) methods,
classification-based fingerprinting treats positioning as a classification problem:
assign query fingerprints to discrete location classes or zones.

Key approaches:
    - Direct classification: Each reference point (RP) is a class
    - Zone-based classification: Group RPs into zones/regions
    - Hierarchical classification: Coarse (floor/region) → Fine (RP-level)

Classifiers mentioned in the book:
    - Decision Trees / Random Forests
    - Support Vector Machines (SVM)
    - Neural Networks (not implemented here)

Author: Li-Ta Hsu
Date: December 2024
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    SVC = None
    LabelEncoder = None

from .types import Fingerprint, FingerprintDatabase, Location


@dataclass
class ClassificationLocalizer:
    """
    Classification-based fingerprinting localizer.
    
    This localizer treats positioning as a classification problem, as discussed
    in Chapter 5, Section 5.2. The classifier predicts a discrete location class
    (RP index, zone ID, or grid cell) from the fingerprint features.
    
    Attributes:
        classifier: Trained scikit-learn classifier (RandomForest, SVM, etc.)
        locations: Reference point locations, shape (M, d)
        class_to_location: Mapping from class label to location
        floor_ids: Floor identifiers for each class
        label_encoder: Encoder for class labels
        meta: Metadata dictionary
    
    References:
        Chapter 5, Section 5.2: Pattern Recognition Approaches
    """
    
    classifier: object  # scikit-learn classifier
    locations: np.ndarray
    class_to_location: dict  # Maps class_id -> location
    floor_ids: np.ndarray
    label_encoder: object  # LabelEncoder
    meta: dict
    
    def predict(
        self,
        z: Fingerprint,
        floor_id: Optional[int] = None,
        return_proba: bool = False,
    ) -> Tuple[Location, dict]:
        """
        Predict location using classification.
        
        Args:
            z: Query fingerprint, shape (N,)
            floor_id: Optional floor constraint. If provided, only considers
                     classes on that floor.
            return_proba: If True, returns class probabilities in info dict.
        
        Returns:
            Tuple of (predicted_location, info_dict)
        """
        # Reshape for sklearn (expects 2D input)
        z_2d = z.reshape(1, -1)
        
        # Predict class
        predicted_class = self.classifier.predict(z_2d)[0]
        
        # Get location for predicted class
        predicted_location = self.class_to_location[predicted_class]
        
        # Build info dict
        info = {"predicted_class": predicted_class}
        
        if return_proba and hasattr(self.classifier, "predict_proba"):
            probas = self.classifier.predict_proba(z_2d)[0]
            classes = self.classifier.classes_
            info["class_probabilities"] = dict(zip(classes, probas))
            info["top_k_classes"] = classes[np.argsort(probas)[::-1][:5]]
        
        return predicted_location, info


def fit_classifier(
    db: FingerprintDatabase,
    classifier_type: Literal["random_forest", "svm"] = "random_forest",
    zone_type: Literal["rp", "grid", "cluster"] = "rp",
    floor_id: Optional[int] = None,
    **classifier_kwargs,
) -> ClassificationLocalizer:
    """
    Fit a classification-based localizer from fingerprint database.
    
    This function implements the classification approach discussed in Chapter 5,
    Section 5.2, where positioning is framed as a pattern recognition problem.
    
    Args:
        db: FingerprintDatabase containing training data
        classifier_type: Type of classifier:
                        - "random_forest": Random Forest (book default)
                        - "svm": Support Vector Machine
        zone_type: How to define location classes:
                  - "rp": Each RP is a separate class (fine-grained)
                  - "grid": Grid-based zones (coarse-grained)
                  - "cluster": Cluster-based zones (adaptive)
        floor_id: Optional floor to train on (None = all floors)
        **classifier_kwargs: Additional arguments for the classifier
                            (e.g., n_estimators for RandomForest)
    
    Returns:
        Trained ClassificationLocalizer
    
    Raises:
        ImportError: If scikit-learn is not installed
        ValueError: If zone_type is invalid or insufficient data
    
    Examples:
        >>> # Direct classification (each RP is a class)
        >>> classifier = fit_classifier(
        ...     db,
        ...     classifier_type="random_forest",
        ...     zone_type="rp",
        ...     n_estimators=100
        ... )
        
        >>> # Zone-based classification
        >>> classifier = fit_classifier(
        ...     db,
        ...     classifier_type="svm",
        ...     zone_type="grid",
        ...     floor_id=0
        ... )
    
    References:
        Chapter 5, Section 5.2: Pattern Recognition for fingerprinting
        mentions Random Forests, Decision Trees, and SVM classifiers.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for classification-based fingerprinting. "
            "Install it with: pip install scikit-learn"
        )
    
    # Filter by floor if specified
    if floor_id is not None:
        mask = db.get_floor_mask(floor_id)
        locations = db.locations[mask]
        features = db.get_mean_features()[mask]
        floor_ids = db.floor_ids[mask]
    else:
        locations = db.locations
        features = db.get_mean_features()
        floor_ids = db.floor_ids
    
    # Create class labels based on zone_type
    if zone_type == "rp":
        # Each RP is a separate class
        class_labels = np.arange(len(locations))
        class_to_location = {i: locations[i] for i in range(len(locations))}
    
    elif zone_type == "grid":
        # Grid-based zones (not implemented yet, would require grid parameters)
        raise NotImplementedError(
            "Grid-based zone classification not yet implemented. Use 'rp' for now."
        )
    
    elif zone_type == "cluster":
        # Cluster-based zones (not implemented yet, would require clustering)
        raise NotImplementedError(
            "Cluster-based zone classification not yet implemented. Use 'rp' for now."
        )
    
    else:
        raise ValueError(
            f"Unknown zone_type '{zone_type}'. Use 'rp', 'grid', or 'cluster'."
        )
    
    # Create and fit classifier
    if classifier_type == "random_forest":
        # Random Forest (book default)
        default_rf_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
        }
        default_rf_params.update(classifier_kwargs)
        classifier = RandomForestClassifier(**default_rf_params)
    
    elif classifier_type == "svm":
        # Support Vector Machine
        default_svm_params = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "random_state": 42,
        }
        default_svm_params.update(classifier_kwargs)
        classifier = SVC(**default_svm_params, probability=True)
    
    else:
        raise ValueError(
            f"Unknown classifier_type '{classifier_type}'. "
            f"Use 'random_forest' or 'svm'."
        )
    
    # Fit classifier
    classifier.fit(features, class_labels)
    
    # Create label encoder for interpretability
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)
    
    return ClassificationLocalizer(
        classifier=classifier,
        locations=locations,
        class_to_location=class_to_location,
        floor_ids=floor_ids,
        label_encoder=label_encoder,
        meta=db.meta.copy(),
    )


def hierarchical_localize(
    z: Fingerprint,
    db: FingerprintDatabase,
    coarse_method: Literal["floor", "random_forest"] = "floor",
    fine_method: Literal["nn", "knn", "map", "posterior_mean"] = "knn",
    **fine_method_kwargs,
) -> Tuple[Location, dict]:
    """
    Hierarchical coarse-to-fine localization.
    
    This function implements the two-step hierarchical approach discussed in
    Chapter 5: first classify into coarse regions (floor, zone), then perform
    fine-grained localization within that region.
    
    Args:
        z: Query fingerprint, shape (N,)
        db: FingerprintDatabase
        coarse_method: Coarse classification method:
                      - "floor": Classify floor first (if multi-floor)
                      - "random_forest": Use RF classifier for coarse regions
        fine_method: Fine localization method:
                    - "nn": Nearest neighbor
                    - "knn": k-Nearest neighbors
                    - "map": Maximum a posteriori (Bayesian)
                    - "posterior_mean": Posterior mean (Bayesian)
        **fine_method_kwargs: Arguments for fine localization method
    
    Returns:
        Tuple of (predicted_location, info_dict)
    
    Examples:
        >>> # Two-step: classify floor, then k-NN
        >>> pos, info = hierarchical_localize(
        ...     query,
        ...     db,
        ...     coarse_method="floor",
        ...     fine_method="knn",
        ...     k=5
        ... )
        >>> print(f"Coarse: Floor {info['coarse_floor']}, Fine: {pos}")
    
    References:
        Chapter 5, Section 5.2: Discusses hierarchical classification approach
        for improved accuracy and efficiency.
    """
    info = {"coarse_method": coarse_method, "fine_method": fine_method}
    
    # Step 1: Coarse classification
    if coarse_method == "floor":
        # Floor classification (simple approach for multi-floor)
        if db.n_floors == 1:
            # Single floor, skip coarse step
            floor_id = db.floor_ids[0]
            info["coarse_floor"] = int(floor_id)
        else:
            # Classify floor using simple nearest neighbor
            from .deterministic import nn_localize
            
            # NN without floor constraint to get initial floor estimate
            pos_nn = nn_localize(z, db, floor_id=None)
            
            # Find which floor this position belongs to
            # (Find closest RP and use its floor)
            distances = np.linalg.norm(db.locations - pos_nn, axis=1)
            closest_rp_idx = np.argmin(distances)
            floor_id = db.floor_ids[closest_rp_idx]
            info["coarse_floor"] = int(floor_id)
    
    elif coarse_method == "random_forest":
        # Use Random Forest for coarse region classification
        # For simplicity, classify floor first
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for random_forest coarse method")
        
        # Train floor classifier if multi-floor
        if db.n_floors > 1:
            features = db.get_mean_features()
            floor_labels = db.floor_ids
            
            rf_floor = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_floor.fit(features, floor_labels)
            
            floor_id = rf_floor.predict(z.reshape(1, -1))[0]
            info["coarse_floor"] = int(floor_id)
        else:
            floor_id = db.floor_ids[0]
            info["coarse_floor"] = int(floor_id)
    
    else:
        raise ValueError(
            f"Unknown coarse_method '{coarse_method}'. Use 'floor' or 'random_forest'."
        )
    
    # Step 2: Fine localization within coarse region
    if fine_method == "nn":
        from .deterministic import nn_localize
        
        pos = nn_localize(z, db, floor_id=floor_id, **fine_method_kwargs)
    
    elif fine_method == "knn":
        from .deterministic import knn_localize
        
        # Default k=5 if not provided
        if "k" not in fine_method_kwargs:
            fine_method_kwargs["k"] = 5
        
        pos = knn_localize(z, db, floor_id=floor_id, **fine_method_kwargs)
    
    elif fine_method == "map":
        from .probabilistic import fit_gaussian_naive_bayes, map_localize
        
        # Fit model on the coarse floor
        model = fit_gaussian_naive_bayes(db)
        pos = map_localize(z, model, floor_id=floor_id)
    
    elif fine_method == "posterior_mean":
        from .probabilistic import fit_gaussian_naive_bayes, posterior_mean_localize
        
        # Fit model on the coarse floor
        model = fit_gaussian_naive_bayes(db)
        pos = posterior_mean_localize(z, model, floor_id=floor_id, **fine_method_kwargs)
    
    else:
        raise ValueError(
            f"Unknown fine_method '{fine_method}'. "
            f"Use 'nn', 'knn', 'map', or 'posterior_mean'."
        )
    
    info["fine_position"] = pos
    
    return pos, info


