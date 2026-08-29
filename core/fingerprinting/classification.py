"""Classification-based fingerprinting methods for Chapter 5.

This module implements pattern recognition classifiers as described in Section 5.2
of the book. Unlike deterministic (NN, k-NN) or probabilistic (Bayesian) methods,
classification-based fingerprinting treats positioning as a classification problem:
assign query fingerprints to discrete reference-point classes.

Key approaches:
    - Direct classification: Each reference point (RP) is a class
    - Zone-based classification: Not implemented; callers should use RP classes
    - Hierarchical classification: Coarse (floor/region) → Fine (RP-level)

Classifiers mentioned in the book:
    - Decision Trees / Random Forests
    - Support Vector Machines (SVM)
    - Neural Networks (not implemented here)

Author: Li-Ta Hsu
Date: December 2024
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    SVC = None
    LabelEncoder = None

from .types import Fingerprint, FingerprintDatabase, Location


class HierarchicalLocalizationResult(NamedTuple):
    """Tuple-compatible result from hierarchical fingerprint localization.

    Attributes:
        estimated_position_m: Estimated location, shape ``(d,)``. Units and
            coordinate frame match ``FingerprintDatabase.locations``; the
            database may use local x-y meters, ENU meters, or another documented
            survey frame.
        diagnostics: Diagnostic dictionary with coarse and fine localization
            details.

    Existing callers can still unpack this as ``position, info`` or access the
    compatibility properties ``.position`` and ``.info``.
    """

    estimated_position_m: Location
    diagnostics: dict

    @property
    def position(self) -> Location:
        """Compatibility alias for ``estimated_position_m``."""
        return self.estimated_position_m

    @property
    def info(self) -> dict:
        """Compatibility alias for ``diagnostics``."""
        return self.diagnostics


@dataclass
class ClassificationLocalizer:
    """
    Classification-based fingerprinting localizer.

    This localizer treats positioning as a classification problem, as discussed
    in Chapter 5, Section 5.2. The classifier predicts a discrete location class
    (currently one class per RP) from the fingerprint features.

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
        floor_id: int | None = None,
        return_proba: bool = False,
    ) -> tuple[Location, dict]:
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
            info["class_probabilities"] = dict(zip(classes, probas, strict=True))
            info["top_k_classes"] = classes[np.argsort(probas)[::-1][:5]]

        return predicted_location, info


def fit_classifier(
    db: FingerprintDatabase,
    classifier_type: Literal["random_forest", "svm"] = "random_forest",
    zone_type: Literal["rp"] = "rp",
    floor_id: int | None = None,
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
                  - "rp": Each RP is a separate class (implemented)
        floor_id: Optional floor to train on (None = all floors)
        **classifier_kwargs: Additional arguments for the classifier
                            (e.g., n_estimators for RandomForest)

    Returns:
        Trained ClassificationLocalizer

    Raises:
        ImportError: If scikit-learn is not installed
        ValueError: If zone_type is invalid or insufficient data
        NotImplementedError: If an old experimental zone_type such as "grid"
            or "cluster" is requested. Those modes are not part of the
            supported public API yet.

    Examples:
        >>> # Direct classification (each RP is a class)
        >>> classifier = fit_classifier(
        ...     db,
        ...     classifier_type="random_forest",
        ...     zone_type="rp",
        ...     n_estimators=100
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
        raise NotImplementedError(
            "zone_type='grid' is an experimental mode that is not implemented. "
            "Use zone_type='rp', the only supported public option."
        )

    elif zone_type == "cluster":
        raise NotImplementedError(
            "zone_type='cluster' is an experimental mode that is not implemented. "
            "Use zone_type='rp', the only supported public option."
        )

    else:
        raise ValueError(
            f"Unknown zone_type '{zone_type}'. Use 'rp', the only supported option."
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


def fit_floor_classifier(
    db: FingerprintDatabase,
    n_estimators: int = 50,
    random_state: int = 42,
) -> "RandomForestClassifier":
    """Train a Random Forest classifier for floor detection (offline).

    This factory produces a model that can be passed to
    ``hierarchical_localize(coarse_model=...)`` so that inference never
    re-trains the classifier.

    Args:
        db: Fingerprint database with multi-floor data.
        n_estimators: Number of trees in the forest.
        random_state: Seed for reproducibility.

    Returns:
        Fitted ``RandomForestClassifier`` ready for ``predict()``.

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If the database contains only one floor.

    References:
        Chapter 5, Section 5.2: Classification-based coarse localisation.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for floor classification. "
            "Install it with: pip install scikit-learn"
        )
    if db.n_floors <= 1:
        raise ValueError(
            "fit_floor_classifier requires a multi-floor database "
            f"(got {db.n_floors} floor(s))."
        )

    features = db.get_mean_features()
    floor_labels = db.floor_ids

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    clf.fit(features, floor_labels)
    return clf


def hierarchical_localize(
    z: Fingerprint,
    db: FingerprintDatabase,
    coarse_method: Literal["floor", "random_forest"] = "floor",
    fine_method: Literal["nn", "knn", "map", "posterior_mean"] = "knn",
    coarse_model: object | None = None,
    **fine_method_kwargs,
) -> HierarchicalLocalizationResult:
    """Hierarchical coarse-to-fine localization.

    Implements the two-step approach from Chapter 5: first classify into a
    coarse region (floor / zone), then run fine-grained localization within
    that region.

    Args:
        z: Query fingerprint, shape (N,).
        db: FingerprintDatabase.
        coarse_method: Coarse classification strategy:
            - ``"floor"``: nearest-neighbour floor classification.
            - ``"random_forest"``: Random Forest floor classifier.
        fine_method: Fine localization method (``"nn"``, ``"knn"``,
            ``"map"``, ``"posterior_mean"``).
        coarse_model: Optional pre-trained coarse classifier (e.g. the
            return value of :func:`fit_floor_classifier`).  When provided
            the classifier is used directly instead of being re-trained on
            every call.
        **fine_method_kwargs: Forwarded to the fine localization function.

    Returns:
        HierarchicalLocalizationResult with named ``estimated_position_m`` and
        ``diagnostics`` fields. Existing callers can still unpack it as
        ``predicted_location, info_dict`` or use compatibility properties
        ``.position`` and ``.info``. Position units and frame match
        ``db.locations``.

    Examples:
        >>> # Offline: train once
        >>> clf = fit_floor_classifier(db)
        >>> # Online: re-use the trained model
        >>> pos, info = hierarchical_localize(
        ...     query, db,
        ...     coarse_method="random_forest",
        ...     coarse_model=clf,
        ...     fine_method="knn", k=5,
        ... )

    References:
        Chapter 5, Section 5.2: Hierarchical classification approach.
    """
    info = {"coarse_method": coarse_method, "fine_method": fine_method}

    # Step 1: Coarse classification
    if coarse_method == "floor":
        if db.n_floors == 1:
            floor_id = db.floor_ids[0]
            info["coarse_floor"] = int(floor_id)
        else:
            # Match in *fingerprint* space and read the floor off the winning
            # RP. This used to call nearest_neighbor_localize, which returns an (x, y)
            # location with the floor already discarded, then pick the RP
            # nearest that location in x-y. A multi-floor grid stacks its RPs
            # at the same coordinates -- the shipped ch5 database has 363 RPs
            # on 121 distinct (x, y) -- so the argmin was a tie between one RP
            # per floor and always resolved to the lowest index. It returned
            # floor 0 for every query, scoring that floor's base rate (~33% on
            # three floors) and reading as a hard problem rather than a bug.
            distances = np.linalg.norm(db.get_mean_features() - z, axis=1)
            closest_rp_idx = int(np.argmin(distances))
            floor_id = db.floor_ids[closest_rp_idx]
            info["coarse_floor"] = int(floor_id)

    elif coarse_method == "random_forest":
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for random_forest coarse method")

        if db.n_floors > 1:
            if coarse_model is not None:
                clf = coarse_model
            else:
                clf = fit_floor_classifier(db)
            floor_id = clf.predict(z.reshape(1, -1))[0]
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
        from .deterministic import nearest_neighbor_localize

        pos = nearest_neighbor_localize(z, db, floor_id=floor_id, **fine_method_kwargs)

    elif fine_method == "knn":
        from .deterministic import k_nearest_neighbor_localize

        # Default k=5 if not provided
        if "k" not in fine_method_kwargs:
            fine_method_kwargs["k"] = 5

        pos = k_nearest_neighbor_localize(
            z, db, floor_id=floor_id, **fine_method_kwargs
        )

    elif fine_method == "map":
        from .probabilistic import (
            fit_gaussian_naive_bayes,
            maximum_a_posteriori_localize,
        )

        # Fit model on the coarse floor
        model = fit_gaussian_naive_bayes(db)
        pos = maximum_a_posteriori_localize(z, model, floor_id=floor_id)

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

    return HierarchicalLocalizationResult(estimated_position_m=pos, diagnostics=info)


def hierarchical_fingerprint_localize(
    z: Fingerprint,
    db: FingerprintDatabase,
    coarse_method: Literal["floor", "random_forest"] = "floor",
    fine_method: Literal["nn", "knn", "map", "posterior_mean"] = "knn",
    coarse_model: object | None = None,
    **fine_method_kwargs,
) -> HierarchicalLocalizationResult:
    """Descriptive alias for :func:`hierarchical_localize`."""
    return hierarchical_localize(
        z,
        db,
        coarse_method=coarse_method,
        fine_method=fine_method,
        coarse_model=coarse_model,
        **fine_method_kwargs,
    )
