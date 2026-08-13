"""`hierarchical_localize(coarse_method="floor")` returned floor 0 for everything.

The coarse step called `nn_localize`, which returns an ``(x, y)`` location with
the floor already discarded, then looked for the RP nearest that location *in
x-y* and read its floor. A multi-floor survey stacks its RPs at the same
coordinates -- the shipped ch5 grid has 363 RPs on 121 distinct ``(x, y)``,
three per position -- so the argmin was a three-way tie that always resolved to
the lowest index, which is floor 0.

Nothing failed. The function returned a valid floor, downstream localization ran
normally, and the measured accuracy came out at floor 0's base rate: 32.7% on a
three-floor database, right at the 33.3% chance level. `example_classification`
printed that as "Floor classification accuracy: 29.0%" and read as a hard
problem rather than a broken one.

`test_coarse_floor_is_not_constant` is the test that would have caught it, and
it is the cheap shape worth reaching for whenever a classifier underperforms:
before tuning anything, check that its output varies at all. The accuracy test
alone is weaker than it looks -- 33% on three balanced classes is exactly what a
constant predictor scores, so a threshold set anywhere near chance would have
passed the bug.

Author: Li-Ta Hsu
References: Chapter 5, Section 5.2
"""

import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from core.fingerprinting import (
    fit_classifier,
    hierarchical_localize,
    load_fingerprint_database,
)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sim" / "ch5_wifi_fingerprint_grid"
QUERY_NOISE_DBM = 3.0
N_QUERIES = 120


def _load():
    return load_fingerprint_database(DB_PATH)


def _queries(db, n=N_QUERIES, noise=QUERY_NOISE_DBM, seed=42):
    """Noisy queries drawn from random RPs, with their true floors."""
    feats = db.get_mean_features()
    floors = np.asarray(db.floor_ids)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = int(rng.integers(0, db.n_reference_points))
        out.append((feats[idx] + rng.standard_normal(db.n_features) * noise,
                    int(floors[idx])))
    return out


class TestTheDatabaseHasTheStructureThatBrokeIt(unittest.TestCase):
    """Pins the precondition, so the bug's cause stays legible."""

    def test_reference_points_are_stacked_across_floors(self):
        db = _load()
        locations = np.asarray(db.locations)
        unique_xy = np.unique(locations, axis=0)

        self.assertLess(len(unique_xy), len(locations))
        self.assertEqual(len(locations), len(unique_xy) * db.n_floors)

    def test_floors_are_separable_in_fingerprint_space(self):
        """If this fails, a floor-accuracy failure is the data, not the code."""
        db = _load()
        feats = db.get_mean_features()
        floors = np.asarray(db.floor_ids)

        means = np.array([feats[floors == f].mean() for f in sorted(set(floors))])
        within = float(np.mean([feats[floors == f].std() for f in sorted(set(floors))]))
        separation = float(np.abs(np.diff(means)).mean())

        self.assertGreater(separation / within, 1.5)


class TestCoarseFloorClassification(unittest.TestCase):
    """The floor step has to actually use the fingerprint."""

    def test_coarse_floor_is_not_constant(self):
        """The discriminating test: the old code answered 0 every time.

        Deliberately makes no accuracy claim. A constant predictor scores the
        majority class's base rate, which on three balanced floors is ~33% and
        survives any threshold set near chance -- so 'is it varying at all'
        catches this failure where 'is it accurate enough' can miss it.
        """
        db = _load()
        predicted = {
            hierarchical_localize(z, db, coarse_method="floor", fine_method="knn", k=5)[1]["coarse_floor"]
            for z, _ in _queries(db)
        }

        self.assertEqual(predicted, set(int(f) for f in np.unique(db.floor_ids)))

    def test_coarse_floor_is_accurate_well_above_chance(self):
        db = _load()
        hits = [
            hierarchical_localize(z, db, coarse_method="floor", fine_method="knn", k=5)[1]["coarse_floor"] == floor
            for z, floor in _queries(db)
        ]

        self.assertGreater(float(np.mean(hits)), 0.90)

    def test_coarse_floor_equals_the_nearest_fingerprint_floor(self):
        """Pins the definition, not just the score."""
        db = _load()
        feats = db.get_mean_features()
        floors = np.asarray(db.floor_ids)

        for z, _ in _queries(db, n=25):
            expected = int(floors[int(np.argmin(np.linalg.norm(feats - z, axis=1)))])
            _, info = hierarchical_localize(
                z, db, coarse_method="floor", fine_method="knn", k=5)

            self.assertEqual(info["coarse_floor"], expected)


class TestExactHitsAreAnArtefactOfTheQueryDesign(unittest.TestCase):
    """Why the example prints an exact-hit column next to its RMSE.

    Queries are made by perturbing an RP's fingerprint, so the true position is
    always exactly on a reference point. A method that returns a single RP can
    score a true zero; one that interpolates between neighbours cannot, however
    close it lands. Comparing the two on RMSE alone credits MAP for a property
    of the evaluation. These tests pin the asymmetry so the caveat the example
    prints stays true, or fails loudly when it stops being.
    """

    def test_map_scores_exact_zeros_and_knn_never_does(self):
        db = _load()
        locations = np.asarray(db.locations)
        feats = db.get_mean_features()
        rng = np.random.default_rng(42)

        map_zero = knn_zero = 0
        trials = 60
        for _ in range(trials):
            idx = int(rng.integers(0, db.n_reference_points))
            z = feats[idx] + rng.standard_normal(db.n_features) * QUERY_NOISE_DBM
            truth = locations[idx]
            pos_map, _ = hierarchical_localize(
                z, db, coarse_method="floor", fine_method="map")
            pos_knn, _ = hierarchical_localize(
                z, db, coarse_method="floor", fine_method="knn", k=5)
            map_zero += np.linalg.norm(pos_map - truth) < 1e-9
            knn_zero += np.linalg.norm(pos_knn - truth) < 1e-9

        self.assertGreater(map_zero / trials, 0.20)
        self.assertEqual(knn_zero, 0)


class TestTrainingRecallIsNotAccuracy(unittest.TestCase):
    """Test 1 of the example reported the first as if it were the second."""

    @classmethod
    def setUpClass(cls):
        cls.db = _load()
        cls.feats = cls.db.get_mean_features()
        cls.rf = fit_classifier(
            cls.db, classifier_type="random_forest", zone_type="rp", n_estimators=100)

    def test_recall_on_training_vectors_is_perfect_by_construction(self):
        """One sample per class: fitting them is guaranteed, not informative."""
        hits = [
            np.allclose(self.rf.predict(self.feats[i])[0], self.db.locations[i], atol=0.1)
            for i in range(self.db.n_reference_points)
        ]

        self.assertEqual(float(np.mean(hits)), 1.0)

    def test_held_out_accuracy_is_strictly_lower(self):
        """The measurement that can distinguish a model from a lookup table."""
        rng = np.random.default_rng(7)
        hits = []
        for _ in range(150):
            idx = int(rng.integers(0, self.db.n_reference_points))
            z = self.feats[idx] + rng.standard_normal(self.db.n_features) * 2.0
            hits.append(
                np.allclose(self.rf.predict(z)[0], self.db.locations[idx], atol=0.1))

        self.assertLess(float(np.mean(hits)), 1.0)
        self.assertGreater(float(np.mean(hits)), 0.5)


if __name__ == "__main__":
    unittest.main()
