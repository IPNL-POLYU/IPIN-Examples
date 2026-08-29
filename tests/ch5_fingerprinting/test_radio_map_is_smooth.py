"""The radio map must be a smooth function of position, because that is the
only property fingerprinting exploits.

This file replaces ``test_query_shadowing_dominates.py``, which pinned the
defect rather than the fix. That file asserted the gap between nearest-neighbour
accuracy and the grid's own quantisation floor **stays open** -- deliberately,
in the pattern this repo uses for a known defect nobody is fixing yet ("written
to fail the moment it is repaired"). It has now been repaired, so the assertion
is inverted here rather than deleted.

What was wrong: the generator drew shadow fading independently for every
(RP, AP, sample), so the 4 dB that should describe *where the walls are* was
scrambling the map instead. Measured on the shipped grid database, 200 queries
on floor 0:

    clean map (no shadowing) + noiseless query   2.27 m   <- 5 m grid floor
    shipped map              + noiseless query   6.93 m   before
                                                 3.39 m   after

**A warning about the helper this file replaced**, because the shape recurs:
its ``_nn_rmse(db, shadow_dbm)`` varied shadowing by overwriting
``shadow_fading_std_dBm`` in a copy of the path-loss config. Under the split
model that key no longer reaches the query's shadowing at all -- the field comes
from ``db.meta['shadow_field']`` -- so the knob became inert while the test kept
passing and kept reporting a difference, which was pure RNG wobble. A parameter
that silently stops being read is indistinguishable from one that has no effect.
The switch here is ``include_shadowing``, a boolean the query generator actually
branches on.

Author: Li-Ta Hsu
References: Chapter 5, Sections 5.1-5.3
"""

import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np
import pytest

from ch5_fingerprinting.example_comparison import _generate_queries_pathloss
from core.fingerprinting import (
    ShadowingField,
    load_fingerprint_database,
    nn_localize,
)

DATABASE = Path("data/sim/ch5_wifi_fingerprint_grid")
GRID_SPACING_M = 5.0

#: RMS distance from a uniformly placed query to the nearest node of a square
#: grid of this spacing, which is the best any nearest-neighbour method can do.
#: sqrt(2 * s^2 / 12) for a cell of side s -- measured at 2.27 m for s = 5 m.
QUANTISATION_FLOOR_M = np.sqrt(2 * GRID_SPACING_M**2 / 12)


@pytest.fixture(scope="module")
def db():
    return load_fingerprint_database(DATABASE)


def _queries(db, *, include_shadowing, fast_std, noise_std=0.0, n=200, seed=42):
    """Query set at a chosen point of the noise model."""
    config = dict(db.meta["path_loss_model"])
    config["fast_fading_std_dBm"] = fast_std
    np.random.seed(seed)
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        return _generate_queries_pathloss(
            db,
            np.asarray(db.meta["ap_positions"]),
            config,
            n_queries=n,
            floor_id=0,
            noise_std=noise_std,
            include_shadowing=include_shadowing,
        )


def _nn_rmse(db, queries, truth):
    estimates = np.array(
        [nn_localize(q, db, metric="euclidean", floor_id=0) for q in queries]
    )
    return float(
        np.sqrt(np.mean(np.linalg.norm(estimates - np.asarray(truth), axis=1) ** 2))
    )


def test_the_shipped_map_decomposes_into_a_field_and_fast_fading(db):
    """The stored bytes must match the model the metadata declares.

    Both terms, separately. Checking only the total would pass for any split
    that happens to sum to the same variance -- including the old one, where all
    of it was per-sample.
    """
    field = ShadowingField.from_meta(db.meta)
    config = db.meta["path_loss_model"]
    aps = np.asarray(db.meta["ap_positions"])
    floor_height = db.meta["floor_height"]

    residuals = []
    for floor in db.floor_list:
        mask = db.get_floor_mask(floor)
        xy = db.locations[mask]
        z = floor * floor_height + 1.5
        distance = np.maximum(
            np.linalg.norm(
                np.column_stack([xy, np.full(len(xy), z)])[:, None, :] - aps[None],
                axis=2,
            ),
            0.1,
        )
        mean = (
            config["P0_dBm"]
            - 10 * config["path_loss_exponent"] * np.log10(distance)
            - np.abs(floor - (aps[:, 2] / floor_height).astype(int))[None, :]
            * config["floor_attenuation_dB"]
        )
        residuals.append(db.features[mask] - mean - field(xy, floor))

    # What is left once path loss and the location's own shadowing are removed
    # is fast fading, and nothing else.
    leftover = np.concatenate(residuals)
    assert leftover.std() == pytest.approx(config["fast_fading_std_dBm"], rel=0.10), (
        f"after removing path loss and S_ap(p) the map still varies by "
        f"{leftover.std():.3f} dB against a declared fast fading of "
        f"{config['fast_fading_std_dBm']} dB"
    )
    assert abs(leftover.mean()) < 0.2


def test_a_query_consistent_with_the_map_approaches_the_grid_floor(db):
    """The assertion the old file made in reverse.

    A noiseless query at position p carries ``S_ap(p)``, the same shadowing the
    map was built with, so the only things between nearest neighbour and the
    quantisation floor are the map's own 1.5 dB of fast fading and the field
    changing between reference points 5 m apart. Measured: 3.39 m against a
    2.27 m floor, where the old model gave 6.93 m -- three times the floor.

    The bound is 2x the floor. Both sides of it were measured: the old model
    sits at 3.05x and the new one at 1.49x, so the gate falls between the defect
    it must catch and the value it has to tolerate rather than beside either.
    """
    queries, truth, _ = _queries(db, include_shadowing=True, fast_std=0.0)

    rmse = _nn_rmse(db, queries, truth)

    assert rmse < 2.0 * QUANTISATION_FLOOR_M, (
        f"nearest neighbour scores {rmse:.2f} m against a grid floor of "
        f"{QUANTISATION_FLOOR_M:.2f} m; the radio map is not a smooth function "
        f"of position"
    )


def test_a_query_that_ignores_the_shadowing_is_much_worse(db):
    """The old model's regime, kept runnable so the comparison is not prose.

    Redrawing shadowing per query -- equivalently, omitting it -- makes the
    query inconsistent with the map at its own position. That is not a small
    effect and it is not noise: it is the correlation the method runs on.
    """
    consistent, truth, _ = _queries(db, include_shadowing=True, fast_std=0.0)
    ignoring, truth_b, _ = _queries(db, include_shadowing=False, fast_std=0.0)
    assert np.array_equal(truth, truth_b)  # same positions, one term removed

    assert _nn_rmse(db, ignoring, truth) > 2.0 * _nn_rmse(db, consistent, truth)


def test_fast_fading_costs_less_than_the_shadowing_field_would(db):
    """The split is a real split: the per-sample term is the smaller one.

    If ``fast_fading_std_dBm`` were set to the shadowing std this would fail,
    which is the point -- it is the assertion that distinguishes the new model
    from the old one wearing two parameter names.
    """
    config = db.meta["path_loss_model"]
    assert config["fast_fading_std_dBm"] < 0.5 * config["shadow_fading_std_dBm"]

    queries_a, truth, _ = _queries(db, include_shadowing=True, fast_std=0.0)
    queries_b, _, _ = _queries(
        db, include_shadowing=True, fast_std=config["fast_fading_std_dBm"]
    )
    queries_c, _, _ = _queries(
        db, include_shadowing=True, fast_std=config["shadow_fading_std_dBm"]
    )

    honest = _nn_rmse(db, queries_b, truth) - _nn_rmse(db, queries_a, truth)
    if_it_were_shadowing = _nn_rmse(db, queries_c, truth) - _nn_rmse(
        db, queries_a, truth
    )

    assert 0.0 < honest < 0.5 * if_it_were_shadowing


def test_the_scenario_labels_now_name_the_dominant_term(db):
    """The complaint the deleted file was written about, re-asked.

    It found that the labelled sweep (1 to 5 dBm of extra noise) and the
    unmentioned per-query shadowing were comparable, so the labels named half
    the story. The unmentioned term is now 1.5 dB of fast fading rather than
    4 dB of redrawn shadowing, so what the labels omit must be the *smaller*
    half.
    """
    base, truth, _ = _queries(db, include_shadowing=True, fast_std=0.0)
    fast_std = db.meta["path_loss_model"]["fast_fading_std_dBm"]

    with_fast, _, _ = _queries(db, include_shadowing=True, fast_std=fast_std)
    labelled_low, _, _ = _queries(
        db, include_shadowing=True, fast_std=fast_std, noise_std=1.0
    )
    labelled_high, _, _ = _queries(
        db, include_shadowing=True, fast_std=fast_std, noise_std=5.0
    )

    unlabelled = _nn_rmse(db, with_fast, truth) - _nn_rmse(db, base, truth)
    swept = _nn_rmse(db, labelled_high, truth) - _nn_rmse(db, labelled_low, truth)

    assert 0.0 < unlabelled < swept
