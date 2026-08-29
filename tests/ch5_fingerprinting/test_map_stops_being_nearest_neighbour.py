"""MAP is 1-NN on a single-sample survey, and stops being it on a repeat one.

Chapter 5's comparison table used to show ``MAP`` and ``NN (Euclidean)`` scoring
identically to the digit at every noise level, and ``Posterior Mean`` matching
``Post.Mean (k=10)``. Both were exact, not coincidental, and neither had
anything to do with the methods:

    With a globally constant sigma the Gaussian log-likelihood is
    ``const - ||z - mu_i||^2 / (2 sigma^2)``, monotone in Euclidean distance,
    so ``argmax`` over it is ``argmin`` over distance. Eq. (5.4) *is* Eq. (5.1).

A single-sample database always produces that constant sigma, because there is
no second sample to take a standard deviation of and ``fit_gaussian_naive_bayes``
falls back to ``min_std``. Every one of the three shipped databases was
single-sample, so the chapter demonstrated probabilistic fingerprinting using a
model with no probabilistic content, and nothing said so.

This file pins both halves:

- the degeneracy, as the theorem it is, so nobody "fixes" the single-sample
  case by tuning something;
- its absence on ``ch5_wifi_fingerprint_multisamples``, the ten-visit survey
  added for exactly this, so the chapter has one database where Eq. (5.6) has
  parameters to estimate.

Author: Li-Ta Hsu
References: Chapter 5, Eqs. (5.1), (5.4)-(5.6)
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
    fit_gaussian_naive_bayes,
    load_fingerprint_database,
    map_localize,
    nn_localize,
)
from core.fingerprinting.probabilistic import log_posterior

SINGLE = Path("data/sim/ch5_wifi_fingerprint_grid")
MULTI = Path("data/sim/ch5_wifi_fingerprint_multisamples")

#: Below the databases' 1.5 dB fast fading, so the estimated sigma is what the
#: model uses rather than the floor. See ``fit_gaussian_naive_bayes``: a floor
#: above the true per-visit spread erases the per-RP sigma entirely.
NUMERICAL_FLOOR = 0.5


@pytest.fixture(scope="module")
def db_single():
    return load_fingerprint_database(SINGLE)


@pytest.fixture(scope="module")
def db_multi():
    return load_fingerprint_database(MULTI)


@pytest.fixture(scope="module")
def queries(db_single):
    np.random.seed(42)
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        fingerprints, truth, _ = _generate_queries_pathloss(
            db_single,
            np.asarray(db_single.meta["ap_positions"]),
            db_single.meta["path_loss_model"],
            n_queries=200,
            floor_id=0,
            noise_std=1.0,
        )
    return fingerprints, truth


def _disagreement(db, model, fingerprints):
    """Fraction of queries where MAP and 1-NN choose different reference points."""
    differ = sum(
        not np.allclose(
            map_localize(z, model, floor_id=0),
            nn_localize(z, db, metric="euclidean", floor_id=0),
        )
        for z in fingerprints
    )
    return differ / len(fingerprints)


def _median_max_posterior(model, fingerprints):
    return float(
        np.median(
            [np.exp(log_posterior(z, model, floor_id=0)).max() for z in fingerprints]
        )
    )


def test_a_multisample_database_ships(db_multi):
    """Eq. (5.6) needs samples to estimate from, and now the chapter has some."""
    assert db_multi.has_multiple_samples
    assert db_multi.n_samples_per_rp >= 10


def test_the_single_sample_model_says_out_loud_that_its_sigma_is_constant(
    db_single,
):
    """A reader must not have to measure this to discover it.

    ``repr`` and ``sigma_summary`` both name the consequence, so printing the
    model is enough.
    """
    model = fit_gaussian_naive_bayes(db_single, min_std=2.0)

    assert model.sigma_is_constant
    assert "1-NN" in repr(model)
    assert "1-NN" in model.sigma_summary()


def test_map_is_exactly_nearest_neighbour_on_a_single_sample_survey(
    db_single, queries
):
    """Not approximately. Every query, on the nose.

    Asserted as an equality rather than a bound because it is a theorem about
    the likelihood, not a measurement: a single failure here would mean the
    monotone relationship had broken, which is worth knowing.
    """
    fingerprints, _ = queries
    model = fit_gaussian_naive_bayes(db_single, min_std=2.0)

    assert _disagreement(db_single, model, fingerprints) == 0.0


def test_the_multisample_model_estimates_a_sigma_that_varies(db_multi):
    """The precondition for anything below it. Variety, not quality.

    A constant predictor scores its base rate and a constant sigma reproduces
    1-NN; in both cases the check that costs one line is whether the output
    varies at all.
    """
    model = fit_gaussian_naive_bayes(db_multi, min_std=NUMERICAL_FLOOR)

    assert not model.sigma_is_constant
    assert len(np.unique(model.stds)) > 100


def test_map_stops_being_nearest_neighbour_on_a_repeat_survey(db_multi, queries):
    """The headline property this change exists to produce.

    Measured at 22% of queries. The gate is 5%: well above the 0% the
    single-sample database gives by construction, and far enough below the
    measured value to survive a change of seed or query count.

    Do not read a large number here as a better method -- see the example's
    "What a repeat survey buys" section, where MAP at this floor is slightly
    *worse* than 1-NN. With one fast-fading std for the whole building the
    per-(RP, AP) sigma varies only by estimation noise. The claim under test is
    that Eq. (5.4) is no longer arithmetically identical to Eq. (5.1), which is
    a statement about the model, not about accuracy.
    """
    fingerprints, _ = queries
    model = fit_gaussian_naive_bayes(db_multi, min_std=NUMERICAL_FLOOR)

    assert _disagreement(db_multi, model, fingerprints) > 0.05


def test_the_posterior_is_not_a_delta(db_single, db_multi, queries):
    """Eq. (5.5) must be able to differ from Eq. (5.4).

    It could not: on the old database at the library's default floor the median
    maximum posterior weight was 1.0000, so the posterior mean was the MAP
    estimate and the top-10 truncation was the full sum.

    The floor is what sets this, and the assertion is scoped accordingly: at
    ``min_std = 2.0``, the value the chapter's comparison uses and roughly the
    2.09 dB a query really disagrees with its nearest RP by, the posterior
    spreads over more than one reference point on both databases.
    """
    fingerprints, _ = queries

    for db in (db_single, db_multi):
        model = fit_gaussian_naive_bayes(db, min_std=2.0)
        weight = _median_max_posterior(model, fingerprints)

        assert weight < 0.95, (
            f"median maximum posterior weight is {weight:.4f}: the posterior is "
            f"a delta and Eq. (5.5) cannot differ from Eq. (5.4)"
        )


def test_raising_the_floor_trades_one_property_for_the_other(db_multi, queries):
    """The finding that neither end of the sweep is the answer.

    A floor above the fast-fading std erases the estimated sigma, and MAP
    collapses back onto 1-NN even on a ten-visit survey. This is what stops
    someone "improving" the chapter by raising ``min_std`` and quietly undoing
    the multi-sample database.
    """
    fingerprints, _ = queries

    low = fit_gaussian_naive_bayes(db_multi, min_std=NUMERICAL_FLOOR)
    high = fit_gaussian_naive_bayes(db_multi, min_std=2.5)

    assert _disagreement(db_multi, low, fingerprints) > _disagreement(
        db_multi, high, fingerprints
    )
    assert _median_max_posterior(low, fingerprints) > _median_max_posterior(
        high, fingerprints
    )
