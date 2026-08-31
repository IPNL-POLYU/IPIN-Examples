"""A repeat survey must reach Eq. (5.6): sigma estimated per RP, not assumed.

Like its neighbour ``test_topk_posterior_mean.py``, this file used to be a
script -- three ``test_``-named functions holding **zero** assertions, printing
``[OK]`` lines and ending in ``return True``. pytest called all three and
reported three passes, so nothing here could ever have gone red. The printed
lines say what the author meant to check, and those are the assertions now.

Three properties separate a multi-sample database from a single-sample one, and
only the third is about localisation:

- A single-sample database has no variance to estimate, so
  :func:`fit_gaussian_naive_bayes` fills sigma with ``min_std`` everywhere.
  That collapses the Gaussian log-likelihood to a monotone function of
  Euclidean distance, which is why MAP is exactly 1-NN there -- a theorem, not
  a coincidence, and the reason this database format exists.
- A multi-sample database estimates mu and sigma from the samples, so sigma
  varies by (RP, AP) and ``min_std`` becomes a floor rather than the whole
  model.
- Varying sigma changes the answer: an RP whose fingerprint is unreliable must
  lose weight in the posterior.

Author: Li-Ta Hsu
References: Chapter 5, Section 5.1.3; Eq. (5.6).
"""

import dataclasses

import numpy as np
import pytest

from core.fingerprinting import (
    FingerprintDatabase,
    fit_gaussian_naive_bayes,
    map_localize,
    posterior_mean_localize,
)
from core.fingerprinting.probabilistic import log_posterior

SINGLE_SAMPLE_MIN_STD = 2.0


def _single_sample_db():
    """Four reference points, one visit each, three APs."""
    return FingerprintDatabase(
        locations=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float),
        features=np.array(
            [
                [-50.0, -60.0, -70.0],
                [-60.0, -50.0, -80.0],
                [-70.0, -80.0, -50.0],
                [-80.0, -70.0, -60.0],
            ]
        ),
        floor_ids=np.zeros(4, dtype=int),
        meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"},
    )


#: Visits per reference point, matching ``ch5_wifi_fingerprint_multisamples``.
#:
#: The deleted file used five, and five is not enough to assert on: the sample
#: standard deviation of n draws has a relative spread of about
#: ``1 / sqrt(2 (n - 1))``, so 35% at n=5. Measured over 64 seeds, the ordering
#: below survives 59 of them at five visits and **64 of 64** at ten, where every
#: estimate also lands within a factor of two of its truth. A test that passes
#: on 92% of seeds is a test whose green means "seed 42", so the survey depth is
#: the shipped one.
SAMPLES_PER_RP = 10

#: The per-RP spreads ``_multi_sample_db`` draws with, in dB.
TRUE_SIGMA_PER_RP = np.array([1.0, 3.0, 6.0])


def _multi_sample_db(seed=42):
    """Three reference points surveyed ten times, with sigma set per RP.

    RP 0 is quiet (1 dB), RP 1 middling (3 dB), RP 2 noisy (6 dB), so the
    estimated sigma has something to recover and an ordering to preserve.
    """
    rng = np.random.default_rng(seed)
    features = np.zeros((3, SAMPLES_PER_RP, 2))
    for rp, (mu, sigma) in enumerate(
        [((-50.0, -60.0), 1.0), ((-60.0, -50.0), 3.0), ((-70.0, -80.0), 6.0)]
    ):
        features[rp] = np.asarray(mu) + rng.standard_normal((SAMPLES_PER_RP, 2)) * sigma
    return FingerprintDatabase(
        locations=np.array([[0, 0], [10, 0], [10, 10]], dtype=float),
        features=features,
        floor_ids=np.zeros(3, dtype=int),
        meta={
            "ap_ids": ["AP1", "AP2"],
            "unit": "dBm",
            "n_samples_per_rp": SAMPLES_PER_RP,
        },
    )


def test_one_visit_per_point_leaves_nothing_to_estimate():
    """A single-sample database must report exactly ``min_std``, everywhere.

    The old TEST 1 printed ``All stds = min_std? True`` and asserted nothing.
    The value matters beyond bookkeeping: a constant sigma is what makes
    Eq. (5.4) identical to Eq. (5.1), which is the chapter's headline finding
    about this database.
    """
    db = _single_sample_db()
    assert not db.has_multiple_samples
    assert db.n_samples_per_rp is None  # not 1: there is no sample axis at all

    model = fit_gaussian_naive_bayes(db, min_std=SINGLE_SAMPLE_MIN_STD)

    assert model.means.shape == db.features.shape
    assert model.stds.shape == db.features.shape
    assert np.all(model.stds == SINGLE_SAMPLE_MIN_STD)
    assert model.sigma_is_constant


def test_a_single_sample_model_still_localises():
    """Backward compatibility, as a result rather than as a print.

    Both estimators must return a finite position inside the surveyed area --
    the old TEST 1 printed the two estimates and checked neither.
    """
    db = _single_sample_db()
    model = fit_gaussian_naive_bayes(db, min_std=SINGLE_SAMPLE_MIN_STD)
    query = np.array([-55.0, -65.0, -75.0])

    for estimate in (
        map_localize(query, model, floor_id=0),
        posterior_mean_localize(query, model, floor_id=0),
    ):
        assert np.all(np.isfinite(estimate))
        assert np.all(estimate >= db.locations.min(axis=0))
        assert np.all(estimate <= db.locations.max(axis=0))

    # MAP picks a reference point; the posterior mean is free not to.
    assert np.min(
        np.linalg.norm(db.locations - map_localize(query, model, 0), axis=1)
    ) == (pytest.approx(0.0, abs=1e-9))


def test_repeat_visits_estimate_sigma_and_preserve_its_ordering():
    """Eq. (5.6) on a real survey: sigma comes from the samples, in order.

    The old TEST 2 printed the three per-RP averages next to the three sigmas
    they were drawn with and let the reader compare them. What can be asserted
    from ten visits is the *ordering*, plus each estimate landing within a
    factor of two of its truth; both hold for all 64 seeds measured, and the
    observed ratios span 0.57 to 1.44, so the factor of two is a gate the noise
    clears rather than one it grazes.
    """
    db = _multi_sample_db()
    assert db.has_multiple_samples
    assert db.n_samples_per_rp == SAMPLES_PER_RP

    model = fit_gaussian_naive_bayes(db, min_std=0.5)

    assert not model.sigma_is_constant
    assert model.means.shape == (3, 2)
    assert model.stds.shape == (3, 2)

    per_rp = model.stds.mean(axis=1)
    assert per_rp[0] < per_rp[1] < per_rp[2], (
        f"estimated sigma per RP is {per_rp}, which does not preserve the "
        f"{TRUE_SIGMA_PER_RP} ordering the samples were drawn with"
    )
    assert np.all(per_rp > TRUE_SIGMA_PER_RP / 2) and np.all(
        per_rp < TRUE_SIGMA_PER_RP * 2
    ), f"estimated sigma {per_rp} is not within a factor of two of {TRUE_SIGMA_PER_RP}"


def test_min_std_is_a_floor_on_a_repeat_survey_and_the_whole_model_without_one():
    """The same argument means two different things to the two database shapes.

    This is the distinction the old file's TEST 1 and TEST 2 straddled without
    ever checking: raising ``min_std`` past every estimated sigma must erase the
    variation and put a multi-sample model back into the single-sample regime.
    """
    db = _multi_sample_db()

    estimated = fit_gaussian_naive_bayes(db, min_std=0.5)
    assert estimated.stds.min() > 0.5  # nothing was floored
    assert not estimated.sigma_is_constant

    floored = fit_gaussian_naive_bayes(db, min_std=50.0)
    assert np.all(floored.stds == 50.0)
    assert floored.sigma_is_constant


def test_widening_a_reference_points_sigma_widens_its_basin():
    """Varying sigma changes the answer -- in the direction opposite the old note.

    The deleted TEST 3 printed ``OK Model 2 shifts toward RP0`` behind an
    ``if``, over a comment reading "high sigma at RP1 reduces its influence".
    Asserted, that is false, and the ``if`` is why nobody found out: it printed
    nothing at all when the shift did not happen, which is what happens.

    Two things were wrong with it. Its "identical but for the variance"
    databases were not identical -- redrawing RP 1's ten samples at eight times
    the spread moves its *mean* as well, by up to 3.6 dB here, so the
    comparison confounded mu with sigma. And the intuition is backwards.
    Eq. (5.6) divides each residual by sigma before squaring it, and the
    ``-log sigma`` normalisation that penalises a wide Gaussian is only additive
    while the quadratic term grows with the square of the residual. So a noisy
    reference point explains a *distant* query better than a quiet one does: it
    claims more of the fingerprint space, not less.

    That is not a curiosity. It is the mechanism behind the chapter's finding
    that MAP loses to 1-NN on the repeat survey -- weighting by the noise a
    survey can measure hands territory to the points whose fingerprints are
    least trustworthy.

    Sigma is varied here by editing the fitted model, so mu is bit-identical on
    both sides and only sigma moves.
    """
    locations = np.array([[0, 0], [10, 0]], dtype=float)
    rng = np.random.default_rng(100)
    features = np.zeros((2, 10, 2))
    features[0] = -50.0 + rng.standard_normal((10, 2)) * 1.0
    features[1] = -60.0 + rng.standard_normal((10, 2)) * 1.0
    db = FingerprintDatabase(
        locations=locations,
        features=features,
        floor_ids=np.zeros(2, dtype=int),
        meta={"ap_ids": ["AP1", "AP2"], "unit": "dBm"},
    )

    quiet = fit_gaussian_naive_bayes(db, min_std=0.5)
    noisy_rp1 = dataclasses.replace(
        quiet, stds=quiet.stds * np.array([[1.0, 1.0], [8.0, 8.0]])
    )
    assert np.array_equal(noisy_rp1.means, quiet.means)  # only sigma moved
    assert np.array_equal(noisy_rp1.stds[0], quiet.stds[0])

    def weight_of_rp1(model, query):
        return float(np.exp(log_posterior(query, model, floor_id=0))[1])

    # Standing on RP 0's own fingerprint, the noisy RP 1 still gains ground.
    at_rp0 = quiet.means[0]
    assert weight_of_rp1(quiet, at_rp0) < 1e-9
    assert weight_of_rp1(noisy_rp1, at_rp0) > 1e-3

    # Halfway between the two fingerprints the basin flips outright: the
    # posterior mean moves from RP 0 to RP 1 on sigma alone.
    midway = (quiet.means[0] + quiet.means[1]) / 2
    assert np.linalg.norm(
        posterior_mean_localize(midway, quiet, floor_id=0) - locations[0]
    ) == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(
        posterior_mean_localize(midway, noisy_rp1, floor_id=0) - locations[1]
    ) == pytest.approx(0.0, abs=1e-6)
