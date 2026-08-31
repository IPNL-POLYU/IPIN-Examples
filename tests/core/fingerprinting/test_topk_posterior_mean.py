"""Truncating Eq. (5.5) to the top k candidates must change the answer and not
the estimator.

This file used to be a *script*: four ``test_``-named functions holding zero
assertions, printing ``[OK]``/``[FAIL]``/``[WARNING]`` and ending in
``return True``. pytest collected all four, called them, discarded the return
value and reported four passes -- so the file could only ever be green, whatever
``posterior_mean_localize`` did. That is the shape CLAUDE.md records as failing
silently by never speaking at all: it looks like coverage from every angle
except the one that decides whether it runs. The printed checks say what the
author meant to assert, and they are the assertions here.

One of the four claims did not survive being measured, and that is the useful
part. ``test_topk_speedup`` benchmarked wall-clock milliseconds and reported a
"speedup"; but ``posterior_mean_localize`` calls ``log_posterior`` over **every**
reference point before it truncates, so top-k trims an O(M*d) weighted sum while
the O(M*N) likelihood stands. On the shipped grid database that is 3993
operations against 3650 -- 1.09x, not the 2.86x the chapter README used to
advertise. The honest assertion is structural rather than temporal:
:func:`test_truncation_does_not_avoid_the_dominant_term` counts the reference
points the likelihood is asked about and requires it to be all of them.

Author: Li-Ta Hsu
References: Chapter 5, Eq. (5.5); Section 5.1.2 ("a calculation based on the
    top k candidates is typically sufficient").
"""

from pathlib import Path

import numpy as np
import pytest

from core.fingerprinting import (
    FingerprintDatabase,
    fit_gaussian_naive_bayes,
    load_fingerprint_database,
    posterior_mean_localize,
)
from core.fingerprinting import probabilistic as probabilistic_module

DATABASE = Path("data/sim/ch5_wifi_fingerprint_grid")

#: The truncation this chapter recommends, and the one the comparison figure
#: plots as "Post.Mean (k=10)".
K = 10


@pytest.fixture(scope="module")
def db():
    return load_fingerprint_database(DATABASE)


@pytest.fixture(scope="module")
def model(db):
    return fit_gaussian_naive_bayes(db, min_std=2.0)


@pytest.fixture(scope="module")
def queries(db):
    """Reference-point fingerprints perturbed the way a live query differs."""
    rng = np.random.default_rng(42)
    mask = db.get_floor_mask(0)
    features = db.get_mean_features()[mask]
    picks = rng.integers(0, len(features), size=40)
    return features[picks] + rng.standard_normal((40, db.n_features)) * 2.0


def _tiny_model(n_rps=4):
    """A hand-written database small enough that k = M is reachable."""
    locations = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)[:n_rps]
    features = np.array(
        [
            [-50.0, -60.0, -70.0],
            [-60.0, -50.0, -80.0],
            [-70.0, -80.0, -50.0],
            [-80.0, -70.0, -60.0],
        ]
    )[:n_rps]
    db = FingerprintDatabase(
        locations=locations,
        features=features,
        floor_ids=np.zeros(n_rps, dtype=int),
        meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"},
    )
    return db, fit_gaussian_naive_bayes(db, min_std=2.0)


def test_top_k_none_is_the_default_and_is_the_full_sum(model, queries):
    """``top_k=None`` must be the untruncated Eq. (5.5), on every query.

    The claim the old TEST 1 printed but never checked. It computed one
    ``top_k=None`` estimate and reported ``[OK]`` for having computed it.
    """
    for query in queries:
        default = posterior_mean_localize(query, model, floor_id=0)
        explicit = posterior_mean_localize(query, model, floor_id=0, top_k=None)
        assert np.array_equal(default, explicit)


def test_keeping_every_candidate_reproduces_the_full_sum():
    """``top_k = M`` truncates nothing, so it must agree with the full sum.

    Not the same code path: the top-k branch selects, gathers and renormalises,
    where the full branch sums the posterior as it stands. Agreement to float
    noise is the assertion that the renormalisation is a no-op when nothing was
    dropped.
    """
    _, model = _tiny_model()
    query = np.array([-55.0, -65.0, -75.0])

    full = posterior_mean_localize(query, model, floor_id=0)
    everything = posterior_mean_localize(
        query, model, floor_id=0, top_k=model.n_reference_points
    )

    assert everything == pytest.approx(full, abs=1e-12)


def test_truncation_converges_to_the_full_sum_as_k_grows(model, queries):
    """More candidates cannot be further from the full sum than fewer.

    The old TEST 2 computed exactly this comparison and printed ``[OK]`` or
    ``[WARNING]`` for the outcome, which is a report rather than a check.

    Both bounds below were measured rather than argued, over these 40 queries
    on the shipped grid: the worst displacement is 1.529 m at k=1, 0.074 m at
    k=3 and 0.000686 m at k=10. So 0.01 m sits 15x above what k=10 does and
    0.02 m sits 3.7x below what k=3 does -- between the behaviour each gate has
    to tolerate and the one it has to catch, rather than beside either.
    """
    worst_k3 = 0.0
    worst_k10 = 0.0
    for query in queries:
        full = posterior_mean_localize(query, model, floor_id=0)
        k3 = posterior_mean_localize(query, model, floor_id=0, top_k=3)
        k10 = posterior_mean_localize(query, model, floor_id=0, top_k=K)

        error_k3 = float(np.linalg.norm(full - k3))
        error_k10 = float(np.linalg.norm(full - k10))
        assert error_k10 <= error_k3 + 1e-12, (
            f"keeping {K} candidates lands {error_k10:.4f} m from the full "
            f"posterior mean while keeping 3 lands {error_k3:.4f} m from it"
        )
        worst_k3 = max(worst_k3, error_k3)
        worst_k10 = max(worst_k10, error_k10)

    # The book's claim, as a number: k=10 is "typically sufficient" here means
    # the truncation is invisible at centimetre scale, and k=3 is not.
    assert worst_k10 < 0.01, f"k={K} displaces the estimate by {worst_k10:.4f} m"
    assert worst_k3 > 0.02, (
        f"k=3 displaces the estimate by only {worst_k3:.4f} m, so this test is "
        f"no longer comparing a truncated posterior against an untruncated one"
    )
    assert worst_k3 > 10 * worst_k10, (
        f"k=3 and k={K} now displace the estimate by {worst_k3:.6f} m and "
        f"{worst_k10:.6f} m, which is not a truncation the posterior notices"
    )


def test_truncation_does_not_avoid_the_dominant_term(model, monkeypatch):
    """top-k trims the weighted sum, not the likelihood -- so it saves little.

    This replaces a wall-clock benchmark that printed a speedup nothing checked.
    A measured millisecond cannot be asserted on: it differs per run and per
    machine, which is why this repository counts operations instead of timing
    them. What *is* exact is that ``log_posterior`` is evaluated over the whole
    database either way, so no truncation can touch the O(M*N) term.
    """
    seen = []
    real_log_posterior = probabilistic_module.log_posterior

    def counting_log_posterior(z, model_, floor_id=None):
        seen.append(model_.n_reference_points)
        return real_log_posterior(z, model_, floor_id=floor_id)

    monkeypatch.setattr(probabilistic_module, "log_posterior", counting_log_posterior)

    query = np.zeros(model.n_features) - 60.0
    posterior_mean_localize(query, model, floor_id=0)
    posterior_mean_localize(query, model, floor_id=0, top_k=K)

    assert seen == [model.n_reference_points] * 2, (
        f"the likelihood was evaluated over {seen} reference points; if top-k "
        f"ever narrows it, the cost figures in Chapter 5 are wrong"
    )


def test_keeping_one_candidate_returns_a_reference_point():
    """``top_k=1`` renormalises a single weight to 1, so it is MAP's location.

    The old TEST 4 computed the distance to the nearest RP and printed ``[OK]``
    only when it was below 1e-6, saying nothing at all when it was not.
    """
    db, model = _tiny_model()
    query = np.array([-55.0, -65.0, -75.0])

    estimate = posterior_mean_localize(query, model, floor_id=0, top_k=1)

    distances = np.linalg.norm(db.locations - estimate, axis=1)
    assert distances.min() == pytest.approx(0.0, abs=1e-9), (
        f"top_k=1 returned {estimate}, which is {distances.min():.6f} m from "
        f"the nearest reference point"
    )


@pytest.mark.parametrize("bad_k", [0, -1, 5])
def test_an_out_of_range_k_is_rejected(bad_k):
    """0, negative, and more candidates than exist must all raise.

    The old file returned ``False`` from the test function on the paths that
    should have failed, which pytest discards -- so a silently-accepted
    ``top_k=0`` would have been reported as a pass.
    """
    _, model = _tiny_model()  # 4 reference points, so 5 is out of range
    query = np.array([-55.0, -65.0, -75.0])

    with pytest.raises(ValueError):
        posterior_mean_localize(query, model, floor_id=0, top_k=bad_k)
