"""The shadowing field must be a *field*: correlated, reproducible, everywhere.

``ShadowingField`` replaced a per-draw ``np.random.randn() * sigma``. Three
properties distinguish the replacement from what it replaced, and a test that
does not pin all three would pass for the thing being removed:

- **Correlated.** Two points a metre apart must see nearly the same shadowing.
  An independent draw satisfies every marginal check and fails this one.
- **A function of position.** Calling it twice at one point must return one
  answer. This is the property the old model lacked entirely, and the reason a
  query could not be made consistent with the map.
- **Reconstructible.** A consumer of a dataset must be able to rebuild the exact
  field the generator used, or the map and the query can only ever disagree.

The marginal-variance check is the weakest of the four and is here for scale,
not for discrimination: the old model passed it too.

Author: Li-Ta Hsu
References: Chapter 5, Section 5.1 (radio map construction).
"""

import numpy as np
import pytest

from core.fingerprinting import DEFAULT_DECORRELATION_M, ShadowingField

SIGMA = 4.0
LENGTH = 8.0
SEED = 42


@pytest.fixture(scope="module")
def field():
    return ShadowingField.build(
        n_aps=8, n_floors=3, sigma_dB=SIGMA, seed=SEED, decorrelation_m=LENGTH
    )


@pytest.fixture(scope="module")
def samples(field):
    """A dense sample of one field, pooled over floors and APs."""
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 50.0, size=(3000, 2))
    return points, np.stack([field(points, f) for f in range(3)])


def test_the_marginal_std_is_the_sigma_it_was_asked_for(samples):
    """Scale check: ``shadow_fading_std_dBm`` still means what it says.

    The construction has marginal variance sigma^2 exactly, in expectation. One
    *realisation* on a bounded domain does not, and both bounds here were set by
    measuring 32 seeds rather than by argument -- the first draft asserted
    ``|mean| < 0.15 * sigma`` and seed 42 failed it at 0.62, which was the field
    behaving correctly and the test being wrong.

    Across 32 seeds the per-realisation spread has mean 3.994 dB (so the
    ensemble value is sigma, to a tenth of a percent) and standard deviation
    0.144 dB, with a range of 3.68 to 4.40; the per-realisation domain *mean*
    has standard deviation 0.280 dB and never exceeded 0.63. A true
    Gaussian-process realisation on a domain only six correlation lengths wide
    behaves the same way: a real building can simply have a systematically
    strong AP.
    """
    _, values = samples

    assert values.mean() == pytest.approx(0.0, abs=0.25 * SIGMA)
    assert values.std() == pytest.approx(SIGMA, rel=0.15)


def test_the_ensemble_recovers_sigma_even_though_one_draw_does_not():
    """Averaging over seeds is what makes the previous test's bounds honest.

    Without this, "within 15%" could be hiding a construction that is
    systematically 10% hot. Over 16 seeds the mean realisation std must land on
    sigma tightly, and the per-realisation scatter must be small but nonzero --
    zero scatter would mean the seed was not reaching the field.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 50.0, size=(1200, 2))

    spreads = []
    centres = []
    for seed in range(16):
        one = ShadowingField.build(
            n_aps=8, n_floors=3, sigma_dB=SIGMA, seed=seed, decorrelation_m=LENGTH
        )
        values = np.stack([one(points, f) for f in range(3)])
        spreads.append(values.std())
        centres.append(values.mean())

    assert np.mean(spreads) == pytest.approx(SIGMA, rel=0.04)
    assert np.mean(centres) == pytest.approx(0.0, abs=0.1 * SIGMA)
    assert 0.0 < np.std(spreads) < 0.1 * SIGMA


def test_nearby_points_see_nearly_the_same_shadowing(field):
    """The property an independent per-draw term cannot have.

    Half a metre apart is 1/16 of the correlation length, where the target
    covariance is exp(-0.5^2 / (2 * 8^2)) = 0.998. Independent draws at sigma =
    4 dB would differ by 5.7 dB rms; this must differ by almost nothing.
    """
    rng = np.random.default_rng(1)
    a = rng.uniform(0.0, 50.0, size=(2000, 2))
    b = a + rng.normal(0.0, 0.35, size=a.shape)  # ~0.5 m apart

    difference = (field(a, 0) - field(b, 0)).std()
    independent_draws = SIGMA * np.sqrt(2.0)

    assert difference < 0.1 * independent_draws


def test_the_covariance_follows_the_kernel_it_claims(samples, field):
    """Squared-exponential, out to and beyond the correlation length.

    Normalised by the realisation's own variance, so this tests the *shape* of
    the covariance rather than re-testing the scale the previous test covers.
    """
    _, values = samples
    variance = float((values**2).mean())

    rng = np.random.default_rng(2)
    a = rng.uniform(0.0, 50.0, size=(3000, 2))
    s_a = np.stack([field(a, f) for f in range(3)])

    for radius in (2.0, 5.0, 8.0, 16.0):
        angle = rng.uniform(0.0, 2 * np.pi, size=len(a))
        b = a + radius * np.column_stack([np.cos(angle), np.sin(angle)])
        s_b = np.stack([field(b, f) for f in range(3)])

        empirical = float((s_a * s_b).mean()) / variance
        target = np.exp(-(radius**2) / (2 * LENGTH**2))

        assert empirical == pytest.approx(target, abs=0.06), (
            f"at r = {radius} m the correlation is {empirical:.3f}, "
            f"target {target:.3f}"
        )


def test_it_is_a_function_of_position_not_a_draw(field):
    """Two calls at one point agree. The old model's defect, in one line."""
    point = np.array([[13.7, 41.2]])

    assert np.array_equal(field(point, 1), field(point, 1))


def test_arbitrary_positions_not_only_a_lattice(field):
    """A query stands between reference points, so the field must too."""
    between = np.array([[2.5, 7.5], [13.3, 28.9], [49.999, 0.001]])
    values = field(between, 0)

    assert values.shape == (3, 8)
    assert np.all(np.isfinite(values))


def test_the_same_seed_rebuilds_the_same_field(field):
    """Reproducible from its parameters alone, which is what ``seed`` promises."""
    twin = ShadowingField.build(
        n_aps=8, n_floors=3, sigma_dB=SIGMA, seed=SEED, decorrelation_m=LENGTH
    )
    points = np.array([[1.0, 2.0], [30.0, 44.0]])

    assert np.array_equal(field(points, 2), twin(points, 2))


def test_a_different_seed_gives_a_different_field(field):
    """The seed is load-bearing, not decorative."""
    other = ShadowingField.build(
        n_aps=8, n_floors=3, sigma_dB=SIGMA, seed=SEED + 1, decorrelation_m=LENGTH
    )
    points = np.array([[1.0, 2.0], [30.0, 44.0]])

    assert not np.allclose(field(points, 0), other(points, 0))


def test_each_ap_and_floor_is_drawn_independently_of_the_others(field):
    """Adding an AP must not redraw the ones already there.

    Realisation (floor, ap) comes from ``default_rng([seed, floor, ap])``, so
    the 4-AP building is the 8-AP building with four APs switched off, and the
    dense / sparse / baseline surveys of one building see one radio
    environment. Drawing the whole array in one call would silently fail this.
    """
    fewer = ShadowingField.build(
        n_aps=4, n_floors=3, sigma_dB=SIGMA, seed=SEED, decorrelation_m=LENGTH
    )
    points = np.array([[5.0, 5.0], [25.0, 35.0]])

    assert np.array_equal(field(points, 0)[:, :4], fewer(points, 0))

    # Different floors are different buildings' worth of walls.
    assert not np.allclose(field(points, 0), field(points, 1))


def test_metadata_round_trips(field):
    """A dataset consumer rebuilds the generator's field, exactly."""
    rebuilt = ShadowingField.from_meta({"shadow_field": field.to_meta()})
    points = np.array([[11.0, 22.0], [33.0, 44.0]])

    assert np.array_equal(field(points, 1), rebuilt(points, 1))


def test_the_default_correlation_length_exceeds_the_survey_grid():
    """A survey cannot represent a field that varies faster than it samples.

    The Chapter 5 databases are surveyed on a 5 m grid, and the sparsest is
    10 m. The default has to sit above the baseline grid or fingerprinting is
    being demonstrated in a building it cannot work in.
    """
    assert DEFAULT_DECORRELATION_M > 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_aps": 0},
        {"n_floors": 0},
        {"n_modes": 0},
        {"sigma_dB": -1.0},
        {"decorrelation_m": 0.0},
    ],
)
def test_it_refuses_parameters_that_are_not_a_field(kwargs):
    """Fail fast at the boundary rather than returning a shape full of noise."""
    base = {"n_aps": 4, "n_floors": 1, "sigma_dB": SIGMA, "seed": SEED}
    base.update(kwargs)

    with pytest.raises(ValueError):
        ShadowingField.build(**base)


def test_it_refuses_positions_and_floors_that_do_not_line_up(field):
    """Shape errors here would otherwise broadcast into a plausible answer."""
    with pytest.raises(ValueError):
        field(np.zeros((3, 3)), 0)  # 3-D coordinates

    with pytest.raises(ValueError):
        field(np.zeros((3, 2)), [0, 1])  # two floors for three points

    with pytest.raises(ValueError):
        field(np.zeros((3, 2)), 9)  # no such floor
