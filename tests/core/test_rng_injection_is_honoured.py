"""The rng parameter on core's random functions has to actually be used.

`test_core_library_takes_its_randomness_from_the_caller` in
tests/test_repo_conventions.py checks that no library function reaches for the
global stream. That is a check on the *shape* of the code, and shape is not
behaviour: a function could accept `rng`, ignore it, and still pass, because the
ratchet only sees that `np.random.foo(...)` is absent.

So this file pins the three properties that make injection worth having:

1. the same Generator gives the same draws, whatever np.random.seed says,
2. a different Generator gives different draws,
3. the default still follows np.random.seed, so existing callers and every
   committed figure are unaffected.

**The first is the one that catches an accepted-but-ignored rng**, and that was
established by mutation rather than by reasoning: making _rng return np.random
unconditionally turns it red. A fourth assertion -- that the injected result
differs from the default -- was written here first and then deleted, because an
ignored rng satisfies it too. Two consecutive draws from the global stream
differ whether or not anyone is listening to the argument, so it discriminated
nothing while reading like the main event.

Author: Li-Ta Hsu
"""

import numpy as np
import pytest

from core.estimators.particle_filter import ParticleFilter
from core.rf.measurement_models import simulate_rss_measurement
from core.slam.scan_generation import generate_dense_wall_scan


def _scan(rng=None):
    pose = np.array([0.0, 0.0, 0.0])
    walls = np.array([[[-5.0, 5.0], [5.0, 5.0]], [[-5.0, -5.0], [5.0, -5.0]]])
    return np.asarray(generate_dense_wall_scan(pose, walls, noise_std=0.1, rng=rng))


def _rss(rng=None):
    rss, _info = simulate_rss_measurement(
        anchor_pos=np.zeros(2),
        agent_pos=np.array([10.0, 0.0]),
        p_ref_dbm=-40.0,
        sigma_long_db=4.0,
        sigma_short_linear=0.5,
        n_samples_avg=4,
        rng=rng,
    )
    return np.asarray(rss, dtype=float)


def _particles(rng=None):
    filt = ParticleFilter(
        process_model=lambda x, u, dt: x,
        likelihood_func=lambda z, x: 1.0,
        n_particles=32,
        x0=np.zeros(2),
        P0=np.eye(2),
        rng=rng,
    )
    return np.asarray(filt.particles)


DRAWERS = {
    "generate_dense_wall_scan": _scan,
    "simulate_rss_measurement": _rss,
    "ParticleFilter": _particles,
}


@pytest.mark.parametrize("name", sorted(DRAWERS), ids=sorted(DRAWERS))
def test_the_same_generator_repeats_whatever_the_global_seed_is(name):
    """An injected Generator must be immune to np.random.seed."""
    draw = DRAWERS[name]
    np.random.seed(1)
    first = draw(rng=np.random.default_rng(7))
    np.random.seed(999)
    second = draw(rng=np.random.default_rng(7))

    assert np.array_equal(first, second), (
        f"{name} gave different results for the same Generator under different "
        f"global seeds, so it is still reading np.random somewhere."
    )


@pytest.mark.parametrize("name", sorted(DRAWERS), ids=sorted(DRAWERS))
def test_a_different_generator_gives_different_draws(name):
    """Guard the guard: a constant output would satisfy the test above."""
    draw = DRAWERS[name]
    assert not np.array_equal(
        draw(rng=np.random.default_rng(7)), draw(rng=np.random.default_rng(8))
    ), f"{name} returned the same values for two different Generators."


@pytest.mark.parametrize("name", sorted(DRAWERS), ids=sorted(DRAWERS))
def test_the_default_still_follows_the_global_seed(name):
    """Existing callers seed with np.random.seed, and must keep working.

    This is why the default is np.random rather than a fresh default_rng(): a
    fresh Generator would ignore the seed, and every committed figure in this
    repository depends on it.
    """
    draw = DRAWERS[name]
    np.random.seed(5)
    first = draw()
    np.random.seed(5)
    second = draw()

    assert np.array_equal(first, second), (
        f"{name} is no longer reproducible under np.random.seed, so the "
        f"examples that seed that way have stopped being reproducible."
    )
