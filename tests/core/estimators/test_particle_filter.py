"""Conformance tests for the particle filter (book Section 3.3, SIR filter).

Covers the weight update (Eq. 3.34), unbiased systematic resampling, and the
overall behaviour as a valid Bayes filter (the PF posterior mean approaches the
Kalman-filter posterior on a linear-Gaussian problem with many particles).
"""

import numpy as np

from core.estimators.kalman_filter import KalmanFilter
from core.estimators.particle_filter import ParticleFilter


def _gauss_pdf(err, cov):
    k = len(err)
    return float(
        np.exp(-0.5 * err @ np.linalg.solve(cov, err))
        / np.sqrt((2 * np.pi) ** k * np.linalg.det(cov))
    )


def test_weight_update_matches_eq_3_34():
    """After update, weights equal normalized (prior_weight * likelihood)."""
    np.random.seed(0)
    R = np.array([[0.5]])
    H = np.array([[1.0, 0.0]])

    def pm(x, u, dt):
        return x

    def lik(z, x):
        return _gauss_pdf(z - H @ x, R)

    pf = ParticleFilter(
        pm, lik, n_particles=200, x0=np.array([0.0, 0.0]),
        P0=np.eye(2), resample_threshold=0.0,  # never resample
    )
    particles = pf.particles.copy()
    z = np.array([0.3])
    expected = np.array([lik(z, particles[i]) for i in range(200)])
    expected /= expected.sum()

    pf.update(z)
    np.testing.assert_allclose(pf.weights, expected, rtol=1e-10, atol=1e-12)


def test_systematic_resampling_is_unbiased():
    """Systematic resampling reproduces the weighted mean of the particles."""
    np.random.seed(1)
    pf = ParticleFilter(
        lambda x, u, dt: x, lambda z, x: 1.0, n_particles=50000,
        x0=np.array([0.0]), P0=np.array([[1.0]]),
    )
    # Impose a skewed weight distribution, then resample.
    pf.particles = np.linspace(-3, 3, 50000).reshape(-1, 1)
    w = np.exp(-0.5 * ((pf.particles[:, 0] - 1.0) / 0.7) ** 2)
    pf.weights = w / w.sum()
    weighted_mean = np.sum(pf.weights * pf.particles[:, 0])

    pf._resample()
    # After resampling weights are uniform; the empirical mean should match.
    resampled_mean = pf.particles[:, 0].mean()
    assert abs(resampled_mean - weighted_mean) < 0.02
    np.testing.assert_allclose(pf.weights, np.ones(50000) / 50000)


def test_effective_sample_size():
    """N_eff = 1/sum(w^2): N for uniform, 1 for a single dominant particle."""
    pf = ParticleFilter(
        lambda x, u, dt: x, lambda z, x: 1.0, n_particles=100,
        x0=np.array([0.0]), P0=np.array([[1.0]]),
    )
    assert np.isclose(pf._effective_sample_size(), 100.0)
    pf.weights = np.zeros(100)
    pf.weights[0] = 1.0
    assert np.isclose(pf._effective_sample_size(), 1.0)


def test_pf_approximates_kf_one_step():
    """One predict+update: PF posterior mean approaches the KF posterior."""
    np.random.seed(3)
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([0.05, 0.05])
    R = np.array([[0.4]])
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2) * 0.5

    kf = KalmanFilter(F, Q, H, R, x0.copy(), P0.copy())

    def pm(x, u, dt):
        return F @ x + np.random.multivariate_normal(np.zeros(2), Q)

    def lik(z, x):
        return _gauss_pdf(z - H @ x, R)

    pf = ParticleFilter(pm, lik, n_particles=40000, x0=x0.copy(), P0=P0.copy())

    z = np.array([0.9])
    kf.predict()
    kf.update(z)
    pf.predict()
    pf.update(z)

    assert np.linalg.norm(pf.state - kf.state) < 0.1
