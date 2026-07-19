"""Conformance tests for the Unscented Kalman Filter (book Section 3.2.4).

Verifies the sigma-point weights (Eq. 3.24) and the predict/update pipeline
(Eqs. 3.25-3.30). Locks the key property that the UKF reduces to the linear
Kalman filter on a linear system -- which requires re-drawing sigma points from
the predicted (x_hat^-, P^-) (see docs/book_errata.md E-02).
"""

import numpy as np

from core.estimators.kalman_filter import KalmanFilter
from core.estimators.unscented_kalman_filter import UnscentedKalmanFilter


def test_sigma_point_weights_match_eq_3_24():
    """Weights: W0^m = lam/(n+lam), W0^c adds (1-a^2+b), rest 1/(2(n+lam))."""
    n = 3
    alpha, beta, kappa = 0.5, 2.0, 0.0
    ukf = UnscentedKalmanFilter(
        lambda x, u, dt: x, lambda x: x,
        lambda dt: np.zeros((n, n)), lambda: np.zeros((n, n)),
        np.zeros(n), np.eye(n), alpha=alpha, beta=beta, kappa=kappa,
    )
    lam = alpha**2 * (n + kappa) - n
    assert np.isclose(ukf.Wm[0], lam / (n + lam))
    assert np.isclose(ukf.Wc[0], lam / (n + lam) + (1 - alpha**2 + beta))
    assert np.allclose(ukf.Wm[1:], 1.0 / (2 * (n + lam)))
    assert np.allclose(ukf.Wc[1:], 1.0 / (2 * (n + lam)))
    # Mean weights sum to 1.
    assert np.isclose(ukf.Wm.sum(), 1.0)


def test_ukf_matches_kf_on_linear_system():
    """A correct UKF must equal the linear KF exactly on a linear system.

    Locks E-02: the update re-draws sigma points from (x_hat^-, P^-), so Q is
    included in the innovation covariance. Reusing chi_i^- (book Eq. 3.27) would
    break this equality.
    """
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.1, 0.0], [0.0, 0.1]])
    R = np.array([[0.5]])
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2) * 0.5

    kf = KalmanFilter(F, Q, H, R, x0.copy(), P0.copy())
    ukf = UnscentedKalmanFilter(
        lambda x, u, dt: F @ x, lambda x: H @ x,
        lambda dt: Q, lambda: R, x0.copy(), P0.copy(),
        alpha=1.0, beta=2.0, kappa=0.0,
    )
    for z_val in [0.7, 1.9, 2.5, 3.1]:
        z = np.array([z_val])
        kf.predict()
        kf.update(z)
        ukf.predict()
        ukf.update(z)
        np.testing.assert_allclose(ukf.state, kf.state, atol=1e-9)
        np.testing.assert_allclose(ukf.covariance, kf.covariance, atol=1e-9)


def test_ukf_range_tracking_converges():
    """Sanity: 2D constant-velocity tracked from two-anchor range (observable)."""
    dt = 0.1
    anchors = np.array([[0.0, 0.0], [20.0, 0.0]])

    def f(x, u, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        return F @ x

    def h(x):
        return np.linalg.norm(anchors - x[:2], axis=1)

    def Q(dt):
        return 0.02 * np.eye(4)

    def R():
        return 0.25 * np.eye(2)

    x_true = np.array([10.0, 5.0, -0.5, 0.3])
    ukf = UnscentedKalmanFilter(
        f, h, Q, R, np.array([8.0, 7.0, 0.0, 0.0]), np.diag([5.0, 5.0, 2.0, 2.0])
    )
    rng = np.random.default_rng(0)
    for _ in range(60):
        x_true = f(x_true, None, dt) + rng.multivariate_normal(np.zeros(4), Q(dt))
        z = h(x_true) + rng.multivariate_normal(np.zeros(2), R())
        ukf.predict(dt=dt)
        ukf.update(z)
    assert np.linalg.norm(ukf.state[:2] - x_true[:2]) < 2.0
    assert np.all(np.linalg.eigvalsh(ukf.covariance) > 0)
