"""Conformance tests for the linear Kalman filter (book Section 3.2.1).

Verifies the five Kalman-filter equations (3.11)/(3.12) predict and
(3.17)/(3.18)/(3.19) update, and locks the covariance update to the correct
form ``(I - K H) P`` rather than the printed Eqs. (3.19)/(3.20), which contain
a spurious ``F_k`` (see docs/book_errata.md, E-01).
"""

import numpy as np

from core.estimators.kalman_filter import KalmanFilter


def _setup():
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[0.05, 0.0], [0.0, 0.05]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.3]])
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2) * 0.5
    return F, Q, H, R, x0, P0


def test_predict_matches_eqs_3_11_3_12():
    """Predict: x = F x + u (3.11); P = F P F^T + Q (3.12)."""
    F, Q, H, R, x0, P0 = _setup()
    u = np.array([0.1, -0.2])
    kf = KalmanFilter(F, Q, H, R, x0, P0)
    kf.predict(u=u)
    np.testing.assert_allclose(kf.state, F @ x0 + u, atol=1e-12)
    np.testing.assert_allclose(kf.covariance, F @ P0 @ F.T + Q, atol=1e-12)


def test_update_matches_eqs_3_17_3_18():
    """Update: gain (3.18) and state (3.17) match the closed forms."""
    F, Q, H, R, x0, P0 = _setup()
    kf = KalmanFilter(F, Q, H, R, x0, P0)
    kf.predict()
    x_pred, P_pred = kf.state.copy(), kf.covariance.copy()
    z = np.array([0.7])

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_expected = x_pred + K @ (z - H @ x_pred)

    kf.update(z)
    np.testing.assert_allclose(kf.state, x_expected, atol=1e-12)


def test_covariance_update_is_symmetric_and_correct():
    """Lock E-01: posterior P equals (I - K H) P_pred and stays symmetric.

    The printed Eqs. (3.19)/(3.20) 'P - F K H P' would be non-symmetric here
    (F != I); this test guards against reverting to that invalid form.
    """
    F, Q, H, R, x0, P0 = _setup()
    kf = KalmanFilter(F, Q, H, R, x0, P0)
    kf.predict()
    P_pred = kf.covariance.copy()

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    P_correct = (np.eye(2) - K @ H) @ P_pred
    P_book_typo = P_pred - F @ K @ H @ P_pred  # spurious F_k

    kf.update(np.array([0.7]))

    # Correct form, and a valid (symmetric, PSD) covariance.
    np.testing.assert_allclose(kf.covariance, P_correct, atol=1e-12)
    np.testing.assert_allclose(kf.covariance, kf.covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(kf.covariance) > 0)
    # The book-typo form is different (and not symmetric) -> must NOT be used.
    assert not np.allclose(kf.covariance, P_book_typo)
    assert not np.allclose(P_book_typo, P_book_typo.T)


def test_constant_velocity_converges():
    """Sanity: 1D constant-velocity tracking converges."""
    F, Q, H, R, x0, P0 = _setup()
    kf = KalmanFilter(F, Q, H, R, x0, P0)
    rng = np.random.default_rng(42)
    x_true = x0.copy()
    for _ in range(60):
        x_true = F @ x_true + rng.multivariate_normal([0, 0], Q)
        z = H @ x_true + rng.normal(0, np.sqrt(R[0, 0]))
        kf.predict()
        kf.update(z)
    assert abs(kf.state[0] - x_true[0]) < 0.6
    assert abs(kf.state[1] - x_true[1]) < 0.4
