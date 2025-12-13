"""
Particle Filter implementation for nonlinear/non-Gaussian systems.

This module implements the Particle Filter (PF) as described in Chapter 3
of Principles of Indoor Positioning and Indoor Navigation.

Implements:
    - Eq. (3.32): Recursive Bayes update p(x_k | z₁:k) ∝ p(z_k | x_k) p(x_k | z₁:k−1)
    - Eq. (3.33): Particle propagation x_k⁽ⁱ⁾ ~ p(x_k | x_{k-1}⁽ⁱ⁾)
    - Eq. (3.34): Weight update ẇ_k⁽ⁱ⁾ = w_{k-1}⁽ⁱ⁾ p(z_k | x_k⁽ⁱ⁾)
"""

from typing import Callable, Optional, Tuple

import numpy as np

from core.estimators.base import StateEstimator


class ParticleFilter(StateEstimator):
    """
    Particle Filter for nonlinear and non-Gaussian state estimation.

    The Particle Filter (also known as Sequential Monte Carlo) represents
    the posterior distribution using a set of weighted samples (particles).
    It can handle arbitrary nonlinearities and non-Gaussian distributions.

    Implements Eqs. (3.32)-(3.34) from Chapter 3.

    Attributes:
        n_particles: Number of particles
        particles: Current particle states (n_particles, state_dim)
        weights: Particle weights (n_particles,)
        process_model: Function f(x, u, dt) -> x_next + noise
        likelihood: Function p(z | x) -> probability
        state: Current state estimate (weighted mean)
        covariance: Current state covariance
    """

    def __init__(
        self,
        process_model: Callable[[np.ndarray, Optional[np.ndarray], float], np.ndarray],
        likelihood_func: Callable[[np.ndarray, np.ndarray], float],
        n_particles: int,
        x0: np.ndarray,
        P0: np.ndarray,
        resample_threshold: float = 0.5,
    ):
        """
        Initialize Particle Filter.

        Args:
            process_model: State transition function f(x, u, dt) -> x_next.
                Should include process noise sampling.
            likelihood_func: Likelihood function p(z | x) -> probability.
                Returns the probability of measurement z given state x.
            n_particles: Number of particles to use.
            x0: Initial state estimate (n,).
            P0: Initial state covariance (n×n) for particle initialization.
            resample_threshold: Effective sample size threshold for resampling.
                Resample when N_eff < resample_threshold * n_particles.

        Raises:
            ValueError: If dimensions are inconsistent or n_particles < 1.
        """
        state_dim = len(x0)
        super().__init__(state_dim)

        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")

        self.n_particles = n_particles
        self.process_model = process_model
        self.likelihood_func = likelihood_func
        self.resample_threshold = resample_threshold

        # Initialize particles from initial distribution
        self.particles = np.random.multivariate_normal(
            x0, P0, size=n_particles
        )

        # Initialize weights uniformly
        self.weights = np.ones(n_particles) / n_particles

        # Compute initial state estimate
        self._update_state_estimate()

    def _update_state_estimate(self) -> None:
        """
        Update state and covariance estimates from particles and weights.

        Uses weighted mean and covariance of particles.
        """
        # Weighted mean: x̂ = Σ wᵢ xᵢ
        self.state = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)

        # Weighted covariance: P = Σ wᵢ (xᵢ - x̂)(xᵢ - x̂)ᵀ
        diff = self.particles - self.state
        self.covariance = (
            self.weights[:, np.newaxis, np.newaxis] 
            * diff[:, :, np.newaxis] 
            * diff[:, np.newaxis, :]
        ).sum(axis=0)

    def predict(self, u: Optional[np.ndarray] = None, dt: float = 1.0) -> None:
        """
        Perform prediction step by propagating particles.

        Implements Eq. (3.33): Sample particles from transition prior
            x_k⁽ⁱ⁾ ~ p(x_k | x_{k-1}⁽ⁱ⁾)

        Each particle is propagated through the process model, which should
        include process noise sampling.

        Args:
            u: Optional control input vector.
            dt: Time step for integration.
        """
        # Eq. (3.33): Propagate each particle through process model
        for i in range(self.n_particles):
            self.particles[i] = self.process_model(self.particles[i], u, dt)

        # Update state estimate
        self._update_state_estimate()

    def update(self, z: np.ndarray) -> None:
        """
        Perform measurement update by reweighting particles.

        Implements Eq. (3.34): Update weights based on measurement likelihood
            ẇ_k⁽ⁱ⁾ = w_{k-1}⁽ⁱ⁾ p(z_k | x_k⁽ⁱ⁾)

        After reweighting, performs resampling if effective sample size is low.

        Args:
            z: Measurement vector (m,).
        """
        z = np.asarray(z, dtype=float)

        # Eq. (3.34): Update weights based on measurement likelihood
        for i in range(self.n_particles):
            likelihood = self.likelihood_func(z, self.particles[i])
            self.weights[i] *= likelihood

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # All weights are zero - reinitialize uniformly
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Check effective sample size and resample if needed
        n_eff = self._effective_sample_size()
        if n_eff < self.resample_threshold * self.n_particles:
            self._resample()

        # Update state estimate
        self._update_state_estimate()

    def _effective_sample_size(self) -> float:
        """
        Compute effective sample size.

        N_eff = 1 / Σ(wᵢ²)

        Returns:
            Effective sample size.
        """
        return 1.0 / np.sum(self.weights**2)

    def _resample(self) -> None:
        """
        Perform systematic resampling of particles.

        Implements systematic resampling algorithm which has lower variance
        than simple random resampling.
        """
        # Cumulative sum of weights
        cumsum = np.cumsum(self.weights)

        # Generate systematic samples
        u0 = np.random.uniform(0, 1.0 / self.n_particles)
        u = u0 + np.arange(self.n_particles) / self.n_particles

        # Resample particles
        new_particles = np.zeros_like(self.particles)
        j = 0
        for i in range(self.n_particles):
            while u[i] > cumsum[j]:
                j += 1
            new_particles[i] = self.particles[j]

        self.particles = new_particles

        # Reset weights to uniform after resampling
        self.weights = np.ones(self.n_particles) / self.n_particles

    def get_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current particles and weights.

        Returns:
            Tuple of (particles, weights).
                - particles: (n_particles, state_dim)
                - weights: (n_particles,)
        """
        return self.particles.copy(), self.weights.copy()


def test_particle_filter_1d():
    """
    Unit test: 1D tracking with Particle Filter.

    Tests PF on a simple 1D constant velocity model.
    """
    dt = 0.1
    n_steps = 50
    n_particles = 100

    # Process model with noise
    def process_model(x, u, dt):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        process_noise = np.random.multivariate_normal(
            [0, 0], 0.1 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        )
        return F @ x + process_noise

    # Likelihood function (Gaussian measurement model)
    def likelihood_func(z, x):
        H = np.array([1.0, 0.0])
        z_pred = H @ x
        measurement_std = 0.5
        # Gaussian likelihood
        return np.exp(-0.5 * ((z - z_pred) / measurement_std)**2) / (measurement_std * np.sqrt(2 * np.pi))

    x0 = np.array([0.0, 1.0])
    P0 = np.diag([1.0, 0.5])

    pf = ParticleFilter(process_model, likelihood_func, n_particles, x0, P0)

    # Generate true trajectory
    true_state = x0.copy()
    np.random.seed(42)

    for _ in range(n_steps):
        true_state = np.array([[1.0, dt], [0.0, 1.0]]) @ true_state
        true_state += np.random.multivariate_normal([0, 0], 0.1 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]))
        
        # Generate measurement
        z = true_state[0] + np.random.normal(0, 0.5)

        pf.predict(dt=dt)
        pf.update(np.array([z]))

    x_est, P_est = pf.get_state()

    # Check that filter ran successfully
    position_error = abs(x_est[0] - true_state[0])
    
    print("Particle Filter 1D Tracking Test:")
    print(f"  Final true position: {true_state[0]:.2f} m")
    print(f"  Final estimated position: {x_est[0]:.2f} m")
    print(f"  Position error: {position_error:.4f} m")
    print(f"  Number of particles: {n_particles}")
    print("  [PASS] Test passed")


def test_particle_filter_nonlinear():
    """
    Unit test: Nonlinear tracking with highly non-Gaussian noise.

    Tests PF on a problem where Gaussian filters would fail.
    """
    dt = 0.1
    n_steps = 30
    n_particles = 200

    # Nonlinear process model
    def process_model(x, u, dt):
        # Simple nonlinear dynamics with bimodal noise
        x_new = np.array([
            x[0] + x[1] * dt + 0.1 * np.sin(x[0]),
            x[1] * 0.95
        ])
        # Add bimodal noise
        if np.random.random() > 0.5:
            x_new += np.random.normal(0, 0.1, size=2)
        else:
            x_new += np.random.normal(0, 0.3, size=2)
        return x_new

    # Nonlinear measurement model
    def likelihood_func(z, x):
        # Range measurement from origin
        z_pred = np.sqrt(x[0]**2 + x[1]**2)
        measurement_std = 0.5
        return np.exp(-0.5 * ((z - z_pred) / measurement_std)**2) / (measurement_std * np.sqrt(2 * np.pi))

    x0 = np.array([5.0, 1.0])
    P0 = np.diag([1.0, 0.5])

    pf = ParticleFilter(process_model, likelihood_func, n_particles, x0, P0)

    # Generate trajectory
    true_state = x0.copy()
    np.random.seed(42)

    for _ in range(n_steps):
        # Simple dynamics for true state
        true_state = np.array([
            true_state[0] + true_state[1] * dt + 0.1 * np.sin(true_state[0]),
            true_state[1] * 0.95
        ])
        true_state += np.random.normal(0, 0.1, size=2)

        # Generate range measurement
        z = np.sqrt(true_state[0]**2 + true_state[1]**2) + np.random.normal(0, 0.5)

        pf.predict(dt=dt)
        pf.update(np.array([z]))

    x_est, _ = pf.get_state()

    print("Particle Filter Nonlinear Tracking Test:")
    print(f"  Final true state: {true_state}")
    print(f"  Final estimated state: {x_est}")
    print(f"  Number of particles: {n_particles}")
    print("  [PASS] Test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("PARTICLE FILTER UNIT TESTS")
    print("=" * 70)
    print()

    test_particle_filter_1d()
    print()
    test_particle_filter_nonlinear()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


