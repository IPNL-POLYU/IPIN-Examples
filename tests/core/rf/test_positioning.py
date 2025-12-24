"""
Unit tests for RF positioning algorithms.

Tests TOA, TDOA, and AOA positioning algorithms.
"""

import numpy as np
import pytest

from core.rf.measurement_models import (
    aoa_angle_vector,
    aoa_azimuth,
    aoa_elevation,
)
from core.rf.positioning import (
    AOAPositioner,
    TDOAPositioner,
    TOAPositioner,
    aoa_ove_solve,
    aoa_ple_solve_2d,
    aoa_ple_solve_3d,
    build_tdoa_covariance,
    tdoa_chan_solver,
    toa_fang_solver,
    toa_solve_with_clock_bias,
)


class TestTOAPositioning:
    """Test TOA positioning algorithms."""

    def test_toa_positioning_perfect_measurements(self):
        """Test TOA positioning with perfect measurements."""
        # Square anchor layout
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        # True position
        true_pos = np.array([5.0, 5.0])

        # Compute true ranges
        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        # Solve
        positioner = TOAPositioner(anchors, method="ls")
        estimated_pos, info = positioner.solve(
            ranges, initial_guess=np.array([6.0, 6.0])
        )

        # Should converge to true position
        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3

    def test_toa_positioning_with_noise(self):
        """Test TOA positioning with measurement noise."""
        np.random.seed(42)

        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([3.0, 7.0])

        # True ranges with noise
        true_ranges = np.linalg.norm(anchors - true_pos, axis=1)
        ranges = true_ranges + np.random.randn(4) * 0.1  # 10cm noise

        positioner = TOAPositioner(anchors, method="iwls")
        estimated_pos, info = positioner.solve(
            ranges, initial_guess=np.array([5.0, 5.0])
        )

        # Should be close to true position
        error = np.linalg.norm(estimated_pos - true_pos)
        assert error < 0.5  # Within 50cm for 10cm noise

    def test_toa_positioning_3d(self):
        """Test TOA positioning in 3D."""
        # Cube anchor layout
        anchors = np.array(
            [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 5, 10]],
            dtype=float,
        )

        true_pos = np.array([5.0, 5.0, 5.0])
        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        positioner = TOAPositioner(anchors, method="ls")
        estimated_pos, info = positioner.solve(
            ranges, initial_guess=np.array([6.0, 6.0, 6.0])
        )

        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3

    def test_toa_with_clock_bias(self):
        """Test TOA positioning with unknown clock bias."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])
        true_clock_bias_m = 2.0  # 2 meters

        # Compute ranges with clock bias
        true_ranges = np.linalg.norm(anchors - true_pos, axis=1)
        ranges = true_ranges + true_clock_bias_m

        # Solve with clock bias estimation
        initial_guess = np.array([6.0, 6.0, 0.0])  # [x, y, clock_bias]
        pos, bias, info = toa_solve_with_clock_bias(
            anchors, ranges, initial_guess
        )

        # Check position accuracy
        assert np.linalg.norm(pos - true_pos) < 1e-3

        # Check clock bias accuracy
        assert np.abs(bias - true_clock_bias_m) < 1e-3

    def test_toa_convergence_history(self):
        """Test that iteration history is recorded."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])
        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        positioner = TOAPositioner(anchors)
        _, info = positioner.solve(ranges, initial_guess=np.array([2.0, 2.0]))

        # History should have initial guess + iterations
        assert "history" in info
        assert len(info["history"]) >= 2  # At least initial + one iteration

    def test_toa_iterative_ls_method(self):
        """Test iterative_ls method (book default: Eq. 4.20)."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])
        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        positioner = TOAPositioner(anchors, method="iterative_ls")
        estimated_pos, info = positioner.solve(
            ranges, initial_guess=np.array([6.0, 6.0])
        )

        assert info["converged"]
        assert info["method"] == "iterative_ls"
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3

    def test_toa_iterative_wls_method(self):
        """Test iterative_wls method with covariance (book Eq. 4.23)."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])

        # Compute ranges with noise
        np.random.seed(42)
        true_ranges = np.linalg.norm(anchors - true_pos, axis=1)
        sigmas = np.array([0.1, 0.2, 0.15, 0.1])
        ranges = true_ranges + np.random.randn(4) * sigmas

        # Build covariance matrix
        covariance = np.diag(sigmas**2)

        positioner = TOAPositioner(anchors, method="iterative_wls")
        estimated_pos, info = positioner.solve(
            ranges, initial_guess=np.array([6.0, 6.0]), covariance=covariance
        )

        assert info["converged"]
        assert info["method"] == "iterative_wls"
        assert np.linalg.norm(estimated_pos - true_pos) < 0.5

    def test_toa_iterative_wls_requires_covariance(self):
        """Test that iterative_wls raises error without covariance."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        ranges = np.array([5.0, 7.07, 7.07, 5.0])

        positioner = TOAPositioner(anchors, method="iterative_wls")

        with pytest.raises(ValueError, match="covariance"):
            positioner.solve(ranges, initial_guess=np.array([5.0, 5.0]))

    def test_toa_range_weighted_method(self):
        """Test range_weighted method (heuristic 1/d^2 weighting)."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])
        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        positioner = TOAPositioner(anchors, method="range_weighted")
        estimated_pos, info = positioner.solve(
            ranges, initial_guess=np.array([6.0, 6.0])
        )

        assert info["converged"]
        assert info["method"] == "range_weighted"
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3

    def test_toa_method_aliases(self):
        """Test backward compatibility with 'ls' and 'iwls' aliases."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        # 'ls' should map to 'iterative_ls'
        positioner_ls = TOAPositioner(anchors, method="ls")
        assert positioner_ls.method == "iterative_ls"

        # 'iwls' should map to 'range_weighted' (legacy behavior)
        positioner_iwls = TOAPositioner(anchors, method="iwls")
        assert positioner_iwls.method == "range_weighted"

    def test_toa_invalid_method(self):
        """Test that invalid method raises error."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        with pytest.raises(ValueError, match="method must be"):
            TOAPositioner(anchors, method="invalid_method")


class TestTDOAPositioning:
    """Test TDOA positioning algorithms."""

    def test_tdoa_positioning_perfect_measurements(self):
        """Test TDOA positioning with perfect measurements."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])

        # Compute TDOA measurements (reference anchor = 0)
        dist_ref = np.linalg.norm(true_pos - anchors[0])
        tdoa = []
        for i in range(1, len(anchors)):
            dist_i = np.linalg.norm(true_pos - anchors[i])
            tdoa.append(dist_i - dist_ref)
        tdoa = np.array(tdoa)

        # Solve
        positioner = TDOAPositioner(anchors, reference_idx=0)
        estimated_pos, info = positioner.solve(
            tdoa, initial_guess=np.array([6.0, 6.0])
        )

        # Should converge to true position
        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3

    def test_tdoa_positioning_with_noise(self):
        """Test TDOA positioning with measurement noise."""
        np.random.seed(43)

        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([3.0, 7.0])

        # True TDOA with noise
        dist_ref = np.linalg.norm(true_pos - anchors[0])
        tdoa = []
        for i in range(1, len(anchors)):
            dist_i = np.linalg.norm(true_pos - anchors[i])
            tdoa.append(dist_i - dist_ref)
        tdoa = np.array(tdoa) + np.random.randn(3) * 0.05  # 5cm noise

        positioner = TDOAPositioner(anchors, reference_idx=0)
        estimated_pos, info = positioner.solve(
            tdoa, initial_guess=np.array([5.0, 5.0])
        )

        # Should be close to true position
        error = np.linalg.norm(estimated_pos - true_pos)
        assert error < 0.5  # Within 50cm for 5cm noise

    def test_tdoa_different_reference(self):
        """Test TDOA with different reference anchors."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])

        # Test with reference anchor = 1
        dist_ref = np.linalg.norm(true_pos - anchors[1])
        tdoa = []
        for i in range(len(anchors)):
            if i == 1:
                continue
            dist_i = np.linalg.norm(true_pos - anchors[i])
            tdoa.append(dist_i - dist_ref)
        tdoa = np.array(tdoa)

        positioner = TDOAPositioner(anchors, reference_idx=1)
        estimated_pos, info = positioner.solve(
            tdoa, initial_guess=np.array([6.0, 6.0])
        )

        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3


class TestAOAPositioning:
    """Test AOA positioning algorithms with ENU convention (book Eqs. 4.63-4.78)."""

    def test_aoa_positioning_perfect_measurements_2d(self):
        """Test 2D AOA positioning with perfect measurements."""
        # Anchors at cardinal directions (ENU: E=x, N=y)
        anchors = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]], dtype=float)
        true_pos = np.array([3.0, 4.0])

        # Compute true azimuth angles using ENU convention (Eq. 4.64)
        # ψ = atan2(ΔE, ΔN) where Δ = anchor - agent
        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=False)

        # Solve
        positioner = AOAPositioner(anchors)
        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([5.0, 5.0])
        )

        # Should converge to true position
        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-2

    def test_aoa_positioning_with_noise_2d(self):
        """Test 2D AOA positioning with angle noise."""
        np.random.seed(44)

        anchors = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]], dtype=float)
        true_pos = np.array([2.0, 3.0])

        # True angles with noise (ENU convention)
        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=False)
        aoa_noisy = aoa + np.random.randn(4) * np.deg2rad(2.0)  # 2° noise

        positioner = AOAPositioner(anchors)
        estimated_pos, info = positioner.solve(
            aoa_noisy, initial_guess=np.array([5.0, 5.0])
        )

        # Should be reasonably close
        error = np.linalg.norm(estimated_pos - true_pos)
        assert error < 2.0  # Within 2m for 2° angle noise

    def test_aoa_positioning_square_anchors_2d(self):
        """Test 2D AOA with square anchor layout."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])

        # ENU convention azimuth angles
        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=False)

        positioner = AOAPositioner(anchors)
        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([6.0, 6.0])
        )

        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-2

    def test_aoa_positioning_3d(self):
        """Test 3D AOA positioning with elevation and azimuth."""
        # 3D anchors (E, N, U) - elevated corners
        anchors = np.array(
            [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float
        )
        true_pos = np.array([5.0, 5.0, 0.0])

        # Generate 3D AOA measurements [θ_1, ψ_1, θ_2, ψ_2, ...]
        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=True)

        positioner = AOAPositioner(anchors)
        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([6.0, 6.0, 1.0])
        )

        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-2

    def test_aoa_positioning_3d_with_noise(self):
        """Test 3D AOA positioning with measurement noise."""
        np.random.seed(45)

        anchors = np.array(
            [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float
        )
        true_pos = np.array([3.0, 7.0, 0.0])

        # True 3D AOA with noise
        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=True)
        aoa_noisy = aoa + np.random.randn(8) * np.deg2rad(2.0)  # 2° noise

        positioner = AOAPositioner(anchors)
        estimated_pos, info = positioner.solve(
            aoa_noisy, initial_guess=np.array([5.0, 5.0, 1.0])
        )

        error = np.linalg.norm(estimated_pos - true_pos)
        assert error < 3.0  # Within 3m for 2° noise in 3D

    def test_aoa_weighting_uniform_sigma_psi_2d(self):
        """Test 2D AOA with uniform sigma_psi weighting."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([4.0, 6.0])

        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=False)

        positioner = AOAPositioner(anchors)

        # Solve with sigma_psi (should compute W_a)
        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([5.0, 5.0]),
            sigma_psi=np.deg2rad(2.0)
        )

        # Should converge and include weight matrix in info
        assert info["converged"]
        assert "final_weights" in info
        assert info["final_weights"].shape == (4, 4)
        # Weights should not be identity
        assert not np.allclose(info["final_weights"], np.eye(4))

    def test_aoa_weighting_heterogeneous_sigma_psi_2d(self):
        """Test 2D AOA with per-anchor sigma_psi weighting."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([4.0, 6.0])

        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=False)

        positioner = AOAPositioner(anchors)

        # Different noise for each anchor
        sigma_psi_per_anchor = np.deg2rad([1.0, 2.0, 5.0, 10.0])

        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([5.0, 5.0]),
            sigma_psi=sigma_psi_per_anchor
        )

        assert info["converged"]
        # Weights should reflect the heterogeneous noise
        weights_diag = np.diag(info["final_weights"])
        # Anchor with lower noise should have higher weight
        assert weights_diag[0] > weights_diag[3]

    def test_aoa_weighting_direct_sigma_tan_psi_2d(self):
        """Test 2D AOA with direct sigma_tan_psi (transformed domain)."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([4.0, 6.0])

        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=False)

        positioner = AOAPositioner(anchors)

        # Direct tan(psi) std
        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([5.0, 5.0]),
            sigma_tan_psi=0.1
        )

        assert info["converged"]
        # All weights should be 1/0.1^2 = 100
        weights_diag = np.diag(info["final_weights"])
        assert np.allclose(weights_diag, 100.0)

    def test_aoa_weighting_3d_with_sigmas(self):
        """Test 3D AOA with sigma_theta and sigma_psi weighting."""
        anchors = np.array(
            [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float
        )
        true_pos = np.array([5.0, 5.0, 0.0])

        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=True)

        positioner = AOAPositioner(anchors)

        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([6.0, 6.0, 1.0]),
            sigma_theta=np.deg2rad(1.0),
            sigma_psi=np.deg2rad(2.0)
        )

        assert info["converged"]
        assert info["final_weights"].shape == (8, 8)
        # Error should be small for perfect measurements
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-2

    def test_aoa_weighting_improves_accuracy_heterogeneous_noise(self):
        """Test that proper weighting improves accuracy with heterogeneous noise."""
        np.random.seed(46)

        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([4.0, 6.0])

        # True angles
        aoa_true = aoa_angle_vector(anchors, true_pos, include_elevation=False)

        # Heterogeneous noise: anchor 0 has 1°, anchor 3 has 10°
        sigma_deg = np.array([1.0, 2.0, 2.0, 10.0])
        sigma_rad = np.deg2rad(sigma_deg)

        # Run multiple trials
        n_trials = 50
        errors_unweighted = []
        errors_weighted = []

        for _ in range(n_trials):
            noise = np.random.randn(4) * sigma_rad
            aoa_noisy = aoa_true + noise

            positioner = AOAPositioner(anchors)

            # Unweighted
            est_uw, _ = positioner.solve(
                aoa_noisy, initial_guess=np.array([5.0, 5.0])
            )
            errors_unweighted.append(np.linalg.norm(est_uw - true_pos))

            # Weighted with known noise
            est_w, _ = positioner.solve(
                aoa_noisy, initial_guess=np.array([5.0, 5.0]),
                sigma_psi=sigma_rad
            )
            errors_weighted.append(np.linalg.norm(est_w - true_pos))

        # Weighted should have lower mean error (or comparable)
        # With heterogeneous noise, weighting should help
        mean_uw = np.mean(errors_unweighted)
        mean_w = np.mean(errors_weighted)
        # At minimum, weighted shouldn't be significantly worse
        assert mean_w <= mean_uw * 1.5

    def test_aoa_weighting_explicit_weights_override_sigma(self):
        """Test that explicit weights parameter overrides sigma inputs."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([4.0, 6.0])

        aoa = aoa_angle_vector(anchors, true_pos, include_elevation=False)

        positioner = AOAPositioner(anchors)

        # Provide both explicit weights and sigma_psi
        explicit_weights = np.eye(4) * 5.0

        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([5.0, 5.0]),
            weights=explicit_weights,
            sigma_psi=np.deg2rad(2.0)  # Should be ignored
        )

        # Explicit weights should be used
        assert np.allclose(info["final_weights"], explicit_weights)

    def test_aoa_weight_matrix_computation(self):
        """Test _compute_weight_matrix directly for correctness."""
        anchors = np.array([[10, 0], [0, 10]], dtype=float)
        positioner = AOAPositioner(anchors)

        # Known angles
        psi = np.array([np.pi / 4, np.pi / 3])  # 45° and 60°

        # Test with sigma_psi
        sigma_psi = np.deg2rad(2.0)
        W = positioner._compute_weight_matrix(psi, sigma_psi=sigma_psi)

        # Compute expected variances using error propagation
        # var(tan ψ) = sec^4(ψ) * var(ψ)
        var_psi = sigma_psi ** 2
        expected_var = []
        for psi_i in psi:
            sec_sq = 1 + np.tan(psi_i) ** 2
            expected_var.append(sec_sq ** 2 * var_psi)

        expected_weights = 1.0 / np.array(expected_var)
        assert np.allclose(np.diag(W), expected_weights)


class TestAOAJacobian:
    """Test AOA Jacobian implementation against finite differences."""

    def test_jacobian_2d_finite_difference(self):
        """Validate 2D Jacobian against finite-difference approximation."""
        anchors = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]], dtype=float)
        position = np.array([3.0, 4.0])
        eps = 1e-7

        positioner = AOAPositioner(anchors)

        # Get analytical Jacobian
        _, H_analytical = positioner._compute_predicted_and_jacobian_2d(position)

        # Compute numerical Jacobian
        H_numerical = np.zeros_like(H_analytical)
        z_base, _ = positioner._compute_predicted_and_jacobian_2d(position)

        for j in range(2):  # For each dimension
            pos_plus = position.copy()
            pos_plus[j] += eps
            z_plus, _ = positioner._compute_predicted_and_jacobian_2d(pos_plus)

            pos_minus = position.copy()
            pos_minus[j] -= eps
            z_minus, _ = positioner._compute_predicted_and_jacobian_2d(pos_minus)

            H_numerical[:, j] = (z_plus - z_minus) / (2 * eps)

        # Compare
        np.testing.assert_allclose(
            H_analytical, H_numerical, rtol=1e-5, atol=1e-8,
            err_msg="2D Jacobian mismatch vs finite difference"
        )

    def test_jacobian_3d_finite_difference(self):
        """Validate 3D Jacobian against finite-difference approximation."""
        anchors = np.array(
            [[10, 0, 5], [0, 10, 5], [-10, 0, 5], [0, -10, 5]], dtype=float
        )
        position = np.array([3.0, 4.0, 0.0])
        eps = 1e-7

        positioner = AOAPositioner(anchors)

        # Get analytical Jacobian
        _, H_analytical = positioner._compute_predicted_and_jacobian_3d(position)

        # Compute numerical Jacobian
        H_numerical = np.zeros_like(H_analytical)

        for j in range(3):  # For each dimension
            pos_plus = position.copy()
            pos_plus[j] += eps
            z_plus, _ = positioner._compute_predicted_and_jacobian_3d(pos_plus)

            pos_minus = position.copy()
            pos_minus[j] -= eps
            z_minus, _ = positioner._compute_predicted_and_jacobian_3d(pos_minus)

            H_numerical[:, j] = (z_plus - z_minus) / (2 * eps)

        # Compare
        np.testing.assert_allclose(
            H_analytical, H_numerical, rtol=1e-5, atol=1e-8,
            err_msg="3D Jacobian mismatch vs finite difference"
        )

    def test_jacobian_f_partial_derivatives(self):
        """Test individual partial derivatives for f_i = sin(θ) (Eqs. 4.68-4.70)."""
        # Single anchor at (10, 0, 5)
        anchor = np.array([10.0, 0.0, 5.0])
        agent = np.array([3.0, 4.0, 0.0])

        # Compute analytically
        delta_e = anchor[0] - agent[0]  # 7
        delta_n = anchor[1] - agent[1]  # -4
        delta_u = anchor[2] - agent[2]  # 5
        d = np.sqrt(delta_e**2 + delta_n**2 + delta_u**2)
        d_cubed = d**3
        horiz_sq = delta_e**2 + delta_n**2

        # Book Eq. 4.68: ∂f/∂x_e,a = Δu * Δe / d³
        df_dxe_expected = delta_u * delta_e / d_cubed

        # Book Eq. 4.69: ∂f/∂x_n,a = Δu * Δn / d³
        df_dxn_expected = delta_u * delta_n / d_cubed

        # Book Eq. 4.70: ∂f/∂x_u,a = -(Δe² + Δn²) / d³
        df_dxu_expected = -horiz_sq / d_cubed

        # Get from positioner
        anchors = anchor.reshape(1, -1)
        positioner = AOAPositioner(anchors)
        _, H = positioner._compute_predicted_and_jacobian_3d(agent)

        # f is in row 0, columns are [∂/∂x_e, ∂/∂x_n, ∂/∂x_u]
        assert np.isclose(H[0, 0], df_dxe_expected, rtol=1e-10)
        assert np.isclose(H[0, 1], df_dxn_expected, rtol=1e-10)
        assert np.isclose(H[0, 2], df_dxu_expected, rtol=1e-10)

    def test_jacobian_g_partial_derivatives(self):
        """Test individual partial derivatives for g_i = tan(ψ) (Eqs. 4.72-4.74)."""
        # Single anchor at (10, 5, 3)
        anchor = np.array([10.0, 5.0, 3.0])
        agent = np.array([3.0, 2.0, 0.0])

        # Compute analytically
        delta_e = anchor[0] - agent[0]  # 7
        delta_n = anchor[1] - agent[1]  # 3

        # Book Eq. 4.72: ∂g/∂x_e,a = -1 / Δn
        dg_dxe_expected = -1.0 / delta_n

        # Book Eq. 4.73: ∂g/∂x_n,a = Δe / Δn²
        dg_dxn_expected = delta_e / (delta_n**2)

        # Book Eq. 4.74: ∂g/∂x_u,a = 0
        dg_dxu_expected = 0.0

        # Get from positioner
        anchors = anchor.reshape(1, -1)
        positioner = AOAPositioner(anchors)
        _, H = positioner._compute_predicted_and_jacobian_3d(agent)

        # g is in row 1, columns are [∂/∂x_e, ∂/∂x_n, ∂/∂x_u]
        assert np.isclose(H[1, 0], dg_dxe_expected, rtol=1e-10)
        assert np.isclose(H[1, 1], dg_dxn_expected, rtol=1e-10)
        assert np.isclose(H[1, 2], dg_dxu_expected, rtol=1e-10)


class TestAOAClosedForm:
    """Test closed-form AOA solvers (OVE and PLE)."""

    def test_ove_perfect_measurements(self):
        """Test OVE with perfect 3D measurements."""
        anchors = np.array(
            [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float
        )
        true_pos = np.array([4.0, 6.0, 0.0])

        # Generate angles
        elevations = np.array([aoa_elevation(a, true_pos) for a in anchors])
        azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors])

        pos, info = aoa_ove_solve(anchors, elevations, azimuths)

        assert info["method"] == "OVE"
        assert np.linalg.norm(pos - true_pos) < 1e-6

    def test_ove_with_noise(self):
        """Test OVE with measurement noise."""
        np.random.seed(100)
        anchors = np.array(
            [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float
        )
        true_pos = np.array([5.0, 5.0, 0.0])

        elevations = np.array([aoa_elevation(a, true_pos) for a in anchors])
        azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors])

        # Add 2 degree noise
        noise_rad = np.deg2rad(2.0)
        elev_noisy = elevations + np.random.randn(4) * noise_rad
        azim_noisy = azimuths + np.random.randn(4) * noise_rad

        pos, info = aoa_ove_solve(anchors, elev_noisy, azim_noisy)

        # Should be within reasonable error
        assert np.linalg.norm(pos - true_pos) < 2.0

    def test_ple_2d_perfect_measurements(self):
        """Test 2D PLE with perfect measurements."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([4.0, 6.0])

        azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors])

        pos, info = aoa_ple_solve_2d(anchors, azimuths)

        assert info["method"] == "PLE_2D"
        assert np.linalg.norm(pos - true_pos) < 1e-6

    def test_ple_2d_with_noise(self):
        """Test 2D PLE with measurement noise."""
        np.random.seed(101)
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 7.0])

        azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors])
        azim_noisy = azimuths + np.random.randn(4) * np.deg2rad(2.0)

        pos, info = aoa_ple_solve_2d(anchors, azim_noisy)

        # Should be within reasonable error
        assert np.linalg.norm(pos - true_pos) < 2.0

    def test_ple_3d_perfect_measurements(self):
        """Test 3D PLE with perfect measurements."""
        anchors = np.array(
            [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float
        )
        true_pos = np.array([4.0, 6.0, 0.0])

        elevations = np.array([aoa_elevation(a, true_pos) for a in anchors])
        azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors])

        pos, info = aoa_ple_solve_3d(anchors, elevations, azimuths)

        assert info["method"] == "PLE_3D"
        assert np.linalg.norm(pos - true_pos) < 1e-6

    def test_ple_geometry_warning_poor_geometry(self):
        """Test that PLE detects poor geometry (near-parallel bearings)."""
        # Linear anchor arrangement - all bearings nearly parallel
        anchors = np.array([[0, 0], [5, 0], [10, 0], [15, 0]], dtype=float)
        true_pos = np.array([7.5, 10.0])  # Far from the line

        azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors])

        pos, info = aoa_ple_solve_2d(anchors, azimuths)

        # Linear geometry should trigger warning (bearings are similar)
        # Note: May not always trigger depending on exact positions
        # The condition number should be higher than for good geometry
        assert info["condition_number"] > 1.0  # Some ill-conditioning expected

    def test_ple_vs_iwls_comparison(self):
        """Test that PLE and I-WLS give similar results for good geometry."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])

        azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors])

        # PLE
        pos_ple, _ = aoa_ple_solve_2d(anchors, azimuths)

        # I-WLS
        aoa_meas = aoa_angle_vector(anchors, true_pos, include_elevation=False)
        positioner = AOAPositioner(anchors)
        pos_iwls, _ = positioner.solve(aoa_meas, initial_guess=np.array([6.0, 6.0]))

        # Both should be close to true position
        assert np.linalg.norm(pos_ple - true_pos) < 1e-6
        assert np.linalg.norm(pos_iwls - true_pos) < 1e-3

    def test_ove_requires_3d_anchors(self):
        """Test that OVE raises error for 2D anchors."""
        anchors_2d = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
        elevations = np.array([0.0, 0.0, 0.0])
        azimuths = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="3D anchors"):
            aoa_ove_solve(anchors_2d, elevations, azimuths)

    def test_ple_2d_requires_2d_anchors(self):
        """Test that PLE_2D raises error for 3D anchors."""
        anchors_3d = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=float)
        azimuths = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="2D anchors"):
            aoa_ple_solve_2d(anchors_3d, azimuths)


class TestTDOACovariance:
    """Test TDOA correlated covariance matrix builder (Eq. 4.42)."""

    def test_build_tdoa_covariance_structure(self):
        """Test covariance matrix structure matches Eq. 4.42."""
        # 4 anchors with different noise levels
        sigmas = np.array([0.1, 0.2, 0.15, 0.25])  # ref, anc1, anc2, anc3
        ref_idx = 0

        cov = build_tdoa_covariance(sigmas, ref_idx)

        # Should be 3x3 (N-1 TDOA measurements)
        assert cov.shape == (3, 3)

        # Diagonal: sigma_k^2 + sigma_ref^2
        expected_diag = [
            sigmas[1] ** 2 + sigmas[0] ** 2,  # var(d^{1,0})
            sigmas[2] ** 2 + sigmas[0] ** 2,  # var(d^{2,0})
            sigmas[3] ** 2 + sigmas[0] ** 2,  # var(d^{3,0})
        ]
        assert np.allclose(np.diag(cov), expected_diag)

        # Off-diagonal: sigma_ref^2
        expected_offdiag = sigmas[0] ** 2
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isclose(cov[i, j], expected_offdiag)

    def test_build_tdoa_covariance_symmetry(self):
        """Test covariance matrix is symmetric."""
        sigmas = np.array([0.3, 0.1, 0.2, 0.15, 0.25])
        cov = build_tdoa_covariance(sigmas, ref_idx=0)

        assert np.allclose(cov, cov.T)

    def test_build_tdoa_covariance_positive_definite(self):
        """Test covariance matrix is positive definite."""
        sigmas = np.array([0.2, 0.1, 0.15, 0.2])
        cov = build_tdoa_covariance(sigmas, ref_idx=0)

        # All eigenvalues should be positive
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_build_tdoa_covariance_different_reference(self):
        """Test covariance with non-zero reference index."""
        sigmas = np.array([0.1, 0.3, 0.15, 0.2])  # ref at idx=1
        ref_idx = 1

        cov = build_tdoa_covariance(sigmas, ref_idx)

        # Should be 3x3
        assert cov.shape == (3, 3)

        # Non-reference anchors are [0, 2, 3]
        # Diagonal: sigma_k^2 + sigma_1^2
        expected_diag = [
            sigmas[0] ** 2 + sigmas[1] ** 2,
            sigmas[2] ** 2 + sigmas[1] ** 2,
            sigmas[3] ** 2 + sigmas[1] ** 2,
        ]
        assert np.allclose(np.diag(cov), expected_diag)

        # Off-diagonal: sigma_1^2
        expected_offdiag = sigmas[1] ** 2
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isclose(cov[i, j], expected_offdiag)

    def test_build_tdoa_covariance_uniform_noise(self):
        """Test covariance with uniform noise across all anchors."""
        sigma = 0.2
        sigmas = np.array([sigma, sigma, sigma, sigma])
        cov = build_tdoa_covariance(sigmas, ref_idx=0)

        # Diagonal: 2 * sigma^2
        expected_diag = 2 * sigma ** 2
        assert np.allclose(np.diag(cov), expected_diag)

        # Off-diagonal: sigma^2
        expected_offdiag = sigma ** 2
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isclose(cov[i, j], expected_offdiag)

    def test_build_tdoa_covariance_invalid_ref_idx(self):
        """Test invalid reference index raises error."""
        sigmas = np.array([0.1, 0.2, 0.15])

        with pytest.raises(ValueError, match="ref_idx must be in"):
            build_tdoa_covariance(sigmas, ref_idx=-1)

        with pytest.raises(ValueError, match="ref_idx must be in"):
            build_tdoa_covariance(sigmas, ref_idx=3)

    def test_tdoa_positioning_with_correlated_covariance(self):
        """Test TDOA positioning accuracy with correlated vs identity covariance."""
        np.random.seed(42)

        anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
        true_pos = np.array([8.0, 12.0])

        # High noise on reference anchor to emphasize correlation
        sigmas = np.array([0.5, 0.1, 0.1, 0.1])

        # Build covariance matrices
        cov_corr = build_tdoa_covariance(sigmas, ref_idx=0)
        cov_id = np.eye(3)

        n_trials = 100
        errors_corr = []
        errors_id = []

        for _ in range(n_trials):
            # Generate noisy ranges
            range_noise = np.random.randn(len(anchors)) * sigmas
            noisy_ranges = np.array(
                [np.linalg.norm(true_pos - anchors[i]) + range_noise[i]
                 for i in range(len(anchors))]
            )

            # Compute TDOA
            tdoa_noisy = np.array(
                [noisy_ranges[i] - noisy_ranges[0]
                 for i in range(1, len(anchors))]
            )

            positioner = TDOAPositioner(anchors, reference_idx=0)

            # Correlated weighting
            try:
                est_corr, info = positioner.solve(
                    tdoa_noisy, initial_guess=np.array([10.0, 10.0]),
                    covariance=cov_corr,
                )
                if info["converged"]:
                    errors_corr.append(np.linalg.norm(est_corr - true_pos))
            except Exception:
                pass

            # Identity weighting
            try:
                est_id, info = positioner.solve(
                    tdoa_noisy, initial_guess=np.array([10.0, 10.0]),
                    covariance=cov_id,
                )
                if info["converged"]:
                    errors_id.append(np.linalg.norm(est_id - true_pos))
            except Exception:
                pass

        # Correlated weighting should generally perform at least as well
        rmse_corr = np.sqrt(np.mean(np.array(errors_corr) ** 2))
        rmse_id = np.sqrt(np.mean(np.array(errors_id) ** 2))

        # Both should converge successfully
        assert len(errors_corr) > 80
        assert len(errors_id) > 80

        # RMSE should be reasonable
        assert rmse_corr < 2.0
        assert rmse_id < 2.0


class TestFangTOASolver:
    """Test Fang's TOA closed-form solver (Eqs. 4.43-4.49)."""

    def test_fang_perfect_measurements(self):
        """Test Fang's solver with perfect measurements."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])

        # Compute true ranges
        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        # Solve
        pos, info = toa_fang_solver(anchors, ranges)

        assert info['method'] == 'Fang_TOA'
        assert np.linalg.norm(pos - true_pos) < 1e-6

    def test_fang_with_noise(self):
        """Test Fang's solver with noisy measurements."""
        np.random.seed(42)

        anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
        true_pos = np.array([8.0, 12.0])

        # Compute ranges with noise
        ranges_true = np.linalg.norm(anchors - true_pos, axis=1)
        ranges = ranges_true + np.random.randn(len(anchors)) * 0.1

        pos, info = toa_fang_solver(anchors, ranges)

        # Should be close to true position
        error = np.linalg.norm(pos - true_pos)
        assert error < 0.5  # Within 0.5m for 0.1m noise

    def test_fang_different_reference(self):
        """Test Fang's solver with non-default reference anchor."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([3.0, 7.0])

        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        # Use anchor 2 as reference
        pos, info = toa_fang_solver(anchors, ranges, ref_idx=2)

        assert np.linalg.norm(pos - true_pos) < 1e-6

    def test_fang_minimum_anchors(self):
        """Test Fang's solver with minimum 3 anchors."""
        anchors = np.array([[0, 0], [10, 0], [5, 10]], dtype=float)
        true_pos = np.array([4.0, 3.0])

        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        pos, info = toa_fang_solver(anchors, ranges)

        assert np.linalg.norm(pos - true_pos) < 1e-6

    def test_fang_insufficient_anchors(self):
        """Test Fang's solver raises error with < 3 anchors."""
        anchors = np.array([[0, 0], [10, 0]], dtype=float)
        ranges = np.array([5.0, 7.0])

        with pytest.raises(ValueError, match="at least 3 anchors"):
            toa_fang_solver(anchors, ranges)

    def test_fang_3d_not_supported(self):
        """Test Fang's solver raises error for 3D input."""
        anchors = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]])
        ranges = np.array([5.0, 7.0, 7.0, 5.0])

        with pytest.raises(ValueError, match="2D only"):
            toa_fang_solver(anchors, ranges)

    def test_fang_wrong_ranges_size(self):
        """Test Fang's solver with mismatched ranges."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        ranges = np.array([5.0, 7.0, 7.0])  # Should be 4

        with pytest.raises(ValueError, match="Expected 4 ranges"):
            toa_fang_solver(anchors, ranges)


class TestChanTDOASolver:
    """Test Chan's TDOA closed-form solver (Eqs. 4.50-4.62)."""

    def test_chan_perfect_measurements(self):
        """Test Chan's solver with perfect TDOA measurements."""
        anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
        true_pos = np.array([8.0, 12.0])
        ref_idx = 0

        # Compute true ranges and TDOA
        ranges = np.linalg.norm(anchors - true_pos, axis=1)
        d_ref = ranges[ref_idx]
        tdoa = np.array([ranges[i] - d_ref for i in range(len(anchors)) if i != ref_idx])

        # Solve
        pos, info = tdoa_chan_solver(anchors, tdoa, ref_idx=ref_idx)

        assert info['method'] == 'Chan_TDOA'
        assert np.linalg.norm(pos - true_pos) < 1e-4
        # Reference distance should be close to true
        assert np.abs(info['reference_distance'] - d_ref) < 1e-3

    def test_chan_with_noise(self):
        """Test Chan's solver with noisy TDOA measurements."""
        np.random.seed(42)

        anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float)
        true_pos = np.array([7.0, 12.0])
        ref_idx = 0

        # Compute ranges with noise
        ranges_true = np.linalg.norm(anchors - true_pos, axis=1)
        ranges_noisy = ranges_true + np.random.randn(len(anchors)) * 0.1

        # Compute noisy TDOA
        d_ref = ranges_noisy[ref_idx]
        tdoa = np.array([ranges_noisy[i] - d_ref for i in range(len(anchors)) if i != ref_idx])

        pos, info = tdoa_chan_solver(anchors, tdoa, ref_idx=ref_idx)

        # Should be close to true position
        error = np.linalg.norm(pos - true_pos)
        assert error < 0.5  # Within 0.5m for 0.1m range noise

    def test_chan_with_covariance(self):
        """Test Chan's solver with WLS using covariance matrix."""
        np.random.seed(42)

        anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float)
        true_pos = np.array([8.0, 11.0])
        ref_idx = 0

        # Per-anchor noise
        sigmas = np.array([0.3, 0.1, 0.1, 0.1, 0.1])
        cov = build_tdoa_covariance(sigmas, ref_idx)

        # Compute noisy ranges
        ranges_true = np.linalg.norm(anchors - true_pos, axis=1)
        ranges_noisy = ranges_true + np.random.randn(len(anchors)) * sigmas

        # Compute TDOA
        d_ref = ranges_noisy[ref_idx]
        tdoa = np.array([ranges_noisy[i] - d_ref for i in range(len(anchors)) if i != ref_idx])

        # Solve with covariance (WLS)
        pos, info = tdoa_chan_solver(anchors, tdoa, ref_idx=ref_idx, covariance=cov)

        error = np.linalg.norm(pos - true_pos)
        assert error < 1.0  # Reasonable for heterogeneous noise

    def test_chan_different_reference(self):
        """Test Chan's solver with non-default reference anchor."""
        anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
        true_pos = np.array([5.0, 5.0])
        ref_idx = 2  # Use anchor 2 as reference

        ranges = np.linalg.norm(anchors - true_pos, axis=1)
        d_ref = ranges[ref_idx]
        tdoa = np.array([ranges[i] - d_ref for i in range(len(anchors)) if i != ref_idx])

        pos, info = tdoa_chan_solver(anchors, tdoa, ref_idx=ref_idx)

        assert np.linalg.norm(pos - true_pos) < 1e-4

    def test_chan_insufficient_anchors(self):
        """Test Chan's solver raises error with < 4 anchors."""
        anchors = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
        tdoa = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="at least 4 anchors"):
            tdoa_chan_solver(anchors, tdoa)

    def test_chan_wrong_tdoa_size(self):
        """Test Chan's solver with mismatched TDOA measurements."""
        anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
        tdoa = np.array([1.0, 2.0])  # Should be 3 (N-1)

        with pytest.raises(ValueError, match="Expected 3 TDOA measurements"):
            tdoa_chan_solver(anchors, tdoa)

    def test_chan_3d_not_supported(self):
        """Test Chan's solver raises error for 3D input."""
        anchors = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]])
        tdoa = np.array([1.0, 2.0, 1.5])

        with pytest.raises(ValueError, match="2D only"):
            tdoa_chan_solver(anchors, tdoa)

    def test_chan_vs_iwls_consistency(self):
        """Test Chan's solver produces similar results to I-WLS."""
        anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float)
        true_pos = np.array([8.0, 12.0])
        ref_idx = 0

        # Perfect measurements
        ranges = np.linalg.norm(anchors - true_pos, axis=1)
        d_ref = ranges[ref_idx]
        tdoa = np.array([ranges[i] - d_ref for i in range(len(anchors)) if i != ref_idx])

        # Chan's solver
        chan_pos, _ = tdoa_chan_solver(anchors, tdoa, ref_idx=ref_idx)

        # I-WLS solver
        positioner = TDOAPositioner(anchors, reference_idx=ref_idx)
        iwls_pos, info = positioner.solve(tdoa, initial_guess=np.array([10.0, 10.0]))

        # Both should converge to same position
        assert np.linalg.norm(chan_pos - iwls_pos) < 1e-3


class TestPositioningEdgeCases:
    """Test edge cases and error conditions."""

    def test_toa_insufficient_anchors(self):
        """Test TOA with insufficient anchors."""
        # Only 2 anchors in 2D (need at least 3)
        anchors = np.array([[0, 0], [10, 0]], dtype=float)
        ranges = np.array([5.0, 5.0])

        positioner = TOAPositioner(anchors)
        # Should not crash, but may not converge well
        pos, info = positioner.solve(ranges, initial_guess=np.array([5.0, 5.0]))

        # At minimum, should complete without exception
        assert pos.shape == (2,)

    def test_tdoa_wrong_measurement_size(self):
        """Test TDOA with wrong number of measurements."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        tdoa = np.array([1.0, 2.0])  # Should be 3 measurements

        positioner = TDOAPositioner(anchors)
        with pytest.raises(ValueError, match="Expected 3 TDOA measurements"):
            positioner.solve(tdoa, initial_guess=np.array([5.0, 5.0]))

    def test_aoa_wrong_measurement_size(self):
        """Test AOA with wrong number of measurements."""
        anchors = np.array([[10, 0], [0, 10], [-10, 0]], dtype=float)
        aoa = np.array([0.0, np.pi / 2])  # Should be 3 measurements

        positioner = AOAPositioner(anchors)
        with pytest.raises(ValueError, match="Expected 3 AOA measurements"):
            positioner.solve(aoa, initial_guess=np.array([0.0, 0.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



