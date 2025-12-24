"""
TDOA Positioning Examples.

This script demonstrates TDOA positioning algorithms from Chapter 4.

Implements:
    - TDOA measurement model (Eqs. 4.27-4.33)
    - TDOA I-WLS (Eqs. 4.34-4.42)
    - Correlated covariance matrix (Eq. 4.42)
    - Fang's TOA closed-form (Eqs. 4.43-4.49)
    - Chan's TDOA closed-form (Eqs. 4.50-4.62)

Author: Li-Ta Hsu
Date: December 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from core.rf import (
    TDOAPositioner,
    TOAPositioner,
    build_tdoa_covariance,
    tdoa_chan_solver,
    toa_fang_solver,
)


def demo_tdoa_basic():
    """Demonstrate basic TDOA positioning with I-WLS."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic TDOA Positioning (I-WLS)")
    print("=" * 70)

    # Setup anchors (5 anchors in a larger area)
    anchors = np.array(
        [[0, 0], [15, 0], [15, 15], [0, 15], [7.5, 7.5]], dtype=float
    )

    # True position
    true_position = np.array([5.0, 8.0])
    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")

    # Generate TDOA measurements (reference anchor = 0)
    dist_ref = np.linalg.norm(true_position - anchors[0])
    tdoa_measurements = []
    for i in range(1, len(anchors)):
        dist_i = np.linalg.norm(true_position - anchors[i])
        tdoa = dist_i - dist_ref
        tdoa_measurements.append(tdoa)
    tdoa_measurements = np.array(tdoa_measurements)

    print(f"\nTDOA measurements (m): {tdoa_measurements}")

    # Solve using I-WLS
    positioner = TDOAPositioner(anchors, reference_idx=0)
    estimated_position, info = positioner.solve(
        tdoa_measurements, initial_guess=np.array([7.5, 7.5])
    )

    print(f"\nEstimated position: {estimated_position}")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(
        f"Position error: {np.linalg.norm(estimated_position - true_position):.6f} m"
    )

    return anchors, true_position, tdoa_measurements


def demo_tdoa_with_noise():
    """Demonstrate TDOA positioning with measurement noise."""
    print("\n" + "=" * 70)
    print("Demo 2: TDOA Positioning with Measurement Noise")
    print("=" * 70)

    # Setup
    anchors = np.array(
        [[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float
    )
    true_position = np.array([7.0, 12.0])

    # Generate noiseless TDOA
    dist_ref = np.linalg.norm(true_position - anchors[0])
    tdoa_true = np.array(
        [
            np.linalg.norm(true_position - anchors[i]) - dist_ref
            for i in range(1, len(anchors))
        ]
    )

    # Test different noise levels
    noise_levels = [0.0, 0.1, 0.5, 1.0]
    results = []

    print(f"\nTrue position: {true_position}")
    print(f"Testing {len(noise_levels)} noise levels...")

    for noise_std in noise_levels:
        # Add noise
        tdoa_noisy = tdoa_true + np.random.randn(len(tdoa_true)) * noise_std

        # Solve
        positioner = TDOAPositioner(anchors, reference_idx=0)
        est_pos, info = positioner.solve(
            tdoa_noisy, initial_guess=np.array([10.0, 10.0])
        )

        if info["converged"]:
            error = np.linalg.norm(est_pos - true_position)
            results.append(
                {
                    "noise": noise_std,
                    "position": est_pos,
                    "error": error,
                    "iterations": info["iterations"],
                }
            )
        else:
            results.append(
                {
                    "noise": noise_std,
                    "position": None,
                    "error": np.inf,
                    "iterations": info["iterations"],
                }
            )

    # Print results
    print("\n" + "-" * 70)
    print(f"{'Noise (m)':<15} {'Est. Position':<25} {'Error (m)':<12} {'Iters':<8}")
    print("-" * 70)
    for r in results:
        pos_str = (
            f"[{r['position'][0]:.3f}, {r['position'][1]:.3f}]"
            if r["position"] is not None
            else "FAILED"
        )
        error_str = f"{r['error']:.4f}" if r["error"] != np.inf else "FAILED"
        print(
            f"{r['noise']:<15.2f} {pos_str:<25} {error_str:<12} {r['iterations']:<8}"
        )

    return results


def demo_correlated_covariance():
    """
    Demonstrate the impact of correlated covariance for TDOA measurements.

    Implements Eq. (4.42) from Chapter 4:
        The TDOA covariance matrix has off-diagonal terms due to the shared
        reference anchor noise.

    This demo compares:
        - Identity weighting (ignoring correlation)
        - Correlated weighting (proper covariance modeling per Eq. 4.42)
    """
    print("\n" + "=" * 70)
    print("Demo 3: Correlated vs Identity Weighting (Eq. 4.42)")
    print("=" * 70)

    # Setup: 5 anchors with heterogeneous noise levels
    anchors = np.array(
        [[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float
    )

    # True position
    true_position = np.array([7.0, 12.0])
    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")

    # Per-anchor range noise standard deviations (meters)
    # Reference anchor (idx=0) has larger noise to emphasize correlation effect
    sigmas = np.array([0.5, 0.1, 0.15, 0.12, 0.08])
    print(f"\nPer-anchor range noise (sigma, meters):")
    print(f"  Reference (anchor 0): {sigmas[0]:.2f} m (higher noise)")
    print(f"  Other anchors: {sigmas[1:]}")

    # Generate noiseless TDOA measurements
    dist_ref = np.linalg.norm(true_position - anchors[0])
    tdoa_true = np.array(
        [
            np.linalg.norm(true_position - anchors[i]) - dist_ref
            for i in range(1, len(anchors))
        ]
    )

    print(f"\nTrue TDOA measurements: {tdoa_true}")

    # Build correlated covariance matrix (Eq. 4.42)
    cov_correlated = build_tdoa_covariance(sigmas, ref_idx=0)
    print("\nCorrelated covariance matrix (Eq. 4.42):")
    print(cov_correlated)
    print(f"\nDiagonal (var): sigma_k^2 + sigma_ref^2")
    print(f"Off-diagonal (cov): sigma_ref^2 = {sigmas[0]**2:.4f}")

    # Build identity covariance (ignoring correlation)
    cov_identity = np.eye(len(tdoa_true))
    print("\nIdentity covariance matrix:")
    print(cov_identity)

    # Run Monte Carlo simulation
    n_trials = 500
    np.random.seed(42)

    errors_identity = []
    errors_correlated = []

    print(f"\nRunning {n_trials} Monte Carlo trials...")

    for trial in range(n_trials):
        # Generate noisy range measurements
        # Range noise for each anchor
        range_noise = np.random.randn(len(anchors)) * sigmas

        # Compute noisy ranges
        noisy_ranges = np.array(
            [np.linalg.norm(true_position - anchors[i]) + range_noise[i]
             for i in range(len(anchors))]
        )

        # Compute noisy TDOA (range differences relative to reference)
        tdoa_noisy = np.array(
            [noisy_ranges[i] - noisy_ranges[0] for i in range(1, len(anchors))]
        )

        # Solve with identity weighting
        positioner = TDOAPositioner(anchors, reference_idx=0)
        try:
            est_identity, info_id = positioner.solve(
                tdoa_noisy, initial_guess=np.array([10.0, 10.0]),
                covariance=cov_identity,
            )
            if info_id["converged"]:
                errors_identity.append(
                    np.linalg.norm(est_identity - true_position)
                )
        except Exception:
            pass

        # Solve with correlated weighting
        try:
            est_correlated, info_corr = positioner.solve(
                tdoa_noisy, initial_guess=np.array([10.0, 10.0]),
                covariance=cov_correlated,
            )
            if info_corr["converged"]:
                errors_correlated.append(
                    np.linalg.norm(est_correlated - true_position)
                )
        except Exception:
            pass

    # Compute statistics
    errors_identity = np.array(errors_identity)
    errors_correlated = np.array(errors_correlated)

    print("\n" + "-" * 70)
    print("Results Summary:")
    print("-" * 70)
    print(f"{'Weighting':<20} {'RMSE (m)':<15} {'Mean (m)':<15} {'Std (m)':<15}")
    print("-" * 70)

    rmse_id = np.sqrt(np.mean(errors_identity**2))
    rmse_corr = np.sqrt(np.mean(errors_correlated**2))

    print(
        f"{'Identity':<20} {rmse_id:<15.4f} "
        f"{np.mean(errors_identity):<15.4f} {np.std(errors_identity):<15.4f}"
    )
    print(
        f"{'Correlated (Eq.4.42)':<20} {rmse_corr:<15.4f} "
        f"{np.mean(errors_correlated):<15.4f} {np.std(errors_correlated):<15.4f}"
    )

    improvement = (rmse_id - rmse_corr) / rmse_id * 100
    print(f"\nRMSE improvement with correlated weighting: {improvement:.1f}%")

    print("\nKey Insight (from Eq. 4.42):")
    print("  - When reference anchor has larger noise (sigma_ref = 0.5 m),")
    print("    the off-diagonal correlation terms are significant (0.25).")
    print("  - Ignoring this correlation leads to suboptimal position estimates.")
    print("  - Proper covariance modeling per Eq. 4.42 improves accuracy.")

    return errors_identity, errors_correlated


def demo_covariance_sensitivity():
    """
    Demonstrate sensitivity of positioning accuracy to reference anchor noise.

    Shows how the correlated covariance structure matters more when:
        - Reference anchor has higher noise relative to other anchors
        - Off-diagonal terms become dominant
    """
    print("\n" + "=" * 70)
    print("Demo 4: Sensitivity to Reference Anchor Noise")
    print("=" * 70)

    # Setup: 4 anchors
    anchors = np.array(
        [[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float
    )
    true_position = np.array([8.0, 12.0])

    print(f"\nTrue position: {true_position}")
    print(f"Other anchor noise (fixed): sigma = 0.1 m")

    # Test different reference anchor noise levels
    ref_noise_levels = [0.05, 0.1, 0.2, 0.5, 1.0]
    n_trials = 200
    np.random.seed(123)

    results = []

    for ref_sigma in ref_noise_levels:
        sigmas = np.array([ref_sigma, 0.1, 0.1, 0.1])

        # Build covariance matrices
        cov_corr = build_tdoa_covariance(sigmas, ref_idx=0)
        cov_id = np.eye(len(anchors) - 1)

        errors_id = []
        errors_corr = []

        for _ in range(n_trials):
            # Generate noisy ranges
            range_noise = np.random.randn(len(anchors)) * sigmas
            noisy_ranges = np.array(
                [np.linalg.norm(true_position - anchors[i]) + range_noise[i]
                 for i in range(len(anchors))]
            )

            # Compute noisy TDOA
            tdoa_noisy = np.array(
                [noisy_ranges[i] - noisy_ranges[0]
                 for i in range(1, len(anchors))]
            )

            positioner = TDOAPositioner(anchors, reference_idx=0)

            # Identity weighting
            try:
                est_id, info = positioner.solve(
                    tdoa_noisy, initial_guess=np.array([10.0, 10.0]),
                    covariance=cov_id,
                )
                if info["converged"]:
                    errors_id.append(np.linalg.norm(est_id - true_position))
            except Exception:
                pass

            # Correlated weighting
            try:
                est_corr, info = positioner.solve(
                    tdoa_noisy, initial_guess=np.array([10.0, 10.0]),
                    covariance=cov_corr,
                )
                if info["converged"]:
                    errors_corr.append(np.linalg.norm(est_corr - true_position))
            except Exception:
                pass

        rmse_id = np.sqrt(np.mean(np.array(errors_id)**2))
        rmse_corr = np.sqrt(np.mean(np.array(errors_corr)**2))
        improvement = (rmse_id - rmse_corr) / rmse_id * 100 if rmse_id > 0 else 0

        results.append({
            "ref_sigma": ref_sigma,
            "rmse_identity": rmse_id,
            "rmse_correlated": rmse_corr,
            "improvement": improvement,
        })

    # Print results
    print("\n" + "-" * 70)
    print(f"{'Ref sigma (m)':<15} {'RMSE Id (m)':<15} "
          f"{'RMSE Corr (m)':<15} {'Improvement':<15}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['ref_sigma']:<15.2f} {r['rmse_identity']:<15.4f} "
            f"{r['rmse_correlated']:<15.4f} {r['improvement']:<15.1f}%"
        )

    print("\nKey Insight:")
    print("  - When reference noise >> other anchor noise, improvement is larger")
    print("  - At sigma_ref = sigma_other, correlation still matters")
    print("  - Proper covariance modeling is always beneficial")

    return results


def demo_visualize_covariance():
    """Visualize the correlated covariance matrix structure."""
    print("\n" + "=" * 70)
    print("Demo 5: Visualizing Covariance Matrix Structure (Eq. 4.42)")
    print("=" * 70)

    # Example with 5 anchors
    sigmas = np.array([0.3, 0.1, 0.15, 0.2, 0.12])
    ref_idx = 0

    print(f"\nPer-anchor sigmas: {sigmas}")
    print(f"Reference anchor index: {ref_idx}")
    print(f"Reference sigma: {sigmas[ref_idx]:.2f} m")

    cov = build_tdoa_covariance(sigmas, ref_idx)

    print("\nCovariance Matrix (4x4 for 4 TDOA measurements):")
    print("-" * 50)
    print("Eq. 4.42 structure:")
    print("  Diagonal[i,i] = sigma_i^2 + sigma_ref^2")
    print("  Off-diag[i,j] = sigma_ref^2 (shared reference noise)")
    print("-" * 50)

    # Print matrix with labels
    non_ref = [i for i in range(len(sigmas)) if i != ref_idx]
    header = "        " + "".join([f"d^{i},{ref_idx}     " for i in non_ref])
    print(header)

    for i, row_idx in enumerate(non_ref):
        row_str = f"d^{row_idx},{ref_idx}  "
        for j in range(len(cov)):
            row_str += f"{cov[i, j]:.4f}    "
        print(row_str)

    print(f"\nOff-diagonal value (sigma_ref^2): {sigmas[ref_idx]**2:.4f}")

    # Show diagonal derivation
    print("\nDiagonal derivation:")
    for i, anc_idx in enumerate(non_ref):
        diag_val = sigmas[anc_idx]**2 + sigmas[ref_idx]**2
        print(
            f"  var(d^{anc_idx},{ref_idx}) = "
            f"{sigmas[anc_idx]:.2f}^2 + {sigmas[ref_idx]:.2f}^2 = {diag_val:.4f}"
        )

    # Create figure
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cov, cmap='Blues')
        ax.set_title('TDOA Covariance Matrix (Eq. 4.42)', fontsize=12)
        ax.set_xlabel('TDOA measurement index')
        ax.set_ylabel('TDOA measurement index')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Covariance (m^2)')

        # Add annotations
        for i in range(len(cov)):
            for j in range(len(cov)):
                text = ax.text(j, i, f'{cov[i, j]:.3f}',
                              ha='center', va='center', color='black', fontsize=9)

        plt.tight_layout()
        plt.savefig('ch4_rf_point_positioning/figs/tdoa_covariance_matrix.png',
                    dpi=150, bbox_inches='tight')
        print("\nFigure saved to: ch4_rf_point_positioning/figs/tdoa_covariance_matrix.png")
        plt.close()
    except Exception as e:
        print(f"\nCould not save figure: {e}")

    return cov


def demo_geometry_effect():
    """Demonstrate the effect of anchor geometry on TDOA accuracy."""
    print("\n" + "=" * 70)
    print("Demo 6: Effect of Anchor Geometry on TDOA Accuracy")
    print("=" * 70)

    true_position = np.array([5.0, 5.0])

    # Good geometry: anchors surrounding the target
    good_anchors = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float
    )

    # Poor geometry: anchors on one side
    poor_anchors = np.array(
        [[0, 0], [2, 0], [4, 0], [6, 0]], dtype=float
    )

    noise_std = 0.2  # meters

    print(f"\nTrue position: {true_position}")
    print(f"Noise std: {noise_std} m")

    # Test with good geometry
    print("\n--- Good Geometry (surrounding) ---")
    dist_ref = np.linalg.norm(true_position - good_anchors[0])
    tdoa_good = np.array(
        [
            np.linalg.norm(true_position - good_anchors[i]) - dist_ref
            for i in range(1, len(good_anchors))
        ]
    )
    tdoa_good_noisy = tdoa_good + np.random.randn(len(tdoa_good)) * noise_std

    positioner_good = TDOAPositioner(good_anchors, reference_idx=0)
    est_good, info_good = positioner_good.solve(
        tdoa_good_noisy, initial_guess=np.array([5.0, 5.0])
    )

    if info_good["converged"]:
        error_good = np.linalg.norm(est_good - true_position)
        print(f"Estimated position: {est_good}")
        print(f"Position error: {error_good:.4f} m")
    else:
        print("Failed to converge")
        error_good = np.inf

    # Test with poor geometry
    print("\n--- Poor Geometry (collinear) ---")
    dist_ref = np.linalg.norm(true_position - poor_anchors[0])
    tdoa_poor = np.array(
        [
            np.linalg.norm(true_position - poor_anchors[i]) - dist_ref
            for i in range(1, len(poor_anchors))
        ]
    )
    tdoa_poor_noisy = tdoa_poor + np.random.randn(len(tdoa_poor)) * noise_std

    positioner_poor = TDOAPositioner(poor_anchors, reference_idx=0)
    est_poor, info_poor = positioner_poor.solve(
        tdoa_poor_noisy, initial_guess=np.array([5.0, 5.0])
    )

    if info_poor["converged"]:
        error_poor = np.linalg.norm(est_poor - true_position)
        print(f"Estimated position: {est_poor}")
        print(f"Position error: {error_poor:.4f} m")
    else:
        print("Failed to converge")
        error_poor = np.inf

    # Summary
    print("\n" + "-" * 70)
    print("Geometry Comparison:")
    print("-" * 70)
    print(f"{'Geometry':<30} {'Position Error (m)':<20}")
    print("-" * 70)
    if error_good != np.inf:
        print(f"{'Good (surrounding)':<30} {error_good:<20.4f}")
    else:
        print(f"{'Good (surrounding)':<30} {'FAILED':<20}")
    if error_poor != np.inf:
        print(f"{'Poor (collinear)':<30} {error_poor:<20.4f}")
    else:
        print(f"{'Poor (collinear)':<30} {'FAILED':<20}")

    print(
        "\nNote: Poor geometry leads to higher errors and potential convergence issues."
    )


def demo_fang_toa_solver():
    """
    Demonstrate Fang's closed-form TOA positioning (Eqs. 4.43-4.49).

    Compares:
        - Fang's closed-form solution (no iteration needed)
        - I-WLS iterative solution (requires initial guess)
    """
    print("\n" + "=" * 70)
    print("Demo 7: Fang's TOA Closed-Form vs I-WLS (Eqs. 4.43-4.49)")
    print("=" * 70)

    # Setup: 4 anchors in a square
    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    true_position = np.array([7.0, 12.0])

    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")

    # Compute true ranges
    ranges_true = np.linalg.norm(anchors - true_position, axis=1)
    print(f"True ranges: {ranges_true}")

    # Test with perfect measurements
    print("\n--- Perfect Measurements ---")

    # Fang's closed-form
    fang_pos, fang_info = toa_fang_solver(anchors, ranges_true)
    fang_error = np.linalg.norm(fang_pos - true_position)

    # I-WLS iterative
    positioner = TOAPositioner(anchors, method='iwls')
    iwls_pos, iwls_info = positioner.solve(
        ranges_true, initial_guess=np.array([10.0, 10.0])
    )
    iwls_error = np.linalg.norm(iwls_pos - true_position)

    print(f"Fang:  position={fang_pos}, error={fang_error:.6f} m")
    print(f"I-WLS: position={iwls_pos}, error={iwls_error:.6f} m, "
          f"iters={iwls_info['iterations']}")

    # Test with noisy measurements
    print("\n--- Noisy Measurements (Monte Carlo) ---")
    noise_levels = [0.1, 0.3, 0.5, 1.0]
    n_trials = 200
    np.random.seed(42)

    results = []
    for noise_std in noise_levels:
        fang_errors = []
        iwls_errors = []

        for _ in range(n_trials):
            ranges_noisy = ranges_true + np.random.randn(len(anchors)) * noise_std

            # Fang's method (no initial guess needed)
            try:
                f_pos, _ = toa_fang_solver(anchors, ranges_noisy)
                fang_errors.append(np.linalg.norm(f_pos - true_position))
            except Exception:
                pass

            # I-WLS (requires initial guess)
            try:
                i_pos, i_info = positioner.solve(
                    ranges_noisy, initial_guess=np.array([10.0, 10.0])
                )
                if i_info['converged']:
                    iwls_errors.append(np.linalg.norm(i_pos - true_position))
            except Exception:
                pass

        fang_rmse = np.sqrt(np.mean(np.array(fang_errors)**2))
        iwls_rmse = np.sqrt(np.mean(np.array(iwls_errors)**2))

        results.append({
            'noise': noise_std,
            'fang_rmse': fang_rmse,
            'iwls_rmse': iwls_rmse,
            'fang_success': len(fang_errors),
            'iwls_success': len(iwls_errors),
        })

    print(f"\n{'Noise (m)':<12} {'Fang RMSE':<15} {'I-WLS RMSE':<15} "
          f"{'Fang Success':<15} {'I-WLS Success':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['noise']:<12.2f} {r['fang_rmse']:<15.4f} {r['iwls_rmse']:<15.4f} "
              f"{r['fang_success']:<15} {r['iwls_success']:<15}")

    print("\nKey Insights:")
    print("  - Fang's method is non-iterative (no initial guess required)")
    print("  - I-WLS can refine estimates with proper weighting")
    print("  - Both methods sensitive to noise and geometry (GDOP)")

    return results


def demo_chan_tdoa_solver():
    """
    Demonstrate Chan's closed-form TDOA positioning (Eqs. 4.50-4.62).

    Compares:
        - Chan's closed-form solution (no iteration needed)
        - I-WLS iterative TDOA solution (requires initial guess)
    """
    print("\n" + "=" * 70)
    print("Demo 8: Chan's TDOA Closed-Form vs I-WLS (Eqs. 4.50-4.62)")
    print("=" * 70)

    # Setup: 5 anchors for good geometry
    anchors = np.array(
        [[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float
    )
    true_position = np.array([8.0, 12.0])
    ref_idx = 0

    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")
    print(f"Reference anchor: {ref_idx}")

    # Compute true ranges and TDOA
    ranges_true = np.linalg.norm(anchors - true_position, axis=1)
    d_ref = ranges_true[ref_idx]
    tdoa_true = np.array([ranges_true[i] - d_ref
                          for i in range(len(anchors)) if i != ref_idx])

    print(f"True reference distance: {d_ref:.4f} m")
    print(f"True TDOA measurements: {tdoa_true}")

    # Test with perfect measurements
    print("\n--- Perfect Measurements ---")

    # Chan's closed-form
    chan_pos, chan_info = tdoa_chan_solver(anchors, tdoa_true, ref_idx=ref_idx)
    chan_error = np.linalg.norm(chan_pos - true_position)

    # I-WLS iterative
    positioner = TDOAPositioner(anchors, reference_idx=ref_idx)
    iwls_pos, iwls_info = positioner.solve(
        tdoa_true, initial_guess=np.array([10.0, 10.0])
    )
    iwls_error = np.linalg.norm(iwls_pos - true_position)

    print(f"Chan:  position={chan_pos}, error={chan_error:.6f} m")
    print(f"       reference distance estimate={chan_info['reference_distance']:.4f} m")
    print(f"I-WLS: position={iwls_pos}, error={iwls_error:.6f} m, "
          f"iters={iwls_info['iterations']}")

    # Test with noisy measurements (correlated noise)
    print("\n--- Noisy Measurements (Monte Carlo with Correlated Noise) ---")
    noise_levels = [0.1, 0.3, 0.5, 1.0]
    n_trials = 200
    np.random.seed(42)

    # Per-anchor noise (uniform for simplicity)
    results = []
    for noise_std in noise_levels:
        sigmas = np.ones(len(anchors)) * noise_std
        cov = build_tdoa_covariance(sigmas, ref_idx=ref_idx)

        chan_errors = []
        iwls_errors = []

        for _ in range(n_trials):
            # Generate noisy ranges
            ranges_noisy = ranges_true + np.random.randn(len(anchors)) * noise_std

            # Compute noisy TDOA
            tdoa_noisy = np.array([ranges_noisy[i] - ranges_noisy[ref_idx]
                                   for i in range(len(anchors)) if i != ref_idx])

            # Chan's method (with WLS using covariance)
            try:
                c_pos, _ = tdoa_chan_solver(
                    anchors, tdoa_noisy, ref_idx=ref_idx, covariance=cov
                )
                chan_errors.append(np.linalg.norm(c_pos - true_position))
            except Exception:
                pass

            # I-WLS (with covariance)
            try:
                i_pos, i_info = positioner.solve(
                    tdoa_noisy, initial_guess=np.array([10.0, 10.0]),
                    covariance=cov,
                )
                if i_info['converged']:
                    iwls_errors.append(np.linalg.norm(i_pos - true_position))
            except Exception:
                pass

        chan_rmse = np.sqrt(np.mean(np.array(chan_errors)**2)) if chan_errors else np.inf
        iwls_rmse = np.sqrt(np.mean(np.array(iwls_errors)**2)) if iwls_errors else np.inf

        results.append({
            'noise': noise_std,
            'chan_rmse': chan_rmse,
            'iwls_rmse': iwls_rmse,
            'chan_success': len(chan_errors),
            'iwls_success': len(iwls_errors),
        })

    print(f"\n{'Noise (m)':<12} {'Chan RMSE':<15} {'I-WLS RMSE':<15} "
          f"{'Chan Success':<15} {'I-WLS Success':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['noise']:<12.2f} {r['chan_rmse']:<15.4f} {r['iwls_rmse']:<15.4f} "
              f"{r['chan_success']:<15} {r['iwls_success']:<15}")

    print("\nKey Insights:")
    print("  - Chan's method is non-iterative, estimates position + ref distance")
    print("  - Chan's WLS step uses correlated covariance (Eq. 4.62)")
    print("  - I-WLS requires initial guess but can iterate to better solution")
    print("  - Both methods benefit from proper covariance modeling")

    return results


def demo_closed_form_comparison():
    """
    Comprehensive comparison of closed-form and iterative solvers.

    Compares:
        - TOA: Fang vs I-WLS
        - TDOA: Chan vs I-WLS
    """
    print("\n" + "=" * 70)
    print("Demo 9: Comprehensive Closed-Form vs Iterative Comparison")
    print("=" * 70)

    # Setup
    anchors = np.array(
        [[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float
    )
    true_position = np.array([7.5, 11.0])
    ref_idx = 0

    print(f"\nTrue position: {true_position}")
    print(f"Anchors: {len(anchors)} beacons")

    # Compute true measurements
    ranges_true = np.linalg.norm(anchors - true_position, axis=1)
    d_ref = ranges_true[ref_idx]
    tdoa_true = np.array([ranges_true[i] - d_ref
                          for i in range(len(anchors)) if i != ref_idx])

    # Test parameters
    noise_std = 0.3
    n_trials = 500
    np.random.seed(123)

    # Results storage
    toa_fang_err = []
    toa_iwls_err = []
    tdoa_chan_err = []
    tdoa_iwls_err = []

    toa_positioner = TOAPositioner(anchors, method='iwls')
    tdoa_positioner = TDOAPositioner(anchors, reference_idx=ref_idx)
    sigmas = np.ones(len(anchors)) * noise_std
    cov = build_tdoa_covariance(sigmas, ref_idx=ref_idx)

    print(f"\nRunning {n_trials} Monte Carlo trials (noise_std={noise_std} m)...")

    for _ in range(n_trials):
        # Generate noisy ranges
        ranges_noisy = ranges_true + np.random.randn(len(anchors)) * noise_std
        tdoa_noisy = np.array([ranges_noisy[i] - ranges_noisy[ref_idx]
                               for i in range(len(anchors)) if i != ref_idx])

        # TOA: Fang
        try:
            pos, _ = toa_fang_solver(anchors, ranges_noisy)
            toa_fang_err.append(np.linalg.norm(pos - true_position))
        except Exception:
            pass

        # TOA: I-WLS
        try:
            pos, info = toa_positioner.solve(
                ranges_noisy, initial_guess=np.array([10.0, 10.0])
            )
            if info['converged']:
                toa_iwls_err.append(np.linalg.norm(pos - true_position))
        except Exception:
            pass

        # TDOA: Chan
        try:
            pos, _ = tdoa_chan_solver(anchors, tdoa_noisy, ref_idx=ref_idx, covariance=cov)
            tdoa_chan_err.append(np.linalg.norm(pos - true_position))
        except Exception:
            pass

        # TDOA: I-WLS
        try:
            pos, info = tdoa_positioner.solve(
                tdoa_noisy, initial_guess=np.array([10.0, 10.0]), covariance=cov
            )
            if info['converged']:
                tdoa_iwls_err.append(np.linalg.norm(pos - true_position))
        except Exception:
            pass

    # Compute statistics
    def stats(errors):
        if len(errors) == 0:
            return np.inf, np.inf, np.inf, 0
        e = np.array(errors)
        return np.sqrt(np.mean(e**2)), np.mean(e), np.std(e), len(e)

    print("\n" + "-" * 80)
    print(f"{'Method':<25} {'RMSE (m)':<12} {'Mean (m)':<12} {'Std (m)':<12} {'Success':<10}")
    print("-" * 80)

    rmse, mean, std, n = stats(toa_fang_err)
    print(f"{'TOA Fang (closed-form)':<25} {rmse:<12.4f} {mean:<12.4f} {std:<12.4f} {n:<10}")

    rmse, mean, std, n = stats(toa_iwls_err)
    print(f"{'TOA I-WLS (iterative)':<25} {rmse:<12.4f} {mean:<12.4f} {std:<12.4f} {n:<10}")

    rmse, mean, std, n = stats(tdoa_chan_err)
    print(f"{'TDOA Chan (closed-form)':<25} {rmse:<12.4f} {mean:<12.4f} {std:<12.4f} {n:<10}")

    rmse, mean, std, n = stats(tdoa_iwls_err)
    print(f"{'TDOA I-WLS (iterative)':<25} {rmse:<12.4f} {mean:<12.4f} {std:<12.4f} {n:<10}")

    print("\nSummary:")
    print("  - Closed-form methods (Fang, Chan) don't need initial guess")
    print("  - Iterative methods (I-WLS) can refine estimates iteratively")
    print("  - TOA methods require range measurements; TDOA uses range differences")
    print("  - TDOA eliminates need for clock synchronization between agent & beacons")
    print("  - All methods benefit from good geometry (low GDOP)")

    # Create comparison figure
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # TOA comparison
        ax1 = axes[0]
        data = [toa_fang_err, toa_iwls_err]
        tick_labels = ['Fang\n(closed-form)', 'I-WLS\n(iterative)']
        bp = ax1.boxplot(data, tick_labels=tick_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        ax1.set_ylabel('Position Error (m)')
        ax1.set_title('TOA Positioning Comparison')
        ax1.grid(True, alpha=0.3)

        # TDOA comparison
        ax2 = axes[1]
        data = [tdoa_chan_err, tdoa_iwls_err]
        tick_labels = ['Chan\n(closed-form)', 'I-WLS\n(iterative)']
        bp = ax2.boxplot(data, tick_labels=tick_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightyellow')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('TDOA Positioning Comparison')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Closed-Form vs Iterative Solvers (noise={noise_std}m)', fontsize=12)
        plt.tight_layout()
        plt.savefig('ch4_rf_point_positioning/figs/closed_form_comparison.png',
                    dpi=150, bbox_inches='tight')
        print("\nFigure saved: ch4_rf_point_positioning/figs/closed_form_comparison.png")
        plt.close()
    except Exception as e:
        print(f"\nCould not save figure: {e}")

    return {
        'toa_fang': toa_fang_err,
        'toa_iwls': toa_iwls_err,
        'tdoa_chan': tdoa_chan_err,
        'tdoa_iwls': tdoa_iwls_err,
    }


def main():
    """Run all TDOA positioning examples."""
    print("\n" + "=" * 70)
    print("Chapter 4: TDOA Positioning Examples")
    print("=" * 70)

    # Run demos
    demo_tdoa_basic()
    demo_tdoa_with_noise()
    demo_correlated_covariance()  # Demo 3: Correlated vs Identity weighting
    demo_covariance_sensitivity()  # Demo 4: Sensitivity analysis
    demo_visualize_covariance()  # Demo 5: Visualize covariance structure
    demo_geometry_effect()  # Demo 6
    demo_fang_toa_solver()  # Demo 7: Fang's TOA closed-form
    demo_chan_tdoa_solver()  # Demo 8: Chan's TDOA closed-form
    demo_closed_form_comparison()  # Demo 9: Comprehensive comparison

    print("\n" + "=" * 70)
    print("All TDOA examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()



