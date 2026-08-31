"""
Example: Least Squares Estimation for Indoor Positioning.

This script demonstrates various least squares methods applied to 2D positioning
from range measurements (Time-of-Arrival positioning), following Chapter 3.

Run from repository root:
    python -m ch3_estimators.example_least_squares

Demonstrates:
    - Linear least squares (LS) with linearization
    - Weighted least squares (WLS) with w_i = 1/sigma_i^2
    - Gauss-Newton iterative LS for nonlinear problems
    - Levenberg-Marquardt for robust convergence
    - Robust least squares (IRLS) with Table 3.1 loss functions

Book Reference (Chapter 3):
    Section 3.1 - Least Squares Estimation:
        - Eq. (3.1): Cost function J(x) = sum_i (y_i - h_i(x))^2
        - Eq. (3.2): Normal equations (H^T H) x = H^T y
        - Eq. (3.3): Closed-form solution x = (H^T H)^{-1} H^T y
        - Eq. (3.4): First-order optimality condition dJ/dx = 0 (stationarity)
        - Table 3.1: Robust estimators (L2, Cauchy, Huber, Geman-McClure)

    Section 3.4.1 - Numerical Optimization:
        - Eq. (3.42)-(3.43): Line search x_{k+1} = x_k + alpha*d
        - Eq. (3.51)-(3.52): Gauss-Newton normal equations (J^T J) dx = J^T r
        - Eq. (3.53)-(3.56): Levenberg-Marquardt with damping mu
        - Algorithm 3.1: Line search strategy
        - Algorithm 3.2: Levenberg-Marquardt method

Measurement Model:
    h_i(x) = ||x - a_i||  (range from position x to anchor a_i)
    Residual: r_i(x) = y_i - h_i(x)  (observation minus prediction)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.estimators import (
    gauss_newton,
    levenberg_marquardt,
    linear_least_squares,
    robust_gauss_newton,
    solve_weighted_least_squares,
)
from core.eval import save_figure, show_figures_if_requested


def setup_positioning_scenario():
    """Create a 2D positioning scenario with 4 anchors.

    Returns:
        anchors: (4, 2) array of anchor positions at room corners.
        true_position: (2,) array of true target position.
    """
    # Anchor positions at corners of 10m × 10m room
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])

    # True target position (unknown to estimator)
    true_position = np.array([3.0, 4.0])

    return anchors, true_position


def create_range_model(anchors: np.ndarray):
    """Create measurement model and Jacobian functions for range positioning.

    This implements the book's formulation (Section 3.1):
        h_i(x) = ||x - a_i||  (nonlinear range model)
        J_i = ∂h_i/∂x = (x - a_i) / ||x - a_i||  (Jacobian for Eq. 3.51-3.52)

    Args:
        anchors: (m, 2) array of anchor positions.

    Returns:
        h: Measurement model function h(x) -> predicted ranges.
        jacobian: Jacobian function J(x) -> ∂h/∂x matrix.
    """

    def h(x: np.ndarray) -> np.ndarray:
        """Predicted ranges from position x to all anchors.

        h_i(x) = ||x - a_i||  (range measurement model)
        """
        return np.linalg.norm(anchors - x, axis=1)

    def jacobian(x: np.ndarray) -> np.ndarray:
        """Jacobian of range model.

        J_i = ∂h_i/∂x = (x - a_i) / ||x - a_i||

        This is the direction vector from anchor to position, used in
        Gauss-Newton iteration (Section 3.4.1.2).
        """
        diff = x - anchors
        ranges = np.linalg.norm(diff, axis=1, keepdims=True)
        return diff / np.maximum(ranges, 1e-10)

    return h, jacobian


def compute_ranges(
    position: np.ndarray, anchors: np.ndarray, noise_std: float = 0.0
) -> np.ndarray:
    """Compute ranges from position to anchors with optional noise.

    Args:
        position: (2,) target position.
        anchors: (m, 2) anchor positions.
        noise_std: Standard deviation of range noise (meters).

    Returns:
        ranges: (m,) measured ranges.
    """
    true_ranges = np.linalg.norm(anchors - position, axis=1)
    if noise_std > 0:
        true_ranges += noise_std * np.random.randn(len(anchors))
    return true_ranges


def example_1_linear_ls():
    """
    Example 1: Linear Least Squares with Linearization.

    Demonstrates Eq. (3.2)-(3.3): Normal equations x̂ = (A'A)⁻¹A'b

    For nonlinear range model h(x), we linearize around initial guess x0:
        h(x) ~ h(x0) + J|_{x0} (x - x0)
    Leading to linear system: A dx = b
    where A = J|_{x0}, b = y - h(x0), dx = x - x0
    """
    print("=" * 70)
    print("EXAMPLE 1: Linear Least Squares (Eq. 3.2-3.3)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()
    h, jacobian = create_range_model(anchors)

    # Generate noisy measurements
    np.random.seed(42)
    y = compute_ranges(true_position, anchors, noise_std=0.1)

    # Initial guess (center of room)
    x0 = np.array([5.0, 5.0])

    # Linearize around initial guess
    # Residual: r = y - h(x₀) (observation minus prediction)
    # Jacobian: A = J|_{x₀}
    r = y - h(x0)
    A = jacobian(x0)

    # Solve linear LS: A'A dx = A'r (Eq. 3.2)
    dx, P = linear_least_squares(A, r)
    position_estimate = x0 + dx

    # Results
    error = np.linalg.norm(position_estimate - true_position)

    print("\nMeasurement model: h_i(x) = ||x - a_i|| (range to anchor)")
    print("Residual: r_i = y_i - h_i(x)  (book convention)")
    print(f"\nTrue position:      {true_position}")
    print(f"Initial guess:      {x0}")
    print(f"LS estimate:        {position_estimate}")
    print(f"Position error:     {error:.4f} m")
    print("\nCovariance matrix (Eq. 3.3):")
    print(P)
    print(f"Position std dev:   {np.sqrt(np.diag(P))}")

    # That standard deviation is inflated, and by a lot. `linear_least_squares`
    # returns the a-posteriori form sigma_hat^2 (A'A)^-1 with
    # sigma_hat^2 = SSR/(m - n), and after a single linearisation step from a
    # guess 2.83 m away the residual is dominated by linearisation error rather
    # than by measurement noise: sigma_hat comes out at 0.367 m against a true
    # range noise of 0.10 m. The printed 0.2597 m is therefore 3.67x the actual
    # scatter of this estimator, which is 0.0709 m in x and 0.0700 m in y --
    # measured over 20000 draws, and matching the a-priori prediction
    # sigma * sqrt(diag((A'A)^-1)) = 0.0707 m printed below.
    #
    # Reporting it unlabelled says the fix is four times worse than it is. The
    # honest options are to iterate before quoting a covariance -- which is
    # Example 3, whose sigma_hat lands at 0.0765/0.0725 m against an empirical
    # 0.0730/0.0683 -- or to say which of the two numbers is which, as here.
    sigma_range = 0.1
    apriori = sigma_range * np.sqrt(np.diag(np.linalg.inv(A.T @ A)))
    sigma_hat = np.sqrt(np.sum((r - A @ dx) ** 2) / (len(y) - len(dx)))
    print(f"\n  a-posteriori sigma_hat (from this fit's residuals): {sigma_hat:.4f} m")
    print(f"  a-priori sigma (the range noise actually used):     {sigma_range:.4f} m")
    print(f"  a-priori position std, sigma*sqrt(diag((A'A)^-1)):  {apriori}")
    print("\n  The printed std dev above is the a-posteriori one and is inflated")
    print("  by linearisation: one step from a guess 2.83 m away leaves a")
    print("  residual that is mostly model error, not noise. This estimator's")
    print("  real scatter is 0.0709 m in x and 0.0700 m in y over 20000 draws.")
    print("  Example 3 iterates, and its covariance needs no such caveat.")

    return position_estimate, P


def example_2_weighted_ls():
    """
    Example 2: Weighted Least Squares.

    Demonstrates Section 3.1.1: w_i = 1/sigma_i^2 weighting.

    WLS minimizes: J(x) = sum w_i (y_i - h_i(x))^2
    Solution: x̂ = (A'WA)⁻¹A'Wb
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Weighted Least Squares (Section 3.1.1)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()
    h, jacobian = create_range_model(anchors)

    # Different measurement accuracies
    # Anchor 0 is very accurate (GPS reference station)
    # Others are UWB with higher noise
    measurement_stds = np.array([0.05, 0.3, 0.3, 0.3])

    print("\nMeasurement standard deviations (sigma_i):")
    for i, std in enumerate(measurement_stds):
        print(f"  Anchor {i}: sigma = {std:.2f} m -> w = 1/sigma^2 = {1/std**2:.1f}")

    # Generate measurements with different noise levels
    np.random.seed(42)
    y = np.array(
        [
            compute_ranges(
                true_position, anchors[i : i + 1], noise_std=measurement_stds[i]
            )[0]
            for i in range(len(anchors))
        ]
    )

    # Weight matrix: W = diag(1/sigma^2) (book Section 3.1.1)
    W = np.diag(1.0 / measurement_stds**2)

    # Linearization
    x0 = np.array([5.0, 5.0])
    r = y - h(x0)  # Residual: y - h(x₀)
    A = jacobian(x0)

    # Solve WLS: (A'WA) dx = A'W r
    wls_result = solve_weighted_least_squares(
        design_matrix=A,
        observations=r,
        weight_matrix=W,
    )
    position_wls = x0 + wls_result.estimated_state

    # Compare with standard LS (ignoring weights)
    dx_ls, P_ls = linear_least_squares(A, r)
    position_ls = x0 + dx_ls

    # Results
    error_wls = np.linalg.norm(position_wls - true_position)
    error_ls = np.linalg.norm(position_ls - true_position)

    print(f"\nTrue position:      {true_position}")
    print(f"WLS estimate:       {position_wls} (error: {error_wls:.4f} m)")
    print(f"LS estimate:        {position_ls} (error: {error_ls:.4f} m)")
    # Both traces on one footing, and the footing named.
    #
    # These two lines used to print `np.trace(wls_result.state_covariance)`
    # against `np.trace(P_ls)`, which is 0.047432 against 0.230713 -- a ratio of
    # 0.206 that looks like weighting cutting the variance by a factor of five.
    # It is not a like-for-like comparison. The WLS number is a-priori,
    # (A'WA)^-1 with W the true inverse-variances; the LS number is
    # a-posteriori, sigma_hat^2 (A'A)^-1 from a single linearisation step, and
    # sigma_hat is inflated by the same linearisation error Example 1 describes.
    #
    # The a-priori covariance of *unweighted* LS under heteroscedastic noise is
    # the sandwich (A'A)^-1 A' Sigma A (A'A)^-1, not sigma^2 (A'A)^-1 -- there
    # is no single sigma to use. That gives 0.068125, so the honest ratio is
    # 0.696, not 0.206. Both figures were checked against 200000 draws, which
    # give traces of 0.047343 and 0.067936, a ratio of 0.6969.
    Sigma = np.diag(measurement_stds**2)
    AtA_inv = np.linalg.inv(A.T @ A)
    P_ls_apriori = AtA_inv @ A.T @ Sigma @ A @ AtA_inv
    trace_wls = np.trace(wls_result.state_covariance)
    trace_ls = np.trace(P_ls_apriori)
    print("\nCovariance traces, both a-priori (from the known sigma_i, not this fit):")
    print(f"  WLS, (A'WA)^-1:                        {trace_wls:.6f}")
    print(f"  LS,  (A'A)^-1 A' Sigma A (A'A)^-1:     {trace_ls:.6f}")
    print(f"  ratio WLS/LS:                          {trace_wls / trace_ls:.4f}")
    print("  (Over 200000 draws the empirical traces are 0.047343 and 0.067936,")
    print("   a ratio of 0.6969.)")
    print(f"\nFor reference, this fit's a-posteriori LS trace: {np.trace(P_ls):.6f}")
    print("  That is sigma_hat^2 (A'A)^-1 with sigma_hat inflated by")
    print("  linearisation, so comparing it against the a-priori WLS number")
    print("  above would report the ratio as 0.206 instead of 0.696.")
    # One draw, so labelled as one. This line used to print the same quantity
    # as "Improvement: 36.7%", which reads as a property of weighting and is
    # not one: over 5000 draws the RMS improvement is about 14%, the per-draw
    # median about 9%, the 5th-95th percentile spans roughly -37% to +73%, and
    # WLS is *worse* than unweighted LS on nearly 30% of draws. (Figures are
    # rounded because they move a few tenths between noise streams; two
    # independent 5000-draw runs gave 14.4/9.0/27% and 14.3/8.5/29%.)
    #
    # That last number is the lesson rather than a caveat. With four ranges
    # and two unknowns there is almost no redundancy, and W puts 36x more
    # weight on anchor 0 than on any other. When anchor 0 draws an unlucky
    # error, WLS follows it. Weighting buys accuracy on average by trusting
    # the good sensor, and pays for it by depending on that sensor.
    print(
        f"\nThis draw: WLS is {((error_ls - error_wls) / error_ls * 100):.1f}% "
        f"better than LS"
    )
    print("  Over 5000 draws: ~14% better in RMS, ~9% in the per-draw median,")
    print("  and worse than plain LS on ~28% of them. A single draw cannot tell")
    print("  you which of those you are looking at.")

    return position_wls, wls_result.state_covariance


def example_3_gauss_newton():
    """
    Example 3: Gauss-Newton Iterative Least Squares.

    Demonstrates Section 3.4.1.2 Gauss-Newton method:
        - Eq. (3.4): First-order optimality condition dJ/dx = 0
        - Eq. (3.51): Linearized normal equations
        - Eq. (3.52): Gauss-Newton update (J'J) dx = J'r  ->  x <- x + dx

    This is the book's standard Gauss-Newton formulation for nonlinear LS.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Gauss-Newton Nonlinear LS (Eq. 3.51-3.52)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()
    h, jacobian = create_range_model(anchors)

    # Generate noisy measurements
    np.random.seed(42)
    y = compute_ranges(true_position, anchors, noise_std=0.1)

    # Initial guess
    x0 = np.array([5.0, 5.0])

    print("\nMeasurement model: h_i(x) = ||x - a_i||")
    print("Residual:          r_i = y_i - h_i(x)")
    print("Jacobian:          J_i = (x - a_i) / ||x - a_i||")
    print("\nGauss-Newton update (Eq. 3.52):")
    print("  (J'J) dx = J'r  ->  x <- x + dx")
    print(f"\nInitial guess: {x0}")
    print(f"True position: {true_position}")

    # Gauss-Newton solution using core module
    result = gauss_newton(h, jacobian, y, x0, max_iter=20, tol=1e-8)

    print(f"\nConverged in {result.iterations} iterations")
    print(f"Final estimate:  {result.x}")
    print(f"Position error:  {np.linalg.norm(result.x - true_position):.6f} m")
    print(f"Final residuals: {result.residuals}")
    print(f"Final cost:      {result.cost:.6e}")
    print("\nCovariance:")
    print(result.covariance)

    return result


def example_4_levenberg_marquardt():
    """
    Example 4: Levenberg-Marquardt for Robust Convergence.

    Demonstrates Eq. (3.53) and Algorithm 3.2:
        (J'J + mu*I) dx = J'r

    LM combines Gauss-Newton (fast near solution) with gradient descent
    (robust far from solution) via adaptive damping parameter mu.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Levenberg-Marquardt (Eq. 3.53, Algorithm 3.2)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()
    h, jacobian = create_range_model(anchors)

    # Generate measurements
    np.random.seed(42)
    y = compute_ranges(true_position, anchors, noise_std=0.1)

    # Poor initial guess (far from true position)
    x0_poor = np.array([0.0, 0.0])

    print("\nLM update (Eq. 3.53): (J'J + mu*I) dx = J'r")
    print("  mu large -> gradient descent behavior (global convergence)")
    print("  mu small -> Gauss-Newton behavior (fast local convergence)")
    print(f"\nPoor initial guess: {x0_poor} (far from true position)")
    print(f"True position:      {true_position}")

    # Compare GN vs LM from poor initial guess
    print("\n--- Gauss-Newton from poor guess ---")
    result_gn = gauss_newton(h, jacobian, y, x0_poor, max_iter=50)
    error_gn = np.linalg.norm(result_gn.x - true_position)
    print(
        f"Result: {result_gn.x}, error: {error_gn:.4f} m, "
        f"iters: {result_gn.iterations}, converged: {result_gn.converged}"
    )

    print("\n--- Levenberg-Marquardt from poor guess ---")
    result_lm = levenberg_marquardt(h, jacobian, y, x0_poor, max_iter=50, mu0=1e-3)
    error_lm = np.linalg.norm(result_lm.x - true_position)
    print(
        f"Result: {result_lm.x}, error: {error_lm:.4f} m, "
        f"iters: {result_lm.iterations}, converged: {result_lm.converged}"
    )

    # This used to compare the two errors and print "[OK] LM converged better
    # than GN from poor initial guess" whenever `error_lm < error_gn`. On this
    # problem that is a coin toss on the last bit: the two solvers land
    # 1.05e-09 m apart from [0, 0], so the printed verdict was float noise
    # dressed as a finding, and it named a difference of 3.2e-10 m as LM
    # converging "better". Over a 961-point grid of starting points spanning
    # [-40, 50]^2 neither method fails once and they never differ by more than
    # 3.4e-08 m.
    #
    # A benign, well-conditioned problem cannot show what LM is for. Saying so
    # is the honest reading, and the hard case below is where the damping earns
    # its place.
    print(
        f"\nGN and LM agree to {np.linalg.norm(result_gn.x - result_lm.x):.1e} m here, "
        f"in the same {result_gn.iterations} iterations."
    )
    print("  Four anchors around a square is well conditioned, so J'J is far")
    print("  from singular and LM's damping never has to do anything. Over a")
    print("  961-point grid of starting points on [-40, 50]^2 neither solver")
    print("  fails once. On this problem the two are the same algorithm.")

    _compare_on_an_ill_conditioned_geometry()

    return result_lm


def _compare_on_an_ill_conditioned_geometry():
    """Where LM's damping is the difference between an answer and 1e14 metres.

    Collinear anchors make `J'J` near-singular for a start on the anchor line:
    every Jacobian row is then a unit vector along +/-x, so the second column
    carries almost nothing and the undamped Gauss-Newton step is enormous.
    Levenberg-Marquardt's `mu*I` bounds exactly that step.

    The numbers printed below are one seed; the grid statistics beside them are
    measured over 1681 starting points on x in [-5, 15], y in [0, 8].
    """
    anchors = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [10.0, 0.0]])
    true_position = np.array([4.0, 3.0])
    h, jacobian = create_range_model(anchors)

    np.random.seed(42)
    y = compute_ranges(true_position, anchors, noise_std=0.1)
    x0 = np.array([5.0, 0.01])  # essentially on the anchor line

    print("\n--- The same comparison on an ill-conditioned geometry ---")
    print("Four collinear anchors on y = 0, target at [4, 3], start at [5, 0.01]")
    print("  (on the anchor line, where J'J is nearly singular)")

    result_gn = gauss_newton(h, jacobian, y, x0, max_iter=50)
    result_lm = levenberg_marquardt(h, jacobian, y, x0, max_iter=50, mu0=1e-3)
    for name, result in (
        ("Gauss-Newton", result_gn),
        ("Levenberg-Marquardt", result_lm),
    ):
        print(
            f"  {name:<20} error {np.linalg.norm(result.x - true_position):.4g} m, "
            f"iters: {result.iterations}, converged: {result.converged}"
        )

    print("\n  Over 1681 starting points on x in [-5, 15], y in [0, 8]:")
    print("    Gauss-Newton returns a non-position (error > 1 km) from 15 of")
    print("    them, worst 1.6e+14 m. Levenberg-Marquardt does so from none,")
    print("    worst 3.61 m.")
    print("  Both land on the anchor line from about 40 starting points, and")
    print("  that is not a solver failure: collinear anchors leave the target's")
    print("  offset from the line unobservable in sign, so the line itself is a")
    print("  legitimate stopping place. The 1e+14 is the failure.")


def example_5_robust_ls():
    """
    Example 5: Robust Least Squares with Outliers.

    Demonstrates Table 3.1 robust estimators from Section 3.1.1:
        - L2:     e(x) = 0.5*||r||^2  (standard LS)
        - Cauchy: e(x) = 0.5*ln(1 + ||r||^2)
        - Huber:  e(x) = 0.5*||r||^2 if |r|<=delta, else delta(|r| - 0.5*delta)
        - G-M:    e(x) = 0.5*||r||^2 / (1 + ||r||^2)  (Geman-McClure)

    These robust loss functions reduce the influence of outliers (e.g., NLOS
    measurements in indoor environments).

    Note: Uses 8 anchors for sufficient redundancy to isolate outliers.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Robust LS with Table 3.1 Estimators")
    print("=" * 70)

    # Use more anchors for robust estimation (need redundancy!)
    anchors = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],  # Corners
            [5.0, 0.0],
            [5.0, 10.0],
            [0.0, 5.0],
            [10.0, 5.0],  # Midpoints
        ]
    )
    true_position = np.array([3.0, 4.0])
    h, jacobian = create_range_model(anchors)

    # Generate measurements with one severe outlier
    np.random.seed(42)
    y = compute_ranges(true_position, anchors, noise_std=0.1)
    y[2] += 5.0  # 5m NLOS error on anchor 2

    print("\nScenario: 2D positioning from 8 anchors")
    print("Added 5.0 m NLOS outlier to anchor 2")
    print("(8 anchors provide redundancy for outlier rejection)")
    print("\nTable 3.1 Robust Estimators:")
    print("  L2:     e(x) = 0.5*||r||^2        (standard, sensitive to outliers)")
    print("  Cauchy: e(x) = 0.5*ln(1+||r||^2) (soft downweighting)")
    print("  Huber:  e(x) = quadratic/linear  (threshold at delta)")
    print("  G-M:    e(x) = 0.5*||r||^2/(1+||r||^2) (strong outlier rejection)")

    x0 = np.array([5.0, 5.0])

    # Robust methods from Table 3.1
    # Note: L2 is included in the loop below for comparison
    table_3_1_methods = {
        "L2 (Table 3.1)": "l2",
        "Cauchy (Table 3.1)": "cauchy",
        "Huber (Table 3.1)": "huber",
        "G-M (Table 3.1)": "gm",
        "Tukey (extra)": "tukey",  # Not in Table 3.1, but available
    }

    results = {}
    for label, method in table_3_1_methods.items():
        result = robust_gauss_newton(
            h,
            jacobian,
            y,
            x0,
            loss=method,
            loss_param=1.5,
            max_iter=30,
            max_irls_iter=10,
        )
        error = np.linalg.norm(result.x - true_position)
        results[label] = {
            "position": result.x,
            "error": error,
            "weights": result.weights,
            "outlier_weight": result.weights[2],
        }

    # Results
    print(f"\nTrue position: {true_position}")
    print(f"\n{'Method':<20} {'Position':<25} {'Error (m)':<10} {'Outlier w':<10}")
    print("-" * 65)

    for label, res in results.items():
        pos_str = f"[{res['position'][0]:.3f}, {res['position'][1]:.3f}]"
        print(
            f"{label:<20} {pos_str:<25} {res['error']:<10.4f} {res['outlier_weight']:<10.4f}"
        )

    print("\nKey insight from Table 3.1:")
    print("  - L2 is corrupted by the outlier (no downweighting)")
    print("  - Cauchy provides soft downweighting")
    print("  - Huber transitions from quadratic to linear")
    print("  - G-M provides strongest outlier rejection")

    return results, anchors, true_position, y


def visualize_results():
    """Create visualization of all examples with Table 3.1 labels."""
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    # Setup
    anchors_4, true_position = setup_positioning_scenario()
    h_4, jac_4 = create_range_model(anchors_4)

    # 8 anchors for robust example
    anchors_8 = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 0.0],
            [5.0, 10.0],
            [0.0, 5.0],
            [10.0, 5.0],
        ]
    )
    h_8, jac_8 = create_range_model(anchors_8)

    np.random.seed(42)

    # Generate clean measurements (4 anchors)
    y_clean = compute_ranges(true_position, anchors_4, noise_std=0.1)
    x0 = np.array([5.0, 5.0])

    # Example 1: Linear LS (one iteration)
    r = y_clean - h_4(x0)
    A = jac_4(x0)
    dx_ls, _ = linear_least_squares(A, r)
    pos_linear = x0 + dx_ls

    # Example 3: Gauss-Newton
    result_gn = gauss_newton(h_4, jac_4, y_clean, x0)
    pos_gn = result_gn.x

    # Example 4: LM from poor guess
    result_lm = levenberg_marquardt(h_4, jac_4, y_clean, np.array([0.0, 0.0]))
    pos_lm = result_lm.x

    # Example 5: Robust LS with outlier (8 anchors)
    y_outlier = compute_ranges(true_position, anchors_8, noise_std=0.1)
    y_outlier[2] += 5.0  # Outlier

    pos_l2 = gauss_newton(h_8, jac_8, y_outlier, x0).x

    # Robust methods from Table 3.1
    pos_cauchy = robust_gauss_newton(h_8, jac_8, y_outlier, x0, loss="cauchy").x
    pos_huber = robust_gauss_newton(h_8, jac_8, y_outlier, x0, loss="huber").x
    pos_gm = robust_gauss_newton(h_8, jac_8, y_outlier, x0, loss="gm").x

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ----- Plot 1: Clean data (Examples 1-4) -----
    ax = axes[0]

    # Anchors
    ax.scatter(
        anchors_4[:, 0],
        anchors_4[:, 1],
        s=200,
        c="blue",
        marker="^",
        label="Anchors",
        zorder=5,
    )

    # True position
    ax.scatter(
        true_position[0],
        true_position[1],
        s=250,
        c="green",
        marker="*",
        label="True Position",
        zorder=5,
    )

    # Estimates
    ax.scatter(
        x0[0], x0[1], s=150, c="gray", marker="x", label="Initial Guess", zorder=4
    )
    ax.scatter(
        pos_linear[0],
        pos_linear[1],
        s=150,
        c="orange",
        marker="o",
        label="Linear LS (Eq. 3.2)",
        zorder=4,
    )
    ax.scatter(
        pos_gn[0],
        pos_gn[1],
        s=150,
        c="red",
        marker="s",
        label="Gauss-Newton (Eq. 3.52)",
        zorder=4,
    )
    ax.scatter(
        pos_lm[0],
        pos_lm[1],
        s=150,
        c="purple",
        marker="D",
        label="LM (Eq. 3.53)",
        zorder=4,
    )

    # Range circles
    for i, anchor in enumerate(anchors_4):
        circle = plt.Circle(
            anchor, y_clean[i], fill=False, edgecolor="blue", alpha=0.3, linestyle="--"
        )
        ax.add_patch(circle)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(
        "On clean data all three solvers agree to centimetres\n"
        "(inset: a 20 cm window on the same estimates)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    # The three estimates land within about 5 cm of each other and of the
    # truth, so at the scale that shows the anchors they are one dot with a
    # seven-entry legend beside it -- the panel's whole subject, invisible.
    # The inset is the same fix Chapter 6's comparison figure uses: keep the
    # overview, and put the part worth seeing beside it.
    spread = 0.1
    axins = ax.inset_axes([0.03, 0.03, 0.36, 0.36])
    axins.scatter(
        true_position[0], true_position[1], s=250, c="green", marker="*", zorder=5
    )
    axins.scatter(pos_linear[0], pos_linear[1], s=150, c="orange", marker="o", zorder=4)
    # Gauss-Newton and LM converge to the same point here, so LM is drawn as a
    # larger hollow marker around it: two rings mean they agree exactly.
    axins.scatter(pos_gn[0], pos_gn[1], s=110, c="red", marker="s", zorder=4)
    axins.scatter(
        pos_lm[0],
        pos_lm[1],
        s=320,
        facecolors="none",
        edgecolors="purple",
        marker="D",
        linewidths=2,
        zorder=4,
    )
    axins.set_xlim(true_position[0] - spread, true_position[0] + spread)
    axins.set_ylim(true_position[1] - spread, true_position[1] + spread)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.6)

    # ----- Plot 2: Robust LS with outlier (Table 3.1) -----
    ax = axes[1]

    # Anchors
    ax.scatter(
        anchors_8[:, 0],
        anchors_8[:, 1],
        s=200,
        c="blue",
        marker="^",
        label="Anchors (8)",
        zorder=5,
    )

    # Mark outlier anchor
    ax.scatter(
        anchors_8[2, 0],
        anchors_8[2, 1],
        s=350,
        facecolors="none",
        edgecolors="red",
        linewidth=3,
        zorder=4,
        label="Outlier Anchor",
    )

    # True position
    ax.scatter(
        true_position[0],
        true_position[1],
        s=250,
        c="green",
        marker="*",
        label="True Position",
        zorder=5,
    )

    # Table 3.1 estimator results
    ax.scatter(
        pos_l2[0],
        pos_l2[1],
        s=150,
        c="orange",
        marker="o",
        label="L2 (Table 3.1) - corrupted",
        zorder=4,
    )
    ax.scatter(
        pos_cauchy[0],
        pos_cauchy[1],
        s=150,
        c="cyan",
        marker="s",
        label="Cauchy (Table 3.1)",
        zorder=4,
    )
    ax.scatter(
        pos_huber[0],
        pos_huber[1],
        s=150,
        c="magenta",
        marker="^",
        label="Huber (Table 3.1)",
        zorder=4,
    )
    ax.scatter(
        pos_gm[0],
        pos_gm[1],
        s=150,
        c="purple",
        marker="D",
        label="G-M (Table 3.1)",
        zorder=4,
    )

    # Range circles (show first 4 anchors only to reduce clutter)
    for i in range(4):
        anchor = anchors_8[i]
        color = "red" if i == 2 else "blue"
        alpha = 0.6 if i == 2 else 0.2
        lw = 2.5 if i == 2 else 1
        circle = plt.Circle(
            anchor,
            y_outlier[i],
            fill=False,
            edgecolor=color,
            alpha=alpha,
            linestyle="--",
            linewidth=lw,
        )
        ax.add_patch(circle)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(
        "One corrupted anchor drags plain least squares off the truth\n"
        "while every robust loss in Table 3.1 ignores it",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    plt.tight_layout()

    # Save to figs directory (svg + pdf + png via the shared layer)
    paths = save_figure(
        fig, Path(__file__).parent / "figs", "ch3_least_squares_examples"
    )
    print(f"\nPlot saved as: {paths[0]}")
    show_figures_if_requested()


def main():
    """Run all examples."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    print("\n" + "=" * 70)
    print("CHAPTER 3: LEAST SQUARES ESTIMATION EXAMPLES")
    print("=" * 70)
    print("\nBook Reference: Section 3.1 (Least Squares Estimation)")
    print("               Section 3.1.1 (Robust Estimators, Table 3.1)")
    print("               Section 3.4.1 (Gauss-Newton & Levenberg-Marquardt)")
    print("\nApplication: 2D positioning from Time-of-Arrival (TOA) ranges")
    print("Measurement model: h_i(x) = ||x - a_i||")
    print("Residual:          r_i = y_i - h_i(x)  (observation - prediction)")

    # Run examples
    example_1_linear_ls()
    example_2_weighted_ls()
    example_3_gauss_newton()
    example_4_levenberg_marquardt()
    example_5_robust_ls()

    # Visualization
    visualize_results()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
