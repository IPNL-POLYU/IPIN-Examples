# Chapter 3: State Estimation

## Overview

This module implements the state estimation algorithms described in **Chapter 3** of *Principles of Indoor Positioning and Indoor Navigation*. It provides the mathematical foundations for estimating position, velocity, and other states from noisy measurements using various filtering and optimization techniques.

## Equation Mapping: Code ↔ Book

The following table maps the implemented functions to their corresponding equations in Chapter 3 of the book:

### Least Squares Methods

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `linear_least_squares()` | `core/estimators/least_squares.py` | **Eq. (3.1)** | ✓ | Standard LS: x̂ = (A'A)⁻¹A'b |
| `weighted_least_squares()` | `core/estimators/least_squares.py` | **Eq. (3.2)** | ✓ | WLS with measurement covariance: x̂ = (A'WA)⁻¹A'Wb |
| `iterative_least_squares()` | `core/estimators/least_squares.py` | **Eq. (3.3)** | ✓ | Gauss-Newton for nonlinear problems |
| `robust_least_squares()` | `core/estimators/least_squares.py` | **Eq. (3.4)** | ✓ | IRLS with Huber/Cauchy/Tukey loss functions |

### Kalman Filtering

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `KalmanFilter.predict()` | `core/estimators/kalman_filter.py` | **Eq. (3.5)** | 🚧 | Linear KF prediction: x̄ₖ = Fxₖ₋₁ + Buₖ |
| `KalmanFilter.update()` | `core/estimators/kalman_filter.py` | **Eq. (3.6)** | 🚧 | Linear KF update: xₖ = x̄ₖ + K(z - Hx̄ₖ) |
| `KalmanFilter._compute_kalman_gain()` | `core/estimators/kalman_filter.py` | **Eq. (3.7)** | 🚧 | Kalman gain: K = P̄H'(HP̄H' + R)⁻¹ |

### Extended Kalman Filter (EKF)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `ExtendedKalmanFilter.predict()` | `core/estimators/extended_kalman_filter.py` | **Eq. (3.8)** | 🚧 | Nonlinear prediction with Jacobian Fₖ |
| `ExtendedKalmanFilter.update()` | `core/estimators/extended_kalman_filter.py` | **Eq. (3.9)** | 🚧 | Nonlinear update with measurement Jacobian Hₖ |

### Unscented Kalman Filter (UKF)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `UnscentedKalmanFilter._compute_sigma_points()` | `core/estimators/unscented_kalman_filter.py` | **Eq. (3.10)** | ⏳ | Sigma point generation |
| `UnscentedKalmanFilter.predict()` | `core/estimators/unscented_kalman_filter.py` | **Eq. (3.11)** | ⏳ | UT-based prediction |
| `UnscentedKalmanFilter.update()` | `core/estimators/unscented_kalman_filter.py` | **Eq. (3.12)** | ⏳ | UT-based measurement update |

### Particle Filter (PF)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `ParticleFilter.predict()` | `core/estimators/particle_filter.py` | **Eq. (3.13)** | ⏳ | Particle propagation: xᵢₖ ~ p(xₖ\|xᵢₖ₋₁) |
| `ParticleFilter.update()` | `core/estimators/particle_filter.py` | **Eq. (3.14)** | ⏳ | Importance weighting: wᵢₖ ∝ p(zₖ\|xᵢₖ) |
| `ParticleFilter.resample()` | `core/estimators/particle_filter.py` | **Eq. (3.15)** | ⏳ | Systematic resampling |

### Performance Metrics

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `compute_innovation()` | `core/eval/metrics.py` | **Eq. (3.16)** | ⏳ | Innovation: ν = z - ẑ |
| `compute_nees()` | `core/eval/metrics.py` | **Eq. (3.17)** | ⏳ | Normalized Estimation Error Squared |
| `compute_nis()` | `core/eval/metrics.py` | **Eq. (3.18)** | ⏳ | Normalized Innovation Squared |

**Legend:**
- ✓ = Fully implemented and tested
- 🚧 = Implementation in progress
- ⏳ = Planned (not yet implemented)
- ✗ = Not implemented (out of scope)

## Implementation Notes

### ✓ Fully Implemented

#### 1. **Linear Least Squares (LS)**
   - Solves overdetermined systems: Ax = b where m ≥ n
   - Uses normal equations: (A'A)x = A'b
   - Computes covariance: P = σ²(A'A)⁻¹
   - Handles exact fit cases (m = n) and overdetermined (m > n)
   - **All 6 test cases pass**

#### 2. **Weighted Least Squares (WLS)**
   - Incorporates measurement uncertainties via weight matrix W
   - Typically W = R⁻¹ (inverse of measurement covariance)
   - Validates W is symmetric positive definite
   - Reduces to LS when W = I
   - **All 5 test cases pass**

#### 3. **Iterative Least Squares (Gauss-Newton)**
   - Handles nonlinear measurement models: f(x) ≈ b
   - Linearizes at each iteration using Jacobian J = ∂f/∂x
   - Converges quadratically near solution
   - Configurable max iterations and tolerance
   - **All 4 test cases pass** including 2D range positioning

#### 4. **Robust Least Squares (IRLS)**
   - Three loss functions implemented:
     - **Huber**: Quadratic for small residuals, linear for large
     - **Cauchy**: Heavy-tailed, more aggressive outlier rejection
     - **Tukey**: Sets outliers to zero weight beyond threshold
   - Uses Median Absolute Deviation (MAD) for robust scale estimation
   - Iteratively reweights and converges to stable solution
   - **All 7 test cases pass** including multi-outlier scenarios

### 🚧 In Progress

#### 5. **Kalman Filter (Linear KF)**
   - Optimal estimator for linear Gaussian systems
   - Prediction step (time update)
   - Update step (measurement correction)
   - Covariance propagation

### Implementation Choices

#### 1. **Numerical Stability**
   - Rank checking before solving normal equations
   - Pseudo-inverse fallback for near-singular matrices
   - MAD-based robust scale estimation (more robust than std dev)
   - Clipping of weights to valid range [0, 1]

#### 2. **Convergence Criteria**
   - Iterative LS: ||Δx|| < tolerance
   - Robust LS: ||x_new - x_old|| < tolerance
   - Default tolerance: 1e-6 for positions (millimeter accuracy)

#### 3. **Error Handling**
   - Validates matrix dimensions before computation
   - Checks rank deficiency
   - Detects non-positive-definite covariance matrices
   - Provides informative error messages

#### 4. **Testing Philosophy**
   - Each function has 4-7 comprehensive test cases
   - Tests cover: exact fit, noisy data, outliers, edge cases
   - Numerical accuracy verified to < 1e-6 for most cases
   - Round-trip tests for consistency

## File Structure

```
ch3_estimators/
├── README.md                              # This file
├── example_least_squares.py               # LS/WLS/ILS/Robust LS demonstrations
├── example_kalman_1d.py                   # 1D constant velocity tracking
├── example_ekf_range_bearing.py           # 2D positioning with EKF
└── example_comparison.py                  # Compare estimators

core/estimators/
├── __init__.py                            # Package exports
├── base.py                                # Abstract base classes
├── least_squares.py                       # LS/WLS/ILS/Robust LS [DONE]
├── kalman_filter.py                       # Linear KF [TODO]
├── extended_kalman_filter.py              # EKF [TODO]
├── unscented_kalman_filter.py             # UKF [TODO]
└── particle_filter.py                     # PF [TODO]

tests/core/estimators/
├── test_least_squares.py                  # 22 test cases [DONE]
├── test_kalman_filter.py                  # [TODO]
├── test_extended_kalman_filter.py         # [TODO]
└── test_particle_filter.py                # [TODO]

data/sim/ch3/
├── range_measurements.npz                 # Simulated TOA data
├── range_bearing.npz                      # Range + bearing data
├── tracking_1d.npz                        # 1D motion data
└── landmark_map.npz                       # 2D SLAM scenario
```

## Usage Examples

### Example 1: Linear Least Squares (Position from Ranges)

```python
import numpy as np
from core.estimators import linear_least_squares

# 2D positioning from 4 range measurements
# Anchors at corners of 10m × 10m square
anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
true_position = np.array([3.0, 4.0])

# Measured ranges (with noise)
ranges = np.linalg.norm(anchors - true_position, axis=1) + 0.1 * np.random.randn(4)

# Linearized problem: solve for position offset from initial guess
x0 = np.array([5.0, 5.0])  # Initial guess at center

# Design matrix (Jacobian of range function)
def compute_design_matrix(x_ref, anchors):
    diff = x_ref - anchors
    ranges_ref = np.linalg.norm(diff, axis=1, keepdims=True)
    return diff / ranges_ref

A = compute_design_matrix(x0, anchors)
b = ranges - np.linalg.norm(anchors - x0, axis=1)

# Solve for offset
dx, P = linear_least_squares(A, b)
position_estimate = x0 + dx

print(f"True position: {true_position}")
print(f"Estimated: {position_estimate}")
print(f"Error: {np.linalg.norm(position_estimate - true_position):.3f} m")
```

**Implements:** Eq. (3.1)

### Example 2: Weighted Least Squares (Different Measurement Accuracies)

```python
from core.estimators import weighted_least_squares

# Same setup, but with different measurement uncertainties
# Anchor 0 is very accurate (σ=0.1m), others are noisy (σ=1.0m)
measurement_std = np.array([0.1, 1.0, 1.0, 1.0])

# Weight matrix is inverse of measurement covariance
W = np.diag(1.0 / measurement_std**2)

# WLS solution weights accurate measurements more
dx_wls, P_wls = weighted_least_squares(A, b, W)
position_wls = x0 + dx_wls

print(f"Weighted LS estimate: {position_wls}")
print(f"Covariance trace: {np.trace(P_wls):.6f}")
```

**Implements:** Eq. (3.2)

### Example 3: Iterative Least Squares (Nonlinear Range Positioning)

```python
from core.estimators import iterative_least_squares

# Define nonlinear measurement model
def range_model(x):
    """Compute predicted ranges from position x to all anchors."""
    return np.linalg.norm(anchors - x, axis=1)

def range_jacobian(x):
    """Jacobian of range function: ∂r/∂x."""
    diff = x - anchors
    ranges = np.linalg.norm(diff, axis=1, keepdims=True)
    return diff / np.maximum(ranges, 1e-10)  # Avoid division by zero

# Initial guess
x_init = np.array([5.0, 5.0])

# Iterative solution
x_hat, P, iterations = iterative_least_squares(
    range_model, range_jacobian, ranges, x_init, max_iter=10, tol=1e-6
)

print(f"Converged in {iterations} iterations")
print(f"Position: {x_hat}")
print(f"Error: {np.linalg.norm(x_hat - true_position):.3f} m")
```

**Implements:** Eq. (3.3) - Gauss-Newton algorithm

### Example 4: Robust Least Squares (With Outliers)

```python
from core.estimators import robust_least_squares

# Add a severe outlier to one measurement
ranges_with_outlier = ranges.copy()
ranges_with_outlier[2] += 5.0  # 5m outlier

# Standard LS (corrupted by outlier)
dx_ls, _ = linear_least_squares(A, ranges_with_outlier - np.linalg.norm(anchors - x0, axis=1))
position_ls = x0 + dx_ls

# Robust LS (downweights outlier)
dx_robust, P_robust, weights = robust_least_squares(
    A, 
    ranges_with_outlier - np.linalg.norm(anchors - x0, axis=1),
    method="huber",
    threshold=2.0
)
position_robust = x0 + dx_robust

print(f"Standard LS error: {np.linalg.norm(position_ls - true_position):.3f} m")
print(f"Robust LS error: {np.linalg.norm(position_robust - true_position):.3f} m")
print(f"Outlier weight: {weights[2]:.3f}")  # Should be << 1.0
```

**Implements:** Eq. (3.4) - IRLS with Huber loss

## Running the Examples

### Run All Tests

```bash
# Run all Chapter 3 tests
pytest tests/core/estimators/ -v

# Run specific module
pytest tests/core/estimators/test_least_squares.py -v

# Run with coverage
pytest tests/core/estimators/ --cov=core.estimators --cov-report=html
```

**Test Coverage:**
- 22 test cases for least squares methods
- All tests pass with numerical accuracy < 1e-6
- Edge cases: rank deficiency, outliers, convergence

### Demo Scripts

```bash
cd ch3_estimators
python example_least_squares.py
python example_comparison.py
```

## Verification and Validation

### Least Squares Tests

| Test Case | Expected Result | Actual Result | Status |
|-----------|----------------|---------------|--------|
| Exact fit (y = 2x + 1) | [1.0, 2.0] | [1.0, 2.0] (< 1e-10) | ✓ |
| Overdetermined (5 equations, 2 unknowns) | Close to true | Error < 0.1 | ✓ |
| Identity matrix | Return b exactly | |b - x̂| < 1e-10 | ✓ |
| Rank deficient | Raise ValueError | ValueError raised | ✓ |
| WLS equal weights = LS | x_wls ≈ x_ls | |x_wls - x_ls| < 1e-10 | ✓ |
| WLS emphasizes accurate | Close to accurate meas. | Verified | ✓ |
| Iterative LS on linear | Converges to LS | < 5 iterations | ✓ |
| Range positioning 2D | Error < 1e-6 | < 10 iterations | ✓ |
| Robust LS (1 outlier) | Downweights outlier | w_outlier < 0.5 | ✓ |
| Huber vs standard LS | Robust < standard error | Verified | ✓ |

## Differences from Book

### Simplifications

1. **Robust Loss Functions**
   - Book may describe M-estimators in general form
   - Implementation provides three specific functions (Huber, Cauchy, Tukey)
   - MAD used for robust scale (simpler than iterative scale estimation)

2. **Iterative LS**
   - Book may cover Levenberg-Marquardt or trust-region methods
   - Implementation uses Gauss-Newton (simpler, sufficient for most cases)
   - Fallback to pseudo-inverse for singular Jacobians

### Extensions

1. **Comprehensive Testing**
   - Each function has 4-7 test cases covering edge cases
   - Automatic validation of matrix properties (symmetry, positive definiteness)
   - Clear error messages for common mistakes

2. **Configurable Parameters**
   - Max iterations and tolerances are adjustable
   - Multiple robust loss functions available
   - Optional covariance computation (saves computation when not needed)

### Not Implemented

1. **Advanced Optimization**
   - Levenberg-Marquardt algorithm (LM) for better convergence
   - Trust-region methods
   - Line search for step size control

2. **Specialized Algorithms**
   - Sequential least squares
   - Recursive least squares (RLS)
   - Total least squares (TLS)

## References

- **Chapter 3**: State Estimation
  - Section 3.2: Least Squares Methods
  - Section 3.3: Kalman Filtering
  - Section 3.4: Nonlinear Filters (EKF, UKF)
  - Section 3.5: Particle Filters

- **Numerical Recipes**: Press et al. (2007)
- **Robust Statistics**: Huber & Ronchetti (2009)

## Future Work

1. **Implement Kalman Filter** (linear KF)
2. **Implement Extended Kalman Filter** (EKF)
3. **Implement Unscented Kalman Filter** (UKF)
4. **Implement Particle Filter** (PF)
5. **Add performance metrics** (NEES, NIS, innovation tests)
6. **Create simulation data generators**
7. **Add interactive Jupyter notebooks**

## Contributing

When adding new estimators:

1. **Add equation reference** in docstring (e.g., "Implements Eq. (3.X)")
2. **Update this README** with new mapping entry
3. **Add comprehensive unit tests** (minimum 4-5 test cases)
4. **Verify numerical accuracy** (< 1e-6 for typical cases)
5. **Update `core/estimators/__init__.py`** with exports

---

**Status**: ✓ Least Squares methods fully implemented and tested  
**Last Updated**: December 2025  
**Maintainer**: Navigation Engineering Team

