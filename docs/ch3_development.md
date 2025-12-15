# Chapter 3: Development Notes

> **Note:** This document contains implementation details, design decisions, and development notes for Chapter 3. For student-facing documentation, see [ch3_estimators/README.md](../ch3_estimators/README.md).

## Implementation Status

| Estimator | Status | Test Cases | Notes |
|-----------|--------|------------|-------|
| Linear Least Squares | Complete | 6 | Eq. (3.1) |
| Weighted Least Squares | Complete | 5 | Eq. (3.2) |
| Iterative Least Squares | Complete | 4 | Eq. (3.3), Gauss-Newton |
| Robust Least Squares | Complete | 7 | Eq. (3.4), Huber/Cauchy/Tukey |
| Kalman Filter | Complete | 3 | Eqs. (3.11)-(3.19) |
| Extended Kalman Filter | Complete | 2 | Eqs. (3.21)-(3.22) |
| Unscented Kalman Filter | Complete | 2 | Eqs. (3.24)-(3.30) |
| Particle Filter | Complete | 2 | Eqs. (3.33)-(3.34) |
| Factor Graph Optimization | Complete | 2 | Eqs. (3.35)-(3.43) |

**Total: 33 test cases, all passing**

## Implementation Notes

### Linear Least Squares (LS)

- Solves overdetermined systems: Ax = b where m ≥ n
- Uses normal equations: (A'A)x = A'b
- Computes covariance: P = σ²(A'A)⁻¹
- Handles exact fit cases (m = n) and overdetermined (m > n)

### Weighted Least Squares (WLS)

- Incorporates measurement uncertainties via weight matrix W
- Typically W = R⁻¹ (inverse of measurement covariance)
- Validates W is symmetric positive definite
- Reduces to LS when W = I

### Iterative Least Squares (Gauss-Newton)

- Handles nonlinear measurement models: f(x) ≈ b
- Linearizes at each iteration using Jacobian J = ∂f/∂x
- Converges quadratically near solution
- Configurable max iterations and tolerance

### Robust Least Squares (IRLS)

Three loss functions implemented:
- **Huber**: Quadratic for small residuals, linear for large
- **Cauchy**: Heavy-tailed, more aggressive outlier rejection
- **Tukey**: Sets outliers to zero weight beyond threshold

Uses Median Absolute Deviation (MAD) for robust scale estimation.

### Kalman Filter (Linear KF)

- Optimal estimator for linear Gaussian systems
- Prediction step: Eqs. (3.11)-(3.12)
- Update step: Eqs. (3.17)-(3.19)
- Covariance propagation with Joseph form for numerical stability
- Supports both constant and time-varying system matrices

### Extended Kalman Filter (EKF)

- Handles nonlinear process and measurement models
- Linearization via Jacobian matrices: Eqs. (3.21)-(3.22)
- Prediction and update steps for nonlinear systems
- Tested on range-only and bearing-only tracking

### Unscented Kalman Filter (UKF)

- Sigma point-based approach for nonlinear systems: Eqs. (3.24)-(3.30)
- No Jacobian computation required
- Better handling of highly nonlinear transformations than EKF
- Configurable parameters (alpha, beta, kappa)

### Particle Filter (PF)

- Monte Carlo approach for non-Gaussian distributions: Eqs. (3.32)-(3.34)
- Can handle arbitrary nonlinearities and multimodal distributions
- Systematic resampling to prevent particle degeneracy
- Configurable number of particles

### Factor Graph Optimization (FGO)

- Batch optimization approach: Eqs. (3.35)-(3.43)
- Represents estimation problem as a graph of variables and factors
- Gauss-Newton and gradient descent solvers
- Can smooth entire trajectory using all measurements

## Design Decisions

### Numerical Stability

- Rank checking before solving normal equations
- Pseudo-inverse fallback for near-singular matrices
- MAD-based robust scale estimation (more robust than std dev)
- Clipping of weights to valid range [0, 1]

### Convergence Criteria

- Iterative LS: ||Δx|| < tolerance
- Robust LS: ||x_new - x_old|| < tolerance
- Default tolerance: 1e-6 for positions (millimeter accuracy)

### Error Handling

- Validates matrix dimensions before computation
- Checks rank deficiency
- Detects non-positive-definite covariance matrices
- Provides informative error messages

### Testing Philosophy

- Each function has 4-7 comprehensive test cases
- Tests cover: exact fit, noisy data, outliers, edge cases
- Numerical accuracy verified to < 1e-6 for most cases
- Round-trip tests for consistency

## Verification and Validation

### Least Squares Tests

| Test Case | Expected Result | Actual Result | Status |
|-----------|----------------|---------------|--------|
| Exact fit (y = 2x + 1) | [1.0, 2.0] | [1.0, 2.0] (< 1e-10) | ✓ |
| Overdetermined (5 equations, 2 unknowns) | Close to true | Error < 0.1 | ✓ |
| Identity matrix | Return b exactly | \|b - x̂\| < 1e-10 | ✓ |
| Rank deficient | Raise ValueError | ValueError raised | ✓ |
| WLS equal weights = LS | x_wls ≈ x_ls | \|x_wls - x_ls\| < 1e-10 | ✓ |
| WLS emphasizes accurate | Close to accurate meas. | Verified | ✓ |
| Iterative LS on linear | Converges to LS | < 5 iterations | ✓ |
| Range positioning 2D | Error < 1e-6 | < 10 iterations | ✓ |
| Robust LS (1 outlier) | Downweights outlier | w_outlier < 0.5 | ✓ |
| Huber vs standard LS | Robust < standard error | Verified | ✓ |

### Running Tests

```bash
# Run all Chapter 3 tests
pytest tests/core/estimators/ -v

# Run specific module
pytest tests/core/estimators/test_least_squares.py -v

# Run with coverage
pytest tests/core/estimators/ --cov=core.estimators --cov-report=html
```

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

3. **Performance Metrics**
   - NEES (Normalized Estimation Error Squared) - Eq. (3.17)
   - NIS (Normalized Innovation Squared) - Eq. (3.18)
   - Innovation consistency tests

## Future Work

1. Add performance metrics (NEES, NIS, innovation tests)
2. Create simulation data generators
3. Add interactive Jupyter notebooks
4. Add Levenberg-Marquardt optimization for FGO
5. Implement Recursive Least Squares (RLS)

## Contributing

When adding new estimators:

1. **Add equation reference** in docstring (e.g., "Implements Eq. (3.X)")
2. **Update ch3_estimators/README.md** with new equation mapping entry
3. **Add comprehensive unit tests** (minimum 4-5 test cases)
4. **Verify numerical accuracy** (< 1e-6 for typical cases)
5. **Update `core/estimators/__init__.py`** with exports

## References

- **Chapter 3**: State Estimation
  - Section 3.2: Least Squares Methods
  - Section 3.3: Kalman Filtering
  - Section 3.4: Nonlinear Filters (EKF, UKF)
  - Section 3.5: Particle Filters
  - Section 3.6: Factor Graph Optimization

- **Numerical Recipes**: Press et al. (2007)
- **Robust Statistics**: Huber & Ronchetti (2009)

---

**Last Updated**: December 2025  
**Maintainer**: Navigation Engineering Team

