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
- Joseph form for numerical stability

## Design Decisions

### Numerical Stability

- Rank checking before solving normal equations
- Pseudo-inverse fallback for near-singular matrices
- MAD-based robust scale estimation

### Testing Philosophy

- Each function has 4-7 comprehensive test cases
- Tests cover: exact fit, noisy data, outliers, edge cases
- Numerical accuracy verified to < 1e-6 for most cases

## Future Work

1. Add performance metrics (NEES, NIS, innovation tests)
2. Add Levenberg-Marquardt optimization
3. Implement Recursive Least Squares (RLS)

---

**Last Updated**: December 2025


