# Chapter 4: Development Notes

> **Note:** This document contains implementation details, design decisions, and development notes for Chapter 4. For student-facing documentation, see [ch4_rf_point_positioning/README.md](../ch4_rf_point_positioning/README.md).

## Implementation Status

| Feature | Status | Test Cases | Notes |
|---------|--------|------------|-------|
| TOA Range Measurement | Complete | 6 | Eqs. (4.1)-(4.3), (4.6)-(4.7) |
| RSS Path-Loss Model | Complete | 5 | Eqs. (4.11)-(4.13) |
| TDOA Measurements | Complete | 4 | Eqs. (4.27)-(4.33) |
| AOA Measurements | Complete | 4 | Eqs. (4.63)-(4.66) |
| TOA Positioning | Complete | 6 | Eqs. (4.14)-(4.26) |
| TDOA Positioning | Complete | 5 | Eqs. (4.34)-(4.42) |
| AOA Positioning | Complete | 5 | Eqs. (4.63)-(4.67) |
| DOP Computation | Complete | 5 | Section 4.5 |
| Fang Algorithm | Planned | - | Eqs. (4.43)-(4.48) |
| Chan Algorithm | Planned | - | Eq. (4.58) |

## Implementation Notes

### Coordinate Frames
- All positions use ENU (East-North-Up) local frame
- 2D positioning: shape (N, 2) for [x, y]
- 3D positioning: shape (N, 3) for [x, y, z]

### Iterative Solvers
- Default: Iterative Weighted LS (I-WLS)
- Max iterations: 10 (typically converges in 3-5)
- Convergence tolerance: 1e-6 meters

### Measurement Noise Models
- TOA: Gaussian range noise (σ ≈ 0.05-0.5 m for UWB/Wi-Fi)
- RSS: Log-normal shadow fading (σ ≈ 3-8 dB)
- TDOA: Correlated range difference noise
- AOA: Gaussian angle noise (σ ≈ 1-5° typical)

## Performance Characteristics

### Typical Performance (with noise)
- TOA with 10cm noise: RMSE ≈ 0.2-0.4 m (4 anchors, square)
- RSS with 3dB noise: RMSE ≈ 1-2 m (path-loss model)
- TDOA with 5cm noise: RMSE ≈ 0.2-0.3 m (4 anchors, square)
- AOA with 2° noise: RMSE ≈ 0.5-1.0 m (4 anchors, 10m distance)

## Future Work

1. Complete closed-form algorithms (Fang, Chan for TDOA)
2. Add OVE and 3D PLE methods for AOA
3. Hybrid positioning (TOA+AOA, RSS+AOA)

---

**Test Coverage:** 52 test cases, >95% code coverage  
**Last Updated:** December 2025


