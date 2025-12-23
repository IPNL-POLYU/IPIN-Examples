# Chapter 3: State Estimation

## Overview

This module implements the state estimation algorithms described in **Chapter 3** of *Principles of Indoor Positioning and Indoor Navigation*. It provides the mathematical foundations for estimating position, velocity, and other states from noisy measurements using various filtering and optimization techniques.

## Quick Start

```bash
# Run individual examples
python -m ch3_estimators.example_least_squares
python -m ch3_estimators.example_kalman_1d
python -m ch3_estimators.example_ekf_range_bearing

# Run with pre-generated dataset
python -m ch3_estimators.example_ekf_range_bearing --data ch3_estimator_nonlinear

# Run comprehensive comparison of all estimators
python -m ch3_estimators.example_comparison
```

## 📂 Dataset Connection

| Example Script | Dataset | Description |
|----------------|---------|-------------|
| `example_ekf_range_bearing.py` | `data/sim/ch3_estimator_nonlinear/` | Moderate nonlinearity (circular trajectory) |
| `example_ekf_range_bearing.py` | `data/sim/ch3_estimator_high_nonlinear/` | High nonlinearity (figure-8 trajectory) |

**Load dataset manually:**
```python
import numpy as np
import json
from pathlib import Path

path = Path("data/sim/ch3_estimator_nonlinear")
t = np.loadtxt(path / "time.txt")
beacons = np.loadtxt(path / "beacons.txt")
true_states = np.loadtxt(path / "ground_truth_states.txt")
range_meas = np.loadtxt(path / "range_measurements.txt")
bearing_meas = np.loadtxt(path / "bearing_measurements.txt")
config = json.load(open(path / "config.json"))
```

## Equation Reference

### Least Squares Methods

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `linear_least_squares()` | `core/estimators/least_squares.py` | Eq. (3.1) | Standard LS: x̂ = (A'A)⁻¹A'b |
| `weighted_least_squares()` | `core/estimators/least_squares.py` | Eq. (3.2) | WLS with measurement covariance |
| `iterative_least_squares()` | `core/estimators/least_squares.py` | Eq. (3.3) | Gauss-Newton for nonlinear problems |
| `robust_least_squares()` | `core/estimators/least_squares.py` | Eq. (3.4) | IRLS with Huber/Cauchy/Tukey loss |

### Kalman Filtering

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `KalmanFilter.predict()` | `core/estimators/kalman_filter.py` | Eq. (3.11)-(3.12) | Linear KF prediction step |
| `KalmanFilter.update()` | `core/estimators/kalman_filter.py` | Eq. (3.17)-(3.19) | Linear KF update step |

### Extended Kalman Filter (EKF)

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `ExtendedKalmanFilter.predict()` | `core/estimators/extended_kalman_filter.py` | Eq. (3.21)-(3.22) | Nonlinear prediction with Jacobian |
| `ExtendedKalmanFilter.update()` | `core/estimators/extended_kalman_filter.py` | Eq. (3.21) | Nonlinear update with measurement Jacobian |

### Unscented Kalman Filter (UKF)

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `UnscentedKalmanFilter._generate_sigma_points()` | `core/estimators/unscented_kalman_filter.py` | Eq. (3.24) | Sigma point generation |
| `UnscentedKalmanFilter.predict()` | `core/estimators/unscented_kalman_filter.py` | Eq. (3.25) | UT-based prediction |
| `UnscentedKalmanFilter.update()` | `core/estimators/unscented_kalman_filter.py` | Eq. (3.30) | UT-based measurement update |

### Particle Filter (PF)

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `ParticleFilter.predict()` | `core/estimators/particle_filter.py` | Eq. (3.33) | Particle propagation |
| `ParticleFilter.update()` | `core/estimators/particle_filter.py` | Eq. (3.34) | Importance weighting |

### Factor Graph Optimization (FGO)

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `FactorGraph.optimize()` | `core/estimators/factor_graph.py` | Eq. (3.35) | MAP estimation |
| `FactorGraph._gauss_newton()` | `core/estimators/factor_graph.py` | Eq. (3.38) | Gauss-Newton optimization |

## Examples

### Example 1: Least Squares Methods

```bash
python -m ch3_estimators.example_least_squares
```

Demonstrates Linear LS, Weighted LS, Iterative LS, and Robust LS for 2D positioning from range measurements. 

**Key Insight:** The Robust LS example uses **8 anchors** (not 4) to demonstrate proper outlier rejection. This illustrates a critical requirement: robust estimation needs sufficient measurement redundancy to isolate and downweight outliers. With only 4 anchors, there are insufficient degrees of freedom for the robust estimator to distinguish the outlier from valid measurements.

### Example 2: 1D Kalman Filter Tracking

```bash
python ch3_estimators/example_kalman_1d.py
```

Demonstrates constant-velocity Kalman Filter for 1D position and velocity estimation.

### Example 3: EKF Range-Bearing Tracking

```bash
python ch3_estimators/example_ekf_range_bearing.py
```

Demonstrates Extended Kalman Filter for 2D trajectory estimation with nonlinear range measurements.

### Example 4: Estimator Comparison

```bash
python ch3_estimators/example_comparison.py
```

Compares EKF, UKF, Particle Filter, and Factor Graph Optimization on the same 2D tracking problem.

## Expected Output

### Example 1: Least Squares Methods

**Console output:**
```
======================================================================
CHAPTER 3: LEAST SQUARES EXAMPLES
======================================================================

Example 1: Linear Least Squares (2D Positioning)
  True position: [3.0, 4.0] m
  LS estimate:   [3.02, 3.92] m
  Error: 0.080 m

Example 4: Robust Least Squares (8 anchors, 5.0m outlier)
  Standard LS error: 1.29 m (corrupted by outlier)
  Huber LS error:    0.08 m (93.5% improvement)
  Cauchy LS error:   0.03 m (97.4% improvement)
  Tukey LS error:    0.04 m (97.2% improvement)
  Outlier weight:    0.025 (Huber), 0.0016 (Cauchy), 0.0 (Tukey)
  
Note: Uses 8 anchors (not 4) to provide sufficient redundancy for robust estimation
```

**Generated figure:** `figs/ch3_least_squares_examples.png`

![Least Squares Examples](figs/ch3_least_squares_examples.png)

**Figure Description:**

### Left Panel: Example 1-3 (LS Methods - Clean Data)
- **Geometry**: 4 anchors (blue triangles) at corners of test area
- **Measurements**: Blue dashed circles show clean range measurements to each anchor
- **Markers**:
  - Green star (★): True position at (3, 4) m
  - Gray × : Initial guess at (5, 5) m
  - Orange ●: Linear LS estimate (very close to true position)
  - Red ■: Iterative LS estimate (overlaps with true position)
- **Result**: Both methods converge accurately when data is clean

### Right Panel: Example 4 (Robust LS with Outlier)
- **Geometry**: **8 anchors** (blue triangles) - note the increased redundancy!
- **Outlier Visualization**:
  - Top-right anchor highlighted with **red circle** (⭕)
  - **Red dashed arc** shows the corrupted 5.0m outlier measurement
  - Blue dashed arcs show correct measurements from other 7 anchors
- **Markers**:
  - Green star (★): True position at (3, 4) m
  - Gray × : Initial guess at (5, 5) m
  - Yellow ●: Standard LS estimate - **pulled toward outlier** at ~(3.5, 2.8) m, error = 1.29m
  - Purple ♦: Robust LS (Huber) - **rejects outlier**, stays near true position, error = 0.08m
- **Key Visual Insight**: The yellow Standard LS is visibly displaced from the true position (green star) toward the outlier direction, while the purple Robust LS successfully stays close to the true position by downweighting the red-circled outlier measurement.

### Critical Observation: 8 Anchors Enable Robust Estimation

**Why 8 anchors instead of 4?**
- With 4 anchors (left panel), only 2 degrees of freedom remain after solving for position
- With 8 anchors (right panel), 6 degrees of freedom provide sufficient redundancy
- The 7 good measurements overpower the 1 outlier, making it clearly identifiable
- Robust loss functions can distinguish the large residual and downweight it to near-zero
- **Result**: 93-97% error reduction compared to Standard LS

This visual comparison demonstrates why real-world robust positioning systems require generous measurement redundancy (typically 2-3× the minimum) to reliably handle outliers.

---

### Example 2: 1D Kalman Filter

**Console output:**
```
======================================================================
CHAPTER 3: 1D KALMAN FILTER TRACKING
======================================================================

Scenario: Constant velocity motion with noisy position measurements
  Duration: 10.0 s
  Measurement noise: 0.5 m

Results:
  Mean position error: 0.15 m
  Mean velocity error: 0.32 m/s
  Filter converges within 2-3 seconds
```

**Generated figure:** `figs/ch3_kalman_1d_tracking.png`

![1D Kalman Filter Tracking](figs/ch3_kalman_1d_tracking.png)

- **Top-left**: Position tracking with ±2σ confidence bounds
- **Top-right**: Velocity estimation (unobserved state estimated from position changes)
- **Bottom panels**: Estimation errors over time - errors stay within measurement noise level

---

### Example 3: EKF Range-Bearing

**Console output:**
```
======================================================================
CHAPTER 3: EKF RANGE-BEARING TRACKING
======================================================================

Scenario: 2D curved trajectory with range measurements to 4 landmarks
  Duration: 20.0 s
  Range measurement std: 0.5 m

Results:
  Mean position RMSE: 0.35 m
  Mean velocity RMSE: 0.72 m/s
```

**Generated figure:** `figs/ch3_ekf_range_bearing.png`

![EKF Range-Bearing Tracking](figs/ch3_ekf_range_bearing.png)

- **Top-left**: 2D trajectory comparison (true vs EKF estimate)
- **Top-right**: Position estimation error over time
- **Bottom-left**: X and Y position components over time
- **Bottom-right**: Velocity estimation error

---

### Example 4: Estimator Comparison

**Console output:**
```
======================================================================
CHAPTER 3: COMPARISON OF STATE ESTIMATORS
======================================================================

Scenario: 2D tracking with range measurements from 4 anchors
  Duration: 15.0 s
  Range measurement std: 0.5 m

Results Summary:
  Estimator    RMSE (m)    Computation Time
  ---------    --------    ----------------
  EKF          0.32        0.016 s
  UKF          0.31        0.017 s
  PF           0.45        1.178 s
  FGO          0.28        0.231 s
```

**Generated figure:** `figs/ch3_estimator_comparison.png`

![Estimator Comparison](figs/ch3_estimator_comparison.png)

- **Top-left**: Trajectory comparison - all estimators track the true path
- **Top-right**: Position error over time for each estimator
- **Bottom-left**: Cumulative Distribution Function (CDF) of errors
- **Bottom-right**: Computational cost comparison - EKF/UKF fastest, PF slowest

## File Structure

```
ch3_estimators/
├── README.md                        # This file
├── example_least_squares.py         # LS/WLS/ILS/Robust LS demonstrations
├── example_kalman_1d.py             # 1D constant velocity tracking
├── example_ekf_range_bearing.py     # 2D positioning with EKF
├── example_comparison.py            # Compare all estimators
└── figs/                            # Generated figures
    ├── ch3_least_squares_examples.png
    ├── ch3_kalman_1d_tracking.png
    ├── ch3_ekf_range_bearing.png
    └── ch3_estimator_comparison.png

core/estimators/
├── least_squares.py                 # LS/WLS/ILS/Robust LS
├── kalman_filter.py                 # Linear Kalman Filter
├── extended_kalman_filter.py        # EKF
├── unscented_kalman_filter.py       # UKF
├── particle_filter.py               # Particle Filter
└── factor_graph.py                  # Factor Graph Optimization
```

## Important Notes on Robust Estimation

### Minimum Anchor Requirements for Robust Methods

Robust least squares requires **sufficient measurement redundancy** to isolate and downweight outliers effectively. This is a critical requirement often overlooked in textbook examples.

#### Recommended Minimum Anchors:
- **2D positioning**: 6-8 anchors recommended (2 unknowns + 4-6 DOF redundancy)
- **3D positioning**: 8-10 anchors recommended (3 unknowns + 5-7 DOF redundancy)

#### Why Standard Minimum Is Insufficient:

With only the theoretical minimum number of anchors (4 for 2D, 5 for 3D), there is insufficient overdetermination for robust methods to work reliably:

**Problem with 4 anchors in 2D:**
- Only 2 degrees of freedom (DOF) for redundancy
- When an outlier is present, the LS solution converges to a biased position
- From this biased position, **all 4 residuals appear similar** in magnitude
- No single residual stands out as anomalously large
- Robust loss functions cannot distinguish the outlier from valid measurements
- Result: Robust LS performs identically to standard LS (fails to reject outlier)

**Solution with 8 anchors in 2D:**
- 6 degrees of freedom for redundancy
- Even with one outlier, the majority of measurements constrain the solution accurately
- The outlier measurement produces a **clearly distinguishable large residual**
- Robust loss functions (Huber/Cauchy/Tukey) successfully identify and downweight it
- Result: 93-97% error reduction compared to standard LS (as shown in Example 4)

#### Practical Demonstration

See **Example 4** in `example_least_squares.py` and the right panel of the figure above, which clearly demonstrates:
- Standard LS with outlier: 1.29m error
- Robust LS (8 anchors): 0.03-0.08m error
- Improvement: 93-97%

This improvement is only possible with sufficient anchor redundancy.

## Additional Documentation

### User Guides
- **[Estimator Selection Guide](../docs/guides/ch3_estimator_selection.md)** - Comprehensive guide for choosing the right estimator for your application (500+ lines covering when to use LS/KF/EKF/UKF/PF/FGO)

### Engineering/Technical Documentation
For developers and maintainers, detailed implementation notes are available in [`docs/engineering/`](../docs/engineering/):

- **[Complete Implementation Summary](../docs/engineering/complete_implementation_summary.md)** - Master overview of all ch3 improvements
- **[Production Fixes](../docs/engineering/ch3_production_fixes.md)** - Critical fixes (angle wrapping, singularity handling, observability checks)
- **[Robustness Improvements](../docs/engineering/ch3_robustness_improvements.md)** - Input validation, shared models, unit tests
- **[Bugfix Summary](../docs/engineering/ch3_bugfix_summary.md)** - Robust LS fix (4 anchors → 8 anchors)

## Book References

- **Section 3.2**: Least Squares Methods
- **Section 3.3**: Kalman Filtering
- **Section 3.4**: Nonlinear Filters (EKF, UKF)
- **Section 3.5**: Particle Filters
- **Section 3.6**: Factor Graph Optimization

