# Ch3 Estimator Comparison Dataset: When to Use Which Estimator?

## Overview

This dataset demonstrates **when to use KF vs. EKF vs. UKF vs. PF** by comparing their performance on the same nonlinear tracking problem. It showcases the critical importance of matching the estimator to the system characteristics.

**Key Learning Objective**: Understand that estimator choice depends on system linearity and noise characteristics - wrong choice can degrade performance by 10×!

## Dataset Purpose

### Learning Goals
1. **KF Assumptions**: Linear system + Gaussian noise (optimal when valid)
2. **EKF Limitations**: Breaks down with high nonlinearity (linearization error)
3. **UKF Advantage**: Better handling of nonlinearity than EKF
4. **PF Robustness**: Handles non-Gaussian noise and outliers
5. **Performance Trade-offs**: Accuracy vs. computational cost

### Implemented Equations
- **Eqs. (3.11-3.19)**: Linear Kalman Filter
  ```
  Prediction: x̂ₖ|ₖ₋₁ = F x̂ₖ₋₁, Pₖ|ₖ₋₁ = F Pₖ₋₁ F' + Q
  Update: x̂ₖ = x̂ₖ|ₖ₋₁ + K(z - H x̂ₖ|ₖ₋₁)
  ```

- **Eq. (3.21)**: Extended Kalman Filter
  ```
  Uses Jacobians: Fₖ = ∂f/∂x, Hₖ = ∂h/∂x
  Linearizes around current estimate
  ```

- **Eqs. (3.24-3.30)**: Unscented Kalman Filter
  ```
  Uses sigma points (no Jacobians needed!)
  Better captures nonlinearity via sampling
  ```

- **Eqs. (3.32-3.34)**: Particle Filter
  ```
  Represents posterior with weighted samples
  Handles arbitrary nonlinearity and noise
  ```

## Dataset Variants

| Variant | Trajectory | Nonlinearity | Outliers | KF | EKF | UKF | PF | Key Lesson |
|---------|------------|--------------|----------|-----|-----|-----|----|-----------| 
| **Linear** | Straight line | None | 0% | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | **KF optimal for linear!** |
| **Nonlinear** | Circle | Moderate | 0% | ❌ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | **EKF/UKF handle nonlinearity** |
| **High Nonlinear** | Figure-8 | High | 0% | ❌ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | **UKF better than EKF!** |
| **Outliers** | Circle | Moderate | 10% | ❌ | ⭐ | ⭐ | ⭐⭐⭐ | **PF robust to outliers!** |

**Generate variants**:
```bash
python scripts/generate_ch3_estimator_comparison_dataset.py --preset linear
python scripts/generate_ch3_estimator_comparison_dataset.py --preset nonlinear
python scripts/generate_ch3_estimator_comparison_dataset.py --preset high_nonlinearity
python scripts/generate_ch3_estimator_comparison_dataset.py --preset outliers
```

## Files

### Trajectory Data
- `time.txt`: Time vector [N×1] (seconds)
- `ground_truth_states.txt`: True states [N×4] (x, y, vx, vy in m, m/s)

### Environment Data
- `beacons.txt`: Beacon positions [M×2] (x, y in m)

### Measurements
- `range_measurements.txt`: Range measurements [N×M] (meters)
- `bearing_measurements.txt`: Bearing measurements [N×M] (radians)

### Configuration
- `config.json`: All dataset parameters and statistics

## Loading Data

### Python
```python
import numpy as np
from pathlib import Path

# Load dataset
data_dir = Path("data/sim/ch3_estimator_nonlinear")

time = np.loadtxt(data_dir / "time.txt")
states = np.loadtxt(data_dir / "ground_truth_states.txt")
beacons = np.loadtxt(data_dir / "beacons.txt")
ranges = np.loadtxt(data_dir / "range_measurements.txt")
bearings = np.loadtxt(data_dir / "bearing_measurements.txt")

print(f"Loaded {len(time)} samples over {time[-1]:.1f}s")
print(f"State: position (x,y) and velocity (vx,vy)")
print(f"Beacons: {len(beacons)}")
print(f"Measurements: ranges and bearings to beacons")
```

## Configuration Parameters

### Trajectory Configuration
```json
{
  "trajectory": {
    "type": "circular",
    "duration_s": 30.0,
    "dt_s": 0.1,
    "num_samples": 300
  }
}
```

**Key Parameters**:
- **type**: Trajectory shape (linear, circular, figure8)
- **duration**: 30s trajectory
- **dt**: 0.1s sampling (10 Hz)

### Measurement Configuration
```json
{
  "measurements": {
    "range_noise_std_m": 0.5,
    "bearing_noise_std_deg": 5.0,
    "outlier_rate": 0.0
  }
}
```

**Key Parameters**:
- **range_noise**: 0.5m std dev (typical UWB)
- **bearing_noise**: 5° std dev
- **outlier_rate**: 0% (or 10% for outliers variant)

## Quick Start Examples

### Example 1: Run All Four Estimators
```python
from core.estimators import (
    KalmanFilter, ExtendedKalmanFilter,
    UnscentedKalmanFilter, ParticleFilter
)
import numpy as np

# Load data
states = np.loadtxt("data/sim/ch3_estimator_nonlinear/ground_truth_states.txt")
ranges = np.loadtxt("data/sim/ch3_estimator_nonlinear/range_measurements.txt")
beacons = np.loadtxt("data/sim/ch3_estimator_nonlinear/beacons.txt")

# Initialize estimators (simplified)
# KF: Assumes linear motion model (will fail on circular!)
# EKF: Linearizes nonlinear range measurements
# UKF: Uses sigma points for nonlinear measurements
# PF: Uses 1000 particles

# Run all estimators and compare errors
# (See ch3_estimators/example_comparison.py for full implementation)
```

**Expected Results** (Nonlinear/Circular):
- KF: Large error (~5m) - **wrong assumption!**
- EKF: Moderate error (~0.8m) - handles nonlinearity
- UKF: Low error (~0.6m) - better than EKF
- PF: Low error (~0.7m) - accurate but slower

**Learning Point**: KF fails when assumptions violated!

### Example 2: Visualize Trajectory and Measurements
```python
import matplotlib.pyplot as plt
import numpy as np

# Load data
states = np.loadtxt("data/sim/ch3_estimator_nonlinear/ground_truth_states.txt")
beacons = np.loadtxt("data/sim/ch3_estimator_nonlinear/beacons.txt")
ranges = np.loadtxt("data/sim/ch3_estimator_nonlinear/range_measurements.txt")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Trajectory
ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='True trajectory')
ax1.scatter(beacons[:, 0], beacons[:, 1], s=200, c='red', marker='^', label='Beacons')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Circular Trajectory (Nonlinear!)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Range measurements over time
ax2.plot(ranges)
ax2.set_xlabel('Time step')
ax2.set_ylabel('Range (m)')
ax2.set_title('Range Measurements to 4 Beacons')
ax2.legend([f'Beacon {i}' for i in range(4)])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Learning Point**: Circular motion → nonlinear range measurements!

## Parameter Effects

### Effect of System Nonlinearity

| Trajectory | Nonlinearity | KF Error | EKF Error | UKF Error | PF Error | Winner |
|------------|--------------|----------|-----------|-----------|----------|--------|
| **Linear** | None | ~0.5m | ~0.5m | ~0.6m | ~0.8m | **KF (optimal!)** |
| **Circular** | Moderate | ~5.0m | ~0.8m | ~0.6m | ~0.7m | **UKF/PF** |
| **Figure-8** | High | ~8.0m | ~1.5m | ~0.7m | ~0.8m | **UKF/PF** |

**Generate comparison**:
```bash
python scripts/generate_ch3_estimator_comparison_dataset.py --preset linear
python scripts/generate_ch3_estimator_comparison_dataset.py --preset nonlinear
python scripts/generate_ch3_estimator_comparison_dataset.py --preset high_nonlinearity
```

**Learning Point**: Higher nonlinearity → bigger gap between EKF and UKF!

### Effect of Outliers

| Outlier Rate | KF Error | EKF Error | UKF Error | PF Error | Winner |
|--------------|----------|-----------|-----------|----------|--------|
| 0% (clean) | ~5m | ~0.8m | ~0.6m | ~0.7m | UKF |
| 10% | ~8m | ~2.0m | ~1.5m | ~0.8m | **PF (robust!)** |

**Generate comparison**:
```bash
python scripts/generate_ch3_estimator_comparison_dataset.py --preset nonlinear
python scripts/generate_ch3_estimator_comparison_dataset.py --preset outliers
```

**Learning Point**: PF naturally robust to outliers (Particle diversity)!

### Computational Cost

| Estimator | State Dim | Complexity | Relative Speed | When to Use |
|-----------|-----------|------------|----------------|-------------|
| **KF** | 4 | O(n³) | 1× (fastest) | **Linear systems only** |
| **EKF** | 4 | O(n³) | ~1× | **Moderate nonlinearity** |
| **UKF** | 4 | O(n³) | ~3× | **High nonlinearity** |
| **PF** | 4, 1000 particles | O(Np) | ~100× | **Non-Gaussian, outliers** |

**Learning Point**: Trade-off between accuracy and speed!

## Experiments

### Experiment 1: Estimator Comparison on Nonlinear Problem

**Objective**: Compare all four estimators on circular trajectory.

**Procedure**:
1. Load nonlinear dataset (circular motion)
2. Run KF, EKF, UKF, PF with same initial conditions
3. Compute RMSE for each estimator
4. Compare computation times

**Expected Results**:
- KF: ~5m RMSE (fails due to wrong assumption)
- EKF: ~0.8m RMSE (handles nonlinearity)
- UKF: ~0.6m RMSE (25% better than EKF!)
- PF: ~0.7m RMSE (accurate but 100× slower)

**Code**: See `ch3_estimators/example_comparison.py`

**Learning Point**: UKF worth the 3× cost for high nonlinearity!

### Experiment 2: When Does EKF Break Down?

**Objective**: Find the nonlinearity level where EKF degrades significantly.

**Procedure**:
1. Generate trajectories with increasing nonlinearity (linear → circular → figure-8)
2. Run EKF and UKF on each
3. Plot error vs. nonlinearity level

**Expected Results**:
- Linear: EKF ≈ UKF (~0.5m)
- Circular: EKF slightly worse (~0.8m vs ~0.6m)
- Figure-8: EKF much worse (~1.5m vs ~0.7m)

**Learning Point**: EKF linearization error grows with nonlinearity!

### Experiment 3: Particle Filter Robustness

**Objective**: Study PF performance with outliers.

**Procedure**:
1. Generate datasets with 0%, 5%, 10% outliers
2. Run EKF, UKF, PF on each
3. Compare degradation

**Expected Results**:
- 0% outliers: EKF/UKF best
- 10% outliers: PF degrades less (0.7m → 0.8m vs. 0.8m → 2.0m for EKF)

**Learning Point**: PF particle diversity naturally handles outliers!

## Performance Metrics (Nonlinear/Circular)

| Metric | KF | EKF | UKF | PF | Notes |
|--------|-----|-----|-----|-----|----|
| **Position RMSE** | ~5.0m | ~0.8m | ~0.6m | ~0.7m | UKF best |
| **Velocity RMSE** | ~2.0m/s | ~0.3m/s | ~0.2m/s | ~0.3m/s | UKF best |
| **Max Error** | ~10m | ~2.0m | ~1.5m | ~1.8m | UKF best |
| **Computation** | 1× | ~1× | ~3× | ~100× | KF fastest |
| **Convergence** | No | Yes | Yes | Yes | KF fails |

**Key Insights**:
- KF fails (wrong assumption: linear system)
- EKF handles moderate nonlinearity
- UKF 25% better than EKF (better linearization)
- PF accurate but 100× slower

**Dataset Specifications**:
- 300 samples over 30s (10 Hz)
- 4 beacons in square configuration
- Circular trajectory (10m radius, 0.3 rad/s)
- Range noise: 0.5m, Bearing noise: 5°

## Book Connection

### Chapter 3: State Estimation

This dataset directly demonstrates Chapter 3 estimator trade-offs:

1. **Linear Kalman Filter (Eqs. 3.11-3.19)**
   - **Optimal** for linear Gaussian systems
   - Assumes: xₖ = F xₖ₋₁ + w, z = H x + v
   - **Fails** when assumptions violated!

2. **Extended Kalman Filter (Eq. 3.21)**
   - Linearizes: Fₖ = ∂f/∂x|ₓ, Hₖ = ∂h/∂x|ₓ
   - Handles moderate nonlinearity
   - **Linearization error** grows with nonlinearity

3. **Unscented Kalman Filter (Eqs. 3.24-3.30)**
   - Uses sigma points (deterministic sampling)
   - **No Jacobians** needed!
   - Better captures nonlinearity than EKF

4. **Particle Filter (Eqs. 3.32-3.34)**
   - Represents posterior with samples
   - Handles arbitrary nonlinearity and noise
   - **Most flexible** but computationally expensive

**Decision Tree**:
```
Is system linear?
├─ Yes → Use KF (optimal!)
└─ No → Is computational cost a concern?
    ├─ Yes → Use EKF (if moderate nonlinearity)
    │         or UKF (if high nonlinearity)
    └─ No → Is noise non-Gaussian or have outliers?
        ├─ Yes → Use PF (most robust!)
        └─ No → Use UKF (best accuracy/cost trade-off)
```

## Common Issues & Solutions

### Issue 1: KF Gives Large Errors

**Symptoms**: KF error ~5m while EKF/UKF error ~0.6m

**Likely Cause**: System is nonlinear (circular motion)

**Solution**: Use EKF or UKF instead:
```python
# Wrong: KF assumes linear
kf = KalmanFilter(F_linear, Q, H_linear, R)  # Will fail!

# Right: EKF handles nonlinearity
ekf = ExtendedKalmanFilter(f_nonlinear, Q, h_nonlinear, R)  # ✓
```

**Learning Point**: Always match estimator to system characteristics!

### Issue 2: EKF Diverges on Figure-8

**Symptoms**: EKF error grows over time, UKF remains stable

**Likely Cause**: High nonlinearity breaks EKF linearization

**Solution**: Use UKF (or PF):
```python
# EKF may diverge with high nonlinearity
ekf = ExtendedKalmanFilter(f, Q, h, R)  # Struggles

# UKF handles high nonlinearity better
ukf = UnscentedKalmanFilter(f, Q, h, R)  # ✓
```

**Learning Point**: UKF worth 3× cost for high nonlinearity!

### Issue 3: All Estimators Fail with Outliers

**Symptoms**: Occasional large spikes in error for EKF/UKF

**Likely Cause**: Outlier measurements (10% in outliers variant)

**Solution**: Use Particle Filter or robust estimation:
```python
# EKF/UKF sensitive to outliers
ekf.update(z_outlier)  # Large error spike

# PF naturally robust (particle diversity)
pf.update(z_outlier)  # Minimal impact ✓
```

## Troubleshooting

### Error: EKF requires Jacobian functions

**Cause**: EKF needs ∂f/∂x and ∂h/∂x

**Fix**: Provide Jacobian functions:
```python
def f_jacobian(x):
    # Return Jacobian of process model
    return F_k

def h_jacobian(x):
    # Return Jacobian of measurement model  
    return H_k

ekf = ExtendedKalmanFilter(f, Q, h, R, f_jac=f_jacobian, h_jac=h_jacobian)
```

### Warning: UKF sigma points negative

**Cause**: Covariance matrix not positive definite

**Fix**: Add small regularization:
```python
P = P + 1e-6 * np.eye(state_dim)  # Regularize
```

## Next Steps

After understanding estimator trade-offs:

1. **Chapter 8**: Apply estimators to sensor fusion (IMU + UWB)
2. **Chapter 6**: Use estimators for dead-reckoning
3. **Adaptive Filtering**: Switch estimators based on conditions
4. **Robust Estimation**: M-estimators for outlier rejection

## Citation

```bibtex
@book{IPIN2024,
  title={Principles of Indoor Positioning and Indoor Navigation},
  author={[Authors]},
  year={2024},
  chapter={3},
  note={State Estimation}
}
```

## License

This dataset is part of the IPIN Book Examples repository. See repository LICENSE for details.

---

**Dataset Version**: 1.0  
**Last Updated**: December 2024  
**Contact**: See repository README for contact information

