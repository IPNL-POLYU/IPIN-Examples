# Chapter 8: Sensor Fusion API Reference

Quick reference for `core.fusion` module functions and data structures.

---

## Data Structures

### `StampedMeasurement`

```python
@dataclass(frozen=True)
class StampedMeasurement:
    t: float              # timestamp (seconds)
    sensor: str           # sensor identifier
    z: np.ndarray         # measurement vector (m,)
    R: np.ndarray         # covariance matrix (m×m)
    meta: dict            # optional metadata
```

**Usage**:
```python
from core.fusion import StampedMeasurement
import numpy as np

meas = StampedMeasurement(
    t=1.234,
    sensor='uwb_range',
    z=np.array([5.67]),
    R=np.array([[0.01]]),
    meta={'anchor_id': 3}
)
```

### `TimeSyncModel`

```python
@dataclass(frozen=True)
class TimeSyncModel:
    offset: float = 0.0   # time offset (seconds)
    drift: float = 0.0    # clock drift (dimensionless)
    
    def to_fusion_time(self, t_sensor: float) -> float
    def to_sensor_time(self, t_fusion: float) -> float
    def is_synchronized(self, tolerance: float = 1e-6) -> bool
```

**Transform**: `t_fusion = (1 + drift) * t_sensor + offset`

---

## Innovation Monitoring (Eqs. 8.5-8.6)

### `innovation(z, z_pred)` → Eq. (8.5)

Compute measurement innovation (residual).

```python
y = innovation(z, z_pred)  # y = z - h(x̂)
```

- **Args**: `z` (m,), `z_pred` (m,)
- **Returns**: Innovation `y` (m,)
- **Example**:
  ```python
  z = np.array([5.2, 3.1])
  z_pred = np.array([5.0, 3.0])
  y = innovation(z, z_pred)  # [0.2, 0.1]
  ```

### `innovation_covariance(H, P_pred, R)` → Eq. (8.6)

Compute innovation covariance.

```python
S = innovation_covariance(H, P_pred, R)  # S = H P H^T + R
```

- **Args**: `H` (m×n), `P_pred` (n×n), `R` (m×m)
- **Returns**: Innovation covariance `S` (m×m)
- **Example**:
  ```python
  H = np.eye(2)
  P_pred = np.diag([0.5, 0.3])
  R = np.diag([0.1, 0.1])
  S = innovation_covariance(H, P_pred, R)  # [[0.6, 0], [0, 0.4]]
  ```

---

## Robust Weighting (Eq. 8.7)

### `scale_measurement_covariance(R, weight)` → Eq. (8.7)

Scale measurement covariance for robust down-weighting.

```python
R_scaled = scale_measurement_covariance(R, weight)  # R ← w * R
```

- **Args**: `R` (m×m), `weight` (scalar ≥ 0)
- **Returns**: Scaled covariance `R_scaled` (m×m)
- **Usage**: `weight > 1` reduces confidence (outlier), `weight < 1` increases confidence
- **Example**:
  ```python
  R = np.diag([0.1, 0.2])
  R_robust = scale_measurement_covariance(R, weight=10.0)
  # [[1.0, 0], [0, 2.0]] - inflated for outlier
  ```

### `huber_weight(residual, threshold)`

Huber robust weight function.

```python
w = huber_weight(r, k)  # w=1 if |r|≤k, else k/|r|
```

- **Args**: `residual` (scalar), `threshold` k (typical: 1.345)
- **Returns**: Weight w ∈ (0, 1]
- **Example**:
  ```python
  huber_weight(0.5, threshold=1.345)   # 1.0 (inlier)
  huber_weight(3.0, threshold=1.345)   # 0.448 (outlier)
  ```

### `cauchy_weight(residual, scale)`

Cauchy robust weight function.

```python
w = cauchy_weight(r, c)  # w = 1 / (1 + (r/c)²)
```

- **Args**: `residual` (scalar), `scale` c (typical: 2.385)
- **Returns**: Weight w ∈ (0, 1]
- **Example**:
  ```python
  cauchy_weight(0.0, scale=2.385)   # 1.0 (perfect)
  cauchy_weight(2.385, scale=2.385) # 0.5 (at scale)
  cauchy_weight(10.0, scale=2.385)  # 0.054 (strong outlier)
  ```

### `compute_normalized_innovation(y, S)`

Normalize innovation by covariance (whitening).

```python
y_norm = compute_normalized_innovation(y, S)  # y_norm = L^{-1} y
```

- **Args**: `y` (m,), `S` (m×m, positive definite)
- **Returns**: Normalized innovation `y_norm` (m,)
- **Use**: Prepare innovation for element-wise robust weight computation

---

## Chi-Square Gating (Eqs. 8.8-8.9)

### `mahalanobis_distance_squared(y, S)` → Eq. (8.8)

Compute squared Mahalanobis distance.

```python
d_sq = mahalanobis_distance_squared(y, S)  # d² = y^T S^{-1} y
```

- **Args**: `y` (m,), `S` (m×m, positive definite)
- **Returns**: Scalar d² ≥ 0
- **Distribution**: d² ~ χ²(m) if measurement is consistent
- **Example**:
  ```python
  y = np.array([3.0, 4.0])
  S = np.eye(2)
  d_sq = mahalanobis_distance_squared(y, S)  # 25.0
  ```

### `chi_square_gate(y, S, alpha)` → Eq. (8.9)

Chi-square gating decision.

```python
accept = chi_square_gate(y, S, alpha=0.05)  # d² < χ²(m, α)?
```

- **Args**: `y` (m,), `S` (m×m), `alpha` (significance level)
- **Returns**: `True` to accept, `False` to reject
- **Alpha values**:
  - 0.01 → 99% confidence (conservative)
  - 0.05 → 95% confidence (standard)
  - 0.10 → 90% confidence (aggressive)
- **Example**:
  ```python
  y_small = np.array([0.1, 0.2])
  chi_square_gate(y_small, np.eye(2), alpha=0.05)  # True
  
  y_large = np.array([5.0, 5.0])
  chi_square_gate(y_large, np.eye(2), alpha=0.05)  # False
  ```

### `chi_square_threshold(dof, alpha)`

Get chi-square critical value.

```python
threshold = chi_square_threshold(m, alpha)  # χ²(m, α)
```

- **Args**: `dof` m (degrees of freedom), `alpha` (significance)
- **Returns**: Critical value (scalar)
- **Example**:
  ```python
  chi_square_threshold(dof=2, alpha=0.05)  # 5.991
  chi_square_threshold(dof=3, alpha=0.05)  # 7.815
  ```

### `chi_square_bounds(dof, alpha)`

Get two-sided chi-square bounds for consistency monitoring.

```python
lower, upper = chi_square_bounds(m, alpha)
```

- **Args**: `dof` m, `alpha` (default 0.05 for 95% interval)
- **Returns**: `(lower, upper)` tuple
- **Use**: Plot NIS/NEES with bounds; should be within bounds ~(1-α)% of time
- **Example**:
  ```python
  lower, upper = chi_square_bounds(dof=2, alpha=0.05)
  # (0.051, 7.378) - 95% interval
  ```

---

## Common Patterns

### Pattern 1: EKF Update with Gating

```python
from core.fusion import innovation, innovation_covariance, chi_square_gate

# Predict
x_pred, P_pred = ekf.predict(u, dt)

# Measurement arrives
z = np.array([...])
R = np.array([[...]])

# Innovation
z_pred = h(x_pred)  # measurement prediction
y = innovation(z, z_pred)

# Innovation covariance
H = jacobian_h(x_pred)
S = innovation_covariance(H, P_pred, R)

# Gate
if chi_square_gate(y, S, alpha=0.05):
    # Accept: perform EKF update
    ekf.update(z, R)
else:
    # Reject: skip update (outlier)
    print("Measurement rejected by chi-square gate")
```

### Pattern 2: Robust Down-Weighting (Soft Gating)

```python
from core.fusion import (
    compute_normalized_innovation,
    huber_weight,
    scale_measurement_covariance
)

# Normalize innovation
y_norm = compute_normalized_innovation(y, S)

# Compute robust weight (Huber example)
# For vector measurements, use max residual
r_max = np.max(np.abs(y_norm))
w = huber_weight(r_max, threshold=2.0)

# If outlier, inflate R (reduce confidence)
if w < 1.0:
    inflation_factor = 1.0 / w
    R_robust = scale_measurement_covariance(R, inflation_factor)
else:
    R_robust = R

# Update with scaled covariance
ekf.update(z, R_robust)
```

### Pattern 3: Multi-Sensor Time Ordering

```python
from core.fusion import StampedMeasurement, TimeSyncModel

# Define sensor time sync models
sync_imu = TimeSyncModel(offset=0.0, drift=0.0)  # reference
sync_uwb = TimeSyncModel(offset=-0.05, drift=0.0001)  # 50ms behind, 100ppm drift

# Collect measurements
measurements = [
    StampedMeasurement(t=1.0, sensor='imu', z=imu_data, R=R_imu),
    StampedMeasurement(t=1.0, sensor='uwb', z=uwb_data, R=R_uwb),
    # ...
]

# Apply time sync and sort
synced_meas = []
for m in measurements:
    if m.sensor == 'imu':
        t_fus = sync_imu.to_fusion_time(m.t)
    elif m.sensor == 'uwb':
        t_fus = sync_uwb.to_fusion_time(m.t)
    
    synced_meas.append(StampedMeasurement(
        t=t_fus, sensor=m.sensor, z=m.z, R=m.R, meta=m.meta
    ))

# Sort by fusion time
synced_meas.sort(key=lambda m: m.t)

# Process in time order
for m in synced_meas:
    if m.sensor == 'imu':
        ekf.predict(m.z, dt)
    elif m.sensor == 'uwb':
        ekf.update(m.z, m.R)
```

---

## References

- **Chapter 8, Section 8.3**: Tuning and Robustness
- **Chapter 8, Section 8.5**: Temporal Calibration and Synchronization
- **Eq. (8.5)**: Innovation y = z - h(x̂)
- **Eq. (8.6)**: Innovation covariance S = H P H^T + R
- **Eq. (8.7)**: Robust R scaling R ← w(y) R
- **Eq. (8.8)**: Mahalanobis distance d² = y^T S^{-1} y
- **Eq. (8.9)**: Chi-square gate d² < χ²(m, α)

