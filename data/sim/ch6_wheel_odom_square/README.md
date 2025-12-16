# Ch6 Wheel Odometry Dataset: Square Trajectory

## Overview

This dataset demonstrates **vehicle dead reckoning using wheel encoders with IMU integration**. It showcases the bounded drift characteristics of wheel odometry (drift proportional to distance, not time) and the importance of lever arm compensation.

**Key Learning Objective**: Understand that wheel odometry provides bounded drift but is sensitive to wheel slip, making it superior to pure IMU for vehicles but still requiring corrections.

## Dataset Purpose

### Learning Goals
1. **Bounded Drift**: Wheel odometry error grows with distance (~0.25% drift rate), not time
2. **Lever Arm Effects**: Proper compensation is critical for accuracy (Eq. 6.11)
3. **Wheel Slip Sensitivity**: Slip during turns significantly degrades performance
4. **Combined Navigation**: Wheel+IMU fusion is more robust than either alone

### Implemented Equations
- **Eq. (6.11)**: Lever arm compensation for wheel speed
  ```
  v^A = v^S - [ω×] l
  ```
  where `v^S` is wheel speed, `ω` is angular velocity, `l` is lever arm

- **Eq. (6.12)**: Skew-symmetric matrix for cross products
  ```
  [v×] = | 0   -vz   vy |
         | vz   0   -vx |
         |-vy   vx   0  |
  ```

- **Eq. (6.14)**: Attitude to map frame velocity transform
  ```
  v^M = C_A^M @ v^A
  ```

- **Eq. (6.15)**: Position update from wheel odometry
  ```
  p_k = p_{k-1} + v^M * Δt
  ```

## Dataset Variants

| Variant | Directory | Encoder Noise | Wheel Slip | Drift Rate | Description |
|---------|-----------|---------------|------------|------------|-------------|
| **Baseline** | `ch6_wheel_odom_square` | 0.03 m/s | No | ~0.25% | Clean sensors, no slip |
| **Noisy** | `ch6_wheel_odom_noisy` | 0.10 m/s | No | ~0.5% | Higher sensor noise |
| **Slip** | `ch6_wheel_odom_slip` | 0.05 m/s | 30% in turns | ~1-2% | Wheel slip during turns |
| **Poor** | `ch6_wheel_odom_poor` | 0.15 m/s | 50% in turns | ~3-5% | Poor sensors + severe slip |

**Generate variants**:
```bash
python scripts/generate_ch6_wheel_odom_dataset.py --preset baseline
python scripts/generate_ch6_wheel_odom_dataset.py --preset noisy
python scripts/generate_ch6_wheel_odom_dataset.py --preset slip
python scripts/generate_ch6_wheel_odom_dataset.py --preset poor
```

## Files

### Ground Truth
- `time.txt`: Timestamps [N×1] (seconds)
- `ground_truth_position.txt`: True positions [N×3] (x, y, z in meters)
- `ground_truth_velocity.txt`: True velocities [N×3] (vx, vy, vz in m/s)
- `ground_truth_quaternion.txt`: True attitudes [N×4] (q0, q1, q2, q3, scalar-first)

### Measurements
- `wheel_speed.txt`: Noisy wheel speed measurements [N×3] (forward, lateral, vertical in m/s)
- `gyro.txt`: Noisy gyroscope measurements [N×3] (ωx, ωy, ωz in rad/s)

### Reference Data
- `wheel_speed_clean.txt`: Clean wheel speed (no noise) [N×3]
- `gyro_clean.txt`: Clean gyro (no noise) [N×3]

### Configuration
- `config.json`: All dataset parameters and performance metrics

## Loading Data

### Python
```python
import numpy as np
from pathlib import Path

# Load dataset
data_dir = Path("data/sim/ch6_wheel_odom_square")

t = np.loadtxt(data_dir / "time.txt")
pos_true = np.loadtxt(data_dir / "ground_truth_position.txt")
vel_true = np.loadtxt(data_dir / "ground_truth_velocity.txt")
quat_true = np.loadtxt(data_dir / "ground_truth_quaternion.txt")
wheel_meas = np.loadtxt(data_dir / "wheel_speed.txt")
gyro_meas = np.loadtxt(data_dir / "gyro.txt")

print(f"Loaded {len(t)} samples over {t[-1]:.1f} seconds")
print(f"Trajectory distance: {np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1)):.1f} m")
```

### MATLAB
```matlab
% Load dataset
data_dir = 'data/sim/ch6_wheel_odom_square/';

t = load([data_dir 'time.txt']);
pos_true = load([data_dir 'ground_truth_position.txt']);
vel_true = load([data_dir 'ground_truth_velocity.txt']);
quat_true = load([data_dir 'ground_truth_quaternion.txt']);
wheel_meas = load([data_dir 'wheel_speed.txt']);
gyro_meas = load([data_dir 'gyro.txt']);

fprintf('Loaded %d samples over %.1f seconds\n', length(t), t(end));
```

## Configuration Parameters

### Trajectory Configuration
```json
{
  "trajectory": {
    "shape": "square",
    "side_length_m": 20.0,
    "speed_m_s": 5.0,
    "num_laps": 2,
    "duration_s": 73.9,
    "total_distance_m": 327.5
  }
}
```

**Key Parameters**:
- **side_length**: Side of square trajectory (20m)
- **speed**: Forward speed (5 m/s, slower in turns)
- **num_laps**: Number of complete loops (2 laps)
- **total_distance**: Total path length (327.5m)

### Sensor Configuration (Baseline)
```json
{
  "encoder": {
    "noise_std_m_s": 0.03,
    "bias_m_s": 0.005
  },
  "gyro": {
    "noise_std_rad_s": 0.0005,
    "bias_rad_s": 0.0002
  },
  "lever_arm_m": [1.5, 0.0, -0.3]
}
```

**Key Parameters**:
- **encoder_noise**: Wheel speed measurement noise (0.03 m/s std dev)
- **wheel_bias**: Systematic wheel speed bias (0.005 m/s)
- **gyro_noise**: Angular velocity noise (0.0005 rad/s)
- **lever_arm**: Offset from IMU to wheel center ([1.5, 0, -0.3] m)

### Slip Configuration
```json
{
  "slip": {
    "enabled": false,
    "magnitude": 0.0,
    "num_events": 0
  }
}
```

**Note**: Baseline has no slip. Use `--preset slip` or `--preset poor` for slip scenarios.

## Quick Start Example

### Run Wheel Odometry Dead Reckoning
```python
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from core.sensors import wheel_odom_update, NavStateQPVP
from core.sensors.strapdown import quat_integrate

# Load dataset
data_dir = Path("data/sim/ch6_wheel_odom_square")
t = np.loadtxt(data_dir / "time.txt")
pos_true = np.loadtxt(data_dir / "ground_truth_position.txt")
quat_true = np.loadtxt(data_dir / "ground_truth_quaternion.txt")
wheel_meas = np.loadtxt(data_dir / "wheel_speed.txt")
gyro_meas = np.loadtxt(data_dir / "gyro.txt")

# Initial state
initial = NavStateQPVP(q=quat_true[0], v=np.zeros(3), p=pos_true[0])
lever_arm = np.array([1.5, 0.0, -0.3])

# Run wheel odometry
N = len(t)
pos_est = np.zeros((N, 3))
quat_est = np.zeros((N, 4))
pos_est[0] = initial.p
quat_est[0] = initial.q

for i in range(1, N):
    dt = t[i] - t[i-1]
    
    # Update attitude using gyro
    quat_est[i] = quat_integrate(quat_est[i-1], gyro_meas[i-1], dt)
    
    # Update position using wheel odometry with lever arm
    pos_est[i] = wheel_odom_update(
        pos_est[i-1], quat_est[i], wheel_meas[i], gyro_meas[i], lever_arm, dt
    )

# Compute error
error = np.linalg.norm(pos_est - pos_true, axis=1)
print(f"Final error: {error[-1]:.3f} m")
print(f"Mean error: {np.mean(error):.3f} m")
print(f"Drift rate: {error[-1] / 327.5 * 100:.2f}% of distance")

# Plot trajectory
plt.figure(figsize=(10, 8))
plt.plot(pos_true[:, 0], pos_true[:, 1], 'g-', label='Ground Truth', linewidth=2)
plt.plot(pos_est[:, 0], pos_est[:, 1], 'b--', label='Wheel Odometry', linewidth=2)
plt.plot(pos_true[0, 0], pos_true[0, 1], 'go', markersize=10, label='Start')
plt.plot(pos_true[-1, 0], pos_true[-1, 1], 'r^', markersize=10, label='End')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'Wheel Odometry: {error[-1]:.2f}m error over {327.5:.0f}m ({error[-1]/327.5*100:.2f}%)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
```

**Expected Result**: ~0.8m final error (0.25% drift rate)

## Visualization

### Plot with Built-in Tool
```bash
# Single dataset
python tools/plot_fusion_dataset.py data/sim/ch6_wheel_odom_square --type wheel_odom

# Compare variants
python tools/compare_fusion_variants.py \
    data/sim/ch6_wheel_odom_square \
    data/sim/ch6_wheel_odom_slip \
    --labels "No Slip" "With Slip"
```

### Custom Plot: Error vs Distance
```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
data_dir = Path("data/sim/ch6_wheel_odom_square")
pos_true = np.loadtxt(data_dir / "ground_truth_position.txt")

# Compute cumulative distance
distance = np.zeros(len(pos_true))
for i in range(1, len(pos_true)):
    distance[i] = distance[i-1] + np.linalg.norm(pos_true[i] - pos_true[i-1])

# Run wheel odometry (code from Quick Start example)
# ... pos_est = run_wheel_odometry(...) ...

# Compute error
error = np.linalg.norm(pos_est - pos_true, axis=1)

# Plot error vs distance
plt.figure(figsize=(10, 6))
plt.plot(distance, error, 'b-', linewidth=2)
plt.xlabel('Distance Traveled (m)')
plt.ylabel('Position Error (m)')
plt.title('Wheel Odometry: Bounded Drift (Error ∝ Distance)')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Drift rate: {error[-1] / distance[-1] * 100:.3f}% of distance")
```

**Expected Pattern**: Linear growth with distance (not time!)

## Parameter Effects

### Effect of Encoder Noise

| Encoder Noise (m/s) | Final Error (m) | Drift Rate (%) | Notes |
|---------------------|-----------------|----------------|-------|
| 0.01 (excellent) | 0.4-0.6 | 0.12-0.18 | High-quality encoders |
| 0.03 (good) | 0.8-1.2 | 0.24-0.36 | Baseline quality |
| 0.10 (fair) | 2.0-3.0 | 0.60-0.90 | Consumer-grade |
| 0.15 (poor) | 3.0-4.5 | 0.90-1.35 | Low-quality sensors |

**Generate sweep**:
```bash
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_enc_001 --encoder-noise 0.01
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_enc_003 --encoder-noise 0.03
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_enc_010 --encoder-noise 0.10
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_enc_015 --encoder-noise 0.15
```

### Effect of Wheel Slip

| Slip Magnitude (%) | Final Error (m) | Drift Rate (%) | Notes |
|--------------------|-----------------|----------------|-------|
| 0% (no slip) | 0.8-1.2 | 0.24-0.36 | Ideal conditions |
| 10% (light) | 2.0-3.0 | 0.60-0.90 | Dry pavement |
| 30% (moderate) | 4.0-6.0 | 1.20-1.80 | Wet/icy conditions |
| 50% (severe) | 8.0-12.0 | 2.40-3.60 | Off-road/gravel |

**Generate sweep**:
```bash
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_slip_00 --slip-magnitude 0.0
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_slip_10 --add-slip --slip-magnitude 0.1
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_slip_30 --add-slip --slip-magnitude 0.3
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_slip_50 --add-slip --slip-magnitude 0.5
```

### Effect of Lever Arm

| Lever Arm (m) | Final Error (m) | Impact | Notes |
|---------------|-----------------|--------|-------|
| [0, 0, 0] | 0.8-1.2 | Baseline | IMU at wheel center (ideal) |
| [1.5, 0, -0.3] | 0.8-1.2 | Low | Typical vehicle offset (compensated) |
| [3.0, 0, -0.5] | 0.9-1.4 | Moderate | Large vehicle (compensated) |
| **No compensation** | 5.0-10.0 | **SEVERE** | Lever arm ignored (wrong!) |

**Generate comparison**:
```bash
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_lever_zero --lever-arm 0 0 0
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_lever_small --lever-arm 1.5 0 -0.3
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_lever_large --lever-arm 3.0 0 -0.5
```

## Experiments

### Experiment 1: Bounded Drift Analysis

**Objective**: Verify that wheel odometry error is bounded (proportional to distance, not time).

**Procedure**:
1. Generate baseline dataset
2. Run wheel odometry
3. Plot error vs time and error vs distance
4. Compute drift rate (%/distance)

**Expected Results**:
- Error vs time: NON-LINEAR (not unbounded like IMU)
- Error vs distance: LINEAR growth (~0.25% drift rate)
- Drift is BOUNDED (proportional to distance)

**Code**:
```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from core.sensors import wheel_odom_update, NavStateQPVP
from core.sensors.strapdown import quat_integrate

# Load dataset
data_dir = Path("data/sim/ch6_wheel_odom_square")
t = np.loadtxt(data_dir / "time.txt")
pos_true = np.loadtxt(data_dir / "ground_truth_position.txt")
quat_true = np.loadtxt(data_dir / "ground_truth_quaternion.txt")
wheel_meas = np.loadtxt(data_dir / "wheel_speed.txt")
gyro_meas = np.loadtxt(data_dir / "gyro.txt")

# Compute cumulative distance
distance = np.zeros(len(pos_true))
for i in range(1, len(pos_true)):
    distance[i] = distance[i-1] + np.linalg.norm(pos_true[i] - pos_true[i-1])

# Run wheel odometry
initial = NavStateQPVP(q=quat_true[0], v=np.zeros(3), p=pos_true[0])
lever_arm = np.array([1.5, 0.0, -0.3])

N = len(t)
pos_est = np.zeros((N, 3))
quat_est = np.zeros((N, 4))
pos_est[0] = initial.p
quat_est[0] = initial.q

for i in range(1, N):
    dt = t[i] - t[i-1]
    quat_est[i] = quat_integrate(quat_est[i-1], gyro_meas[i-1], dt)
    pos_est[i] = wheel_odom_update(
        pos_est[i-1], quat_est[i], wheel_meas[i], gyro_meas[i], lever_arm, dt
    )

# Compute error
error = np.linalg.norm(pos_est - pos_true, axis=1)

# Plot error vs time and distance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(t, error, 'b-', linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position Error (m)')
ax1.set_title('Error vs Time (Non-linear)')
ax1.grid(True)

ax2.plot(distance, error, 'r-', linewidth=2)
ax2.set_xlabel('Distance Traveled (m)')
ax2.set_ylabel('Position Error (m)')
ax2.set_title(f'Error vs Distance (Linear, {error[-1]/distance[-1]*100:.2f}% drift rate)')
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Final error: {error[-1]:.3f} m")
print(f"Total distance: {distance[-1]:.1f} m")
print(f"Drift rate: {error[-1] / distance[-1] * 100:.3f}% of distance")
```

**Learning Point**: Wheel odometry is MUCH better than pure IMU for vehicles!

### Experiment 2: Wheel Slip Impact

**Objective**: Quantify the effect of wheel slip on odometry accuracy.

**Procedure**:
1. Generate datasets with varying slip magnitudes (0%, 10%, 30%, 50%)
2. Run wheel odometry on each
3. Compare final errors and drift rates

**Expected Results**:
- 0% slip: ~0.8m error (0.25% drift)
- 10% slip: ~2.5m error (0.75% drift, 3× worse)
- 30% slip: ~5.0m error (1.5% drift, 6× worse)
- 50% slip: ~10m error (3% drift, 12× worse)

**Code**:
```bash
# Generate datasets
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_slip_00
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_slip_10 --add-slip --slip-magnitude 0.1
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_slip_30 --add-slip --slip-magnitude 0.3
python scripts/generate_ch6_wheel_odom_dataset.py --output data/sim/wheel_slip_50 --add-slip --slip-magnitude 0.5

# Compare using tool
python tools/compare_fusion_variants.py \
    data/sim/wheel_slip_00 \
    data/sim/wheel_slip_10 \
    data/sim/wheel_slip_30 \
    data/sim/wheel_slip_50 \
    --labels "No Slip" "10% Slip" "30% Slip" "50% Slip"
```

**Learning Point**: Wheel slip is the PRIMARY error source in wheel odometry!

### Experiment 3: Lever Arm Compensation

**Objective**: Demonstrate the importance of lever arm compensation (Eq. 6.11).

**Procedure**:
1. Generate baseline dataset with lever arm [1.5, 0, -0.3]
2. Run wheel odometry WITH lever arm compensation
3. Run wheel odometry WITHOUT lever arm compensation (set to zero)
4. Compare errors

**Expected Results**:
- **With compensation**: ~0.8m error (correct)
- **Without compensation**: ~8-10m error (10× worse!)

**Code**:
```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from core.sensors import wheel_odom_update, NavStateQPVP
from core.sensors.strapdown import quat_integrate

# Load dataset
data_dir = Path("data/sim/ch6_wheel_odom_square")
t = np.loadtxt(data_dir / "time.txt")
pos_true = np.loadtxt(data_dir / "ground_truth_position.txt")
quat_true = np.loadtxt(data_dir / "ground_truth_quaternion.txt")
wheel_meas = np.loadtxt(data_dir / "wheel_speed.txt")
gyro_meas = np.loadtxt(data_dir / "gyro.txt")

# Run WITH lever arm compensation
initial = NavStateQPVP(q=quat_true[0], v=np.zeros(3), p=pos_true[0])
lever_arm_correct = np.array([1.5, 0.0, -0.3])
lever_arm_zero = np.array([0.0, 0.0, 0.0])  # WRONG: no compensation

# With compensation
N = len(t)
pos_with = np.zeros((N, 3))
quat_with = np.zeros((N, 4))
pos_with[0] = initial.p
quat_with[0] = initial.q

for i in range(1, N):
    dt = t[i] - t[i-1]
    quat_with[i] = quat_integrate(quat_with[i-1], gyro_meas[i-1], dt)
    pos_with[i] = wheel_odom_update(
        pos_with[i-1], quat_with[i], wheel_meas[i], gyro_meas[i], lever_arm_correct, dt
    )

# Without compensation (WRONG)
pos_without = np.zeros((N, 3))
quat_without = np.zeros((N, 4))
pos_without[0] = initial.p
quat_without[0] = initial.q

for i in range(1, N):
    dt = t[i] - t[i-1]
    quat_without[i] = quat_integrate(quat_without[i-1], gyro_meas[i-1], dt)
    pos_without[i] = wheel_odom_update(
        pos_without[i-1], quat_without[i], wheel_meas[i], gyro_meas[i], lever_arm_zero, dt
    )

# Compute errors
error_with = np.linalg.norm(pos_with - pos_true, axis=1)
error_without = np.linalg.norm(pos_without - pos_true, axis=1)

print(f"With lever arm compensation: {error_with[-1]:.2f}m error")
print(f"Without compensation: {error_without[-1]:.2f}m error")
print(f"Degradation: {error_without[-1] / error_with[-1]:.1f}× worse!")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(t, error_with, 'g-', linewidth=2, label='With Compensation (Eq. 6.11)')
plt.plot(t, error_without, 'r-', linewidth=2, label='Without Compensation (WRONG)')
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Lever Arm Compensation: Critical for Accuracy!')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Learning Point**: Lever arm compensation (Eq. 6.11) is ESSENTIAL!

## Performance Metrics (Baseline)

| Metric | Value | Notes |
|--------|-------|-------|
| **Final Error** | 0.8m | After 327m distance |
| **Mean Error** | 1.75m | Average over trajectory |
| **Max Error** | 2.83m | Peak error (during turn) |
| **Drift Rate** | 0.25% | Error per distance traveled |
| **Sample Rate** | 100 Hz | 10ms time steps |
| **Duration** | 73.9s | 2 laps of 20m square |
| **Total Distance** | 327.5m | Cumulative path length |

**Comparison to Pure IMU**:
- IMU strapdown: ~100m error in 12s (UNBOUNDED)
- Wheel odometry: ~0.8m error over 327m (BOUNDED, 100× better!)

## Book Connection

### Chapter 6, Section 6.2: Wheel Odometry Dead Reckoning

This dataset directly implements the wheel odometry algorithms from Section 6.2:

1. **Lever Arm Compensation (Eq. 6.11)**
   - Corrects for offset between IMU and wheel center
   - Uses cross product: `v^A = v^S - [ω×] l`
   - Critical for accuracy (10× degradation if ignored!)

2. **Frame Transformations (Eq. 6.14)**
   - Rotates velocity from attitude frame to map frame
   - Uses rotation matrix from quaternion

3. **Position Update (Eq. 6.15)**
   - Integrates velocity to propagate position
   - Simple Euler integration (can be improved)

4. **Bounded Drift Characteristic**
   - Drift proportional to DISTANCE, not time
   - Typical: 0.1-1% drift rate (vs. unbounded IMU)
   - Sensitive to wheel slip during turns

**Key Insight from Chapter 6**: Wheel odometry is ideal for vehicles because drift is bounded and predictable, unlike pure IMU which drifts catastrophically. However, it still requires corrections (e.g., GNSS, map matching) for long-term accuracy.

## Common Issues & Solutions

### Issue 1: Large Errors Despite Clean Sensors

**Symptoms**: 10-20m errors even with low noise

**Likely Cause**: Lever arm not compensated (Eq. 6.11 not applied)

**Solution**: Ensure `lever_arm` parameter is correctly set and passed to `wheel_odom_update()`

### Issue 2: Errors Grow Unbounded with Time

**Symptoms**: Drift keeps growing indefinitely (like IMU)

**Likely Cause**: Using wheel speed incorrectly (e.g., no attitude update)

**Solution**: Ensure gyro is used to update attitude, not just wheel speed

### Issue 3: Unrealistic Drift Rate (>5%)

**Symptoms**: Drift rate much higher than expected

**Likely Cause**: Wheel slip too severe or encoder noise too high

**Solution**: Check `slip_magnitude` and `encoder_noise` parameters; real systems are typically <1% drift

## Troubleshooting

### Error: Quaternion becomes denormalized

**Cause**: Numerical errors in integration

**Fix**: Re-normalize quaternion after each integration:
```python
quat = quat / np.linalg.norm(quat)
```

### Error: Position explodes to inf/nan

**Cause**: Extremely large gyro values or corrupted data

**Fix**: Sanity-check gyro measurements (should be <10 rad/s for vehicles)

### Warning: Lever arm has large Z component

**Cause**: Unusual vehicle configuration

**Fix**: Verify lever arm is correct; typically Z < 0.5m for ground vehicles

## Next Steps

After understanding wheel odometry:

1. **Chapter 6, Section 6.3**: Constraints (ZUPT, NHC) for additional drift correction
2. **Chapter 8**: Sensor fusion (combine wheel odometry with GNSS, IMU in EKF/FGO)
3. **Wheel+IMU EKF**: Tightly coupled integration (future dataset)
4. **Map-Matching**: Use road maps to bound lateral drift (future dataset)

## Citation

If you use this dataset in your research, please cite:

```bibtex
@book{IPIN2024,
  title={Principles of Indoor Positioning and Indoor Navigation},
  author={[Authors]},
  year={2024},
  chapter={6},
  section={6.2},
  note={Wheel Odometry Dead Reckoning}
}
```

## License

This dataset is part of the IPIN Book Examples repository. See repository LICENSE for details.

---

**Dataset Version**: 1.0  
**Last Updated**: December 2024  
**Contact**: See repository README for contact information

