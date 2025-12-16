# IMU + UWB Fusion Dataset (Time Offset Variant)

## Overview

**Purpose**: Test temporal calibration and synchronization in multi-sensor fusion when sensors have misaligned timestamps and relative clock drift.

**Learning Objectives**:
- Understand temporal calibration importance in multi-rate fusion (Ch8, Section 8.4)
- Observe effects of time offset on innovation residuals
- Learn online time offset estimation techniques
- Study clock drift accumulation and mitigation
- Recognize symptoms of temporal misalignment in NIS plots

**Related Chapter**: Chapter 8 - Sensor Fusion (Section 8.4: Temporal Calibration)

**Related Book Equations**: Time synchronization model, Eqs. (8.19)-(8.21)

---

## Scenario Description

**Identical to baseline EXCEPT**:
- **UWB clock is 50ms behind IMU clock**: `time_offset_sec = -0.05`
- **Relative clock drift**: 100 ppm (parts per million), `clock_drift = 0.0001`

**Temporal Misalignment Details**:
- At t=0: UWB timestamp is 50ms behind true time
- Clock drift: UWB clock runs 0.01% slower than IMU
- At t=60s: Additional 6ms drift accumulated (60s × 0.0001 = 0.006s)
- Total offset at end: 56ms

**Physical Interpretation**:
- Sensors powered by different clocks (IMU: processor clock, UWB: radio clock)
- No hardware timestamp synchronization (e.g., PPS signal)
- Common in low-cost systems where sensors can't share clock
- Typical in systems with multiple microcontrollers or wireless sensors

**All other parameters** (trajectory, IMU noise, UWB noise, anchors) identical to baseline. See `fusion_2d_imu_uwb/README.md` for full details.

---

## Files and Data Structure

Same structure as baseline - see `fusion_2d_imu_uwb/README.md`:
- `truth.npz`: Ground truth (t, p_xy, v_xy, yaw)
- `imu.npz`: IMU measurements (timestamps are "correct")
- `uwb_ranges.npz`: UWB ranges (timestamps are 50ms behind + drift)
- `uwb_anchors.npy`: Anchor positions
- `config.json`: Configuration with temporal calibration parameters

---

## Loading Example

```python
import numpy as np
import json

# Load dataset
dataset_path = 'data/sim/ch8_fusion_2d_imu_uwb_timeoffset'
truth = np.load(f'{dataset_path}/truth.npz')
imu = np.load(f'{dataset_path}/imu.npz')
uwb = np.load(f'{dataset_path}/uwb_ranges.npz')

with open(f'{dataset_path}/config.json') as f:
    config = json.load(f)

# Extract temporal calibration parameters
time_offset = config['temporal_calibration']['time_offset_sec']
clock_drift = config['temporal_calibration']['clock_drift']

print(f"Time offset: {time_offset*1000:.1f} ms (UWB behind IMU)")
print(f"Clock drift: {clock_drift*1e6:.1f} ppm")
print(f"\nAt t=60s:")
print(f"  Initial offset: {time_offset*1000:.1f} ms")
print(f"  Accumulated drift: {60*clock_drift*1000:.1f} ms")
print(f"  Total offset: {(time_offset + 60*clock_drift)*1000:.1f} ms")

# Demonstrate misalignment
t_imu = imu['t']
t_uwb = uwb['t']  # These are the reported (incorrect) timestamps

print(f"\nFirst UWB measurement:")
print(f"  Reported timestamp: {t_uwb[0]:.4f} s")
print(f"  True timestamp: {t_uwb[0] - time_offset:.4f} s")
print(f"  Misalignment: {time_offset*1000:.1f} ms")

# Apply correction manually
t_uwb_corrected = t_uwb - time_offset  # First-order correction (ignoring drift)

print(f"\nAfter simple offset correction:")
print(f"  Corrected timestamp: {t_uwb_corrected[0]:.4f} s")
```

---

## Configuration Parameters

**Key differences from baseline**:

### Temporal Calibration Parameters

- `time_offset_sec`: -0.05 **(Changed from 0.0)** 
  - Negative means UWB is behind (UWB timestamp < true time)
- `clock_drift`: 0.0001 **(Changed from 0.0)**
  - 100 ppm = 0.01% relative frequency error
  - Accumulates linearly: offset(t) = time_offset + clock_drift × t

All other parameters identical to baseline:
- IMU: accel_noise = 0.1 m/s², gyro_noise = 0.01 rad/s
- UWB: range_noise = 0.05 m, rate = 10 Hz, dropout = 5%
- Trajectory: 20m × 15m rectangle, 1 m/s, 60s

---

## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on Fusion Performance | Learning Objective |
|-----------|---------|------------------|------------------------------|-------------------|
| `time_offset` | -0.05 | -0.2 to 0.2 | Larger offset → systematic residuals → higher NIS → may trigger gating | Understand timestamp error propagation |
| `clock_drift` | 0.0001 | 0-0.001 | Larger drift → offset grows over time → time-varying residuals | Learn importance of drift compensation |
| `speed` | 1.0 | 0.5-2.0 | Higher speed → larger position change per time → offset effects amplified | Observe motion-dependency of temporal errors |
| Calibration quality | — | Uncalibrated/offset-only/full | Better calibration → lower residuals → improved accuracy | Compare calibration strategies |

**Key Insight**: Time offset causes systematic innovation because measurement is compared to prediction at wrong time. For 50ms offset at 1 m/s speed: position error ≈ 0.05m, which is comparable to measurement noise (0.05m)! At 2 m/s, position error would be 0.1m, clearly detectable.

**Theoretical Relationship**:
```
Position error ≈ speed × |time_offset|
At turning points: heading error ≈ angular_velocity × |time_offset|
```

---

## Dataset Variants

This is the **time offset variant**. Related datasets:

**1. fusion_2d_imu_uwb/** - Baseline (Synchronized)
   - No time offset, no drift
   - Use for: Establishing synchronized performance baseline
   - Comparison: Time offset variant should show degraded performance

**2. fusion_2d_imu_uwb_nlos/** - NLOS Challenge
   - Different challenge: outlier measurements instead of temporal issues
   - Use for: Robustness experiments (gating, robust losses)

---

## Visualization Example

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data
dataset_path = 'data/sim/ch8_fusion_2d_imu_uwb_timeoffset'
truth = np.load(f'{dataset_path}/truth.npz')
uwb = np.load(f'{dataset_path}/uwb_ranges.npz')
anchors = np.load(f'{dataset_path}/uwb_anchors.npy')

with open(f'{dataset_path}/config.json') as f:
    config = json.load(f)

time_offset = config['temporal_calibration']['time_offset_sec']
clock_drift = config['temporal_calibration']['clock_drift']

# Compute innovations with and without correction
t_uwb = uwb['t']
ranges = uwb['ranges']

# Position at reported timestamps (WRONG)
p_xy_wrong = np.column_stack([
    np.interp(t_uwb, truth['t'], truth['p_xy'][:, 0]),
    np.interp(t_uwb, truth['t'], truth['p_xy'][:, 1])
])

# Position at corrected timestamps (CORRECT)
t_uwb_corrected = t_uwb - time_offset - clock_drift * t_uwb
p_xy_correct = np.column_stack([
    np.interp(t_uwb_corrected, truth['t'], truth['p_xy'][:, 0]),
    np.interp(t_uwb_corrected, truth['t'], truth['p_xy'][:, 1])
])

# Compute range innovations for Anchor 0
anchor0 = anchors[0]
ranges_meas = ranges[:, 0]
valid = ~np.isnan(ranges_meas)

ranges_pred_wrong = np.linalg.norm(p_xy_wrong - anchor0, axis=1)
ranges_pred_correct = np.linalg.norm(p_xy_correct - anchor0, axis=1)

innov_wrong = ranges_meas - ranges_pred_wrong
innov_correct = ranges_meas - ranges_pred_correct

# Plot innovations
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Without correction
axes[0].plot(t_uwb[valid], innov_wrong[valid], 'o-', markersize=3, 
             label='Innovation (uncorrected)', alpha=0.7)
axes[0].axhline(0, color='k', linestyle='--', linewidth=2)
axes[0].fill_between([t_uwb[0], t_uwb[-1]], -0.15, 0.15, 
                       alpha=0.2, color='gray', label='±3σ noise')
axes[0].set_ylabel('Innovation (m)')
axes[0].set_title('Anchor 0 Innovations WITHOUT Temporal Correction')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# With correction
axes[1].plot(t_uwb[valid], innov_correct[valid], 'o-', markersize=3, 
             label='Innovation (corrected)', alpha=0.7, color='green')
axes[1].axhline(0, color='k', linestyle='--', linewidth=2)
axes[1].fill_between([t_uwb[0], t_uwb[-1]], -0.15, 0.15, 
                       alpha=0.2, color='gray', label='±3σ noise')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Innovation (m)')
axes[1].set_title('Anchor 0 Innovations WITH Temporal Correction')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fusion_timeoffset_innovations.svg')

print(f"Innovation statistics:")
print(f"  Without correction - Mean: {np.mean(innov_wrong[valid]):.4f} m, Std: {np.std(innov_wrong[valid]):.4f} m")
print(f"  With correction    - Mean: {np.mean(innov_correct[valid]):.4f} m, Std: {np.std(innov_correct[valid]):.4f} m")
```

---

## Connection to Book Equations

**Time Synchronization Model** (Ch8, Section 8.4):

Let `t_true` be the true time and `t_sensor` be the sensor-reported timestamp:

```
t_true = t_sensor + Δt₀ + δ × t_sensor
```

Where:
- `Δt₀`: Initial time offset (constant)
- `δ`: Clock drift rate (relative frequency error)

For this dataset:
- `Δt₀ = -0.05 s` (UWB 50ms behind)
- `δ = 0.0001` (100 ppm)

**Effect on Prediction**:

Without correction, fusion compares measurement z(t_uwb) to prediction at state x̂(t_uwb), but should compare to x̂(t_true):

```
Innovation (wrong):    y = z(t) - h(x̂(t))
Innovation (correct):  y = z(t) - h(x̂(t + Δt₀ + δt))
```

**Position Propagation Error**:

During time offset Δt, position changes by ≈ v × Δt:

```
Δp ≈ v × Δt₀ = 1.0 m/s × 0.05 s = 0.05 m
```

This 5cm systematic error is as large as measurement noise!

**Online Estimation**:

Time offset can be estimated online as additional state:

```
Augmented state: x_aug = [x, y, vx, vy, θ, ω, Δt, δ]
```

With appropriate process/measurement models (Ch8, Eqs. 8.19-8.21).

---

## Recommended Experiments

### Experiment 1: Temporal Misalignment Impact

**Objective**: Quantify performance degradation from time offset without correction.

**Setup**:
```bash
# Run fusion WITHOUT temporal correction
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/ch8_fusion_2d_imu_uwb_timeoffset \
    --no-time-correction \
    --output results_no_correction.json

# Run fusion WITH offline temporal correction (if known)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/ch8_fusion_2d_imu_uwb_timeoffset \
    --time-offset -0.05 \
    --clock-drift 0.0001 \
    --output results_with_correction.json

# Run fusion WITH online temporal calibration (estimate offset)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf_augmented \
    --data data/sim/ch8_fusion_2d_imu_uwb_timeoffset \
    --estimate-time-offset \
    --output results_online_calibration.json
```

**Expected Observations**:

| Configuration | Position RMSE | NIS Consistency | Time Offset Estimate |
|---------------|---------------|-----------------|----------------------|
| No correction | 0.18-0.25 m | Poor (~50-60% within bounds) | N/A |
| Offline correction (known) | 0.08-0.12 m | Good (~95% within bounds) | N/A (provided) |
| Online calibration | 0.10-0.15 m | Good after convergence | Converges to -0.05s in ~20-30s |

**Analysis**:
1. Plot position RMSE over time (all 3 cases)
2. Plot NIS values with χ² threshold
3. For online calibration: plot estimated time offset vs. true value
4. Compute convergence time for online estimator

**Key Insight**: Even "small" time offsets (50ms) significantly degrade performance when robot speed is non-negligible. Proper temporal calibration is critical in multi-rate fusion systems.

---

### Experiment 2: Time Offset Sensitivity Sweep

**Objective**: Understand how performance degrades with increasing time offset.

**Setup**:
```bash
# Generate datasets with varying time offset
for offset in -0.2 -0.1 -0.05 -0.02 0.0 0.02 0.05 0.1 0.2; do
    python scripts/generate_fusion_2d_imu_uwb_dataset.py \
        --time-offset $offset \
        --output data/sim/fusion_timeoffset_${offset}
done

# Run uncorrected fusion on each
for offset in -0.2 -0.1 -0.05 -0.02 0.0 0.02 0.05 0.1 0.2; do
    python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
        --data data/sim/fusion_timeoffset_${offset} \
        --no-time-correction \
        --output results_offset_${offset}.json
done
```

**Expected Observations**:
- Offset = 0: Baseline performance (RMSE ~0.10m)
- |Offset| < 20ms: Mild degradation (RMSE ~0.12-0.15m)
- |Offset| = 50ms: Significant degradation (RMSE ~0.20-0.25m)
- |Offset| > 100ms: Severe degradation (RMSE > 0.30m), may trigger gating

**Analysis**:
- Plot RMSE vs. time offset (V-shaped curve, minimum at 0)
- Verify prediction: RMSE_additional ≈ speed × |offset|
- Find "acceptable" offset threshold (e.g., RMSE < 2× baseline)

**Key Insight**: Acceptable time offset depends on application requirements and robot dynamics. High-speed robots need tighter synchronization.

---

### Experiment 3: Online Time Offset Estimation

**Objective**: Learn online temporal calibration by augmenting state with time offset parameter.

**Setup**:

Implement augmented EKF with state:
```python
x = [x, y, vx, vy, θ, ω, Δt]  # 7D state (added Δt)
```

Process model for Δt:
```python
Δt_{k+1} = Δt_k + w_Δt  # Random walk or constant
```

Measurement model (modified):
```python
# Predict state at corrected time
t_corrected = t_uwb + Δt  # Use estimated offset
x_pred = propagate_state_to(t_corrected)
z_pred = h(x_pred)
```

**Run**:
```bash
python -m ch8_sensor_fusion.tc_uwb_imu_ekf_augmented \
    --data data/sim/ch8_fusion_2d_imu_uwb_timeoffset \
    --estimate-time-offset \
    --output results_online.json
```

**Expected Observations**:
- Time offset estimate starts at 0 (or prior guess)
- Converges to -0.05s over ~20-40 seconds
- After convergence: performance matches offline-corrected case
- Estimation uncertainty decreases over time

**Analysis**:
- Plot estimated Δt vs. true value over time
- Plot estimation uncertainty (std from covariance)
- Observe correlation between position uncertainty and Δt uncertainty
- Check observability: Δt is observable when robot is moving

**Key Insight**: Time offset is observable from innovation residuals when robot has motion. Augmented-state approach enables online calibration without external synchronization hardware.

---

## Troubleshooting / Common Student Questions

**Q: Why does time offset affect position but not IMU measurements?**
A: IMU measurements are used for prediction (no comparison to external reference), so internal timestamp is fine. UWB measurements are compared to predicted state, so misalignment creates systematic residuals.

**Q: How can I detect if my real system has time offset?**
A: Symptoms:
1. Systematic (non-zero mean) innovations
2. NIS values consistently elevated but not outliers
3. Innovation direction correlates with motion direction
4. Performance degrades with higher speeds
5. Residuals show periodic pattern (if trajectory is periodic)

**Q: Is 50ms offset realistic?**
A: Yes! Examples:
- Different sensors on different microcontrollers without PPS sync
- Network delays in distributed systems
- Sensor buffering and processing delays
- Wireless transmission delays
Typical: 10-100ms for low-cost systems, <1ms for GPS-disciplined systems.

**Q: Can I just interpolate measurements to fix time offset?**
A: Interpolation only works if you **know** the offset. In practice:
- Offset may be unknown → need online estimation
- Clock drift causes time-varying offset → need continuous tracking
- Better to fix in fusion (augmented state) than pre-process

**Q: What about asynchronous sensors (different, unknown rates)?**
A: This dataset uses known, fixed rates. For truly asynchronous sensors:
- Use event-based fusion (update whenever measurement arrives)
- Each sensor can have own clock model
- More complex but handles arbitrary timing
See Ch8, Section 8.5 for asynchronous fusion.

**Q: How does time offset interact with latency (processing delay)?**
A: Both create temporal mismatch. In practice:
- **Latency**: Measurement delayed by processing time → use state at (t - latency)
- **Time offset**: Timestamp is wrong → correct timestamp before fusion
- **Combined**: t_true = t_reported + offset - latency
Modern systems must handle both!

---

## Generation

This dataset was generated using:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --time-offset -0.05 \
    --clock-drift 0.0001 \
    --output data/sim/ch8_fusion_2d_imu_uwb_timeoffset
```

**Or using preset**:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py --preset time_offset_50ms
```

**Generate all 3 standard variants**:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py --all-variants
```

**Custom temporal calibration experiments**:
```bash
# Larger offset (200ms)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --time-offset -0.2 \
    --output data/sim/fusion_timeoffset_200ms

# Large clock drift only (no initial offset)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --time-offset 0.0 \
    --clock-drift 0.001 \
    --output data/sim/fusion_drift_1000ppm

# Positive offset (IMU behind UWB)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --time-offset 0.05 \
    --output data/sim/fusion_timeoffset_positive
```

---

## References

- **Book Chapter**: Chapter 8, Section 8.4 (Temporal Calibration and Synchronization)
- **Key Equations**: Time synchronization model, Eqs. (8.19)-(8.21)
- **Related Topics**:
  - Ch8, Section 8.5: Asynchronous fusion
  - Ch8, Section 8.1: Multi-rate fusion fundamentals
- **Related Examples**:
  - `ch8_sensor_fusion/tc_uwb_imu_ekf.py` - Main fusion (supports offline correction)
  - `ch8_sensor_fusion/tc_uwb_imu_ekf_augmented.py` - Augmented-state fusion (online estimation)
  - `ch8_sensor_fusion/example_temporal_calibration.py` - Dedicated temporal calibration demo
- **Generation Script**: `scripts/generate_fusion_2d_imu_uwb_dataset.py`
- **Baseline**: See `fusion_2d_imu_uwb/README.md` for synchronized reference


