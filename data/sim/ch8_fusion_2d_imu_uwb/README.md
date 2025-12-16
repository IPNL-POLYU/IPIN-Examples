# IMU + UWB Fusion Dataset (Baseline)

## Overview

**Purpose**: Baseline 2D indoor positioning dataset for testing loosely-coupled (LC) and tightly-coupled (TC) sensor fusion algorithms combining high-rate IMU with low-rate UWB ranging.

**Learning Objectives**:
- Understand multi-rate sensor fusion (100 Hz IMU + 10 Hz UWB)
- Observe how fusion reduces unbounded IMU drift using absolute measurements
- Compare LC vs. TC fusion architectures
- Learn proper filter tuning for heterogeneous sensor rates
- Study innovation monitoring and consistency checking

**Related Chapter**: Chapter 8 - Sensor Fusion

**Related Book Equations**: Eqs. (8.1)-(8.12) for TC fusion, Eqs. (8.13)-(8.18) for LC fusion

---

## Scenario Description

**Trajectory**: 2D rectangular walking path indoors

**Duration**: 60 seconds

**Motion Characteristics**: 
- Constant speed: 1.0 m/s
- Rectangle: 20m × 15m (70m perimeter)
- Counter-clockwise loop starting from (0, 0)
- ~0.86 complete laps

**Sensors**:
- **IMU**: 100 Hz, consumer-grade (smartphone-like)
  - 2D accelerometer: σ = 0.1 m/s²
  - 1D gyroscope (yaw rate): σ = 0.01 rad/s
  - No bias in baseline configuration
- **UWB**: 10 Hz ranging to 4 corner anchors
  - Range noise: σ = 0.05 m (5 cm)
  - Dropout rate: 5% per anchor per measurement
  - No NLOS in baseline

**UWB Anchors** (4 total at rectangle corners):
- Anchor 0: (0.0, 0.0) m - Bottom-left
- Anchor 1: (20.0, 0.0) m - Bottom-right  
- Anchor 2: (20.0, 15.0) m - Top-right
- Anchor 3: (0.0, 15.0) m - Top-left

---

## Files and Data Structure

| File | Shape | Description | Units |
|------|-------|-------------|-------|
| `truth.npz` | | Ground truth states | |
| ├─ `t` | (6000,) | Timestamps | seconds |
| ├─ `p_xy` | (6000, 2) | Positions (x, y) | meters |
| ├─ `v_xy` | (6000, 2) | Velocities (vx, vy) | m/s |
| └─ `yaw` | (6000,) | Heading angle | radians |
| `imu.npz` | | IMU measurements | |
| ├─ `t` | (6000,) | Timestamps | seconds |
| ├─ `accel_xy` | (6000, 2) | 2D accelerations | m/s² |
| └─ `gyro_z` | (6000,) | Yaw rate | rad/s |
| `uwb_ranges.npz` | | UWB range measurements | |
| ├─ `t` | (600,) | Timestamps | seconds |
| └─ `ranges` | (600, 4) | Ranges to 4 anchors (NaN = dropout) | meters |
| `uwb_anchors.npy` | (4, 2) | Anchor positions | meters |
| `config.json` | | Configuration params | see below |

---

## Loading Example

```python
import numpy as np
import json
from pathlib import Path

# Set dataset path
dataset_path = Path('data/sim/ch8_fusion_2d_imu_uwb')

# Load ground truth
truth = np.load(dataset_path / 'truth.npz')
t = truth['t']          # (6000,) timestamps
p_xy = truth['p_xy']    # (6000, 2) positions
v_xy = truth['v_xy']    # (6000, 2) velocities
yaw = truth['yaw']      # (6000,) heading

# Load IMU data
imu = np.load(dataset_path / 'imu.npz')
t_imu = imu['t']              # (6000,) timestamps
accel_xy = imu['accel_xy']    # (6000, 2) accelerations
gyro_z = imu['gyro_z']        # (6000,) yaw rate

# Load UWB data
uwb = np.load(dataset_path / 'uwb_ranges.npz')
t_uwb = uwb['t']              # (600,) timestamps
ranges = uwb['ranges']        # (600, 4) ranges (may contain NaN)

# Load anchor positions
anchors = np.load(dataset_path / 'uwb_anchors.npy')  # (4, 2)

# Load configuration
with open(dataset_path / 'config.json') as f:
    config = json.load(f)
    
print(f"Duration: {config['dataset_info']['duration_sec']} seconds")
print(f"IMU rate: {config['imu']['rate_hz']} Hz")
print(f"UWB rate: {config['uwb']['rate_hz']} Hz")
print(f"Trajectory: {config['trajectory']['width_m']}m × {config['trajectory']['height_m']}m")
print(f"\nIMU samples: {len(t_imu)}")
print(f"UWB samples: {len(t_uwb)}")
print(f"Valid UWB measurements: {np.sum(~np.isnan(ranges))}/{ranges.size}")
```

---

## Configuration Parameters

From `config.json`:

### Trajectory Parameters

- `type`: "rectangular_walk"
- `width_m`: 20.0 (rectangle width)
- `height_m`: 15.0 (rectangle height)
- `speed_m_s`: 1.0 (constant walking speed)

### IMU Parameters

- `rate_hz`: 100.0 (sampling rate)
- `dt_sec`: 0.01 (time step)
- `accel_noise_std_m_s2`: 0.1 (consumer-grade)
- `gyro_noise_std_rad_s`: 0.01 (~0.57°/s)
- `accel_bias_m_s2`: [0.0, 0.0] (no bias in baseline)
- `gyro_bias_rad_s`: 0.0 (no bias in baseline)

### UWB Parameters

- `rate_hz`: 10.0 (measurement rate)
- `n_anchors`: 4
- `range_noise_std_m`: 0.05 (5 cm std)
- `nlos_anchors`: [] (no NLOS in baseline)
- `nlos_bias_m`: 0.5 (not applied in baseline)
- `dropout_rate`: 0.05 (5% dropout probability)

### Temporal Calibration

- `time_offset_sec`: 0.0 (perfectly synchronized)
- `clock_drift`: 0.0 (no relative drift)

---

## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on Fusion Performance | Learning Objective |
|-----------|---------|------------------|------------------------------|-------------------|
| `accel_noise_std` | 0.1 | 0.01-0.5 | Higher → faster IMU drift → fusion relies more on UWB | Understand noise propagation through integration (Ch6, Eq. 6.5) |
| `gyro_noise_std` | 0.01 | 0.001-0.05 | Higher → faster heading drift → larger prediction uncertainty | Observe heading error growth: σ_θ(t) ≈ σ_gyro × √t |
| `range_noise_std` | 0.05 | 0.01-0.5 | Higher → noisier UWB fixes → EKF trusts prediction more | Learn measurement weight balancing (Ch8, Eq. 8.7) |
| `dropout_rate` | 0.05 | 0.1-0.5 | More dropouts → longer IMU-only intervals → larger covariance growth | Understand multi-rate fusion challenges |
| `uwb_rate` | 10 | 1-20 | Lower rate → longer prediction intervals → must handle time-varying DOP | Study update frequency vs. accuracy trade-off |

**Key Insight**: Fusion converts unbounded IMU drift into bounded error by periodically correcting with absolute UWB measurements. Filter covariance grows during IMU-only intervals and resets upon successful UWB updates (Ch8, Eqs. 8.3-8.4).

---

## Dataset Variants

This is the **baseline** configuration. Related variants demonstrate specific challenges:

**1. fusion_2d_imu_uwb_nlos/** - NLOS Robustness Test
   - Anchors 1 and 2 have +0.8m NLOS bias
   - Use for: Testing chi-square gating (Ch8, Eqs. 8.8-8.9) and robust loss functions
   - Expected observation: Gating rejects 60-80% of NLOS measurements, preventing filter corruption

**2. fusion_2d_imu_uwb_timeoffset/** - Temporal Calibration Challenge
   - UWB 50ms behind IMU (time_offset = -0.05s)
   - Clock drift: 100 ppm
   - Use for: Testing time synchronization and calibration (Ch8, Section 8.4)
   - Expected observation: Without correction, RMSE increases 50-100% due to systematic residuals

---

## Visualization Example

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data (as above)
dataset_path = 'data/sim/ch8_fusion_2d_imu_uwb'
truth = np.load(f'{dataset_path}/truth.npz')
uwb = np.load(f'{dataset_path}/uwb_ranges.npz')
anchors = np.load(f'{dataset_path}/uwb_anchors.npy')

# Plot trajectory and anchors
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1], 
        'k-', label='Ground Truth', linewidth=2)
ax.plot(anchors[:, 0], anchors[:, 1], 
        'r^', markersize=15, label='UWB Anchors')

# Annotate anchors
for i, (x, y) in enumerate(anchors):
    ax.text(x, y + 1, f'A{i}', ha='center', fontsize=12, fontweight='bold')

ax.set_xlabel('East (m)', fontsize=12)
ax.set_ylabel('North (m)', fontsize=12)
ax.set_title('2D IMU + UWB Fusion Dataset (Baseline)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.savefig('fusion_2d_baseline_trajectory.svg')

# Plot range measurements over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i in range(4):
    ax = axes[i // 2, i % 2]
    ranges_i = uwb['ranges'][:, i]
    valid_mask = ~np.isnan(ranges_i)
    
    ax.plot(uwb['t'][valid_mask], ranges_i[valid_mask], 
            'o', markersize=3, alpha=0.6, label=f'Measured')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range (m)')
    ax.set_title(f'Anchor {i} Ranges')
    ax.grid(True, alpha=0.3)
    
plt.tight_layout()
plt.savefig('fusion_2d_baseline_ranges.svg')
```

**Quick visualization**:
```bash
python tools/plot_dataset_overview.py data/sim/ch8_fusion_2d_imu_uwb
```

---

## Connection to Book Equations

This dataset is designed to demonstrate:

**Prediction Step** (Ch8, Eqs. 8.1-8.2, TC Fusion)
- IMU measurements propagate state: x̂_k = f(x̂_{k-1}, u_k)
- At 100 Hz, very frequent predictions (every 10ms)
- Covariance grows: P_k = F P_{k-1} F' + Q
- Observable: Position uncertainty increases quadratically during IMU-only intervals

**Update Step** (Ch8, Eqs. 8.5-8.7, TC Fusion)
- UWB ranges provide absolute correction: z_k = h(x̂_k) + v_k
- Innovation: y_k = z_k - h(x̂_k) (Ch8, Eq. 8.5)
- Kalman gain balances prediction vs. measurement: K_k = P_k H' S^{-1} (Ch8, Eq. 8.7)
- Observable: Covariance drops sharply upon successful UWB update

**Innovation Monitoring** (Ch8, Eqs. 8.8-8.9)
- NIS statistic: d² = y' S^{-1} y ~ χ²(m) (Ch8, Eq. 8.8)
- For m=4 ranges, χ²_{0.05}(4) ≈ 9.49
- Expected: ~95% of measurements should have d² < 9.49
- Observable: NIS plot shows consistency (or lack thereof)

**LC vs. TC Comparison** (Ch8, Eqs. 8.13-8.18)
- LC: Two-stage (UWB→position, then fuse positions)
- TC: Direct fusion of raw UWB ranges
- Observable: TC should show ~10-20% better accuracy due to proper uncertainty propagation

---

## Recommended Experiments

### Experiment 1: LC vs. TC Fusion Comparison

**Objective**: Understand architectural differences and performance trade-offs between LC and TC fusion.

**Setup**:
```bash
# Run LC fusion (two-stage)
python -m ch8_sensor_fusion.lc_uwb_imu_ekf --data data/sim/ch8_fusion_2d_imu_uwb

# Run TC fusion (single-stage)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/ch8_fusion_2d_imu_uwb
```

**Expected Observations**:
- LC RMSE: ~0.12-0.15 m (accumulated from two filters)
- TC RMSE: ~0.08-0.12 m (10-20% improvement)
- TC has larger state dimension but better uncertainty handling
- LC is simpler to implement and debug

**Analysis**:
- Plot position errors over time
- Compare covariance traces (LC accumulates more uncertainty)
- Verify: σ_LC² ≈ σ_UWB² + σ_IMU_integration²

**Key Insight**: TC fusion is theoretically optimal (Ch8, Theorem 8.1) but LC is often "good enough" for practical systems.

---

### Experiment 2: IMU Quality Impact on Fusion

**Objective**: Quantify how IMU degradation affects overall fusion performance.

**Setup**:
```bash
# Generate datasets with varying IMU quality
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --preset tactical_imu --output data/sim/fusion_tactical

python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --preset baseline --output data/sim/fusion_consumer

python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --preset degraded_imu --output data/sim/fusion_mems

# Run fusion on each
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/fusion_tactical
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/fusion_consumer
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/fusion_mems
```

**Expected Observations**:
- Tactical IMU: RMSE ~0.08m, very stable between UWB updates
- Consumer IMU: RMSE ~0.12m, moderate drift between updates
- MEMS IMU: RMSE ~0.25m, significant drift, fusion heavily relies on UWB

**Analysis**:
- Plot position error growth between UWB updates
- Measure covariance growth rate: dP/dt ∝ Q (Ch8, Eq. 8.2)
- Verify: Better IMU → less frequent UWB updates needed

**Key Insight**: IMU quality determines prediction uncertainty growth rate, which affects how much the filter trusts measurements vs. predictions.

---

### Experiment 3: Filter Tuning Sensitivity

**Objective**: Understand how measurement covariance R affects fusion performance and consistency.

**Setup**:
Modify fusion script to scale R by factors: 0.1, 0.5, 1.0, 2.0, 5.0

```python
# In tc_uwb_imu_ekf.py, add parameter:
R_scale = 1.0  # vary this: 0.1, 0.5, 1.0, 2.0, 5.0
R = (range_noise_std ** 2) * R_scale * np.eye(4)
```

**Expected Observations**:

| R Scale | RMSE | NIS Consistency | Behavior |
|---------|------|-----------------|----------|
| 0.1 | May diverge | << χ² threshold | Over-confident, ignores prediction |
| 0.5 | Slightly low | Below threshold | Slightly over-trusting measurements |
| 1.0 | Optimal | ~95% within bounds | Balanced, consistent |
| 2.0 | Slightly high | Above threshold | Conservative, under-uses measurements |
| 5.0 | High | >> χ² threshold | Too conservative, misses information |

**Analysis**:
- Plot NIS values over time with χ²_{0.05}(4) = 9.49 threshold
- Count % of NIS values within bounds (should be ~95% for correct R)
- Plot RMSE vs. R scale (U-shaped curve, minimum at true R)

**Key Insight**: Proper covariance tuning is critical. NIS monitoring (Ch8, Eqs. 8.8-8.9) provides objective consistency check.

---

## Troubleshooting / Common Student Questions

**Q: Why does position error grow between UWB updates even with frequent IMU measurements?**
A: IMU-only positioning suffers from unbounded drift (Ch6 fundamental limitation). Each integration step accumulates noise: velocity error grows as √t, position error as t^(3/2). UWB provides absolute corrections that reset this accumulation.

**Q: My filter diverges after a few seconds. What's wrong?**
A: Common causes:
1. Under-estimated R (measurement covariance) → check NIS plot, should be ~95% below χ² threshold
2. Over-estimated Q (process covariance) → increase Q
3. Poor initialization → check initial P₀ matches actual uncertainty
4. Bug in linearization → verify Jacobian H matrix

**Q: LC vs. TC: which should I use in practice?**
A: LC is simpler to implement and debug, "good enough" for many applications. Use TC when:
- Need maximum accuracy (10-20% improvement)
- Have computational resources for larger state
- Sensor correlations are significant
- System is safety-critical

**Q: How do I choose the UWB update rate (10 Hz here)?**
A: Trade-offs:
- Lower rate → less computational cost, but longer IMU drift intervals
- Higher rate → better tracking, but diminishing returns and more processing
- Rule of thumb: UWB rate ≈ (IMU rate) / (10-20) for indoor walking
- Adapt based on IMU quality and motion dynamics

**Q: What if I don't have exactly 4 anchors?**
A: 2D positioning requires minimum 3 anchors (3 ranges → 2D position via trilateration). More anchors improve:
- Geometric dilution of precision (GDOP)
- Robustness to NLOS/outliers
- Observability in degenerate geometries
4 anchors is sweet spot for indoor 2D; 5+ provides redundancy.

---

## Generation

This dataset was generated using:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py
```

**Regenerate with custom parameters**:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.2 \
    --gyro-noise 0.02 \
    --range-noise 0.1 \
    --duration 120 \
    --output data/sim/my_custom_fusion
```

**Use presets**:
```bash
# Tactical-grade IMU
python scripts/generate_fusion_2d_imu_uwb_dataset.py --preset tactical_imu

# High dropout test
python scripts/generate_fusion_2d_imu_uwb_dataset.py --preset high_dropout
```

**Generate all 3 standard variants at once**:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py --all-variants
```

See `scripts/README.md` for more experimentation scenarios.

---

## References

- **Book Chapter**: Chapter 8, Sections 8.1-8.3 (Sensor Fusion)
- **Key Equations**: 
  - TC Prediction: Eqs. (8.1)-(8.2)
  - TC Update: Eqs. (8.5)-(8.7)
  - Innovation Monitoring: Eqs. (8.8)-(8.9)
  - LC Fusion: Eqs. (8.13)-(8.18)
- **Related Examples**: 
  - `ch8_sensor_fusion/lc_uwb_imu_ekf.py` - Loosely-coupled demo
  - `ch8_sensor_fusion/tc_uwb_imu_ekf.py` - Tightly-coupled demo
  - `ch8_sensor_fusion/example_innovation_monitoring.py` - NIS analysis
- **Generation Script**: `scripts/generate_fusion_2d_imu_uwb_dataset.py`
- **Variants**: See `fusion_2d_imu_uwb_nlos/` and `fusion_2d_imu_uwb_timeoffset/`


