# Foot-Mounted IMU with ZUPT Dataset

## Overview

**Purpose**: Demonstrate Zero-Velocity UPdaTes (ZUPT) as THE solution to unbounded IMU drift for foot-mounted applications, achieving ~100× improvement over pure integration.

**Learning Objectives**:
- Understand ZUPT principle: use zero-velocity constraints during stance (Ch6, Section 6.3)
- Observe dramatic drift reduction: unbounded → ~2-5% of distance traveled
- Learn stance detection from IMU statistics (acceleration magnitude, variance)
- Compare IMU-only (unbounded) vs. ZUPT (bounded) performance
- Understand why ZUPT works: resets velocity errors before they integrate to position

**Related Chapter**: Chapter 6 - Dead Reckoning (Section 6.3, ZUPT)

**Related Book Equations**: Eqs. (6.44)-(6.45) for ZUPT constraint, Eq. (6.46) for stance detection

---

## Scenario Description

**Trajectory**: 2D linear walking (forward motion with stop-and-go)

**Duration**: 12 seconds

**Motion Characteristics**:
- 20 steps total
- Step length: 0.7 meters (typical adult)
- Step duration: 0.6 seconds (1.7 steps/second)
- Stance duration: 0.2 seconds (foot stationary, ~33% of time)
- Total distance: 14.0 meters

**Sensors**:
- **Foot-mounted IMU**: 100 Hz, consumer-grade
  - 2D accelerometer: σ = 0.1 m/s²
  - 1D gyroscope (yaw rate): σ = 0.01 rad/s
  - Mounted on foot (experiences high accelerations during swing)

**Key Feature**: Ground truth includes `is_stance` labels indicating when foot is stationary → enables ZUPT constraint.

**Why ZUPT Works**:
1. During stance: foot velocity = 0 (known constraint)
2. Use this to correct velocity estimate from IMU
3. Velocity errors don't integrate to position → bounded drift
4. ~100× better than pure IMU integration

---

## Files and Data Structure

| File | Shape | Description | Units |
|------|-------|-------------|-------|
| `truth.npz` | | Ground truth states | |
| ├─ `t` | (1200,) | Timestamps | seconds |
| ├─ `p_xy` | (1200, 2) | Positions (x, y) | meters |
| ├─ `v_xy` | (1200, 2) | Velocities (vx, vy) | m/s |
| ├─ `yaw` | (1200,) | Heading angle | radians |
| └─ `is_stance` | (1200,) | Stance phase indicator | boolean |
| `imu.npz` | | IMU measurements | |
| ├─ `t` | (1200,) | Timestamps | seconds |
| ├─ `accel_xy` | (1200, 2) | 2D accelerations | m/s² |
| └─ `gyro_z` | (1200,) | Yaw rate | rad/s |
| `config.json` | | Configuration params | see below |

**Important**: `is_stance` provides ground truth for when foot is stationary. In practice, detect stance from IMU statistics (see recommended experiments).

---

## Loading Example

```python
import numpy as np
import json
from pathlib import Path

# Set dataset path
dataset_path = Path('data/sim/ch6_foot_zupt_walk')

# Load ground truth
truth = np.load(dataset_path / 'truth.npz')
t = truth['t']              # (1200,) timestamps
p_xy = truth['p_xy']        # (1200, 2) positions
v_xy = truth['v_xy']        # (1200, 2) velocities
yaw = truth['yaw']          # (1200,) heading
is_stance = truth['is_stance']  # (1200,) stance labels

# Load IMU data
imu = np.load(dataset_path / 'imu.npz')
t_imu = imu['t']              # (1200,) timestamps
accel_xy = imu['accel_xy']    # (1200, 2) accelerations
gyro_z = imu['gyro_z']        # (1200,) yaw rate

# Load configuration
with open(dataset_path / 'config.json') as f:
    config = json.load(f)
    
print(f"Duration: {config['dataset_info']['duration_sec']} seconds")
print(f"Steps: {config['trajectory']['num_steps']}")
print(f"Distance: {config['dataset_info']['total_distance_m']} meters")
print(f"Stance ratio: {config['trajectory']['stance_ratio']*100:.1f}%")
print(f"\nStance samples: {np.sum(is_stance)}/{len(is_stance)}")
```

---

## Configuration Parameters

From `config.json`:

### Trajectory Parameters

- `type`: "walking_linear"
- `num_steps`: 20
- `step_length_m`: 0.7 (typical adult stride)
- `step_duration_sec`: 0.6 (time per complete step)
- `stance_duration_sec`: 0.2 (stationary time per step)
- `swing_duration_sec`: 0.4 (moving time per step)
- `stance_ratio`: 0.335 (33.5% of time in stance)

### IMU Parameters

- `rate_hz`: 100.0 (sampling rate)
- `dt_sec`: 0.01 (time step)
- `accel_noise_std_m_s2`: 0.1
- `gyro_noise_std_rad_s`: 0.01
- `accel_bias_m_s2`: [0.0, 0.0] (no bias in baseline)
- `gyro_bias_rad_s`: 0.0 (no bias in baseline)

### ZUPT Parameters

- `stance_threshold_description`: Ground truth provided via `is_stance`
- `detection_note`: In practice, detect from IMU statistics (see experiments)

---

## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on ZUPT Performance | Learning Objective |
|-----------|---------|------------------|----------------------------|-------------------|
| `stance_duration` | 0.2 | 0.1-0.4 | Longer stance → more frequent corrections → better performance | Understand update frequency importance |
| `step_length` | 0.7 | 0.5-1.0 | Longer steps → more velocity error accumulation between ZUPTs | See swing phase error growth |
| `accel_noise_std` | 0.1 | 0.01-0.5 | Higher noise → but ZUPT still works! (resets before accumulation) | Observe robustness to IMU quality |
| Stance detection threshold | — | varies | Wrong detection → missed corrections or false alarms | Learn detection trade-offs |

**Key Insight**: ZUPT effectiveness depends on:
1. **Stance detection accuracy**: Miss stance → drift accumulates; False alarm → corrupts good velocity
2. **Stance frequency**: More frequent → better (errors reset before growing large)
3. **IMU quality**: Less critical than for pure integration (ZUPT is robust!)

**Performance Comparison** (14m walk, consumer IMU):
- **Pure IMU**: ~50-150m error (350-1000% of distance) - UNUSABLE
- **With ZUPT**: ~0.3-0.7m error (2-5% of distance) - EXCELLENT
- **Improvement**: ~100× better!

---

## Visualization Example

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data (as above)
dataset_path = 'data/sim/ch6_foot_zupt_walk'
truth = np.load(f'{dataset_path}/truth.npz')
imu = np.load(f'{dataset_path}/imu.npz')

# Plot trajectory with stance phases highlighted
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Position over time
axes[0].plot(truth['t'], truth['p_xy'][:, 0], 'b-', linewidth=2, label='Ground Truth')
axes[0].fill_between(truth['t'], 0, truth['p_xy'][:, 0], 
                      where=truth['is_stance'], alpha=0.3, color='red', 
                      label='Stance Phase')
axes[0].set_ylabel('Position X (m)', fontsize=11)
axes[0].set_title('Walking with ZUPT: Trajectory and Stance Phases', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Velocity over time
axes[1].plot(truth['t'], truth['v_xy'][:, 0], 'g-', linewidth=2, label='Velocity')
axes[1].fill_between(truth['t'], 0, np.max(truth['v_xy'][:, 0]), 
                      where=truth['is_stance'], alpha=0.3, color='red',
                      label='Stance (v=0)')
axes[1].set_ylabel('Velocity X (m/s)', fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Acceleration magnitude (for stance detection)
accel_mag = np.linalg.norm(imu['accel_xy'], axis=1)
axes[2].plot(truth['t'], accel_mag, 'r-', linewidth=1, alpha=0.7, label='Accel Magnitude')
axes[2].fill_between(truth['t'], 0, np.max(accel_mag), 
                      where=truth['is_stance'], alpha=0.3, color='red',
                      label='Stance (low accel)')
axes[2].axhline(np.std(accel_mag[truth['is_stance']]), color='k', linestyle='--', 
                label='Stance threshold estimate')
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Accel Mag (m/s²)', fontsize=11)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zupt_trajectory_stance.svg')
```

---

## Connection to Book Equations

This dataset demonstrates the ZUPT method from Ch6, Section 6.3:

**Zero-Velocity Constraint** (Ch6, Eq. 6.44):

When foot is in stance phase:
```
v_measured = 0  (with measurement noise R_zupt)
```

Apply as pseudo-measurement in EKF:
```
Innovation: y = 0 - v̂
Update state: x̂+ = x̂- + K × y
```

**Stance Detection** (Ch6, Eq. 6.46):

Detect stance when:
```
||a_measured|| < threshold_a  AND
variance(a_window) < threshold_var
```

Typical thresholds for foot-mounted:
- threshold_a ≈ 0.5 m/s² (near gravity = 9.81 m/s²)
- threshold_var ≈ 0.01 (m/s²)²

**Why ZUPT Works**:

Without ZUPT:
- Velocity error grows: σ_v(t) = σ_a √t
- Position error grows: σ_p(t) = (σ_a/√2) t^(3/2)
- After 0.4s swing (200 samples): σ_v ≈ 0.06 m/s, σ_p ≈ 0.004 m

With ZUPT (every 0.6s):
- Velocity reset to 0 during stance → σ_v never exceeds ~0.06 m/s
- Position drift limited to single swing phase: ~0.004 m per step
- After 20 steps: total error ~20 × 0.004m = 0.08m (vs. 50-150m without!)

**Error Budget**:

Pure IMU (14m walk, 12s):
- Velocity error @ 12s: σ_v ≈ 0.35 m/s
- Position error @ 12s: σ_p ≈ 125 m (unbounded)

With ZUPT (20 swing phases):
- Velocity error: reset 20 times, max ~0.06 m/s
- Position error: √20 × 0.004m ≈ 0.02 m (bounded!)
- Heading error: similar dramatic improvement

**Key Insight**: ZUPT transforms IMU from unbounded to bounded by resetting velocity errors before they integrate to position. Critical for foot-mounted IMU systems.

---

## Recommended Experiments

### Experiment 1: ZUPT Effectiveness Demonstration

**Objective**: Observe dramatic improvement from ZUPT vs. pure IMU integration.

**Setup**:
```bash
# Generate ZUPT dataset (already done)
python scripts/generate_ch6_zupt_dataset.py

# For comparison, could generate longer strapdown trajectory
python scripts/generate_ch6_strapdown_dataset.py --duration 12 --output data/sim/ch6_strapdown_12s
```

**Run Integration**:
```bash
# Pure IMU integration (unbounded drift)
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/ch6_foot_zupt_walk

# With ZUPT corrections (bounded drift)
python -m ch6_dead_reckoning.example_zupt --data data/sim/ch6_foot_zupt_walk
```

**Expected Observations**:

| Method | Position RMSE @ 14m | Error as % of Distance | Status |
|--------|---------------------|------------------------|--------|
| Pure IMU | 50-150 m | 350-1000% | UNUSABLE |
| With ZUPT | 0.3-0.7 m | 2-5% | EXCELLENT |

**Analysis**:
- Plot both trajectories: IMU drifts off wildly, ZUPT stays close to truth
- Plot error growth: IMU grows continuously, ZUPT bounded
- Count stance corrections: 20 ZUPTs → 20 velocity resets

**Key Insight**: ZUPT is THE technique that makes foot-mounted IMU viable for navigation. ~100× improvement!

---

### Experiment 2: Stance Detection Sensitivity

**Objective**: Understand how stance detection accuracy affects ZUPT performance.

**Setup**:
```python
# Implement stance detection from IMU statistics
def detect_stance_simple(accel, gyro, threshold_accel=0.5, threshold_gyro=0.05):
    """Simple stance detector based on low acceleration and gyro.
    
    Args:
        accel: (N, 2) acceleration measurements
        gyro: (N,) gyroscope measurements
        threshold_accel: Acceleration magnitude threshold (m/s²)
        threshold_gyro: Gyro rate threshold (rad/s)
    
    Returns:
        is_stance_detected: (N,) boolean array
    """
    accel_mag = np.linalg.norm(accel, axis=1)
    
    # Low acceleration and low rotation rate → stance
    is_stance_detected = (accel_mag < threshold_accel) & (np.abs(gyro) < threshold_gyro)
    
    return is_stance_detected

# Test different thresholds
thresholds_accel = [0.3, 0.5, 1.0, 2.0]
for thresh in thresholds_accel:
    detected = detect_stance_simple(accel_xy, gyro_z, threshold_accel=thresh)
    true_positive = np.sum(detected & is_stance) / np.sum(is_stance)
    false_positive = np.sum(detected & ~is_stance) / np.sum(~is_stance)
    print(f"Threshold {thresh}: TP={true_positive:.2f}, FP={false_positive:.2f}")
```

**Expected Observations**:
- Threshold too low (0.3): Misses most stances (low recall)
- Threshold optimal (0.5): Good balance (~90% detection, <5% false alarms)
- Threshold too high (2.0): False alarms during swing (high false positive)

**Analysis**:
- Plot ROC curve: true positive rate vs. false positive rate
- Find optimal threshold balancing missed detections vs. false alarms
- Run ZUPT with detected (imperfect) vs. ground truth stance labels

**Key Insight**: Stance detection is critical. Miss stance → drift accumulates. False alarm → corrupts good velocity estimate. Need robust detector!

---

### Experiment 3: Walking Speed Effect

**Objective**: Understand how walking speed affects ZUPT performance.

**Setup**:
```bash
# Slow walking (longer stance phases)
python scripts/generate_ch6_zupt_dataset.py \
    --preset slow_walk \
    --output data/sim/ch6_zupt_slow

# Fast walking (shorter stance phases)
python scripts/generate_ch6_zupt_dataset.py \
    --preset fast_walk \
    --output data/sim/ch6_zupt_fast

# Run ZUPT on each
python -m ch6_dead_reckoning.example_zupt --data data/sim/ch6_foot_zupt_walk
python -m ch6_dead_reckoning.example_zupt --data data/sim/ch6_zupt_slow
python -m ch6_dead_reckoning.example_zupt --data data/sim/ch6_zupt_fast
```

**Expected Observations**:

| Walking Style | Stance Duration | Stance Ratio | Position RMSE | Error % |
|---------------|-----------------|--------------|---------------|---------|
| Slow | 0.3s | ~38% | 0.2-0.4 m | ~2% |
| Baseline | 0.2s | ~33% | 0.3-0.7 m | ~4% |
| Fast | 0.15s | ~30% | 0.5-1.0 m | ~6% |

**Analysis**:
- Slower walking → more frequent ZUPTs → better performance
- Fast walking → less time in stance → less correction opportunity
- But even fast walking MUCH better than no ZUPT!

**Key Insight**: ZUPT effectiveness increases with longer/more frequent stance phases. Gait analysis important for optimal performance.

---

## Troubleshooting / Common Student Questions

**Q: Why is stance detection so important? Can't I just apply ZUPT all the time?**
A: No! If you apply zero-velocity constraint when foot is moving (false alarm), you're telling the filter "velocity is zero" when it's actually ~1 m/s. This corrupts the estimate worse than no ZUPT. Must detect stance accurately.

**Q: How much error does one missed stance detection cause?**
A: One missed stance (0.2s) allows errors to grow uncorrected:
- Additional velocity error: ~0.02 m/s
- Additional position error: ~0.002 m
Not huge for one miss, but if you miss many → performance degrades significantly.

**Q: Can ZUPT work with other sensors (not foot-mounted)?**
A: ZUPT specifically for foot-mounted because clear stance phases. For vehicles, use "zero-velocity" when stopped (red light, parking). For hand-held, harder (no clear zero-velocity moments) → need other techniques.

**Q: My ZUPT implementation shows "jumps" in position at each stance. Is this wrong?**
A: These jumps are CORRECTIONS! The filter is correcting accumulated drift from swing phase. Jumps should be small (<0.1m). Large jumps (>1m) indicate stance detection errors or bad IMU integration.

**Q: Does ZUPT correct heading errors too?**
A: Yes! Zero-velocity constraint provides 2D information:
- Direct: velocity in both x,y directions = 0
- Indirect: heading errors cause velocity errors, which ZUPT detects and corrects
But heading correction is weaker than position/velocity.

**Q: What if IMU has significant bias?**
A: ZUPT helps estimate and remove bias! During stance:
- Measured acceleration ≠ 0 → must be bias
- Filter can estimate bias as augmented state
- This is called "ZUPT-based IMU calibration"

**Q: How does this compare to the strapdown dataset?**
A: Same IMU quality, but ZUPT adds constraints:
- Strapdown (ch6_strapdown_basic): Pure IMU → 50-150m error in 12s (UNUSABLE)
- ZUPT (this dataset): IMU + constraints → 0.3-0.7m error in 14m (EXCELLENT)
Demonstrates why constraints are necessary for IMU-based navigation.

---

## Generation

This dataset was generated using:
```bash
python scripts/generate_ch6_zupt_dataset.py
```

**Regenerate with custom parameters**:
```bash
# Slow walking (more stance time)
python scripts/generate_ch6_zupt_dataset.py --preset slow_walk

# Fast walking (less stance time)
python scripts/generate_ch6_zupt_dataset.py --preset fast_walk

# More steps
python scripts/generate_ch6_zupt_dataset.py --num-steps 40 --output data/sim/ch6_zupt_long

# Test with noisy IMU
python scripts/generate_ch6_zupt_dataset.py --preset noisy_imu

# Custom gait parameters
python scripts/generate_ch6_zupt_dataset.py \
    --step-length 0.8 --stance-duration 0.15 \
    --output data/sim/ch6_zupt_custom
```

See `scripts/README.md` for more experimentation scenarios.

---

## References

- **Book Chapter**: Chapter 6, Section 6.3 (Zero-Velocity Updates)
- **Key Equations**:
  - ZUPT constraint: Eqs. (6.44)-(6.45)
  - Stance detection: Eq. (6.46)
  - Error analysis: Section 6.3.2
- **Related Examples**:
  - `ch6_dead_reckoning/example_zupt.py` - ZUPT EKF implementation
  - `ch6_dead_reckoning/example_stance_detection.py` - Detector algorithms
- **Generation Script**: `scripts/generate_ch6_zupt_dataset.py`
- **Related Datasets**:
  - `ch6_strapdown_basic/` - Shows the problem (unbounded drift)
  - `fusion_2d_imu_uwb/` - Alternative solution (sensor fusion)

