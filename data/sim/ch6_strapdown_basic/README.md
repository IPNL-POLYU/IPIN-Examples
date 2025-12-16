# IMU Strapdown Integration Dataset

## Overview

**Purpose**: Demonstrate unbounded drift from pure IMU integration without external corrections, the fundamental limitation of dead reckoning.

**Learning Objectives**:
- Understand how IMU errors propagate through integration (Ch6, Section 6.1)
- Observe unbounded position drift: grows as t^(3/2) for noise, t² for bias
- Distinguish between random drift (noise) and systematic drift (bias)
- Quantify drift rates for different IMU grades (tactical, consumer, MEMS)
- Understand why IMU-only positioning requires constraints or fusion

**Related Chapter**: Chapter 6 - Dead Reckoning (Sections 6.1-6.2)

**Related Book Equations**: Eqs. (6.5), (6.9) for IMU error models, Eq. (6.19) for strapdown integration

---

## Scenario Description

**Trajectory**: 2D circular motion (constant radius, constant speed)

**Duration**: 60 seconds

**Motion Characteristics**:
- Radius: 10.0 meters
- Speed: 1.0 m/s (constant)
- Period: ~62.8 seconds (almost one complete lap)
- Centripetal acceleration: 0.1 m/s²

**Sensors**:
- **IMU only**: 100 Hz, consumer-grade
  - 2D accelerometer: σ = 0.1 m/s²
  - 1D gyroscope (yaw rate): σ = 0.01 rad/s (~0.57°/s)
  - No bias in baseline configuration

**Why Circular Motion?**
- Constant non-zero acceleration (tests accelerometer under load)
- Constant angular velocity (tests gyroscope performance)
- Easy to verify: position should return to start after each lap
- Drift becomes obvious when trajectory diverges from circle

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
| `config.json` | | Configuration params | see below |

---

## Loading Example

```python
import numpy as np
import json
from pathlib import Path

# Set dataset path
dataset_path = Path('data/sim/ch6_strapdown_basic')

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

# Load configuration
with open(dataset_path / 'config.json') as f:
    config = json.load(f)
    
print(f"Duration: {config['dataset_info']['duration_sec']} seconds")
print(f"IMU rate: {config['imu']['rate_hz']} Hz")
print(f"Trajectory: circular, radius={config['trajectory']['radius_m']}m")
print(f"\nIMU grade: consumer")
print(f"Accel noise: {config['imu']['accel_noise_std_m_s2']} m/s²")
print(f"Gyro noise: {config['imu']['gyro_noise_std_rad_s']} rad/s")
```

---

## Configuration Parameters

From `config.json`:

### Trajectory Parameters

- `type`: "circular"
- `radius_m`: 10.0 (circle radius)
- `speed_m_s`: 1.0 (constant speed)

### IMU Parameters (Consumer-Grade Baseline)

- `rate_hz`: 100.0 (sampling rate)
- `dt_sec`: 0.01 (time step)
- `accel_noise_std_m_s2`: 0.1
- `gyro_noise_std_rad_s`: 0.01
- `accel_bias_m_s2`: [0.0, 0.0] (no bias in baseline)
- `gyro_bias_rad_s`: 0.0 (no bias in baseline)

---

## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on Position Drift | Learning Objective |
|-----------|---------|------------------|--------------------------|-------------------|
| `accel_noise_std` | 0.1 | 0.01-0.5 | Drift grows as t^(3/2): σ_p(t) ≈ (σ_a/√2) × t^(3/2) | Observe stochastic drift from white noise |
| `gyro_noise_std` | 0.01 | 0.001-0.05 | Heading drift grows as √t: σ_θ(t) ≈ σ_ω × √t | See how heading errors amplify position errors |
| `accel_bias` | [0,0] | [0.05, 0.05] | Drift grows as t²: Δp(t) ≈ ½ b_a × t² | Systematic unbounded divergence |
| `gyro_bias` | 0.0 | 0.001-0.01 | Linear heading drift: Δθ(t) = b_ω × t | Understand bias vs. noise effects |

**Key Insight**: Without external corrections, IMU drift is **unbounded**. Noise causes stochastic growth (~t^(3/2)), bias causes systematic growth (t²). After 60s at 1 m/s:
- Consumer IMU: ~30-50m position error (5-8× distance traveled)
- MEMS IMU: ~150-250m error (25-40× distance traveled)
- Tactical IMU: ~5-10m error (~1× distance traveled)

**Theoretical Predictions** (Ch6, Section 6.1):

Velocity error from accel noise: σ_v(t) = σ_a × √t  
Position error from accel noise: σ_p(t) = (σ_a / √2) × t^(3/2)

For σ_a = 0.1 m/s², t = 60s:  
σ_v(60) ≈ 0.77 m/s  
σ_p(60) ≈ 36 m

---

## Visualization Example

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data (as above)
dataset_path = 'data/sim/ch6_strapdown_basic'
truth = np.load(f'{dataset_path}/truth.npz')

# Plot circular trajectory
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1], 
        'b-', label='Ground Truth', linewidth=2)
ax.plot(truth['p_xy'][0, 0], truth['p_xy'][0, 1], 
        'go', markersize=12, label='Start')
ax.plot(truth['p_xy'][-1, 0], truth['p_xy'][-1, 1], 
        'ro', markersize=12, label='End')

# Draw circle
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(10*np.cos(theta), 10*np.sin(theta), 
        'k--', alpha=0.3, label='Ideal Circle (r=10m)')

ax.set_xlabel('East (m)', fontsize=12)
ax.set_ylabel('North (m)', fontsize=12)
ax.set_title('IMU Strapdown: Circular Trajectory', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.savefig('strapdown_trajectory.svg')

# Plot IMU measurements
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(truth['t'], imu['accel_xy'][:, 0], 'b-', linewidth=0.5, alpha=0.7)
axes[0].set_ylabel('Accel X (m/s²)')
axes[0].set_title('IMU Measurements (Consumer-Grade)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(truth['t'], imu['accel_xy'][:, 1], 'g-', linewidth=0.5, alpha=0.7)
axes[1].set_ylabel('Accel Y (m/s²)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(truth['t'], imu['gyro_z'], 'r-', linewidth=0.5, alpha=0.7)
axes[2].set_ylabel('Gyro Z (rad/s)')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('strapdown_imu_measurements.svg')
```

---

## Connection to Book Equations

This dataset demonstrates fundamental concepts from Ch6:

**IMU Error Model** (Ch6, Eqs. 6.5, 6.9):

Accelerometer: f̃ = f + b_a + n_a  
Gyroscope: ω̃ = ω + b_ω + n_ω

Where:
- f̃, ω̃: measured specific force and angular velocity
- b_a, b_ω: constant biases
- n_a, n_ω: white noise processes

**Strapdown Integration** (Ch6, Eq. 6.19):

```
v_{k+1} = v_k + C(θ_k) f̃_k Δt
p_{k+1} = p_k + v_k Δt + ½ C(θ_k) f̃_k Δt²
θ_{k+1} = θ_k + ω̃_k Δt
```

**Error Propagation**:

Noise integration (Ch6, Section 6.1.2):
- Gyro noise → heading random walk: σ_θ(t) = σ_ω √t
- Accel noise → velocity random walk: σ_v(t) = σ_a √t
- Accel noise → position error: σ_p(t) = (σ_a/√2) t^(3/2)

Bias integration:
- Gyro bias → linear heading drift: Δθ(t) = b_ω t
- Accel bias → quadratic position drift: Δp(t) = ½ b_a t²

**Why Drift is Unbounded**:

IMU measures acceleration and angular velocity (derivatives).  
Navigation requires position and orientation (integrals).  
Each integration step adds noise → errors accumulate → unbounded growth.

Solution: Add constraints (ZUPT) or fuse with absolute measurements (GPS, UWB).

---

## Recommended Experiments

### Experiment 1: IMU Grade Comparison

**Objective**: Quantify drift rates for different IMU qualities.

**Setup**:
```bash
# Generate datasets for different IMU grades
python scripts/generate_ch6_strapdown_dataset.py --preset tactical
python scripts/generate_ch6_strapdown_dataset.py --preset consumer  
python scripts/generate_ch6_strapdown_dataset.py --preset mems

# Or use custom noise levels
python scripts/generate_ch6_strapdown_dataset.py \
    --accel-noise 0.01 --gyro-noise 0.001 \
    --output data/sim/ch6_strapdown_tactical

python scripts/generate_ch6_strapdown_dataset.py \
    --accel-noise 0.5 --gyro-noise 0.05 \
    --output data/sim/ch6_strapdown_mems
```

**Run Integration**:
```bash
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/ch6_strapdown_tactical
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/ch6_strapdown_basic
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/ch6_strapdown_mems
```

**Expected Observations**:

| IMU Grade | σ_a | σ_ω | Position RMSE @ 60s | Drift Rate |
|-----------|-----|-----|---------------------|------------|
| Tactical | 0.01 | 0.001 | ~5-8 m | ~1% of distance |
| Consumer | 0.1 | 0.01 | ~30-50 m | ~5-8% |
| MEMS | 0.5 | 0.05 | ~150-250 m | ~25-40% |

**Analysis**:
- Plot position error vs. time (should show t^(3/2) growth)
- Fit power law: log(error) vs. log(time)
- Verify slope ≈ 1.5 for noise-dominated case
- Compare with theoretical prediction

**Key Insight**: 5× increase in noise → 25× increase in position error after 60s (because t^(3/2) dependence). Better IMU dramatically reduces drift but doesn't eliminate it.

---

### Experiment 2: Noise vs. Bias Effects

**Objective**: Distinguish stochastic drift (noise) from systematic drift (bias).

**Setup**:
```bash
# Pure noise (no bias)
python scripts/generate_ch6_strapdown_dataset.py \
    --accel-noise 0.1 --gyro-noise 0.01 \
    --output data/sim/ch6_strapdown_noise_only

# Pure bias (no noise)
python scripts/generate_ch6_strapdown_dataset.py \
    --accel-noise 0.0 --gyro-noise 0.0 \
    --accel-bias-x 0.05 --accel-bias-y 0.03 --gyro-bias 0.002 \
    --output data/sim/ch6_strapdown_bias_only

# Combined (realistic)
python scripts/generate_ch6_strapdown_dataset.py \
    --preset biased_consumer \
    --output data/sim/ch6_strapdown_combined
```

**Run Integration**:
```bash
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/ch6_strapdown_noise_only
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/ch6_strapdown_bias_only
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/ch6_strapdown_combined
```

**Expected Observations**:
- **Noise only**: Error grows but different each run (stochastic), t^(3/2) growth
- **Bias only**: Error grows systematically (reproducible), t² growth, much faster
- **Combined**: Bias dominates long-term, noise adds variability

**Analysis**:
- Run noise-only case 10 times → different trajectories (Monte Carlo)
- Run bias-only case 10 times → identical trajectories (deterministic)
- Plot error growth rate: bias case should be quadratic

**Key Insight**: Bias is more dangerous than noise because:
1. Quadratic growth (t²) faster than noise (t^(3/2))
2. Systematic → always goes same direction
3. Can't be reduced by averaging

Solution: Calibration to estimate and remove bias.

---

### Experiment 3: Integration Time Effect

**Objective**: Understand how drift scales with time.

**Setup**:
```bash
# Generate datasets with different durations
for dur in 30 60 120 180; do
    python scripts/generate_ch6_strapdown_dataset.py \
        --duration $dur \
        --output data/sim/ch6_strapdown_${dur}s
done
```

**Run Integration**:
```bash
for dur in 30 60 120 180; do
    python -m ch6_dead_reckoning.example_imu_strapdown \
        --data data/sim/ch6_strapdown_${dur}s \
        --output results_${dur}s.json
done
```

**Expected Observations**:

| Duration | Position RMSE (consumer) | Growth Factor |
|----------|--------------------------|---------------|
| 30 s | ~15 m | baseline |
| 60 s | ~36 m | 2.4× (≈ 2^1.5) |
| 120 s | ~86 m | 5.7× (≈ 4^1.5) |
| 180 s | ~140 m | 9.3× (≈ 6^1.5) |

**Analysis**:
- Plot RMSE vs. time on log-log scale
- Verify slope ≈ 1.5 (t^(3/2) relationship)
- Extrapolate: after 5 minutes → ~200m error!

**Key Insight**: Drift grows super-linearly. Doubling time → 2.8× more drift. After a few minutes, IMU-only positioning becomes useless.

---

## Troubleshooting / Common Student Questions

**Q: Why doesn't the integrated trajectory match ground truth even though IMU measurements are "reasonable"?**
A: Small errors (0.1 m/s² noise) seem negligible, but integration amplifies them:
- 0.1 m/s² over 60s → 6 m/s velocity error (if integrated once)
- That 6 m/s over 60s → 180m position error (if integrated twice)
Actual error is smaller (~36m) due to √t and statistical cancellation, but still huge.

**Q: Can I just use a better IMU to solve the drift problem?**
A: Better IMU helps but doesn't eliminate drift. Even tactical-grade IMU (100× better) still drifts ~5m in 60s. For long-term navigation, you need:
- Constraints (ZUPT for foot-mounted, odometry for vehicles)
- Absolute measurements (GPS, UWB, vision)
- Map matching or SLAM

**Q: Why circular trajectory instead of straight line?**
A: Circular motion:
1. Tests accelerometer under load (centripetal acceleration)
2. Tests gyroscope continuously (constant rotation)
3. Makes drift obvious (should return to start, doesn't)
4. Represents realistic vehicular/robotic motion

**Q: My integrated position has huge bias in one direction. Is this wrong?**
A: If you included accel/gyro bias, yes, this is expected! Bias causes systematic drift in one direction. Without bias, drift should be more "random" (though still growing).

**Q: How can I reduce drift without external sensors?**
A: Limited options for IMU-only:
1. Better calibration (remove bias)
2. Better IMU (reduce noise)
3. Zero-velocity updates (ZUPT) if there are stationary periods
4. Motion constraints (e.g., "vehicle moves forward" for cars)

But fundamentally, IMU alone cannot provide bounded positioning.

**Q: What's the difference between this and the fusion dataset?**
A: This dataset: Pure IMU → unbounded drift (demonstrates problem)  
Fusion dataset: IMU + UWB → bounded error (demonstrates solution)

Use this dataset to understand WHY fusion/constraints are needed.

---

## Generation

This dataset was generated using:
```bash
python scripts/generate_ch6_strapdown_dataset.py
```

**Regenerate with custom parameters**:
```bash
# Tactical-grade IMU
python scripts/generate_ch6_strapdown_dataset.py --preset tactical

# MEMS-grade IMU  
python scripts/generate_ch6_strapdown_dataset.py --preset mems

# Custom noise levels
python scripts/generate_ch6_strapdown_dataset.py \
    --accel-noise 0.2 --gyro-noise 0.02 \
    --duration 120

# With bias
python scripts/generate_ch6_strapdown_dataset.py \
    --accel-bias-x 0.05 --gyro-bias 0.001 \
    --output data/sim/ch6_strapdown_biased
```

See `scripts/README.md` for more experimentation scenarios.

---

## References

- **Book Chapter**: Chapter 6, Sections 6.1-6.2 (IMU Strapdown Integration)
- **Key Equations**: 
  - IMU error models: Eqs. (6.5), (6.9)
  - Strapdown integration: Eq. (6.19)
  - Error propagation analysis: Section 6.1.2
- **Related Examples**:
  - `ch6_dead_reckoning/example_imu_strapdown.py` - Integration demo
  - `ch6_dead_reckoning/example_error_analysis.py` - Drift analysis
- **Generation Script**: `scripts/generate_ch6_strapdown_dataset.py`
- **Related Datasets**:
  - `ch6_foot_zupt_walk/` - ZUPT demonstrates how to bound drift
  - `fusion_2d_imu_uwb/` - Fusion demonstrates another solution


