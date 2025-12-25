# Chapter 6: Dead Reckoning

## Overview

This module implements dead reckoning and sensor algorithms described in **Chapter 6** of *Principles of Indoor Positioning and Indoor Navigation*. Dead reckoning propagates position using proprioceptive sensors (IMU, wheel encoders, step counters) without external references.

The module provides simulation-based examples of:
- **IMU strapdown integration** (attitude, velocity, position propagation)
- **Wheel odometry** (vehicle dead reckoning with lever arm compensation)
- **Drift correction constraints** (ZUPT, ZARU, NHC)
- **Pedestrian dead reckoning** (step-and-heading navigation)
- **Environmental sensors** (magnetometer heading, barometric altitude)
- **IMU calibration** (Allan variance noise characterization)

**Key Insight:** Dead reckoning drifts unbounded without corrections. Examples demonstrate both the drift problem and solutions.

## ⚙️ Frame Conventions (IMPORTANT!)

All Chapter 6 algorithms use **explicit frame conventions** via the `FrameConvention` dataclass. This ensures:
- ✅ Correct gravity handling (no drift for stationary IMU)
- ✅ Consistent heading definitions (0° = East in ENU, 0° = North in NED)
- ✅ Support for both ENU and NED coordinate systems

**Default:** ENU (East-North-Up) where:
- x = East, y = North, z = Up
- Heading 0° = East, 90° = North
- Gravity: [0, 0, -9.81] m/s²

```python
from core.sensors import FrameConvention, strapdown_update

# Explicit frame convention (recommended)
frame = FrameConvention.create_enu()
q, v, p = strapdown_update(q, v, p, omega_b, f_b, dt, frame=frame)
```

**📖 See detailed documentation:** [`docs/ch6_frame_conventions.md`](../docs/ch6_frame_conventions.md)

**✅ Validated:** All conventions are tested in `tests/core/test_strapdown_stationary_imu.py` (stationary IMU produces **zero drift**).

## Quick Start

```bash
# Run individual examples
python -m ch6_dead_reckoning.example_imu_strapdown
python -m ch6_dead_reckoning.example_zupt
python -m ch6_dead_reckoning.example_wheel_odometry
python -m ch6_dead_reckoning.example_environment
python -m ch6_dead_reckoning.example_allan_variance

# Run PDR with pre-generated dataset
python -m ch6_dead_reckoning.example_pdr --data ch6_pdr_corridor_walk

# Run comprehensive comparison
python -m ch6_dead_reckoning.example_comparison
```

## 📂 Dataset Connection

| Example Script | Dataset | Description |
|----------------|---------|-------------|
| `example_pdr.py` | `data/sim/ch6_pdr_corridor_walk/` | 40m x 20m corridor walk with IMU data |
| *(manual loading)* | `data/sim/ch6_strapdown_basic/` | Basic IMU strapdown integration |
| *(manual loading)* | `data/sim/ch6_wheel_odom_square/` | Vehicle wheel odometry square path |
| *(manual loading)* | `data/sim/ch6_foot_zupt_walk/` | Foot-mounted IMU with ZUPT |
| *(manual loading)* | `data/sim/ch6_env_sensors_heading_altitude/` | Magnetometer and barometer data |

**Load dataset manually:**
```python
import numpy as np
import json
from pathlib import Path

path = Path("data/sim/ch6_pdr_corridor_walk")
t = np.loadtxt(path / "time.txt")
pos_true = np.loadtxt(path / "ground_truth_position.txt")
heading_true = np.loadtxt(path / "ground_truth_heading.txt")
accel = np.loadtxt(path / "accel.txt")
gyro = np.loadtxt(path / "gyro.txt")
mag = np.loadtxt(path / "magnetometer.txt")
config = json.load(open(path / "config.json"))
```

## Equation Reference

### IMU Strapdown Integration

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `omega_matrix()` | `core/sensors/strapdown.py` | Eq. (6.3) | Skew-symmetric matrix for quaternion kinematics |
| `quat_integrate()` | `core/sensors/strapdown.py` | Eq. (6.2-6.4) | Discrete quaternion integration |
| `vel_update()` | `core/sensors/strapdown.py` | Eq. (6.7) | Velocity update |
| `pos_update()` | `core/sensors/strapdown.py` | Eq. (6.10) | Position update |
| `strapdown_update()` | `core/sensors/strapdown.py` | Eq. (6.2-6.10) | Full strapdown loop |

### Wheel Odometry

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `wheel_speed_to_attitude_velocity()` | `core/sensors/wheel_odometry.py` | Eq. (6.11) | Lever arm compensation |
| `attitude_to_map_velocity()` | `core/sensors/wheel_odometry.py` | Eq. (6.14) | Frame transform |
| `odom_pos_update()` | `core/sensors/wheel_odometry.py` | Eq. (6.15) | Position update |

### Drift Correction Constraints

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `detect_zupt()` | `core/sensors/constraints.py` | Eq. (6.44) | Zero velocity detector |
| `ZuptMeasurementModel.h()` | `core/sensors/constraints.py` | Eq. (6.45) | ZUPT pseudo-measurement |
| `ZaruMeasurementModel.h()` | `core/sensors/constraints.py` | Eq. (6.60) | ZARU pseudo-measurement |
| `NhcMeasurementModel.h()` | `core/sensors/constraints.py` | Eq. (6.61) | NHC pseudo-measurement |

### Pedestrian Dead Reckoning (PDR)

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `total_accel_magnitude()` | `core/sensors/pdr.py` | Eq. (6.46) | Total acceleration magnitude |
| `step_length()` | `core/sensors/pdr.py` | Eq. (6.49) | Weinberg step length model |
| `pdr_step_update()` | `core/sensors/pdr.py` | Eq. (6.50) | 2D position update |

### Environmental Sensors

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `mag_tilt_compensate()` | `core/sensors/environment.py` | Eq. (6.52) | Tilt compensation |
| `mag_heading()` | `core/sensors/environment.py` | Eq. (6.51-6.53) | Heading from magnetometer |
| `pressure_to_altitude()` | `core/sensors/environment.py` | Eq. (6.54) | Barometric altitude |

### Allan Variance

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `allan_variance()` | `core/sensors/calibration.py` | Eq. (6.56-6.58) | IMU noise characterization |

## Expected Output

### IMU Strapdown Example

Running `python ch6_dead_reckoning/example_imu_strapdown.py` produces:

```
=== Chapter 6: IMU Strapdown Integration ===
Scenario: Figure-8 trajectory (100 seconds, 100 Hz IMU)

Configuration:
  IMU Grade:       Consumer (ARW=0.1 deg/sqrt(hr), BI=10 deg/hr)
  Frame:           ENU (East-North-Up)
  Trajectory:      267.9 m total distance

Results (IMU-only, no corrections):
  Final Position Error:  252.0 m (94.1% of distance)
  Max Velocity Error:    5.04 m/s
  Max Attitude Error (Yaw): 359.7 deg
  Drift Rate:            2.520 m/s (UNBOUNDED!)

Key Insight: IMU drift is UNBOUNDED without corrections!
```

**Visual Output:**

![IMU Strapdown Trajectory](figs/imu_strapdown_trajectory.svg)

*Shows the growing position error over time - IMU alone drifts unboundedly.*

### ZUPT Example (Zero-Velocity Update)

Running `python ch6_dead_reckoning/example_zupt.py` produces:

```
=== Chapter 6: Zero-Velocity Update (ZUPT) ===
Scenario: Walking with stops (60 seconds, 61.6m total distance)
Walking Pattern: 5s walk + 2s stop (repeated)
Stance time: 26.7% of trajectory

Results:
  IMU-only RMSE:     110.49 m (179% of distance)
  IMU + ZUPT RMSE:     9.22 m (15% of distance)
  
  Improvement:       91.7% reduction in RMSE

Method: ZUPT-EKF with proper Kalman filter measurement update
        (Eqs. 6.40-6.43 for Kalman filter + Eq. 6.45 for ZUPT)
        Detection: Windowed test statistic (Eq. 6.44)

Key Insight: ZUPT-EKF corrects velocity drift during stance phases!
             Essential for foot-mounted IMU navigation.
             Achieves >90% error reduction.
```

**Important Notes:**
- Uses proper EKF measurement update (not hard-coded v=0)
- Windowed ZUPT detector (Eq. 6.44) for robust detection
- State vector includes biases: [q, v, p, b_g, b_a] (16 states total)
- Covariance properly tracked and updated

### Comprehensive Comparison

Running `python ch6_dead_reckoning/example_comparison.py` generates:

```
===========================================================================
RESULTS - Performance Comparison
===========================================================================

Method                 RMSE [m]  Final [m] Median [m]    90% [m]   % Dist
---------------------------------------------------------------------------
IMU Only                 722.40    1613.46     408.01    1306.61   722.4%
IMU + ZUPT                20.78       0.51      19.32      32.64    20.8%
Wheel Odom                31.86      47.85      22.84      47.85    31.9%
PDR (Mag)                 20.03       2.91      19.27      31.06    20.0%

KEY INSIGHTS:
  1. IMU-only: UNBOUNDED drift (unusable without corrections)
  2. IMU+ZUPT: Dramatic improvement (97.1% RMSE reduction: 722.4m -> 20.8m)
  3. Wheel Odom: BOUNDED drift (~30% of distance)
  4. PDR: BOUNDED, heading-limited (~20% of distance)

Conclusion: Dead reckoning REQUIRES corrections or fusion!
           ZUPT-EKF provides >95% error reduction for foot-mounted IMU.
```

**Visual Outputs:**

![Comparison Trajectories](figs/comparison_trajectories.svg)

*Side-by-side comparison of all DR methods on the same trajectory.*

![Error CDF](figs/comparison_error_cdf.svg)

*Cumulative distribution of position errors for each method.*

### Allan Variance Example

Running `python ch6_dead_reckoning/example_allan_variance.py` produces:

```
=== Allan Variance Analysis ===

Gyroscope Noise Parameters:
  Angle Random Walk:   0.10 deg/sqrt(hr)
  Bias Instability:    10.0 deg/hr
  Rate Random Walk:    0.02 deg/hr/sqrt(hr)

Accelerometer Noise Parameters:
  Velocity Random Walk: 0.05 m/s/sqrt(hr)
  Bias Instability:     0.1 mg
```

**Visual Output:**

![Allan Variance](figs/allan_gyroscope_consumer.svg)

*Allan deviation plot showing noise sources at different averaging times.*

### Environmental Sensors Example

Running `python ch6_dead_reckoning/example_environment.py` produces:

```
=== Chapter 6: Environmental Sensors ===
Scenario: Multi-floor building walk (180 seconds, 3 floors)

Magnetometer Heading:
  RMSE:             103.2 deg
  Max error:        180.0 deg
  Note: Large errors during magnetic disturbances (30-50s, 100-120s)

Barometric Altitude:
  RMSE:             3.04 m
  Floor Accuracy:   44.4%

Key Insight: Environmental sensors provide absolute references!
             Magnetometer bounds heading drift (when clean).
             Barometer provides floor-level positioning.
             BUT sensitive to indoor disturbances (steel, weather).
```

**Notes:**
- High heading RMSE (103°) reflects severe magnetic disturbances in test scenario
- In clean environments, magnetometer RMSE is typically 5-10°
- Barometer provides ~3m accuracy (suitable for floor detection with multi-sensor fusion)

## Performance Summary

Based on actual outputs from `example_comparison.py` (100m trajectory, consumer-grade IMU):

| Method | RMSE | Final Error | Drift Type | Best For |
|--------|------|-------------|------------|----------|
| **IMU Only** | 722.4 m (722%) | 1613.5 m | Unbounded | Never use alone |
| **IMU + ZUPT** | 20.8 m (21%) | 0.5 m | Bounded | Foot-mounted systems |
| **Wheel Odometry** | 31.9 m (32%) | 47.9 m | Bounded | Vehicles |
| **PDR (Mag)** | 20.0 m (20%) | 2.9 m | Bounded | Smartphones |

**Key Findings:**
- ZUPT provides **97.1% RMSE reduction** over IMU-only
- Wheel odometry and PDR both achieve ~20-30% error (bounded drift)
- All corrections dramatically outperform pure IMU integration

## File Structure

```
ch6_dead_reckoning/
├── README.md                      # This file (student documentation)
├── example_imu_strapdown.py       # Pure IMU integration
├── example_zupt.py                # Zero-velocity updates
├── example_pdr.py                 # Pedestrian dead reckoning
├── example_wheel_odometry.py      # Vehicle odometry
├── example_environment.py         # Magnetometer, barometer
├── example_allan_variance.py      # IMU calibration
├── example_comparison.py          # All methods comparison
└── figs/                          # Generated figures (SVG/PDF)

core/sensors/
├── strapdown.py                   # IMU strapdown integration
├── wheel_odometry.py              # Wheel odometry
├── constraints.py                 # ZUPT, ZARU, NHC
├── pdr.py                         # Pedestrian DR
├── environment.py                 # Magnetometer, barometer
└── calibration.py                 # Allan variance
```

## References

- **Chapter 6**: Dead Reckoning and Sensor Fusion
  - Section 6.1: IMU error models and strapdown integration
  - Section 6.2: Wheel odometry
  - Section 6.3: Pedestrian dead reckoning
  - Section 6.4: Environmental sensors
  - Section 6.5: IMU calibration (Allan variance)
  - Section 6.6: Drift correction constraints

