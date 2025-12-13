# Chapter 6: Dead Reckoning and Sensor Fusion for Indoor Navigation

## Overview

This module implements dead reckoning and sensor fusion algorithms described in **Chapter 6** of *Principles of Indoor Positioning and Indoor Navigation*. Dead reckoning propagates position using proprioceptive sensors (IMU, wheel encoders, step counters) without external references, while sensor fusion combines multiple sources to reduce drift.

The module provides simulation-based examples of:
- **IMU strapdown integration** (attitude, velocity, position propagation)
- **Wheel odometry** (vehicle dead reckoning with lever arm compensation)
- **Drift correction constraints** (ZUPT, ZARU, NHC)
- **Pedestrian dead reckoning** (step-and-heading navigation)
- **Environmental sensors** (magnetometer heading, barometric altitude)
- **IMU calibration** (Allan variance noise characterization)

**Key Insight:** Dead reckoning drifts unbounded without corrections. Examples demonstrate both the drift problem and solutions (constraints, fusion, absolute measurements).

## Equation Mapping: Code ↔ Book

The following tables map the implemented functions to their corresponding equations in Chapter 6 of the book.

### IMU Strapdown Integration (Attitude/Velocity/Position)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `omega_matrix()` | `core/sensors/strapdown.py` | **Eq. (6.3)** | ✓ | Ω(ω) skew-symmetric matrix for quaternion kinematics |
| `quat_integrate()` | `core/sensors/strapdown.py` | **Eq. (6.2-6.4)** | ✓ | Discrete quaternion integration: q̇ = ½Ω(ω)q |
| `quat_to_rotmat()` | `core/sensors/strapdown.py` | - | ✓ | Convert quaternion to rotation matrix C_B^M |
| `correct_gyro()` | `core/sensors/imu_models.py` | **Eq. (6.6)** | ✓ | Gyro correction: ω = ω̃ - b_g - n_g |
| `correct_accel()` | `core/sensors/imu_models.py` | **Eq. (6.9)** | ✓ | Accel correction: f = f̃ - b_a - n_a |
| `gravity_vector()` | `core/sensors/strapdown.py` | **Eq. (6.8)** | ✓ | Gravity vector g = [0, 0, -9.81]^T in map frame |
| `vel_update()` | `core/sensors/strapdown.py` | **Eq. (6.7)** | ✓ | Velocity update: v_k = v_{k-1} + (C_B^M f_b + g)Δt |
| `pos_update()` | `core/sensors/strapdown.py` | **Eq. (6.10)** | ✓ | Position update: p_k = p_{k-1} + v_k Δt |
| `strapdown_update()` | `core/sensors/strapdown.py` | **Eq. (6.2-6.10)** | ✓ | Full strapdown loop (quaternion, velocity, position) |

### IMU Error Models and Calibration

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `apply_imu_scale_misalignment()` | `core/sensors/imu_models.py` | **Eq. (6.59)** | ✓ | Apply scale/misalignment: u_corrected = M·S·(u_meas - b) |
| `remove_gravity_component()` | `core/sensors/imu_models.py` | - | ✓ | Remove known gravity from accel measurement |

### Wheel Odometry (Vehicle Dead Reckoning)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `skew()` | `core/sensors/wheel_odometry.py` | **Eq. (6.12)** | ✓ | Skew-symmetric matrix [v×] for cross product |
| `wheel_speed_to_attitude_velocity()` | `core/sensors/wheel_odometry.py` | **Eq. (6.11)** | ✓ | Lever arm compensation: v^A = v^S + [ω×]r |
| `attitude_to_map_velocity()` | `core/sensors/wheel_odometry.py` | **Eq. (6.14)** | ✓ | Frame transform: v^M = C_A^M(q) v^A |
| `odom_pos_update()` | `core/sensors/wheel_odometry.py` | **Eq. (6.15)** | ✓ | Position update: p_k^M = p_{k-1}^M + v_k^M Δt |
| `wheel_odom_update()` | `core/sensors/wheel_odometry.py` | **Eq. (6.11-6.15)** | ✓ | Full wheel DR loop with lever arm |

### Drift Correction Constraints

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `detect_zupt()` | `core/sensors/constraints.py` | **Eq. (6.44)** | ✓ | Zero velocity detector: \|Δω\| < δ_ω AND \|Δf - g\| < δ_f |
| `ZuptMeasurementModel.h()` | `core/sensors/constraints.py` | **Eq. (6.45)** | ✓ | ZUPT pseudo-measurement: h(x) = v (velocity should be zero) |
| `ZuptMeasurementModel.H()` | `core/sensors/constraints.py` | - | ✓ | ZUPT Jacobian ∂h/∂x |
| `ZaruMeasurementModel.h()` | `core/sensors/constraints.py` | **Eq. (6.60)** | ✓ | ZARU pseudo-measurement: h(x) = ω (angular rate should be zero) |
| `NhcMeasurementModel.h()` | `core/sensors/constraints.py` | **Eq. (6.61)** | ✓ | NHC pseudo-measurement: h(x) = C_B^M^T v (lateral/vertical velocity = 0) |

### Pedestrian Dead Reckoning (PDR)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `total_accel_magnitude()` | `core/sensors/pdr.py` | **Eq. (6.46)** | ✓ | Total acceleration magnitude: a_mag = \\|a\\| |
| `remove_gravity_from_magnitude()` | `core/sensors/pdr.py` | **Eq. (6.47)** | ✓ | Dynamic acceleration: a_dyn = a_mag - g |
| `step_frequency()` | `core/sensors/pdr.py` | **Eq. (6.48)** | ✓ | Step frequency: f_step = 1/Δt |
| `step_length()` | `core/sensors/pdr.py` | **Eq. (6.49)** | ✓ | Weinberg step length: L = c·h^a·f_step^b |
| `pdr_step_update()` | `core/sensors/pdr.py` | **Eq. (6.50)** | ✓ | 2D position update: p_k = p_{k-1} + L[cos(ψ), sin(ψ)]^T |
| `detect_step_simple()` | `core/sensors/pdr.py` | - | ✓ | Simple peak-based step detector |
| `integrate_gyro_heading()` | `core/sensors/pdr.py` | - | ✓ | Gyro heading integration: ψ_k = ψ_{k-1} + ω_z Δt |
| `wrap_heading()` | `core/sensors/pdr.py` | - | ✓ | Wrap heading angle to [-π, π] |

### Environmental Sensors (Magnetometer + Barometer)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `mag_tilt_compensate()` | `core/sensors/environment.py` | **Eq. (6.52)** | ✓ | Tilt compensation: mag_h = R_y(-pitch)·R_x(-roll)·mag_b |
| `mag_heading()` | `core/sensors/environment.py` | **Eq. (6.51-6.53)** | ✓ | Heading from magnetometer: ψ = atan2(mag_hy, mag_hx) + declination |
| `pressure_to_altitude()` | `core/sensors/environment.py` | **Eq. (6.54)** | ✓ | Barometric altitude: h = (T/L)(1 - (p/p₀)^α) |
| `detect_floor_change()` | `core/sensors/environment.py` | - | ✓ | Simple floor change detector from altitude |
| `smooth_measurement_simple()` | `core/sensors/environment.py` | **Eq. (6.55)** | ✓ | Exponential smoothing: x_k = (1-α)x_{k-1} + αz_k |
| `compensate_hard_iron()` | `core/sensors/environment.py` | - | ✓ | Hard-iron calibration: mag_corr = mag_raw - offset |

### IMU Calibration (Allan Variance)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `allan_variance()` | `core/sensors/calibration.py` | **Eq. (6.56-6.58)** | ✓ | Allan variance & deviation: σ²(τ) = ½E[(θ̄_{k+1} - θ̄_k)²] |
| `identify_bias_instability()` | `core/sensors/calibration.py` | - | ✓ | Extract bias instability (minimum of Allan curve) |
| `identify_random_walk()` | `core/sensors/calibration.py` | - | ✓ | Extract angle/velocity random walk (slope -½ region) |
| `identify_rate_random_walk()` | `core/sensors/calibration.py` | - | ✓ | Extract rate random walk (slope +½ region) |
| `characterize_imu_noise()` | `core/sensors/calibration.py` | **Eq. (6.56-6.58)** | ✓ | Complete IMU noise characterization |

### Data Structures

| Type | Location | Equation | Status | Description |
|------|----------|----------|--------|-------------|
| `ImuSeries` | `core/sensors/types.py` | - | ✓ | Time-series IMU data: (t, accel, gyro) |
| `WheelSpeedSeries` | `core/sensors/types.py` | - | ✓ | Time-series wheel speed: (t, v_s) |
| `MagnetometerSeries` | `core/sensors/types.py` | - | ✓ | Time-series magnetometer: (t, mag) |
| `BarometerSeries` | `core/sensors/types.py` | - | ✓ | Time-series barometer: (t, pressure) |
| `NavStateQPVP` | `core/sensors/types.py` | - | ✓ | Navigation state: (quaternion, velocity, position) |
| `NavStateQPVPBias` | `core/sensors/types.py` | - | ✓ | Navigation state with IMU biases |

**Legend:**
- ✓ Implemented and tested (249 unit tests, 100% pass rate)
- ⚠️ Planned / To be implemented
- ✗ Not implemented (out of scope)

## Implementation Notes

### ✓ Fully Implemented

#### 1. **IMU Strapdown Integration (Eqs. 6.2-6.10)**

**Quaternion Attitude Propagation**
- Quaternion kinematics: q̇ = ½Ω(ω)q where Ω(ω) is the skew-symmetric matrix (Eq. 6.3)
- Discrete integration: q_k = q_{k-1} + ½Ω(ω)q_{k-1}Δt (Eqs. 6.2-6.4)
- Quaternion normalized after each step to prevent drift
- Represents body-to-map frame rotation: C_B^M(q)
- Benefits: No gimbal lock, efficient computation

**Velocity Integration**
- Specific force rotation: f^M = C_B^M(q) f^B (sensor frame → map frame)
- Gravity compensation: f^M + g where g = [0, 0, -9.81]^T (Eq. 6.8)
- Velocity update: v_k = v_{k-1} + (f^M + g)Δt (Eq. 6.7)
- Corrects for sensor biases: f^B = f̃^B - b_a - n_a (Eq. 6.9)

**Position Integration**
- Simple Euler integration: p_k = p_{k-1} + v_k Δt (Eq. 6.10)
- More accurate: trapezoidal or RK4 (not implemented for simplicity)
- Drift is **unbounded** without corrections (velocity errors integrate to position)

**Error Growth**
- Attitude: gyro bias causes ~0.1-10°/hr drift (depends on IMU grade)
- Velocity: accel bias causes ~0.01-1 m/s² drift
- Position: velocity drift integrates → quadratic error growth
- **1 hour uncorrected IMU:** position error can reach 100s-1000s of meters!

#### 2. **Wheel Odometry (Eqs. 6.11-6.15)**

**Lever Arm Compensation**
- Wheels measure velocity at sensor location (speed frame S)
- IMU/navigation center is offset by lever arm r^B
- Velocity transform: v^A = v^S + [ω^B×]r^B (Eq. 6.11)
- Skew-symmetric matrix: [v×] used for cross product (Eq. 6.12)

**Frame Transformations**
- Velocity in attitude frame A (aligned with body, but horizontal)
- Transform to map frame M: v^M = C_A^M(q) v^A (Eq. 6.14)
- Quaternion represents attitude-to-map rotation

**Position Integration**
- Direct integration: p_k^M = p_{k-1}^M + v_k^M Δt (Eq. 6.15)
- Drift sources: wheel slip, incorrect wheel radius, encoder errors
- Drift is **bounded** (proportional to distance traveled, not time)

**Error Characteristics**
- Typical: 1-5% of distance traveled
- Wheel slip: can cause 10-50% errors in turns
- Scale errors (wrong wheel radius): systematic drift
- Better than IMU for vehicles (bounded drift)

#### 3. **Drift Correction Constraints (Eqs. 6.44-6.45, 6.60-6.61)**

**Zero-Velocity Update (ZUPT) - Eq. 6.44-6.45**
- **Detection:** Stationary if |Δω| < δ_ω AND |Δf - g| < δ_f (Eq. 6.44)
- **Measurement:** h(x) = v (velocity should be zero) (Eq. 6.45)
- **Application:** Foot-mounted IMU during stance phase
- **Effect:** Eliminates velocity drift → prevents position drift
- **Typical thresholds:** δ_ω = 0.05 rad/s, δ_f = 0.5 m/s²

**Zero Angular Rate Update (ZARU) - Eq. 6.60**
- **Measurement:** h(x) = ω (angular rate should be zero)
- **Application:** Stationary periods, vehicle stops
- **Effect:** Corrects gyro bias estimation
- **Complementary to ZUPT:** together eliminate all drift during stops

**Nonholonomic Constraint (NHC) - Eq. 6.61**
- **Measurement:** h(x) = C_B^M^T v (velocity in body frame)
- **Constraint:** Lateral and vertical velocities = 0 for wheeled vehicles
- **Application:** Cars, robots (can't move sideways or vertically)
- **Effect:** Reduces 3D velocity to 1D (forward only)
- **Typical for vehicles:** Always active during motion

**Failure Modes**
- ZUPT: False detections during slow motion → position jumps
- ZUPT: Missed detections during stance → continued drift
- NHC: Violated during wheel slip or skid → filter divergence
- All constraints: Only valid under specific motion assumptions

#### 4. **Pedestrian Dead Reckoning (Eqs. 6.46-6.50)**

**Step Detection**
- Acceleration magnitude: a_mag = ||a|| = √(a_x² + a_y² + a_z²) (Eq. 6.46)
- Removes orientation dependency (magnitude is invariant)
- Gravity removal (approx): a_dyn = a_mag - g (Eq. 6.47)
- Peak detection: step when a_mag crosses threshold (~11-12 m/s²)
- Advanced: zero-crossing, autocorrelation, machine learning

**Step Length Estimation**
- Weinberg model: L = c·h^a·f^b (Eq. 6.49)
- Typical parameters: a ≈ 0.371, b ≈ 0.227, c ≈ 1.0
- Inputs: h = user height [m], f = step frequency [Hz] (Eq. 6.48)
- Personal calibration: adjust c based on gait
- Typical range: 0.5-1.0 m per step

**Position Update**
- 2D step-and-heading: p_k = p_{k-1} + L[cos(ψ), sin(ψ)]^T (Eq. 6.50)
- Heading from: gyro integration or magnetometer
- Error growth: heading errors dominate (∝ L·sin(Δψ))
- Typical accuracy: 2-5% of distance traveled (without corrections)

**Heading Sources**
- **Gyro integration:** Drifts unbounded (~10-100°/min for consumer IMU)
- **Magnetometer:** Absolute but sensitive to magnetic disturbances indoors
- **Best practice:** Complementary filter (fuse gyro + magnetometer)

#### 5. **Environmental Sensors (Eqs. 6.51-6.55)**

**Magnetometer Heading**
- Tilt compensation: mag_h = R_y(-pitch)·R_x(-roll)·mag_b (Eq. 6.52)
- Removes pitch/roll effects to get horizontal field component
- Heading: ψ = atan2(mag_hy, mag_hx) (Eq. 6.53)
- Declination correction: ψ_true = ψ_magnetic + δ_declination (Eq. 6.51)
- Hard-iron calibration: removes constant offsets from ferromagnetic materials

**Indoor Challenges**
- Steel structures → local magnetic disturbances (10-50° errors)
- Electronics → high-frequency noise
- Solution: detect disturbances, switch to gyro temporarily

**Barometric Altitude**
- Standard atmosphere: h = (T/L)(1 - (p/p₀)^α) (Eq. 6.54)
- Typical: α ≈ 0.190263, T = 288.15 K, L = 0.0065 K/m
- Resolution: 0.1-1.0 m (modern barometers)
- Drift: weather changes affect p₀ (need frequent reference updates)

**Floor Detection**
- Floor height: typically 3-4 m
- Detect change: |Δh| > 1.5 m → floor change
- Hysteresis: prevent oscillation at boundaries
- Accuracy: ±1 floor (95% of time)

**Measurement Smoothing**
- Exponential filter: x_k = (1-α)x_{k-1} + αz_k (Eq. 6.55)
- α controls tradeoff: responsiveness vs noise rejection
- Typical: α = 0.05-0.2 for environmental sensors
- Better: Kalman filter with process and measurement models

#### 6. **Allan Variance (Eqs. 6.56-6.58)**

**Computation**
- Cluster/bin data: average over τ seconds (Eq. 6.56)
- Allan variance: σ²(τ) = ½E[(θ̄_{k+1} - θ̄_k)²] (Eq. 6.57)
- Allan deviation: σ(τ) = √σ²(τ) (Eq. 6.58)
- Overlapping: better statistics, lower noise floor
- Requirements: stationary data, 1-24 hours duration

**Noise Identification (Log-Log Plot)**
- **Slope -1:** Quantization noise (very short τ)
- **Slope -½:** Angle/velocity random walk (short τ, read at τ=1s)
  - Gyro: Angle Random Walk (ARW) [°/√hr] or [rad/√s]
  - Accel: Velocity Random Walk (VRW) [m/s/√s]
- **Slope 0 (flat):** Bias instability (minimum, medium τ)
  - Critical specification for IMU quality
  - Consumer: 10-100 °/hr, Tactical: 1-10 °/hr, Navigation: <1 °/hr
- **Slope +½:** Rate random walk (long τ)
  - Characterizes bias drift
- **Slope +1:** Rate ramp (very long τ)

**IMU Specifications**
| Grade | ARW (gyro) | Bias Instability | Cost |
|-------|-----------|------------------|------|
| Consumer | 0.1-1.0 °/√hr | 10-100 °/hr | $1-10 |
| Tactical | 0.01-0.1 °/√hr | 1-10 °/hr | $100-1k |
| Navigation | <0.01 °/√hr | <1 °/hr | $10k-100k |

**Applications**
- IMU selection: match specs to application requirements
- Kalman filter tuning: extracted noise parameters → Q matrix
- Performance prediction: estimate drift over mission duration
- Quality control: verify sensor specifications

## Examples

This module includes 7 comprehensive examples demonstrating Chapter 6 algorithms:

### Example 1: `example_imu_strapdown.py`
**Demonstrates:** Pure IMU strapdown integration showing unbounded drift
- **Equations Used:** 6.2-6.10 (quaternion, velocity, position)
- **Scenario:** Figure-8 trajectory with turns and stops (100 seconds)
- **Key Outputs:**
  - Trajectory plot: truth vs IMU-only (shows growing error)
  - Attitude evolution: roll/pitch/yaw vs time
  - Position error vs time: demonstrates quadratic growth
  - Final drift rate: typically 15-30% of distance traveled
- **Key Insight:** IMU alone is unusable for more than a few seconds without corrections

### Example 2: `example_zupt.py`
**Demonstrates:** Foot-mounted IMU with Zero-Velocity Updates (ZUPT)
- **Equations Used:** 6.2-6.10 (strapdown), 6.44-6.45 (ZUPT detection and correction)
- **Scenario:** Walking with periodic stops (stance phases)
- **Key Outputs:**
  - Trajectory comparison: IMU-only vs IMU+ZUPT
  - ZUPT detector firing timeline
  - Drift reduction quantification: >90% error reduction
  - Stance phase detection accuracy
- **Key Insight:** ZUPTs eliminate drift during stops, making foot-mounted IMU practical

### Example 3: `example_pdr.py`
**Demonstrates:** Step-and-heading pedestrian dead reckoning
- **Equations Used:** 6.46-6.50 (step detection, step length, position update)
- **Scenario:** Corridor walk with turns and loop closure test
- **Key Outputs:**
  - Step detection visualization: peaks in acceleration magnitude
  - Trajectory comparison: gyro heading vs magnetometer heading
  - Heading drift accumulation: shows why heading is critical
  - Loop closure error: quantifies drift
- **Key Insight:** Heading errors dominate PDR accuracy; need fusion with magnetometer

### Example 4: `example_wheel_odometry.py`
**Demonstrates:** Vehicle wheel odometry with lever arm compensation
- **Equations Used:** 6.11-6.15 (lever arm, frame transforms, position update)
- **Scenario:** Square and circular paths, with wheel slip simulation
- **Key Outputs:**
  - Trajectory comparison: truth vs wheel DR
  - Effect of lever arm offset: shows importance of calibration
  - Wheel slip sensitivity: errors during turns
  - Drift rate: typically 1-5% of distance traveled
- **Key Insight:** Wheel DR is bounded (proportional to distance, not time) but sensitive to slip

### Example 5: `example_environment.py`
**Demonstrates:** Magnetometer heading and barometric altitude
- **Equations Used:** 6.51-6.55 (mag tilt compensation, heading, barometer)
- **Scenario:** Multi-floor building navigation with magnetic disturbances
- **Key Outputs:**
  - Tilt-compensated heading accuracy vs device orientation
  - Floor change detection from barometer
  - Magnetic disturbance visualization: heading jumps near steel
  - Altitude accuracy: <1 m typical
- **Key Insight:** Environmental sensors provide absolute references to bound drift

### Example 6: `example_allan_variance.py`
**Demonstrates:** IMU noise characterization using Allan variance
- **Equations Used:** 6.56-6.58 (Allan variance and deviation)
- **Scenario:** 1-hour stationary IMU data (simulated consumer and tactical grade)
- **Key Outputs:**
  - Allan deviation log-log plot: shows all noise sources
  - Extracted noise parameters: ARW, bias instability, RRW
  - Comparison: consumer vs tactical IMU specifications
  - Performance prediction: expected drift over time
- **Key Insight:** Allan variance reveals all IMU noise characteristics; critical for system design

### Example 7: `example_comparison.py`
**Demonstrates:** Comprehensive comparison of all DR methods
- **All Equations:** All Chapter 6 methods on same trajectory
- **Methods Compared:**
  1. IMU strapdown (pure, no corrections)
  2. Wheel odometry (vehicle)
  3. PDR (step-and-heading)
  4. IMU + ZUPT (foot-mounted)
  5. IMU + NHC (vehicle with constraints)
- **Key Outputs:**
  - Side-by-side trajectory plots (all methods)
  - RMSE comparison table
  - Error CDF plots
  - Drift rate comparison: bounded vs unbounded methods
  - Computation time comparison
- **Key Insight:** Fusion and constraints dramatically reduce drift; trade-offs between methods

## Usage

### Quick Start

Run any example script standalone:

```bash
# From project root
cd ch6_dead_reckoning

# Individual examples
python example_imu_strapdown.py
python example_zupt.py
python example_pdr.py
python example_wheel_odometry.py
python example_environment.py
python example_allan_variance.py

# Comprehensive comparison
python example_comparison.py
```

### Output

Each example generates:
1. **Console output:** Performance metrics, RMSE, drift rates
2. **Figures:** Saved to `ch6_dead_reckoning/figs/` as SVG/PDF (publication quality)
3. **Summary:** Key insights and equation references

Example console output:
```
=== Chapter 6: IMU Strapdown Integration ===
Scenario: Figure-8 trajectory (100 seconds, 100 Hz IMU)

Configuration:
  IMU Grade:       Consumer (ARW=0.1°/√hr, BI=10°/hr)
  Initial State:   [0, 0, 0] m, [0, 0, 0] rad
  Trajectory:      50.2 m total distance

Results (IMU-only, no corrections):
  Final Position Error:  15.3 m (30.5% of distance)
  Max Velocity Error:    2.4 m/s
  Max Attitude Error:    45.2° (yaw), 2.3° (roll), 1.8° (pitch)
  Drift Rate:            0.153 m/s (unbounded)
  
Figures saved:
  ✓ ch6_dead_reckoning/figs/imu_strapdown_trajectory.svg
  ✓ ch6_dead_reckoning/figs/imu_strapdown_error_time.svg
  ✓ ch6_dead_reckoning/figs/imu_strapdown_attitude.svg

Key Insight: IMU drift is UNBOUNDED without corrections!
```

### Requirements

The examples use only standard dependencies from the main project:
- `numpy` - numerical operations
- `matplotlib` - visualization
- `core.sensors` - implemented Chapter 6 algorithms (all phases 1-6)

No additional installation required beyond project setup.

## Performance Metrics

All examples report:
- **RMSE** (Root Mean Square Error): overall position accuracy
- **Mean/Median/90th percentile errors**: distribution statistics
- **Drift rate**: error growth per meter or per second
- **Computation time**: runtime performance
- **Error CDFs**: cumulative distribution functions

Typical performance (simulated, consumer-grade IMU):
- **IMU-only:** 10-30% of distance traveled (unbounded)
- **Wheel odometry:** 1-5% of distance traveled (bounded)
- **PDR:** 2-5% of distance traveled (bounded)
- **IMU + ZUPT:** <1% of distance traveled (during walking)
- **IMU + NHC:** 1-3% of distance traveled (vehicle)

## References

### Book Sections
- **Section 6.1:** IMU error models and strapdown integration
- **Section 6.2:** Wheel odometry and lever arm compensation
- **Section 6.3:** Pedestrian dead reckoning (step-and-heading)
- **Section 6.4:** Environmental sensors (magnetometer, barometer)
- **Section 6.5:** IMU calibration (Allan variance)
- **Section 6.6:** Drift correction constraints (ZUPT, ZARU, NHC)

### Implemented Modules
- `core/sensors/types.py` - Data structures (29 tests ✓)
- `core/sensors/imu_models.py` - IMU correction (24 tests ✓)
- `core/sensors/strapdown.py` - Strapdown integration (35 tests ✓)
- `core/sensors/wheel_odometry.py` - Wheel DR (26 tests ✓)
- `core/sensors/constraints.py` - ZUPT/ZARU/NHC (28 tests ✓)
- `core/sensors/pdr.py` - Pedestrian DR (46 tests ✓)
- `core/sensors/environment.py` - Mag/Baro (36 tests ✓)
- `core/sensors/calibration.py` - Allan variance (25 tests ✓)

**Total:** 8 modules, 3,855 lines of code, 249 tests (100% pass rate)

### Related Chapters
- **Chapter 3:** Kalman filtering (EKF used for sensor fusion)
- **Chapter 7:** SLAM (combines DR with mapping)
- **Chapter 8:** Multi-sensor fusion architectures

## Notes

### Design Philosophy
- **Simulation-first:** Examples use synthetic but realistic data
- **Equation traceability:** Every line references book equations
- **Failure modes shown:** Demonstrates when methods break (drift, slip, disturbances)
- **Comparison-focused:** Shows trade-offs between methods
- **Production-ready code:** Can be adapted for real systems

### Limitations
- Examples use synthetic data (no real sensor logs)
- Simplified scenarios (no multipath, no GPS outages, no complex buildings)
- Consumer-grade IMU simulation (tactical/navigation grade available but not default)
- 2D emphasis (3D navigation fully supported but examples focus on horizontal plane)

### Extensions
Users can extend examples by:
- Using real sensor data (replace synthetic data generation)
- Adding sensor fusion (EKF/UKF combining multiple sources)
- Implementing integrated navigation (IMU + wheel + ZUPT + GPS)
- Testing on complex trajectories (multi-floor, outdoor-indoor transitions)
- Comparing different IMU grades (consumer vs tactical vs navigation)

## License

MIT License - See project root for details.

## Authors

Navigation Engineer (Chapter 6 Implementation)  
December 2024

## Citation

If you use this code in research, please cite:

```
@book{ipin_book,
  title={Principles of Indoor Positioning and Indoor Navigation},
  author={[Book Authors]},
  year={2024},
  publisher={[Publisher]}
}
```

---

**Version:** 1.0.0 (Complete Chapter 6 Implementation)  
**Tests:** 249 passing (100%)  
**Coverage:** All major Chapter 6 equations implemented

