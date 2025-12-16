# Ch6 Environmental Sensors Dataset: Multi-Floor Building Walk

## Overview

This dataset demonstrates **magnetometer and barometer sensors for indoor navigation**, specifically for absolute heading estimation and floor detection. It showcases both the benefits (no drift!) and challenges (indoor disturbances) of environmental sensors.

**Key Learning Objective**: Understand that environmental sensors provide absolute measurements that don't drift, but are susceptible to indoor disturbances (magnetic anomalies, weather pressure changes).

## Dataset Purpose

### Learning Goals
1. **Absolute Heading**: Magnetometer provides drift-free heading (unlike gyro integration)
2. **Tilt Compensation**: Must compensate for device tilt to get accurate heading (Eq. 6.52)
3. **Indoor Challenges**: Steel structures and electronics corrupt magnetic field
4. **Floor Detection**: Barometric altitude enables floor identification
5. **Weather Effects**: Atmospheric pressure changes affect altitude measurement

### Implemented Equations
- **Eq. (6.51)**: Magnetometer heading definition
  ```
  ψ = atan2(m_y, m_x) + declination
  ```

- **Eq. (6.52)**: Tilt compensation for magnetometer
  ```
  m_h = R_y(-pitch) * R_x(-roll) * m_b
  Projects body-frame measurement to horizontal plane
  ```

- **Eq. (6.53)**: Heading computation from compensated field
  ```
  ψ = atan2(m_hy, m_hx)
  ```

- **Eq. (6.54)**: Barometric altitude from pressure
  ```
  h = (T_0 / L) * [1 - (p / p_0)^α]
  where α = (g*M)/(R*L), international barometric formula
  ```

- **Eq. (6.55)**: Exponential smoothing filter
  ```
  x_k = (1 - α) * x_{k-1} + α * z_k
  Simple low-pass filter for noisy measurements
  ```

## Dataset Variants

| Variant | Directory | Mag Quality | Floor Detection | Description |
|---------|-----------|-------------|-----------------|-------------|
| **Baseline** | `ch6_env_sensors_heading_altitude` | Clean | 50-70% | Clean sensors, minimal disturbances |
| **Noisy** | `ch6_env_sensors_noisy` | Moderate | 40-60% | Higher sensor noise |
| **Disturbances** | `ch6_env_sensors_disturbances` | Poor (spikes) | 50-70% | Indoor magnetic anomalies |
| **Poor** | `ch6_env_sensors_poor` | Very poor | 30-50% | Poor sensors + disturbances |

**Generate variants**:
```bash
python scripts/generate_ch6_env_sensors_dataset.py --preset baseline
python scripts/generate_ch6_env_sensors_dataset.py --preset noisy
python scripts/generate_ch6_env_sensors_dataset.py --preset disturbances
python scripts/generate_ch6_env_sensors_dataset.py --preset poor
```

## Files

### Ground Truth
- `time.txt`: Timestamps [N×1] (seconds)
- `ground_truth_position.txt`: True 3D positions [N×3] (x, y, z in meters)
- `ground_truth_attitude.txt`: True attitude [N×3] (roll, pitch, yaw in radians)
- `ground_truth_floor.txt`: True floor number [N×1] (0=ground, 1=first, 2=second)

### Measurements
- `magnetometer.txt`: Magnetometer measurements [N×3] (mx, my, mz in microTesla)
- `barometer.txt`: Barometric pressure measurements [N×1] (pressure in Pascals)

### Reference Data
- `magnetometer_clean.txt`: Clean magnetometer (no noise) [N×3]
- `barometer_clean.txt`: Clean pressure (no noise) [N×1]

### Configuration
- `config.json`: All dataset parameters and performance metrics

## Loading Data

### Python
```python
import numpy as np
from pathlib import Path

# Load dataset
data_dir = Path("data/sim/ch6_env_sensors_heading_altitude")

t = np.loadtxt(data_dir / "time.txt")
pos_true = np.loadtxt(data_dir / "ground_truth_position.txt")
att_true = np.loadtxt(data_dir / "ground_truth_attitude.txt")
floor_true = np.loadtxt(data_dir / "ground_truth_floor.txt", dtype=int)
mag_meas = np.loadtxt(data_dir / "magnetometer.txt")
pressure_meas = np.loadtxt(data_dir / "barometer.txt")

print(f"Loaded {len(t)} samples over {t[-1]:.0f} seconds")
print(f"Floors: {np.unique(floor_true)}")
print(f"Max altitude: {pos_true[:, 2].max():.1f} m")
```

### MATLAB
```matlab
% Load dataset
data_dir = 'data/sim/ch6_env_sensors_heading_altitude/';

t = load([data_dir 'time.txt']);
pos_true = load([data_dir 'ground_truth_position.txt']);
att_true = load([data_dir 'ground_truth_attitude.txt']);
floor_true = load([data_dir 'ground_truth_floor.txt']);
mag_meas = load([data_dir 'magnetometer.txt']);
pressure_meas = load([data_dir 'barometer.txt']);

fprintf('Loaded %d samples, %d floors\n', length(t), length(unique(floor_true)));
```

## Configuration Parameters

### Trajectory Configuration
```json
{
  "trajectory": {
    "type": "building_walk",
    "duration_s": 180.0,
    "num_floors": 3,
    "floor_height_m": 3.5,
    "max_altitude_m": 7.0
  }
}
```

**Key Parameters**:
- **num_floors**: Number of floors visited (3: ground, first, second)
- **floor_height**: Height of each floor (3.5m, typical office building)
- **duration**: Total walk time (180s = 3 minutes)

### Sensor Configuration (Baseline)
```json
{
  "sensors": {
    "magnetometer": {
      "noise_std_uT": 1.5,
      "disturbances_enabled": false,
      "num_disturbance_events": 0
    },
    "barometer": {
      "noise_std_Pa": 8.0,
      "weather_drift_Pa": 30.0
    }
  }
}
```

**Key Parameters**:
- **mag_noise**: Magnetometer noise (1.5 microTesla std dev)
- **disturbances**: Indoor magnetic anomalies (steel, electronics)
- **pressure_noise**: Barometer noise (8 Pa ≈ 0.7m altitude uncertainty)
- **weather_drift**: Slow pressure changes (30 Pa ≈ 2.5m altitude drift)

## Quick Start Example

### Magnetometer Heading with Tilt Compensation
```python
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from core.sensors import mag_heading

# Load dataset
data_dir = Path("data/sim/ch6_env_sensors_heading_altitude")
t = np.loadtxt(data_dir / "time.txt")
att_true = np.loadtxt(data_dir / "ground_truth_attitude.txt")
mag_meas = np.loadtxt(data_dir / "magnetometer.txt")

# Extract roll, pitch, true yaw
roll = att_true[:, 0]
pitch = att_true[:, 1]
yaw_true = att_true[:, 2]

# Compute heading from magnetometer with tilt compensation
N = len(t)
heading_est = np.zeros(N)

for k in range(N):
    # Eq. (6.51-6.53): Tilt compensation + heading
    heading_est[k] = mag_heading(mag_meas[k], roll[k], pitch[k], declination=0.0)

# Compute heading error
heading_error = np.abs(heading_est - yaw_true)
heading_error = np.minimum(heading_error, 2 * np.pi - heading_error)  # Wrap to [-pi, pi]
heading_error_deg = np.rad2deg(heading_error)

print(f"Mean heading error: {np.mean(heading_error_deg):.2f} deg")
print(f"Max heading error: {np.max(heading_error_deg):.2f} deg")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(t, np.rad2deg(yaw_true), 'g-', linewidth=2, label='True Heading')
ax1.plot(t, np.rad2deg(heading_est), 'b--', linewidth=2, label='Magnetometer')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Heading (deg)')
ax1.set_title('Magnetometer Heading: Absolute, No Drift!')
ax1.legend()
ax1.grid(True)

ax2.plot(t, heading_error_deg, 'r-', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Heading Error (deg)')
ax2.set_title('Magnetometer Error: Bounded (No Drift)')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

**Expected Result**: Heading error is bounded and doesn't drift over time!

### Barometric Altitude and Floor Detection
```python
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from core.sensors import pressure_to_altitude, detect_floor_change, smooth_measurement_simple

# Load dataset
data_dir = Path("data/sim/ch6_env_sensors_heading_altitude")
t = np.loadtxt(data_dir / "time.txt")
pos_true = np.loadtxt(data_dir / "ground_truth_position.txt")
floor_true = np.loadtxt(data_dir / "ground_truth_floor.txt", dtype=int)
pressure_meas = np.loadtxt(data_dir / "barometer.txt")

# Compute altitude from pressure (Eq. 6.54)
p0 = 101325.0  # Sea level pressure
altitude_est = np.array([pressure_to_altitude(p, p0) for p in pressure_meas])

# Smooth altitude (Eq. 6.55)
altitude_smooth = np.zeros_like(altitude_est)
altitude_smooth[0] = altitude_est[0]
for k in range(1, len(altitude_est)):
    altitude_smooth[k] = smooth_measurement_simple(altitude_smooth[k-1], altitude_est[k], alpha=0.1)

# Detect floor changes
floor_height = 3.5
floor_detected = np.zeros_like(floor_true)
current_floor = 0

for k in range(1, len(t)):
    delta_floor = detect_floor_change(altitude_smooth[k-1], altitude_smooth[k], 
                                       floor_height=floor_height, threshold=1.5)
    current_floor += delta_floor
    floor_detected[k] = max(0, min(2, current_floor))  # Clamp to [0, 2]

# Compute accuracy
altitude_true = pos_true[:, 2]
altitude_error = np.abs(altitude_smooth - altitude_true)
floor_accuracy = np.mean(floor_detected == floor_true) * 100

print(f"Mean altitude error: {np.mean(altitude_error):.2f} m")
print(f"Max altitude error: {np.max(altitude_error):.2f} m")
print(f"Floor detection accuracy: {floor_accuracy:.1f}%")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(t, altitude_true, 'g-', linewidth=2, label='True Altitude')
ax1.plot(t, altitude_smooth, 'b--', linewidth=2, label='Barometer (Smoothed)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Altitude (m)')
ax1.set_title(f'Barometric Altitude: {np.mean(altitude_error):.2f}m mean error')
ax1.legend()
ax1.grid(True)

ax2.plot(t, floor_true, 'g-', linewidth=2, label='True Floor')
ax2.plot(t, floor_detected, 'b--', linewidth=2, label='Detected Floor')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Floor Number')
ax2.set_title(f'Floor Detection: {floor_accuracy:.0f}% accuracy')
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['Ground', 'First', 'Second'])
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

**Expected Result**: ~1-2m altitude error, 50-70% floor detection accuracy

## Visualization

### Plot with Built-in Tool
```bash
# Single dataset
python tools/plot_fusion_dataset.py data/sim/ch6_env_sensors_heading_altitude --type env_sensors

# Compare clean vs. disturbances
python tools/compare_fusion_variants.py \
    data/sim/ch6_env_sensors_heading_altitude \
    data/sim/ch6_env_sensors_disturbances \
    --labels "Clean" "With Disturbances"
```

### Custom Plot: 3D Trajectory with Altitude
```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
data_dir = Path("data/sim/ch6_env_sensors_heading_altitude")
pos_true = np.loadtxt(data_dir / "ground_truth_position.txt")
floor_true = np.loadtxt(data_dir / "ground_truth_floor.txt", dtype=int)

# Plot 3D trajectory
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Color by floor
colors = ['green', 'blue', 'red']
for floor_num in range(3):
    mask = (floor_true == floor_num)
    ax.plot(pos_true[mask, 0], pos_true[mask, 1], pos_true[mask, 2], 
            c=colors[floor_num], linewidth=2, label=f'Floor {floor_num}')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Altitude (m)')
ax.set_title('3D Building Walk Trajectory')
ax.legend()
plt.show()
```

## Parameter Effects

### Effect of Magnetometer Noise

| Mag Noise (μT) | Mean Heading Error (deg) | Notes |
|----------------|--------------------------|-------|
| 0.5 (excellent) | 1-2 | High-quality MEMS, outdoor |
| 1.5 (good) | 2-4 | Baseline quality |
| 4.0 (fair) | 5-8 | Consumer smartphone |
| 6.0 (poor) | 10-15 | Low-cost or uncalibrated |

**Generate sweep**:
```bash
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_mag_05 --mag-noise 0.5
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_mag_15 --mag-noise 1.5
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_mag_40 --mag-noise 4.0
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_mag_60 --mag-noise 6.0
```

### Effect of Magnetic Disturbances

| Disturbance Type | Heading Error (deg) | Altitude Error (m) | Notes |
|------------------|---------------------|--------------------| ------|
| None (outdoor) | 2-4 | 0.3-0.5 | Clean environment |
| Mild (corridor) | 5-10 | 0.3-0.5 | Away from steel |
| Moderate (office) | 10-20 | 0.3-0.5 | Near desks, chairs |
| Severe (elevator) | 30-90 | 0.3-0.5 | Steel structure, motors |

**Generate comparison**:
```bash
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_no_dist --preset baseline
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_with_dist --add-disturbances
```

**Learning Point**: Magnetometers work well outdoors but fail near steel structures!

### Effect of Weather Pressure Changes

| Weather Drift (Pa) | Altitude Drift (m) | Floor Detection Impact | Notes |
|--------------------|--------------------| -----------------------|-------|
| 0 (none) | 0 | None | Ideal conditions |
| 30 (mild) | 2-3 | Moderate | Typical daily variation |
| 80 (moderate) | 6-8 | Significant | Weather front passing |
| 120 (severe) | 10-12 | Severe | Storm conditions |

**Generate sweep**:
```bash
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_weather_00 --weather-drift 0
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_weather_30 --weather-drift 30
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_weather_80 --weather-drift 80
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_weather_120 --weather-drift 120
```

**Learning Point**: Weather changes limit long-term barometric altitude accuracy!

## Experiments

### Experiment 1: Tilt Compensation Necessity

**Objective**: Demonstrate that tilt compensation (Eq. 6.52) is ESSENTIAL for accurate magnetometer heading.

**Procedure**:
1. Generate baseline dataset
2. Compute heading WITH tilt compensation
3. Compute heading WITHOUT tilt compensation (assume level)
4. Compare heading errors

**Expected Results**:
- With tilt compensation: 2-4° error
- Without tilt compensation: 10-30° error (device tilts during walking!)

**Code**:
```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from core.sensors import mag_heading

# Load dataset
data_dir = Path("data/sim/ch6_env_sensors_heading_altitude")
t = np.loadtxt(data_dir / "time.txt")
att_true = np.loadtxt(data_dir / "ground_truth_attitude.txt")
mag_meas = np.loadtxt(data_dir / "magnetometer.txt")

roll = att_true[:, 0]
pitch = att_true[:, 1]
yaw_true = att_true[:, 2]

# WITH tilt compensation
heading_with = np.array([mag_heading(mag_meas[k], roll[k], pitch[k]) for k in range(len(t))])

# WITHOUT tilt compensation (assume level)
heading_without = np.array([mag_heading(mag_meas[k], 0.0, 0.0) for k in range(len(t))])

# Compute errors
error_with = np.abs(heading_with - yaw_true)
error_without = np.abs(heading_without - yaw_true)
error_with = np.minimum(error_with, 2*np.pi - error_with)
error_without = np.minimum(error_without, 2*np.pi - error_without)

print(f"WITH tilt compensation: {np.rad2deg(np.mean(error_with)):.2f} deg error")
print(f"WITHOUT tilt compensation: {np.rad2deg(np.mean(error_without)):.2f} deg error")
print(f"Degradation: {np.mean(error_without) / np.mean(error_with):.1f}x worse!")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(t, np.rad2deg(error_with), 'g-', linewidth=2, label='With Tilt Compensation')
plt.plot(t, np.rad2deg(error_without), 'r-', linewidth=2, label='Without Compensation (WRONG)')
plt.xlabel('Time (s)')
plt.ylabel('Heading Error (deg)')
plt.title('Tilt Compensation: ESSENTIAL for Magnetometer Heading!')
plt.legend()
plt.grid(True)
plt.show()
```

**Learning Point**: Tilt compensation (Eq. 6.52) is NON-NEGOTIABLE!

### Experiment 2: Floor Detection Robustness

**Objective**: Test barometric floor detection under different noise and weather conditions.

**Procedure**:
1. Generate datasets with varying pressure noise and weather drift
2. Run floor detection algorithm
3. Compute floor detection accuracy

**Expected Results**:
- Clean (8 Pa noise, 30 Pa drift): 70-80% accuracy
- Noisy (20 Pa noise, 80 Pa drift): 50-60% accuracy
- Poor (30 Pa noise, 120 Pa drift): 30-40% accuracy

**Code**:
```bash
# Generate datasets
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_clean --preset baseline
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_noisy --preset noisy
python scripts/generate_ch6_env_sensors_dataset.py --output data/sim/env_poor --preset poor

# Check floor detection accuracy in config.json
# Look for "floor_detection_accuracy_percent"
```

**Learning Point**: Floor detection works but requires calibration and is sensitive to weather!

### Experiment 3: Magnetometer vs. Gyro for Heading

**Objective**: Compare magnetometer (absolute) vs. gyro (relative) for heading estimation.

**Procedure**:
1. Use Ch6 PDR dataset (has both magnetometer and gyro)
2. Integrate gyro for heading (drifts!)
3. Use magnetometer for heading (absolute, noisy)
4. Compare heading errors over time

**Expected Results**:
- Gyro: Unbounded drift (~10-50° over 2 minutes)
- Magnetometer: Bounded noise (~2-5° constant)
- Magnetometer is 10× better long-term!

**Code**: See `data/sim/ch6_pdr_corridor_walk/README.md` Experiment 2

**Learning Point**: Absolute heading (magnetometer) is ESSENTIAL for long-term navigation!

## Performance Metrics (Baseline)

| Metric | Magnetometer | Barometer | Notes |
|--------|--------------|-----------|-------|
| **Mean Error** | 2-4 deg | 1.5m | Bounded, no drift |
| **Max Error** | 8-12 deg | 2.8m | Occasional spikes |
| **Floor Detection** | N/A | 50-70% | 3-floor building |
| **Drift Over Time** | None! | ~2m/hour | Weather-dependent |
| **Sample Rate** | 10 Hz | 10 Hz | 0.1s time steps |
| **Duration** | 180s | 180s | 3-minute walk |
| **Altitude Range** | N/A | 0-7m | 3 floors @ 3.5m |

**Key Insight**: Environmental sensors provide absolute measurements (no drift!) but are susceptible to indoor disturbances.

## Book Connection

### Chapter 6, Section 6.4: Environmental Sensors

This dataset directly implements the environmental sensor algorithms from Section 6.4:

1. **Magnetometer Tilt Compensation (Eq. 6.52)**
   - Projects body-frame measurement to horizontal plane
   - Uses roll and pitch from IMU
   - Critical for accurate heading (10× error without it!)

2. **Magnetometer Heading (Eqs. 6.51, 6.53)**
   - Computes heading from horizontal magnetic field
   - Provides absolute reference (no drift!)
   - Applies magnetic declination correction

3. **Barometric Altitude (Eq. 6.54)**
   - International barometric formula
   - Converts pressure to altitude
   - Enables floor detection in buildings

4. **Smoothing Filter (Eq. 6.55)**
   - Exponential smoothing for noisy measurements
   - Simple low-pass filter: `x_k = (1-α)x_{k-1} + αz_k`
   - Trade-off between noise reduction and lag

**Key Insight from Chapter 6**: Environmental sensors complement proprioceptive sensors (IMU, wheel) by providing absolute references that bound drift. However, they suffer from indoor disturbances (magnetic anomalies, weather pressure changes) and must be fused carefully.

## Common Issues & Solutions

### Issue 1: Heading Jumps Wildly (±180°)

**Symptoms**: Magnetometer heading jumps between opposite directions

**Likely Cause**: Sign ambiguity or coordinate frame mismatch

**Solution**: Check magnetometer calibration and coordinate frame:
```python
# Ensure magnetometer is normalized
mag_norm = np.linalg.norm(mag_meas[k])
if mag_norm > 1e-9:
    mag_meas[k] /= mag_norm

# Check sign conventions (depends on sensor frame)
# Try negating one component if heading is 180° off
```

### Issue 2: Floor Detection Always Returns Same Floor

**Symptoms**: Floor number doesn't change despite altitude changes

**Likely Cause**: Threshold too large or altitude not changing enough

**Solution**: Adjust floor detection threshold:
```python
# Reduce threshold for more sensitive detection
delta_floor = detect_floor_change(alt_prev, alt_curr, floor_height=3.5, threshold=1.0)  # was 1.5

# Or check if altitude is actually changing
print(f"Altitude range: {alt_smooth.min():.2f} to {alt_smooth.max():.2f} m")
print(f"Expected range: 0 to {num_floors * floor_height:.2f} m")
```

### Issue 3: Heading Error Much Larger Than Expected

**Symptoms**: Mean heading error > 10° even with clean sensors

**Likely Cause**: Tilt compensation not applied or incorrect roll/pitch

**Solution**: Verify tilt compensation:
```python
# Check roll/pitch values
print(f"Roll range: {np.rad2deg(roll.min()):.1f} to {np.rad2deg(roll.max()):.1f} deg")
print(f"Pitch range: {np.rad2deg(pitch.min()):.1f} to {np.rad2deg(pitch.max()):.1f} deg")

# Ensure tilt compensation is called
heading = mag_heading(mag, roll, pitch)  # Correct
# NOT: heading = mag_heading(mag, 0, 0)  # WRONG: no compensation
```

## Troubleshooting

### Error: Altitude becomes negative or > 100m

**Cause**: Incorrect sea-level pressure reference or pressure measurement corruption

**Fix**: Calibrate reference pressure:
```python
# Use ground-level pressure as reference
p0 = pressure_meas[0]  # First measurement at ground level
altitude = pressure_to_altitude(pressure_meas, p0)
```

### Warning: Magnetometer magnitude changes significantly

**Cause**: Magnetic disturbances or sensor moving through inhomogeneous field

**Fix**: Check magnitude and apply rejection:
```python
MAG_NOMINAL = 50.0  # microTesla (typical Earth field magnitude)
mag_magnitude = np.linalg.norm(mag_meas[k])

if abs(mag_magnitude - MAG_NOMINAL) > 20.0:
    print(f"Warning: Magnetic disturbance detected at t={t[k]:.1f}s")
    # Use previous heading or gyro-integrated heading instead
```

### Error: Floor detection lags behind true floor changes

**Cause**: Smoothing filter too aggressive (alpha too small)

**Fix**: Increase filter responsiveness:
```python
# Increase alpha for faster response (but more noise)
altitude_smooth[k] = smooth_measurement_simple(altitude_smooth[k-1], altitude_raw[k], alpha=0.2)  # was 0.1
```

## Next Steps

After understanding environmental sensors:

1. **Chapter 8**: Sensor fusion (complementary filter for gyro + magnetometer heading)
2. **Magnetic Field Mapping**: Build magnetic field maps for indoor positioning
3. **Barometric Relative Positioning**: Use pressure differences between devices
4. **Multi-Sensor Heading**: Fuse gyro, magnetometer, and visual heading
5. **Advanced Floor Detection**: Machine learning classifiers for robust detection

## Citation

If you use this dataset in your research, please cite:

```bibtex
@book{IPIN2024,
  title={Principles of Indoor Positioning and Indoor Navigation},
  author={[Authors]},
  year={2024},
  chapter={6},
  section={6.4},
  note={Environmental Sensors for Indoor Navigation}
}
```

## License

This dataset is part of the IPIN Book Examples repository. See repository LICENSE for details.

---

**Dataset Version**: 1.0  
**Last Updated**: December 2024  
**Contact**: See repository README for contact information

