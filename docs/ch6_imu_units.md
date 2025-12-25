# Chapter 6: IMU Unit Conventions and Conversions

**Author:** Li-Ta Hsu  
**Date:** December 2025

## Overview

This document explains the explicit unit handling system for IMU specifications in Chapter 6. All unit conversions are handled through the `core.sensors.units` module and the `IMUNoiseParams` dataclass to eliminate ambiguous "deg/hr but used as deg/s" bugs.

## The Problem

IMU datasheets specify parameters in various units:
- **Gyroscope bias**: deg/hr (degrees per hour)
- **Gyroscope ARW**: deg/√hr (degrees per square root hour)
- **Accelerometer bias**: mg (milligravity)
- **Accelerometer VRW**: m/s/√hr (meters per second per square root hour)

However, simulation and navigation algorithms require SI units:
- Gyroscope bias: rad/s (radians per second)
- Gyroscope ARW: rad/√s (radians per square root second)
- Accelerometer bias: m/s² (meters per second squared)
- Accelerometer VRW: m/s/√s (meters per second per square root second)

**Common bug**: Using `np.deg2rad(10.0)` converts 10 degrees to radians (~0.1745 rad), but if the 10 represents "10 deg/hr", the result should be `10 deg/hr = 0.002778 deg/s = 4.848e-05 rad/s`, NOT 0.1745 rad/s!

## The Solution

### 1. Explicit Unit Conversion Functions

All conversions are performed through named functions in `core.sensors.units`:

```python
from core.sensors import units

# Gyroscope conversions
bias_rad_s = units.deg_per_hour_to_rad_per_sec(10.0)  # 10 deg/hr → rad/s
arw_rad_sqrt_s = units.deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.1)  # 0.1 deg/√hr → rad/√s

# Accelerometer conversions
bias_mps2 = units.mg_to_mps2(10.0)  # 10 mg → m/s²
vrw_mps_sqrt_s = units.mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.01)  # 0.01 m/s/√hr → m/s/√s

# Reverse conversions (for display)
bias_deg_hr = units.rad_per_sec_to_deg_per_hour(bias_rad_s)
bias_deg_s = units.rad_per_sec_to_deg_per_sec(bias_rad_s)
```

### 2. Typed IMU Noise Parameters

The `IMUNoiseParams` dataclass explicitly names units in every field:

```python
from core.sensors import IMUNoiseParams, units

# Create consumer-grade IMU parameters
params = IMUNoiseParams(
    gyro_bias_rad_s=units.deg_per_hour_to_rad_per_sec(10.0),  # 10 deg/hr
    gyro_arw_rad_sqrt_s=units.deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.1),  # 0.1 deg/√hr
    gyro_rrw_rad_s_sqrt_s=0.0,  # Not specified
    accel_bias_mps2=units.mg_to_mps2(10.0),  # 10 mg
    accel_vrw_mps_sqrt_s=units.mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.01),  # 0.01 m/s/√hr
    grade='consumer'
)

# Or use factory methods
params = IMUNoiseParams.consumer_grade()
params = IMUNoiseParams.tactical_grade()
params = IMUNoiseParams.navigation_grade()
```

### 3. Human-Readable Formatting

Use formatter functions for diagnostics:

```python
from core.sensors import units

bias_rad_s = units.deg_per_hour_to_rad_per_sec(10.0)

# Format for display
print(units.format_gyro_bias(bias_rad_s))
# Output: "10.00 deg/hr (0.0028 deg/s)"

print(units.format_accel_bias(units.mg_to_mps2(10.0)))
# Output: "10.00 mg (0.0981 m/s²)"

# Or use IMUNoiseParams.format_specs()
params = IMUNoiseParams.consumer_grade()
print(params.format_specs())
# Output:
# IMU Specifications (consumer grade):
#   Gyro Bias:  10.00 deg/hr (0.0028 deg/s)
#   Gyro ARW:   0.10 deg/√hr
#   Accel Bias: 10.00 mg (0.0981 m/s²)
#   Accel VRW:  0.0100 m/s/√hr
```

## Typical IMU Grades

### Consumer Grade
- **Gyro bias**: 10-100 deg/hr (smartphone, tablet IMUs)
- **Gyro ARW**: 0.1-1.0 deg/√hr
- **Accel bias**: 1-100 mg
- **Accel VRW**: 0.01-0.1 m/s/√hr
- **Example**: MPU-6050, BMI160

### Tactical Grade
- **Gyro bias**: 0.1-10 deg/hr (mid-range MEMS, fiber optic gyros)
- **Gyro ARW**: 0.01-0.1 deg/√hr
- **Accel bias**: 0.1-1 mg
- **Accel VRW**: 0.001-0.01 m/s/√hr
- **Example**: Honeywell HG1700, Northrop Grumman LN-200

### Navigation Grade
- **Gyro bias**: 0.001-0.1 deg/hr (ring laser gyros, high-end FOGs)
- **Gyro ARW**: 0.001-0.01 deg/√hr
- **Accel bias**: 0.01-0.1 mg
- **Accel VRW**: 0.0001-0.001 m/s/√hr
- **Example**: Honeywell GG1320, Litton LN-100

## Conversion Formulas

### Time-Based Conversions

**Gyroscope bias** (deg/hr → rad/s):
```
1 deg/hr = (1 deg/hr) × (π/180 rad/deg) × (1 hr/3600 s)
         = π/(180×3600) rad/s
         ≈ 4.848e-06 rad/s

10 deg/hr ≈ 4.848e-05 rad/s ≈ 0.002778 deg/s
```

**Angular Random Walk** (deg/√hr → rad/√s):
```
1 deg/√hr = (1 deg/√hr) × (π/180 rad/deg) × (1 √hr/√3600 √s)
          = π/(180×60) rad/√s
          ≈ 2.909e-04 rad/√s
```

### Accelerometer Conversions

**Milligravity to m/s²**:
```
1 mg = 0.001 × 9.80665 m/s² = 0.00980665 m/s²
10 mg = 0.0980665 m/s²
```

**Standard gravity**: g = 9.80665 m/s² (ISO 80000-3:2006)

## Usage in Examples

All Chapter 6 examples now use explicit unit handling:

```python
from core.sensors import IMUNoiseParams, units

# Create IMU parameters with explicit conversions
imu_params = IMUNoiseParams.consumer_grade()

# Print specifications
print(imu_params.format_specs())

# Add noise to IMU measurements
accel_meas, gyro_meas, accel_bias, gyro_bias = add_imu_noise(
    accel_body, gyro_body, dt, imu_params
)

# Print realized bias (random sample)
print(f"Gyro bias: {units.format_gyro_bias(np.linalg.norm(gyro_bias))}")
```

## Acceptance Criterion

**Criterion**: If you set "10 deg/hr bias," the printed bias should be ~0.0028 deg/s, NOT ~10 deg/s.

**Verification**:
```python
from core.sensors import IMUNoiseParams, units

params = IMUNoiseParams.consumer_grade()
bias_rad_s = params.gyro_bias_rad_s
bias_deg_s = units.rad_per_sec_to_deg_per_sec(bias_rad_s)

print(f"Gyro bias: {units.format_gyro_bias(bias_rad_s)}")
# Output: "10.00 deg/hr (0.0028 deg/s)"

assert abs(bias_deg_s - 0.002778) < 0.0001  # PASS
```

## Related Equations

- **Eq. (6.5)**: Gyroscope measurement model: ω̃ = ω + b_G + n_G
- **Eq. (6.6)**: Gyroscope bias correction: ω = ω̃ - b_G
- **Eq. (6.9)**: Accelerometer measurement model: a_tilde = a + b_A + n_A
- **Eqs. (6.56)-(6.58)**: Allan variance for noise characterization

## References

1. **IEEE Std 952-1997**: IEEE Standard Specification Format Guide and Test Procedure for Single-Axis Interferometric Fiber Optic Gyros.
2. **ISO 80000-3:2006**: Quantities and units — Part 3: Space and time.
3. **Titterton & Weston (2004)**: Strapdown Inertial Navigation Technology, 2nd ed., Chapter 11.

## Best Practices

1. **Always use explicit unit conversions**: Never use bare `np.deg2rad()` for bias conversions.
2. **Always use typed parameters**: Pass `IMUNoiseParams`, not raw dicts.
3. **Always format for display**: Use `units.format_*` functions for printed diagnostics.
4. **Document units in variable names**: `gyro_bias_rad_s`, not `gyro_bias`.
5. **Test your conversions**: Verify that 10 deg/hr = 0.0028 deg/s in your code.

