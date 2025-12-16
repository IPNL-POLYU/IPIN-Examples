# Chapter 6: Development Notes

> **Note:** This document contains implementation details, design decisions, and development notes for Chapter 6. For student-facing documentation, see [ch6_dead_reckoning/README.md](../ch6_dead_reckoning/README.md).

## Implementation Status

| Feature | Status | Test Cases | Notes |
|---------|--------|------------|-------|
| IMU Strapdown Integration | Complete | 35 | Eqs. (6.2)-(6.10) |
| IMU Error Models | Complete | 24 | Eq. (6.6), (6.9), (6.59) |
| Wheel Odometry | Complete | 26 | Eqs. (6.11)-(6.15) |
| ZUPT Detection/Correction | Complete | 10 | Eqs. (6.44)-(6.45) |
| ZARU Constraint | Complete | 8 | Eq. (6.60) |
| NHC Constraint | Complete | 10 | Eq. (6.61) |
| Pedestrian DR | Complete | 46 | Eqs. (6.46)-(6.50) |
| Magnetometer Heading | Complete | 18 | Eqs. (6.51)-(6.53) |
| Barometric Altitude | Complete | 18 | Eqs. (6.54)-(6.55) |
| Allan Variance | Complete | 25 | Eqs. (6.56)-(6.58) |
| Data Types | Complete | 29 | NavState, ImuSeries, etc. |

**Total:** 249 unit tests, 100% pass rate

## Implementation Notes

### IMU Strapdown Integration (Eqs. 6.2-6.10)

**Quaternion Attitude Propagation**
- Discrete integration with normalization after each step
- Represents body-to-map frame rotation
- Benefits: No gimbal lock, efficient computation

**Error Growth**
- Attitude: gyro bias causes ~0.1-10 deg/hr drift
- Velocity: accel bias causes ~0.01-1 m/s^2 drift
- Position: velocity drift integrates to quadratic error

### Drift Correction Constraints

**Zero-Velocity Update (ZUPT)**
- Detection: |delta_omega| < threshold AND |delta_f - g| < threshold
- Effect: Eliminates velocity drift during stops

**Nonholonomic Constraint (NHC)**
- Constraint: Lateral and vertical velocities = 0 for wheeled vehicles

## Performance Characteristics

| Method | RMSE | Drift Type |
|--------|------|------------|
| IMU-only | 10-30% of distance | Unbounded |
| Wheel odometry | 1-5% of distance | Bounded |
| PDR | 2-5% of distance | Bounded |
| IMU + ZUPT | <1% of distance | Bounded |

---

**Test Coverage:** 249 test cases, 100% pass rate  
**Lines of Code:** 3,855  
**Last Updated:** December 2025


