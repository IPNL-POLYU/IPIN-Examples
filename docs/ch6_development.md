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
- Quaternion kinematics: q_dot = 0.5 * Omega(omega) * q
- Discrete integration with normalization after each step
- Represents body-to-map frame rotation
- Benefits: No gimbal lock, efficient computation

**Velocity Integration**
- Specific force rotation: f^M = C_B^M(q) * f^B
- Gravity compensation: f^M + g
- Velocity update: v_k = v_{k-1} + (f^M + g) * dt

**Position Integration**
- Simple Euler integration: p_k = p_{k-1} + v_k * dt
- Drift is **unbounded** without corrections

**Error Growth**
- Attitude: gyro bias causes ~0.1-10 deg/hr drift
- Velocity: accel bias causes ~0.01-1 m/s^2 drift
- Position: velocity drift integrates to quadratic error

### Wheel Odometry (Eqs. 6.11-6.15)

**Lever Arm Compensation**
- Wheels measure velocity at sensor location
- Velocity transform: v^A = v^S + [omega^B x] * r^B
- Skew-symmetric matrix for cross product

**Error Characteristics**
- Typical: 1-5% of distance traveled
- Wheel slip: 10-50% errors in turns
- Better than IMU for vehicles (bounded drift)

### Drift Correction Constraints

**Zero-Velocity Update (ZUPT)**
- Detection: |delta_omega| < threshold AND |delta_f - g| < threshold
- Measurement: h(x) = v (velocity should be zero)
- Effect: Eliminates velocity drift during stops

**Nonholonomic Constraint (NHC)**
- Constraint: Lateral and vertical velocities = 0 for wheeled vehicles
- Application: Cars, robots (can't move sideways)

### Pedestrian Dead Reckoning (Eqs. 6.46-6.50)

**Step Detection**
- Acceleration magnitude: a_mag = ||a||
- Peak detection: step when a_mag crosses threshold (~11-12 m/s^2)

**Step Length Estimation (Weinberg Model)**
- L = c * h^a * f^b
- Typical parameters: a = 0.371, b = 0.227, c = 1.0

**Position Update**
- 2D step-and-heading: p_k = p_{k-1} + L * [cos(psi), sin(psi)]^T
- Heading errors dominate accuracy

### Allan Variance (Eqs. 6.56-6.58)

**Noise Identification (Log-Log Plot)**
- Slope -0.5: Angle/velocity random walk
- Slope 0 (flat): Bias instability
- Slope +0.5: Rate random walk

**IMU Specifications**

| Grade | ARW (gyro) | Bias Instability | Cost |
|-------|-----------|------------------|------|
| Consumer | 0.1-1.0 deg/sqrt(hr) | 10-100 deg/hr | $1-10 |
| Tactical | 0.01-0.1 deg/sqrt(hr) | 1-10 deg/hr | $100-1k |
| Navigation | <0.01 deg/sqrt(hr) | <1 deg/hr | $10k-100k |

## Performance Characteristics

### Typical Performance (Consumer-grade IMU)

| Method | RMSE | Drift Type |
|--------|------|------------|
| IMU-only | 10-30% of distance | Unbounded |
| Wheel odometry | 1-5% of distance | Bounded |
| PDR | 2-5% of distance | Bounded |
| IMU + ZUPT | <1% of distance | Bounded |
| IMU + NHC | 1-3% of distance | Bounded |

### Error Growth

- **IMU-only:** Quadratic in time (unbounded)
- **Wheel odometry:** Linear in distance (bounded)
- **PDR:** Linear in distance (bounded)
- **IMU + ZUPT:** Reset at each stop (bounded)

## Future Enhancements

- Real sensor data integration
- EKF/UKF sensor fusion
- Integrated navigation (IMU + wheel + GPS)
- Multi-floor scenarios
- Comparison of different IMU grades

## Troubleshooting

**IMU drift too large:**
- Check gyro bias estimation
- Apply ZUPT during stops
- Use higher-grade IMU

**Wheel odometry errors in turns:**
- Likely wheel slip
- Apply NHC constraints
- Check lever arm calibration

**PDR heading drift:**
- Use magnetometer fusion
- Check for magnetic disturbances
- Increase ZUPT frequency

---

**Test Coverage:** 249 test cases, 100% pass rate  
**Lines of Code:** 3,855  
**Last Updated:** December 2025

