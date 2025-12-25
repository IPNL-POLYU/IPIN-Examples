# Chapter 6: Frame Conventions and Coordinate Systems

## Author: Li-Ta Hsu
## Date: December 2025

## Overview

This document defines the **unified frame conventions** used throughout all Chapter 6 dead reckoning algorithms. All code in `ch6_dead_reckoning/` and `core/sensors/` follows these conventions to ensure consistency across:
- IMU strapdown integration
- Wheel odometry
- Pedestrian dead reckoning (PDR)
- Magnetometer heading estimation
- All sensor fusion methods

## Frame Convention Dataclass

All Chapter 6 algorithms use the `FrameConvention` dataclass (defined in `core/sensors/types.py`) to explicitly specify coordinate system conventions.

```python
from core.sensors import FrameConvention

# Default: ENU (East-North-Up)
frame = FrameConvention.create_enu()

# Alternative: NED (North-East-Down)
frame_ned = FrameConvention.create_ned()
```

## Coordinate Frames

### Map Frame (M)

The **map frame** is the navigation reference frame where positions, velocities, and trajectories are expressed.

**Default: ENU (East-North-Up)**
- **x-axis**: Points East  
- **y-axis**: Points North  
- **z-axis**: Points Up (perpendicular to local ground plane)  
- **Origin**: Arbitrary local reference point (e.g., building entrance)

**Alternative: NED (North-East-Down)**
- **x-axis**: Points North  
- **y-axis**: Points East  
- **z-axis**: Points Down (toward Earth center)  
- **Origin**: Same concept as ENU

### Body Frame (B)

The **body frame** is fixed to the moving platform (robot, smartphone, pedestrian).

**Standard body frame axes:**
- **x-axis**: Forward direction of motion
- **y-axis**: Left (perpendicular to forward)
- **z-axis**: Up (perpendicular to ground plane)

For a smartphone held upright:
- x = toward top of screen
- y = toward left edge
- z = out of screen (away from user)

### Quaternion Convention

**Scalar-first representation:**  
`q = [q0, q1, q2, q3]` where:
- `q0` = scalar part
- `[q1, q2, q3]` = vector part

**Quaternion meaning:**  
`q` represents **body-to-map rotation** (C_B^M):
```python
v_map = quat_to_rotmat(q) @ v_body
```

**Identity quaternion** (body aligned with map):  
`q = [1, 0, 0, 0]`

## Gravity Handling

### Gravity Vector

For ENU (z-up):
```python
g_map = [0, 0, -9.81]  # downward (negative z)
```

For NED (z-down):
```python
g_map = [0, 0, +9.81]  # downward (positive z)
```

### Accelerometer Measurements

**Key insight:** Accelerometers measure **specific force** (reaction force), NOT gravitational acceleration.

For a **stationary device** in ENU:
- True acceleration: `a_true = [0, 0, 0]` (not moving)
- Accelerometer reading: `f_b = [0, 0, +9.81]` (upward reaction from ground)
- Gravity vector: `g_map = [0, 0, -9.81]` (downward)

**Relationship:**  
`a_true = f_measured + g_gravity`

For stationary in ENU:  
`[0, 0, 0] = [0, 0, +9.81] + [0, 0, -9.81]` ✓

### Velocity Update (Eq. 6.7)

```python
# Correct implementation
a_map = C_B_M @ f_b + g_map

# For stationary in ENU with identity quaternion:
# a_map = I @ [0, 0, +9.81] + [0, 0, -9.81]
#       = [0, 0, +9.81] + [0, 0, -9.81]  
#       = [0, 0, 0]  ✓ (no acceleration)
```

## Heading Convention

### Heading Definition

**Heading** (yaw angle ψ) is measured in the horizontal plane (x-y) of the map frame.

**ENU convention (default):**
- ψ = 0: Points **East** (+x direction)
- ψ = π/2: Points **North** (+y direction)
- ψ = π: Points **West** (-x direction)
- ψ = -π/2 (or 3π/2): Points **South** (-y direction)

**NED convention:**
- ψ = 0: Points **North** (+x direction)
- ψ = π/2: Points **East** (+y direction)
- ψ = π: Points **South** (-x direction)
- ψ = -π/2: Points **West** (-y direction)

### PDR Step Update (Eq. 6.50)

```python
# ENU: heading 0 = East, π/2 = North
displacement = step_length * [cos(ψ), sin(ψ)]
p_next = p_prev + displacement

# Examples (ENU):
# ψ = 0 (East):  displacement = L * [1, 0] = [L, 0]
# ψ = π/2 (North): displacement = L * [0, 1] = [0, L]
```

### Magnetometer Heading

Magnetometer heading must match the frame convention:

```python
from core.sensors import mag_heading, FrameConvention

# ENU frame
frame_enu = FrameConvention.create_enu()
heading = mag_heading(mag_b, roll, pitch, declination=0.0, frame=frame_enu)
# Returns: 0 = East, π/2 = North

# NED frame
frame_ned = FrameConvention.create_ned()  
heading = mag_heading(mag_b, roll, pitch, declination=0.0, frame=frame_ned)
# Returns: 0 = North, π/2 = East
```

## Usage Examples

### Example 1: IMU Strapdown Integration

```python
from core.sensors import FrameConvention, strapdown_update
import numpy as np

# Define frame convention
frame = FrameConvention.create_enu()

# Initial state
q = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
v = np.zeros(3)  # stationary
p = np.zeros(3)  # at origin

# IMU measurements (stationary device in ENU)
omega_b = np.zeros(3)  # no rotation
f_b = np.array([0.0, 0.0, +9.81])  # upward reaction force

# Time step
dt = 0.01  # 10 ms

# Propagate one step
q_next, v_next, p_next = strapdown_update(
    q=q, v=v, p=p,
    omega_b=omega_b,
    f_b=f_b,
    dt=dt,
    g=9.81,
    frame=frame
)

# Result: v_next = [0, 0, 0] (no drift!)
print(f"Velocity: {v_next}")  # [0, 0, 0]
```

### Example 2: Pedestrian Dead Reckoning

```python
from core.sensors import FrameConvention, pdr_step_update
import numpy as np

# Define frame convention
frame = FrameConvention.create_enu()

# Initial position (2D)
p = np.array([0.0, 0.0])  # at origin

# Take a step
step_length = 0.7  # meters
heading = 0.0  # East (in ENU)

# Update position
p_next = pdr_step_update(p, step_length, heading, frame=frame)

print(f"Position after step East: {p_next}")  # [0.7, 0.0]

# Take another step North
heading = np.pi / 2  # North (in ENU)
p_next2 = pdr_step_update(p_next, step_length, heading, frame=frame)

print(f"Position after step North: {p_next2}")  # [0.7, 0.7]
```

### Example 3: Magnetometer Heading

```python
from core.sensors import FrameConvention, mag_heading
import numpy as np

# Define frame convention
frame = FrameConvention.create_enu()

# Magnetometer measurement (simplified: pointing north)
# In reality, includes inclination and declination
mag_b = np.array([20.0, 0.0, -40.0])  # μT (north + down components)

# Device attitude (level)
roll = 0.0
pitch = 0.0

# Compute heading
heading = mag_heading(mag_b, roll, pitch, declination=0.0, frame=frame)

print(f"Heading: {np.rad2deg(heading):.1f}°")  
# Should be ~90° (North) in ENU where 0°=East, 90°=North
```

## Validation: Stationary IMU Test

The frame convention correctness is validated by the **stationary IMU test** in `tests/core/test_strapdown_stationary_imu.py`.

**Acceptance criteria:**
1. **Zero velocity drift** with zero biases (< 1e-10 m/s after 100 seconds) ✅
2. **Zero position drift** with zero biases (< 1e-8 m after 100 seconds) ✅
3. **Works for both ENU and NED frames** ✅
4. **Noise-bounded drift** (not systematic drift) ✅

**Run the test:**
```bash
pytest tests/core/test_strapdown_stationary_imu.py -v
```

All tests pass, confirming:
- Gravity compensation is correct
- Frame conventions are consistent
- No systematic drift for stationary IMU

## Equations Reference

| Equation | Description | Frame Dependency |
|----------|-------------|------------------|
| Eq. (6.2)-(6.4) | Quaternion integration | Body frame angular velocity |
| Eq. (6.7) | Velocity update | **Gravity vector depends on frame** |
| Eq. (6.8) | Gravity vector | ENU: [0,0,-g], NED: [0,0,+g] |
| Eq. (6.10) | Position update | Frame-agnostic (uses velocity) |
| Eq. (6.50) | PDR step update | **Heading convention depends on frame** |
| Eq. (6.51)-(6.53) | Magnetometer heading | **Heading output depends on frame** |

## Migration Guide for Existing Code

If you have existing Chapter 6 code, update it to use `FrameConvention`:

### Before (implicit ENU):
```python
from core.sensors import strapdown_update

q, v, p = strapdown_update(q, v, p, omega_b, f_b, dt)
# Assumes ENU, but not explicit
```

### After (explicit frame convention):
```python
from core.sensors import FrameConvention, strapdown_update

frame = FrameConvention.create_enu()  # Explicit!
q, v, p = strapdown_update(q, v, p, omega_b, f_b, dt, frame=frame)
```

**Note:** The `frame` parameter defaults to ENU if not provided, so existing code still works. However, **explicitly passing the frame** is recommended for clarity.

## Common Pitfalls

### ❌ Wrong: Confusing gravity direction
```python
# WRONG for ENU!
g_map = [0, 0, +9.81]  # This points upward in ENU
```

### ✅ Correct: Use FrameConvention
```python
# CORRECT
frame = FrameConvention.create_enu()
g_map = frame.gravity_vector()  # Returns [0, 0, -9.81] for ENU
```

### ❌ Wrong: Accelerometer reading for stationary
```python
# WRONG! Accelerometer reads reaction force, not gravity
f_b = np.array([0, 0, -9.81])  # This would be falling!
```

### ✅ Correct: Stationary accelerometer
```python
# CORRECT: Stationary accelerometer in ENU reads upward reaction
f_b = np.array([0, 0, +9.81])  # Upward reaction from ground
```

### ❌ Wrong: Mixing heading conventions
```python
# WRONG! Magnetometer heading in ENU (0=East)  
# used with PDR step assuming 0=North
heading_mag = mag_heading(...)  # Returns 0 for East in ENU
p_next = pdr_step_update(p, L, heading_mag)  
# If PDR assumes 0=North, position will be wrong!
```

### ✅ Correct: Consistent frame convention
```python
# CORRECT: Use same frame for both
frame = FrameConvention.create_enu()
heading_mag = mag_heading(mag, roll, pitch, frame=frame)  # 0=East
p_next = pdr_step_update(p, L, heading_mag, frame=frame)  # 0=East
```

## References

- Chapter 6, Section 6.1: Strapdown inertial navigation
- Chapter 6, Section 6.3: Pedestrian dead reckoning
- Chapter 6, Section 6.4: Environmental sensors
- `core/sensors/types.py`: FrameConvention implementation
- `tests/core/test_strapdown_stationary_imu.py`: Validation tests

## Conclusion

The `FrameConvention` dataclass provides a **unified, explicit, and validated** coordinate system definition for all Chapter 6 algorithms. By using this convention consistently:

1. **No ambiguity** about frame definitions
2. **Correct gravity handling** (validated by stationary IMU test)
3. **Consistent heading conventions** across magnetometer and PDR
4. **Support for both ENU and NED** frames
5. **Easy to extend** for other coordinate systems if needed

**Always use `FrameConvention` in your Chapter 6 code!**

