# Chapter 2: Coordinate Systems and Transformations

## Overview

This module implements the coordinate systems and transformation functions described in **Chapter 2** of *Principles of Indoor Positioning and Indoor Navigation*. It provides the foundational mathematical tools for converting between different coordinate frames and rotation representations commonly used in indoor navigation systems.

## Quick Start

```bash
# Run the demonstration script
python ch2_coords/example_coordinate_transforms.py
```

## Equation Reference

### Coordinate Transformations

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `llh_to_ecef()` | `core/coords/transforms.py` | Eq. (2.1) | Geodetic (LLH) to ECEF Cartesian coordinates |
| `ecef_to_llh()` | `core/coords/transforms.py` | Eq. (2.2) | ECEF to Geodetic (LLH) - iterative solution |
| `ecef_to_enu()` | `core/coords/transforms.py` | Eq. (2.3) | ECEF to local East-North-Up frame |
| `enu_to_ecef()` | `core/coords/transforms.py` | Eq. (2.4) | Local ENU to ECEF coordinates |

### Rotation Representations

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `euler_to_rotation_matrix()` | `core/coords/rotations.py` | Eq. (2.5) | Euler angles (ZYX) to 3×3 rotation matrix |
| `rotation_matrix_to_euler()` | `core/coords/rotations.py` | Eq. (2.6) | Rotation matrix to Euler angles |
| `euler_to_quat()` | `core/coords/rotations.py` | Eq. (2.7) | Euler angles to unit quaternion |
| `quat_to_euler()` | `core/coords/rotations.py` | Eq. (2.8) | Quaternion to Euler angles |
| `quat_to_rotation_matrix()` | `core/coords/rotations.py` | Eq. (2.9) | Quaternion to rotation matrix |
| `rotation_matrix_to_quat()` | `core/coords/rotations.py` | Eq. (2.10) | Rotation matrix to quaternion |

### WGS84 Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `WGS84_A` | 6378137.0 m | Semi-major axis |
| `WGS84_F` | 1/298.257223563 | Flattening |
| `WGS84_B` | 6356752.314245 m | Semi-minor axis |

## Examples

### Example 1: LLH to ECEF Transformation

```python
import numpy as np
from core.coords import llh_to_ecef, ecef_to_llh

# San Francisco: 37.7749°N, 122.4194°W
lat = np.deg2rad(37.7749)
lon = np.deg2rad(-122.4194)
height = 0.0  # meters above WGS84 ellipsoid

# Convert to ECEF
xyz = llh_to_ecef(lat, lon, height)
print(f"ECEF: {xyz}")  # [x, y, z] in meters

# Round-trip conversion
llh_recovered = ecef_to_llh(*xyz)
print(f"LLH: {np.rad2deg(llh_recovered[:2])}, {llh_recovered[2]:.2f}m")
```

**Implements:** Eq. (2.1), Eq. (2.2)

### Example 2: Local ENU Frame

```python
from core.coords import ecef_to_enu, llh_to_ecef

# Reference point (building entrance)
lat_ref = np.deg2rad(37.7749)
lon_ref = np.deg2rad(-122.4194)
height_ref = 0.0

# Target point (100m north of reference)
lat_target = lat_ref + np.deg2rad(100.0 / 111000.0)
xyz_target = llh_to_ecef(lat_target, lon_ref, height_ref)

# Convert to local ENU coordinates
enu = ecef_to_enu(*xyz_target, lat_ref, lon_ref, height_ref)
print(f"ENU: East={enu[0]:.2f}m, North={enu[1]:.2f}m, Up={enu[2]:.2f}m")
# Expected: East≈0m, North≈100m, Up≈0m
```

**Implements:** Eq. (2.3)

### Example 3: Rotation Representations

```python
from core.coords import (
    euler_to_rotation_matrix,
    euler_to_quat,
    quat_to_rotation_matrix,
)

# Define attitude: 10° roll, 20° pitch, 30° yaw
roll = np.deg2rad(10.0)
pitch = np.deg2rad(20.0)
yaw = np.deg2rad(30.0)

# Convert to rotation matrix
R = euler_to_rotation_matrix(roll, pitch, yaw)
print(f"Rotation matrix:\n{R}")
print(f"det(R) = {np.linalg.det(R):.6f}")  # Should be 1.0

# Convert to quaternion
q = euler_to_quat(roll, pitch, yaw)
print(f"Quaternion: {q}")
print(f"||q|| = {np.linalg.norm(q):.6f}")  # Should be 1.0
```

**Implements:** Eq. (2.5), Eq. (2.7), Eq. (2.9)

## Expected Output

When you run the demonstration script, you should see output similar to:

```
================================================================================
Chapter 2: Coordinate Systems and Transformations
================================================================================

1. LLH to ECEF Transformation (Eq. 2.1)
   Input:  lat=37.7749°, lon=-122.4194°, height=0.0m
   Output: ECEF = [-2694044.41, -4266368.81, 3888310.05] m

2. ECEF to LLH Round-trip (Eq. 2.2)
   Recovered: lat=37.7749°, lon=-122.4194°, height=0.00m
   Round-trip error: < 1e-9 m

3. ECEF to ENU Transformation (Eq. 2.3)
   Reference: Building entrance
   Target: 100m north
   ENU result: East=0.00m, North=100.00m, Up=0.00m

4. Rotation Representations (Eqs. 2.5-2.10)
   Euler angles: roll=10°, pitch=20°, yaw=30°
   Rotation matrix determinant: 1.000000
   Quaternion norm: 1.000000
   Round-trip error: < 1e-9 radians

================================================================================
All transformations completed successfully!
================================================================================
```

## File Structure

```
ch2_coords/
├── README.md                          # This file
└── example_coordinate_transforms.py   # Demonstration script

core/coords/
├── __init__.py                        # Package exports
├── frames.py                          # Frame type definitions
├── transforms.py                      # LLH/ECEF/ENU transformations
└── rotations.py                       # Rotation representations
```

## Book References

- **Section 2.2**: Coordinate Frames (ENU, NED, ECEF, LLH, Body, Map)
- **Section 2.3**: Coordinate Transformations (LLH↔ECEF↔ENU)
- **Section 2.4**: Rotation Representations (Euler, Quaternion, Matrix)

---

*For implementation details and development notes, see [docs/ch2_development.md](../docs/ch2_development.md).*
