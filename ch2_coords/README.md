# Chapter 2: Coordinate Systems and Transformations

This module implements coordinate transformations and rotation representations for indoor positioning applications, as described in Chapter 2 of the IPIN book.

## Overview

Indoor positioning systems require working with multiple coordinate frames:
- **LLH (Latitude-Longitude-Height)**: Geodetic coordinates using WGS84 ellipsoid
- **ECEF (Earth-Centered Earth-Fixed)**: Global Cartesian frame
- **ENU (East-North-Up)**: Local tangent plane frame for indoor positioning
- **Body frames**: For sensor orientation representation

## Implemented Features

### Coordinate Transformations

1. **LLH ↔ ECEF**
   - `llh_to_ecef()`: Convert geodetic to ECEF Cartesian coordinates
   - `ecef_to_llh()`: Convert ECEF to geodetic (iterative algorithm with pole handling)

2. **ECEF ↔ ENU**
   - `ecef_to_enu()`: Convert ECEF to local East-North-Up frame
   - `enu_to_ecef()`: Convert local ENU back to ECEF

### Rotation Representations

The module supports three rotation representations with full conversion capability:

1. **Euler Angles** (roll-pitch-yaw, ZYX convention)
   - Roll (φ): rotation about x-axis
   - Pitch (θ): rotation about y-axis
   - Yaw (ψ): rotation about z-axis

2. **Rotation Matrices** (3×3 orthogonal matrices in SO(3))
   - Proper handling of orthogonality constraints
   - Determinant = 1.0

3. **Quaternions** (unit quaternions [qw, qx, qy, qz])
   - Normalized representation
   - No gimbal lock issues

### Conversion Functions

All conversions between representations are implemented:
- `euler_to_rotation_matrix()` / `rotation_matrix_to_euler()`
- `euler_to_quat()` / `quat_to_euler()`
- `quat_to_rotation_matrix()` / `rotation_matrix_to_quat()`

Special features:
- Gimbal lock handling at pitch = ±90°
- Shepperd's method for numerically stable matrix-to-quaternion conversion
- Round-trip conversion accuracy < 1e-9

## Frame Definitions

The module defines standard coordinate frames:
- `FRAME_ENU`: East-North-Up local tangent plane
- `FRAME_NED`: North-East-Down local tangent plane
- `FRAME_ECEF`: Earth-Centered Earth-Fixed
- `FRAME_LLH`: Latitude-Longitude-Height geodetic
- `FRAME_BODY`: Body frame (forward-right-down)
- `FRAME_MAP`: Map/world frame for indoor positioning

## Examples

### Basic Usage

```python
import numpy as np
from core.coords import llh_to_ecef, ecef_to_enu, euler_to_quat

# Convert location to ECEF
lat = np.deg2rad(37.7749)  # San Francisco
lon = np.deg2rad(-122.4194)
height = 0.0
xyz = llh_to_ecef(lat, lon, height)

# Define local ENU frame at reference point
lat_ref = np.deg2rad(37.7749)
lon_ref = np.deg2rad(-122.4194)
height_ref = 0.0

# Convert target point to ENU
enu = ecef_to_enu(*xyz_target, lat_ref, lon_ref, height_ref)

# Convert Euler angles to quaternion
roll, pitch, yaw = 0.1, 0.2, 0.3
q = euler_to_quat(roll, pitch, yaw)
```

### Running the Example

```bash
python ch2_coords/example_coordinate_transforms.py
```

This demonstrates:
1. LLH to ECEF transformation
2. ECEF to LLH round-trip
3. Local ENU frame transformations
4. Rotation representations (Euler, matrix, quaternion)
5. Applying rotations to vectors
6. Practical indoor positioning scenario

## Testing

Comprehensive unit tests are provided in `tests/core/coords/`:

```bash
pytest tests/core/coords/ -v
```

**Test Coverage:**
- 46 unit tests (all passing)
- Round-trip conversions (LLH ↔ ECEF ↔ ENU)
- Rotation conversions (Euler ↔ matrix ↔ quaternion)
- Edge cases (poles, gimbal lock, zero rotations)
- Numerical accuracy verification

### Test Summary

#### Coordinate Transformations (`test_transforms.py`)
- ✅ LLH to ECEF at equator, poles, arbitrary locations
- ✅ ECEF to LLH with pole handling (p ≈ 0)
- ✅ Round-trip LLH → ECEF → LLH (accuracy < 1e-3 m)
- ✅ ECEF to ENU relative positioning
- ✅ ENU to ECEF conversion
- ✅ Round-trip ECEF → ENU → ECEF

#### Rotation Conversions (`test_rotations.py`)
- ✅ Euler to rotation matrix (identity, 90°, combined rotations)
- ✅ Rotation matrix properties (orthogonality, det = 1.0)
- ✅ Matrix to Euler with gimbal lock handling
- ✅ Euler to quaternion (normalization, special angles)
- ✅ Quaternion to rotation matrix
- ✅ Matrix to quaternion (Shepperd's method)
- ✅ Cross-conversions (Euler → quat → matrix → Euler)
- ✅ Quaternion double-cover handling

## Code Quality

All code follows PEP 8 and Google Python Style Guide:
- ✅ Type hints for all functions
- ✅ Google-style docstrings with examples
- ✅ No linter errors (black, ruff, flake8, mypy, pylint)
- ✅ 88-character line length
- ✅ Comprehensive documentation

## References

**WGS84 Ellipsoid Parameters:**
- Semi-major axis (a): 6378137.0 m
- Flattening (f): 1/298.257223563
- Semi-minor axis (b): 6356752.314245 m
- First eccentricity squared (e²): 0.00669437999014

**Chapter References:**
- Section 2.2: Coordinate Frames
- Section 2.3: Coordinate Transformations
- Section 2.4: Rotation Representations
- Equations (2.1)-(2.10): Transformation formulas

## Next Steps

This coordinate module provides the foundation for subsequent chapters:
- **Chapter 3**: State estimation algorithms (LS, KF, EKF)
- **Chapter 4**: RF positioning (TOA, TDOA, RSS)
- **Chapter 5**: Fingerprinting methods
- **Chapter 6**: Dead reckoning and PDR
- **Chapter 7**: SLAM algorithms
- **Chapter 8**: Sensor fusion

The transformations implemented here will be used throughout for coordinate frame conversions in positioning algorithms.

