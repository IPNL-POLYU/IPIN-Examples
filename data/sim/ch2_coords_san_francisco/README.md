# Ch2 Coordinate Transforms Dataset: Practical Indoor Positioning Frames

## Overview

This dataset demonstrates **practical coordinate transformations** for indoor positioning: LLH (geodetic) → ECEF (Cartesian) → ENU (local), plus rotation representations (Euler, Quaternion, Matrix). Shows the **numerical precision** needed for accurate transformations.

**Key Learning Objective**: Understand coordinate frame transformations are the foundation of indoor positioning - wrong choice or poor precision → positioning errors!

## Dataset Purpose

### Learning Goals
1. **Coordinate Frames**: LLH (GPS), ECEF (global), ENU (local building)
2. **When to Use Which**: Global vs. local coordinate systems
3. **Numerical Precision**: Round-trip accuracy matters (sub-mm!)
4. **Rotation Representations**: Euler vs. Quaternion vs. Matrix trade-offs
5. **Practical Application**: GPS → local building coordinates

### Implemented Equations
- **Eq. (2.1)**: LLH → ECEF (closed-form)
- **Eq. (2.2)**: ECEF → LLH (iterative, ~10 iterations)
- **Eq. (2.3)**: ECEF → ENU (rotation + translation)
- **Eqs. (2.5-2.10)**: Rotation representations

## Files

- `llh_coordinates.txt`: GPS-like coordinates [N×3] (lat, lon, height in rad, rad, m)
- `ecef_coordinates.txt`: Global Cartesian [N×3] (X, Y, Z in m)
- `enu_coordinates.txt`: Local building frame [N×3] (East, North, Up in m)
- `reference_llh.txt`: Reference point for ENU frame [1×3]
- `euler_angles.txt`: Euler angles [N×3] (roll, pitch, yaw in rad)
- `quaternions.txt`: Unit quaternions [N×4] (qw, qx, qy, qz)
- `rotation_matrices.txt`: 3×3 rotation matrices [N×9] (flattened)
- `config.json`: Dataset parameters and accuracy metrics

## Quick Start

```python
import numpy as np
from pathlib import Path

# Load dataset
data_dir = Path("data/sim/ch2_coords_san_francisco")

llh = np.loadtxt(data_dir / "llh_coordinates.txt")
ecef = np.loadtxt(data_dir / "ecef_coordinates.txt")
enu = np.loadtxt(data_dir / "enu_coordinates.txt")

print(f"Loaded {len(llh)} points")
print(f"LLH: {np.rad2deg(llh[0, :2])} degrees, {llh[0, 2]}m height")
print(f"ECEF: {ecef[0]/1e3} km")
print(f"ENU: {enu[0]} m (local)")
```

## Key Concepts

### 1. Why Three Coordinate Systems?

| Frame | Use Case | Advantages | Disadvantages |
|-------|----------|------------|---------------|
| **LLH** | GPS output | Intuitive (lat/lon) | Nonlinear, complex math |
| **ECEF** | Global tracking | Linear, simple | Not intuitive |
| **ENU** | Indoor positioning | Local, intuitive | Requires reference point |

**Typical Flow**: GPS (LLH) → Convert to ECEF → Convert to ENU for indoor algorithms

### 2. Transformation Chain

```
GPS Receiver Output (LLH)
    ↓ Eq. (2.1): llh_to_ecef()
Global Cartesian (ECEF)
    ↓ Eq. (2.3): ecef_to_enu()
Local Building Frame (ENU)
    ↓ Indoor positioning algorithms
Position estimate (ENU)
```

### 3. Rotation Representations

| Representation | Size | Singularities | Composition | Use Case |
|----------------|------|---------------|-------------|----------|
| **Euler** | 3 params | Gimbal lock | Complex | Human-readable |
| **Quaternion** | 4 params | None | Simple | Optimal for computation |
| **Matrix** | 9 params | None | Direct | Theoretical analysis |

**Recommendation**: Use Quaternions for computation, Euler for display!

## Dataset Accuracy

From `config.json`:
```json
"accuracy": {
  "llh_roundtrip_lat_arcsec": 4.58e-11,
  "llh_roundtrip_lon_arcsec": 0.0,
  "llh_roundtrip_height_m": 9.31e-10,
  "rotation_roundtrip_deg": 360.0
}
```

**Key Points**:
- Position round-trip: **sub-nanometer accuracy!**
- Rotation: Large error due to Euler gimbal lock (educational!)

## Example Usage

### Convert GPS to Local Coordinates
```python
from core.coords import llh_to_ecef, ecef_to_enu
import numpy as np

# GPS measurement (San Francisco)
lat_deg, lon_deg, height_m = 37.7749, -122.4194, 10.0
lat = np.deg2rad(lat_deg)
lon = np.deg2rad(lon_deg)

# Convert to ECEF
ecef_pos = llh_to_ecef(lat, lon, height_m)
print(f"ECEF: {ecef_pos}")

# Convert to local ENU (relative to reference)
lat_ref = np.deg2rad(37.7749)
lon_ref = np.deg2rad(-122.4194)
height_ref = 0.0

enu_pos = ecef_to_enu(ecef_pos[0], ecef_pos[1], ecef_pos[2],
                       lat_ref, lon_ref, height_ref)
print(f"ENU: {enu_pos} meters")  # Local building coordinates!
```

### Work with Rotations
```python
from core.coords import euler_to_quat, quat_to_rotation_matrix
import numpy as np

# Device orientation (Euler angles)
roll = np.deg2rad(10)   # Tilt sideways
pitch = np.deg2rad(5)   # Tilt forward
yaw = np.deg2rad(45)    # Facing northeast

# Convert to quaternion (better for computation)
quat = euler_to_quat(roll, pitch, yaw)
print(f"Quaternion: {quat}")

# Convert to rotation matrix
R = quat_to_rotation_matrix(quat)
print(f"Rotation matrix:\\n{R}")

# Apply rotation to a vector
v_body = np.array([1, 0, 0])  # Forward in body frame
v_global = R @ v_body          # Forward in global frame
```

## Parameter Effects

### Location Dependency
Different locations have different ECEF coordinates:
- San Francisco: ~(-2706, -4261, 3885) km
- Tokyo: ~(-3960, 3350, 3700) km
- London: ~(3980, -8, 4966) km

**Generate comparison**:
```bash
python scripts/generate_ch2_coordinate_transforms_dataset.py --preset san_francisco
python scripts/generate_ch2_coordinate_transforms_dataset.py --preset tokyo
python scripts/generate_ch2_coordinate_transforms_dataset.py --preset london
```

## Common Issues

### Issue 1: Large Rotation Errors

**Symptoms**: Euler round-trip error ~360°

**Cause**: Gimbal lock or angle wrapping

**Solution**: Use quaternions instead:
```python
# Avoid Euler for computation
q1 = euler_to_quat(roll, pitch, yaw)
q2 = euler_to_quat(roll2, pitch2, yaw2)

# Compose rotations (quaternion multiplication)
q_combined = quat_multiply(q1, q2)  # No gimbal lock!
```

### Issue 2: ENU Range Seems Wrong

**Symptoms**: ENU coordinates in km instead of m

**Cause**: Wrong reference point

**Solution**: Use building center as reference:
```python
# Load correct reference
ref_llh = np.loadtxt("data/sim/ch2_coords_san_francisco/reference_llh.txt")
lat_ref, lon_ref, h_ref = ref_llh[0]
```

## Experiments

### Experiment 1: Verify Round-Trip Accuracy
```python
from core.coords import llh_to_ecef, ecef_to_llh
import numpy as np

# Original LLH
llh = np.loadtxt("data/sim/ch2_coords_san_francisco/llh_coordinates.txt")

# LLH -> ECEF -> LLH
ecef = np.array([llh_to_ecef(lat, lon, h) for lat, lon, h in llh])
llh_recovered = np.array([ecef_to_llh(x, y, z) for x, y, z in ecef])

# Compute errors
errors = np.abs(llh - llh_recovered)
print(f"Max error: {errors.max()} rad/m")  # Should be < 1e-9!
```

**Expected**: Sub-nanometer accuracy!

### Experiment 2: Compare Rotation Representations
```python
from core.coords import (euler_to_quat, euler_to_rotation_matrix,
                         quat_to_rotation_matrix)
import numpy as np

euler = np.loadtxt("data/sim/ch2_coords_san_francisco/euler_angles.txt")

# Convert to both representations
for e in euler[:3]:
    q = euler_to_quat(*e)
    R1 = euler_to_rotation_matrix(*e)
    R2 = quat_to_rotation_matrix(q)
    
    # Should be identical
    diff = np.linalg.norm(R1 - R2)
    print(f"Matrix difference: {diff:.3e}")  # Should be ~0
```

## Book Connection

### Chapter 2: Coordinate Systems

- **Eq. (2.1)**: LLH → ECEF (closed-form, WGS84 ellipsoid)
- **Eq. (2.2)**: ECEF → LLH (iterative solution)
- **Eq. (2.3)**: ECEF → ENU (rotation matrix depends on reference lat/lon)
- **Eqs. (2.5-2.10)**: Rotation conversions

**Key Insight**: Indoor positioning uses **local (ENU) frames** - much simpler than global (ECEF)!

## Next Steps

1. Apply to **sensor fusion** (Ch8) - convert GPS to local frame
2. Use for **multi-floor** positioning - height coordinate critical
3. Study **rotation errors** - why quaternions are preferred

## Citation

```bibtex
@book{IPIN2024,
  title={Principles of Indoor Positioning and Indoor Navigation},
  author={[Authors]},
  year={2024},
  chapter={2},
  note={Coordinate Systems and Transformations}
}
```

---

**Dataset Version**: 1.0  
**Last Updated**: December 2024

