# Ch2 Coordinate Transforms Dataset: Practical Indoor Positioning Frames

## Overview

This dataset demonstrates **practical coordinate transformations** for indoor positioning: LLH (geodetic) → ECEF (Cartesian) → ENU (local), plus rotation representations (Euler, Quaternion, Matrix). Shows the **numerical precision** needed for accurate transformations.

**Key Learning Objective**: Understand coordinate frame transformations are the foundation of indoor positioning - wrong choice or poor precision → positioning errors!

## Scenario Description

### Learning Goals
1. **Coordinate Frames**: LLH (GPS), ECEF (global), ENU (local building)
2. **When to Use Which**: Global vs. local coordinate systems
3. **Numerical Precision**: Round-trip accuracy matters (sub-mm!)
4. **Rotation Representations**: Euler vs. Quaternion vs. Matrix trade-offs
5. **Practical Application**: GPS → local building coordinates

### Implemented Equations
- **Eq. (2.9)**: LLH → ECEF (closed-form)
- **Eq. (2.9) inverse**: ECEF → LLH (iterative, ~10 iterations; the book
  gives no closed form and refers to Kaplan & Hegarty)
- **Eq. (2.10)**: ECEF → ENU (rotation + translation)
- **Eqs. (2.14)-(2.17)**: Euler angles → rotation matrix, and its inverse
- **Eq. (2.21)**: quaternion → rotation matrix
- **Eq. (2.23)**: Euler angles → quaternion

## Files and Data Structure

- `llh_coordinates.txt`: GPS-like coordinates [N×3] (lat, lon, height in rad, rad, m)
- `ecef_coordinates.txt`: Global Cartesian [N×3] (X, Y, Z in m)
- `enu_coordinates.txt`: Local building frame [N×3] (East, North, Up in m)
- `reference_llh.txt`: Reference point for ENU frame [1×3]
- `euler_angles.txt`: Euler angles [N×3] (roll, pitch, yaw in rad)
- `quaternions.txt`: Unit quaternions [N×4] (qw, qx, qy, qz)
- `rotation_matrices.txt`: 3×3 rotation matrices [N×9] (flattened)
- `config.json`: Dataset parameters and accuracy metrics

## Loading Example

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

## Configuration Parameters

```python
import json

sf_config = json.load(open(data_dir / "config.json"))

ref = sf_config["reference_point"]
print(f"reference:   {ref['latitude_deg']}, {ref['longitude_deg']} ({ref['location']})")
print(f"building:    {sf_config['building']['size_m']} m across, "
      f"{sf_config['building']['num_points']} points")
print(f"seed:        {sf_config['seed']}")

acc = sf_config["accuracy"]
print(f"LLH round-trip:      {acc['llh_roundtrip_height_m']:.2e} m in height")
print(f"rotation round-trip: {acc['rotation_roundtrip_deg']:.1f} deg")
```

| Parameter | Value | Effect |
|---|---|---|
| `reference_point` | 37.7749, -122.4194 | Origin of the ENU frame. Every ENU coordinate is relative to it |
| `building.size_m` | 50.0 | The footprint the points are drawn in. Check it against the ENU extent below — they have disagreed before |
| `building.num_points` | 20 | Sample count; small enough to read the files by eye |
| `seed` | 42 | Fixed, so the sampled points are reproducible |
| `accuracy.rotation_roundtrip_deg` | 0.0 | Euler -> matrix -> Euler, wrapped. It read 360.0 once, which is the identity rotation reported as total failure |

**The footprint is the check worth running**, because a coordinate bug moves the
data and every check derived from that data together:

```python
sf_enu = np.loadtxt(data_dir / "enu_coordinates.txt")
sf_span_e = sf_enu[:, 0].max() - sf_enu[:, 0].min()
sf_span_n = sf_enu[:, 1].max() - sf_enu[:, 1].min()

print(f"declared building size: {sf_config['building']['size_m']:.1f} m")
print(f"actual ENU footprint:   {sf_span_e:.1f} m east-west, {sf_span_n:.1f} m north-south")
print(f"height range:           {sf_enu[:, 2].min():.1f} - {sf_enu[:, 2].max():.1f} m")
```

Expected output:

```
declared building size: 50.0 m
actual ENU footprint:   46.3 m east-west, 45.6 m north-south
height range:           -0.0 - 15.0 m
```

Twenty points drawn in a 50 m square span a little under 50 m, which is what a
random sample of twenty should do. This dataset once spanned **2666 m**, because
the generator added a per-degree constant to a latitude already in radians, and
nothing compared the result against the size `config.json` declares. See
`tests/ch2_coords/test_dataset_matches_its_config.py`.

## Visualization Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sf_fig, (sf_ax1, sf_ax2) = plt.subplots(1, 2, figsize=(12, 5))

sf_ax1.scatter(sf_enu[:, 0], sf_enu[:, 1], c=sf_enu[:, 2], cmap="viridis", s=70)
sf_ax1.plot(0, 0, "r+", markersize=16, label="reference point")
sf_ax1.set_xlabel("East [m]")
sf_ax1.set_ylabel("North [m]")
sf_ax1.set_title("Building footprint in ENU")
sf_ax1.legend()
sf_ax1.grid(alpha=0.3)
sf_ax1.axis("equal")

sf_ax2.hist(sf_enu[:, 2], bins=10, color="steelblue", edgecolor="white")
sf_ax2.set_xlabel("Up [m]")
sf_ax2.set_ylabel("Points")
sf_ax2.set_title("Floors: height is sampled, not continuous")
sf_ax2.grid(alpha=0.3)

sf_fig.tight_layout()
print("figure built")
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
    ↓ Eq. (2.9): llh_to_ecef()
Global Cartesian (ECEF)
    ↓ Eq. (2.10): ecef_to_enu()
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
  "llh_roundtrip_height_m": 1.86e-09,
  "rotation_roundtrip_deg": 0.0
}
```

**Key Points**:
- Position round-trip: **sub-nanometer accuracy!**
- Rotation round-trip: **exact** — Euler → quaternion → matrix → Euler
  recovers the input to machine precision.

> This field used to read `360.0`, and this README explained it as gimbal
> lock. It was neither gimbal lock nor a rotation error. Yaw is sampled on
> [0, 2π) but recovered on (−π, π], so an exact round-trip of 4.4307 rad came
> back as −1.8525 rad and a raw subtraction called the 2π difference "error".
> **A rotation error of 360° is the identity** — a pipeline reporting one is
> measuring its own subtraction, not its accuracy. The generator now wraps the
> difference to [−π, π] before taking its magnitude.

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
print(f"Rotation matrix:\n{R}")

# Apply rotation to a vector
v_body = np.array([1, 0, 0])  # Forward in body frame
v_global = R @ v_body          # Forward in global frame
```

## Parameter Effects and Learning Experiments

| Parameter | Try | What to watch |
|---|---|---|
| `--preset` | `san_francisco`, `tokyo`, `london` | ECEF changes completely while ENU barely moves. ENU is local by construction; that is the point of having it |
| Reference latitude | 0 deg, 45 deg, 80 deg | Metres per degree of longitude shrinks as `cos(lat)`. A per-degree constant borrowed from one latitude is wrong at another -- which is how this dataset once ended up 57x too large |
| `building.size_m` | 10, 50, 500 | The ENU footprint should follow it linearly. If it does not, the conversion is wrong, not the sampling |
| Height range | single floor vs 15 m | Up is metres in both LLH and ENU, so a unit error in the horizontal leaves the vertical looking correct. One right component out of three is what made the old bug survive |

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

### Issue 1: Rotation round-trip error of exactly ~360°

**Symptom**: comparing Euler angles before and after a round-trip gives ~360°,
or ~2π rad, on the yaw column only.

**Cause**: a branch-cut artifact in *your comparison*, not an error in the
rotation. Yaw is sampled on [0, 2π) here but every recovery function returns
(−π, π], so an exact round-trip of 4.4307 rad comes back as −1.8525 rad. The
two describe the same rotation; subtracting them does not.

This dataset shipped with `"rotation_roundtrip_deg": 360.0` for exactly this
reason, and this README used to explain it as gimbal lock and recommend
quaternions. Neither was true — and note that quaternions would not have helped,
because there was nothing wrong to fix. **A rotation error of 360° is the
identity.** Treat one as a bug in the measurement.

**Fix**: wrap the difference before taking its magnitude.
```py
# Wrap to [-pi, pi] so the comparison respects the branch cut
d = euler_recovered - euler_original
d = (d + np.pi) % (2 * np.pi) - np.pi
error_deg = np.rad2deg(np.abs(d).max())   # 0.0 for this dataset
```

Gimbal lock is real, but it lives at **roll = ±90°** in this book's convention
(see `ch2_coords/figs/ch2_gimbal_lock.png`) and this dataset never goes near it
— roll is sampled within ±30°.

### Issue 2: ENU coordinates come out far larger than the building

**Symptom**: points hundreds of metres or kilometres from the reference, for a
building declared tens of metres across.

**Cause**: an offset in **degrees** (or in metres) added to a coordinate already
in **radians**. This is not hypothetical — it is the bug this dataset shipped
with. The generator computed `building_size_m / 111000.0`, named it
`lat_offset_deg`, and added it straight to a latitude in radians. No `deg2rad`
ran, so every offset was 180/π = 57.3× too large and the declared 50 m footprint
was sampled across **2666 m × 2612 m**.

This README previously listed the same symptom under "wrong reference point".
That was a misdiagnosis: the reference point was correct, and re-deriving it
could not have helped. A frame or unit error is common-mode — it moves the data
and any check recomputed from that same data together — so the only thing that
catches it is comparing against an independent statement of intent, here
`config.json`. See `tests/ch2_coords/test_dataset_matches_its_config.py`.

**Fix**: let the library do the conversion. `enu_to_llh_offset` takes metres and
returns radians, so there is no per-degree quantity to mislay:
```python
import numpy as np
from core.coords import ecef_to_enu, enu_to_llh_offset, llh_to_ecef

lat_ref, lon_ref = np.deg2rad(37.7749), np.deg2rad(-122.4194)

# 25 m east, 25 m north of the reference -- inside a 50 m footprint
dlat, dlon = enu_to_llh_offset(east=25.0, north=25.0, lat=lat_ref)
xyz = llh_to_ecef(lat_ref + dlat, lon_ref + dlon, 0.0)
enu = ecef_to_enu(*xyz, lat_ref, lon_ref, 0.0)
print(f"ENU: [{enu[0]:.2f}, {enu[1]:.2f}, {enu[2]:.2f}] m")   # ~[25, 25, 0]
assert abs(enu[0] - 25.0) < 0.01 and abs(enu[1] - 25.0) < 0.01
```

## Recommended Experiments

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

## Connection to Book Equations

### Chapter 2: Coordinate Systems

- **Eq. (2.9)**: LLH → ECEF (closed-form, WGS84 ellipsoid)
- **Eq. (2.9) inverse**: ECEF → LLH (iterative solution)
- **Eq. (2.10)**: ECEF → ENU (rotation matrix depends on reference lat/lon)
- **Eqs. (2.14)-(2.17)**, **(2.21)**, **(2.23)**: rotation conversions.
  Note 2.1/2.2 are the local body/map *vector definitions* and 2.3 is
  map ↔ body -- this list used to cite them for the geodetic chain.

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

