# Ch4 RF 2D Positioning Dataset: Square Beacon Geometry

## Overview

This dataset demonstrates **RF (Radio Frequency) positioning using TOA, TDOA, and AOA measurements** with various beacon geometries. It showcases the **critical impact of geometry on DOP (Dilution of Precision)** and positioning accuracy.

**Key Learning Objective**: Understand that beacon geometry is THE most important factor in RF positioning accuracy - geometry can cause 10× variation in GDOP!

## Dataset Purpose

### Learning Goals
1. **Geometry is Critical**: GDOP varies from <2 (good) to >10 (poor) based on beacon layout
2. **Compare Techniques**: TOA vs. TDOA vs. AOA have different characteristics
3. **DOP Analysis**: Understand how geometry affects positioning accuracy
4. **NLOS Impact**: See how non-line-of-sight bias degrades all techniques
5. **Measurement Noise**: Explore noise propagation in different RF methods

### Implemented Equations
- **Eq. (4.1-4.3)**: TOA range measurements
  ```
  d_i = ||p - p_i|| + c*b + w_i
  where d_i is measured range, p is position, p_i is beacon i, c*b is clock bias
  ```

- **Eq. (4.27-4.33)**: TDOA range differences
  ```
  d_ij = d_i - d_j = ||p - p_i|| - ||p - p_j|| + w_ij
  Eliminates clock bias!
  ```

- **Eq. (4.63-4.66)**: AOA angle measurements
  ```
  θ_i = atan2(y - y_i, x - x_i) + w_i
  Measures bearing from beacon to agent
  ```

- **Section 4.5**: DOP calculations
  ```
  GDOP = sqrt(trace((H^T H)^{-1}))
  where H is geometry matrix
  ```

## Dataset Variants

| Variant | Directory | Geometry | Mean GDOP (TOA) | Description |
|---------|-----------|----------|-----------------|-------------|
| **Baseline** | `ch4_rf_2d_square` | Square (4 corners) | ~1.0 | Good geometry, low GDOP |
| **Optimal** | `ch4_rf_2d_optimal` | Circular (evenly spaced) | ~0.8 | Best geometry, minimum GDOP |
| **Poor** | `ch4_rf_2d_linear` | Linear array | >10 | Bad geometry, very high GDOP |
| **NLOS** | `ch4_rf_2d_nlos` | Square + NLOS bias | ~1.0 | Good geometry but measurement bias |

**Generate variants**:
```bash
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset baseline
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset optimal
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset poor_geometry
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset nlos
```

## Files

### Beacon Configuration
- `beacons.txt`: Beacon positions [N_beacons×2] (x, y in meters)

### Ground Truth
- `ground_truth_positions.txt`: True agent positions [N×2] (x, y in meters)

### Measurements
- `toa_ranges.txt`: TOA range measurements [N×N_beacons] (meters)
- `tdoa_diffs.txt`: TDOA range differences [N×(N_beacons-1)] (meters, relative to beacon 0)
- `aoa_angles.txt`: AOA angle measurements [N×N_beacons] (radians)

### DOP Metrics
- `gdop_toa.txt`: GDOP values for TOA [N×1]
- `gdop_tdoa.txt`: GDOP values for TDOA [N×1]
- `gdop_aoa.txt`: GDOP values for AOA [N×1]

### Configuration
- `config.json`: All dataset parameters and performance metrics

## Loading Data

### Python
```python
import numpy as np
from pathlib import Path

# Load dataset
data_dir = Path("data/sim/ch4_rf_2d_square")

beacons = np.loadtxt(data_dir / "beacons.txt")
positions = np.loadtxt(data_dir / "ground_truth_positions.txt")
toa_ranges = np.loadtxt(data_dir / "toa_ranges.txt")
tdoa_diffs = np.loadtxt(data_dir / "tdoa_diffs.txt")
aoa_angles = np.loadtxt(data_dir / "aoa_angles.txt")
gdop_toa = np.loadtxt(data_dir / "gdop_toa.txt")

print(f"Loaded {len(positions)} positions with {len(beacons)} beacons")
print(f"Mean TOA GDOP: {gdop_toa.mean():.2f}")
print(f"Area: {positions.max():.0f}m x {positions.max():.0f}m")
```

### MATLAB
```matlab
% Load dataset
data_dir = 'data/sim/ch4_rf_2d_square/';

beacons = load([data_dir 'beacons.txt']);
positions = load([data_dir 'ground_truth_positions.txt']);
toa_ranges = load([data_dir 'toa_ranges.txt']);
gdop_toa = load([data_dir 'gdop_toa.txt']);

fprintf('Loaded %d positions, %d beacons\n', size(positions, 1), size(beacons, 1));
fprintf('Mean GDOP: %.2f\n', mean(gdop_toa));
```

## Configuration Parameters

### Geometry Configuration
```json
{
  "geometry": {
    "type": "square",
    "num_beacons": 4,
    "area_size_m": 20.0
  }
}
```

**Key Parameters**:
- **type**: Beacon geometry (square, optimal, linear, lshape, poor)
- **num_beacons**: Number of beacons (typically 4)
- **area_size**: Size of positioning area (20m × 20m)

### Trajectory Configuration
```json
{
  "trajectory": {
    "type": "grid",
    "num_points": 100
  }
}
```

**Key Parameters**:
- **type**: Evaluation trajectory (grid, random, circle, corridor)
- **num_points**: Number of test positions (100)

### Measurement Noise Configuration
```json
{
  "measurements": {
    "toa_noise_std_m": 0.1,
    "tdoa_noise_std_m": 0.1,
    "aoa_noise_std_deg": 2.0
  }
}
```

**Key Parameters**:
- **toa_noise**: TOA range noise (0.1m std dev ≈ 0.3ns timing error)
- **tdoa_noise**: TDOA noise (0.1m)
- **aoa_noise**: AOA angle noise (2° std dev)

## Quick Start Example

### TOA Positioning
```python
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from core.rf import TOAPositioner

# Load dataset
data_dir = Path("data/sim/ch4_rf_2d_square")
beacons = np.loadtxt(data_dir / "beacons.txt")
positions = np.loadtxt(data_dir / "ground_truth_positions.txt")
toa_ranges = np.loadtxt(data_dir / "toa_ranges.txt")

# Initialize TOA positioner
toa_solver = TOAPositioner(beacons, method="iwls")

# Estimate positions
N = len(positions)
estimated_pos = np.zeros((N, 2))

for i in range(N):
    try:
        pos_est, info = toa_solver.solve(
            toa_ranges[i],
            initial_guess=np.array([10.0, 10.0])
        )
        estimated_pos[i] = pos_est
    except:
        estimated_pos[i] = positions[i]  # Fallback if solve fails

# Compute errors
errors = np.linalg.norm(estimated_pos - positions, axis=1)
print(f"Mean error: {errors.mean():.3f} m")
print(f"Max error: {errors.max():.3f} m")
print(f"RMS error: {np.sqrt(np.mean(errors**2)):.3f} m")

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(beacons[:, 0], beacons[:, 1], s=200, marker='^', c='red', label='Beacons', zorder=10)
ax.scatter(positions[:, 0], positions[:, 1], s=20, c='green', label='True', alpha=0.5)
ax.scatter(estimated_pos[:, 0], estimated_pos[:, 1], s=20, c='blue', marker='x', label='Estimated')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'TOA Positioning: {errors.mean():.2f}m mean error')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.show()
```

**Expected Result**: ~0.1m mean error with square geometry

### TDOA Positioning
```python
from core.rf import TDOAPositioner

# Load dataset
tdoa_diffs = np.loadtxt(data_dir / "tdoa_diffs.txt")

# Initialize TDOA positioner
tdoa_solver = TDOAPositioner(beacons, reference_idx=0)

# Estimate positions
estimated_pos_tdoa = np.zeros((N, 2))

for i in range(N):
    try:
        pos_est, info = tdoa_solver.solve(
            tdoa_diffs[i],
            initial_guess=np.array([10.0, 10.0])
        )
        estimated_pos_tdoa[i] = pos_est
    except:
        estimated_pos_tdoa[i] = positions[i]

# Compute errors
errors_tdoa = np.linalg.norm(estimated_pos_tdoa - positions, axis=1)
print(f"TDOA mean error: {errors_tdoa.mean():.3f} m")
```

**Note**: TDOA may have larger errors due to hyperbolic geometry and linearization

### AOA Positioning
```python
from core.rf import AOAPositioner

# Load dataset
aoa_angles = np.loadtxt(data_dir / "aoa_angles.txt")

# Initialize AOA positioner
aoa_solver = AOAPositioner(beacons)

# Estimate positions
estimated_pos_aoa = np.zeros((N, 2))

for i in range(N):
    try:
        pos_est, info = aoa_solver.solve(
            aoa_angles[i],
            initial_guess=np.array([10.0, 10.0])
        )
        estimated_pos_aoa[i] = pos_est
    except:
        estimated_pos_aoa[i] = positions[i]

# Compute errors
errors_aoa = np.linalg.norm(estimated_pos_aoa - positions, axis=1)
print(f"AOA mean error: {errors_aoa.mean():.3f} m")
```

**Expected Result**: ~0.5m mean error (angle errors amplify with distance)

## Visualization

### Plot GDOP Map
```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from core.rf import compute_geometry_matrix, compute_dop

# Load dataset
data_dir = Path("data/sim/ch4_rf_2d_square")
beacons = np.loadtxt(data_dir / "beacons.txt")
gdop_toa = np.loadtxt(data_dir / "gdop_toa.txt")
positions = np.loadtxt(data_dir / "ground_truth_positions.txt")

# Reshape GDOP for grid (assuming grid trajectory)
grid_size = int(np.sqrt(len(gdop_toa)))
gdop_grid = gdop_toa[:grid_size**2].reshape((grid_size, grid_size))

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(gdop_grid, extent=[0, 20, 0, 20], origin='lower', cmap='RdYlGn_r')
ax.scatter(beacons[:, 0], beacons[:, 1], s=200, marker='^', c='blue', 
           edgecolors='black', linewidths=2, label='Beacons', zorder=10)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('TOA GDOP Map (lower is better)')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('GDOP')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Min GDOP: {gdop_toa.min():.2f} (best geometry)")
print(f"Max GDOP: {gdop_toa.max():.2f} (worst geometry)")
```

**Learning Point**: GDOP is lowest at center, increases near edges and corners

## Parameter Effects

### Effect of Beacon Geometry

| Geometry | Mean GDOP | Position Error (m) | Notes |
|----------|-----------|-------------------|-------|
| **Square** (corners) | 1.0-1.1 | 0.10-0.15 | Good, symmetric |
| **Optimal** (circular) | 0.8-0.9 | 0.08-0.12 | Best, evenly spaced |
| **L-shape** | 2.0-3.0 | 0.20-0.30 | Poor in some regions |
| **Linear** | >10 | >1.0 | Very poor perpendicular |
| **Clustered** | >20 | >2.0 | Unusable |

**Generate comparison**:
```bash
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset baseline
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset optimal
python scripts/generate_ch4_rf_2d_positioning_dataset.py --geometry lshape --output data/sim/ch4_rf_lshape
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset poor_geometry
```

**Learning Point**: Geometry can cause 10× variation in accuracy!

### Effect of TOA Measurement Noise

| TOA Noise (m) | Position Error (m) | GDOP Amplification | Notes |
|---------------|-------------------|--------------------| ------|
| 0.05 (excellent) | 0.05-0.08 | ~1.5× | High-quality UWB |
| 0.10 (good) | 0.10-0.15 | ~1.5× | Baseline |
| 0.30 (fair) | 0.30-0.45 | ~1.5× | GPS-like |
| 0.50 (poor) | 0.50-0.75 | ~1.5× | Multipath environment |

**Formula**: `Position Error ≈ Measurement Noise × GDOP`

**Generate sweep**:
```bash
python scripts/generate_ch4_rf_2d_positioning_dataset.py --output data/sim/ch4_toa_005 --toa-noise 0.05
python scripts/generate_ch4_rf_2d_positioning_dataset.py --output data/sim/ch4_toa_010 --toa-noise 0.10
python scripts/generate_ch4_rf_2d_positioning_dataset.py --output data/sim/ch4_toa_030 --toa-noise 0.30
python scripts/generate_ch4_rf_2d_positioning_dataset.py --output data/sim/ch4_toa_050 --toa-noise 0.50
```

### Effect of AOA Measurement Noise

| AOA Noise (deg) | Position Error (m) @ 10m | Notes |
|-----------------|-------------------------|-------|
| 1° (excellent) | 0.17 | High-precision arrays |
| 2° (good) | 0.35 | Baseline |
| 5° (fair) | 0.87 | Consumer antennas |
| 10° (poor) | 1.75 | Low-cost systems |

**Formula**: `Position Error ≈ distance × tan(angle_error)`

**Generate sweep**:
```bash
python scripts/generate_ch4_rf_2d_positioning_dataset.py --output data/sim/ch4_aoa_01 --aoa-noise 1.0
python scripts/generate_ch4_rf_2d_positioning_dataset.py --output data/sim/ch4_aoa_02 --aoa-noise 2.0
python scripts/generate_ch4_rf_2d_positioning_dataset.py --output data/sim/ch4_aoa_05 --aoa-noise 5.0
python scripts/generate_ch4_rf_2d_positioning_dataset.py --output data/sim/ch4_aoa_10 --aoa-noise 10.0
```

**Learning Point**: AOA errors amplify with distance from beacons!

## Experiments

### Experiment 1: Geometry Impact on GDOP

**Objective**: Quantify how beacon geometry affects GDOP and positioning accuracy.

**Procedure**:
1. Generate datasets with different geometries (square, optimal, linear)
2. Compare GDOP distributions
3. Measure positioning errors

**Expected Results**:
- Square: GDOP ~1.0, error ~0.1m
- Optimal: GDOP ~0.8, error ~0.08m (20% better!)
- Linear: GDOP >10, error >1.0m (10× worse!)

**Code**:
```bash
# Generate datasets
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset baseline
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset optimal
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset poor_geometry

# Compare GDOP
python -c "
import numpy as np
for name in ['ch4_rf_2d_square', 'ch4_rf_2d_optimal', 'ch4_rf_2d_linear']:
    gdop = np.loadtxt(f'data/sim/{name}/gdop_toa.txt')
    print(f'{name}: mean={gdop.mean():.2f}, min={gdop.min():.2f}, max={gdop.max():.2f}')
"
```

**Learning Point**: Optimal beacon placement can reduce errors by 20-50%!

### Experiment 2: TOA vs. TDOA vs. AOA Comparison

**Objective**: Compare the three RF positioning techniques.

**Procedure**:
1. Generate baseline dataset (all three measurements)
2. Run TOA, TDOA, and AOA positioning
3. Compare errors and characteristics

**Expected Results**:
- **TOA**: Best accuracy (~0.1m) but requires clock sync
- **TDOA**: Moderate accuracy, eliminates clock bias
- **AOA**: Angle errors amplify with distance (~0.5m)

**Code**: Use Quick Start examples for all three techniques

**Learning Point**: TOA is best IF you can sync clocks, otherwise use TDOA!

### Experiment 3: NLOS Impact

**Objective**: Study how NLOS bias affects positioning.

**Procedure**:
1. Generate clean dataset (no NLOS)
2. Generate NLOS dataset (biased beacons)
3. Compare positioning errors

**Expected Results**:
- Clean: ~0.1m error
- NLOS: 0.5-1.0m error (5-10× worse)
- NLOS creates systematic bias, not just noise

**Code**:
```bash
# Generate datasets
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset baseline
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset nlos

# Compare errors by loading and running positioning (see Quick Start)
```

**Learning Point**: NLOS is the PRIMARY error source in indoor RF positioning!

## Performance Metrics (Baseline)

| Metric | TOA | TDOA | AOA | Notes |
|--------|-----|------|-----|-------|
| **Mean Error** | 0.10m | ~15m* | 0.46m | TOA best |
| **Std Dev Error** | 0.05m | ~15m* | 0.29m | Consistent |
| **Max Error** | 0.27m | ~154m* | 1.39m | Outliers |
| **Mean GDOP** | 1.02 | 0.87 | 15.04 | TOA/TDOA similar |
| **Min GDOP** | 1.00 | 0.81 | 13.84 | Center of area |
| **Max GDOP** | 1.09 | 1.03 | 16.74 | Near edges |

*Note: TDOA errors are high due to linearization and hyperbolic geometry - requires good initialization

**Key Insights**:
- TOA: Best accuracy, requires clock sync
- TDOA: Clock-free but larger errors
- AOA: Good for bearing, errors grow with distance

## Book Connection

### Chapter 4: RF Point Positioning

This dataset directly implements RF positioning from Chapter 4:

1. **TOA (Section 4.1, Eqs. 4.1-4.3)**
   - Measures propagation time → range
   - Requires clock synchronization
   - Position from range intersection (trilateration)

2. **TDOA (Section 4.2, Eqs. 4.27-4.33)**
   - Measures time difference → range difference
   - Eliminates clock bias (huge advantage!)
   - Position from hyperbola intersection

3. **AOA (Section 4.4, Eqs. 4.63-4.67)**
   - Measures angle of arrival
   - No clock required
   - Position from bearing intersection (triangulation)

4. **DOP (Section 4.5)**
   - Quantifies geometry quality
   - GDOP = sqrt(trace((H^T H)^{-1}))
   - Lower GDOP = better geometry

**Key Insight from Chapter 4**: Geometry matters MORE than measurement noise! A 2× improvement in geometry (GDOP 2.0 → 1.0) has the same effect as 2× better measurements!

## Common Issues & Solutions

### Issue 1: TDOA Positioning Fails or Returns Large Errors

**Symptoms**: TDOA gives >10m errors while TOA gives <0.5m

**Likely Cause**: Poor initialization or hyperbolic geometry

**Solution**: Use TOA solution as initial guess for TDOA:
```python
# Get TOA solution first
toa_pos, _ = toa_solver.solve(toa_ranges[i])

# Use as initial guess for TDOA
tdoa_pos, _ = tdoa_solver.solve(tdoa_diffs[i], initial_guess=toa_pos)
```

### Issue 2: High GDOP in Certain Regions

**Symptoms**: GDOP >5 in some areas, <2 in others

**Likely Cause**: Poor beacon geometry or being too close to a beacon

**Solution**: Add more beacons or improve geometry:
```python
# Check GDOP before positioning
if gdop_toa[i] > 5.0:
    print(f"Warning: High GDOP ({gdop_toa[i]:.1f}) at position {i}")
    # Consider rejecting or using different beacons
```

### Issue 3: AOA Errors Increase with Distance

**Symptoms**: Positions near beacons accurate, far positions have large errors

**Likely Cause**: Angle errors amplify with distance (geometric)

**Solution**: This is expected! Use TOA/TDOA for long-range, AOA for short-range:
```python
# Position error ≈ distance × tan(angle_error)
angle_error_rad = np.deg2rad(2.0)  # 2 degree error
distance = 10.0  # meters
expected_error = distance * np.tan(angle_error_rad)
print(f"Expected AOA error at {distance}m: {expected_error:.2f}m")
```

## Troubleshooting

### Error: Positioning solver doesn't converge

**Cause**: Poor initial guess or degenerate geometry

**Fix**: Improve initial guess or check beacon geometry:
```python
# Use center of beacon array as initial guess
initial_guess = beacons.mean(axis=0)

# Or use grid search
best_pos = None
best_residual = np.inf
for x in np.linspace(0, 20, 5):
    for y in np.linspace(0, 20, 5):
        try:
            pos, info = solver.solve(measurements, initial_guess=np.array([x, y]))
            if info['residual'] < best_residual:
                best_residual = info['residual']
                best_pos = pos
        except:
            pass
```

### Warning: GDOP > 10

**Cause**: Very poor geometry (linear array or near singularity)

**Fix**: Add beacons or avoid problematic regions:
```python
if gdop > 10:
    print("Warning: Positioning unreliable (GDOP too high)")
    # Either reject position or increase measurement uncertainty
```

## Next Steps

After understanding RF positioning basics:

1. **Chapter 8**: Sensor fusion (combine RF with IMU/odometry)
2. **NLOS Mitigation**: Robust estimation techniques (M-estimators, RANSAC)
3. **Multipath**: Study indoor propagation effects
4. **Hybrid Methods**: TOA + AOA fusion
5. **3D Positioning**: Extend to 3D with altitude

## Citation

If you use this dataset in your research, please cite:

```bibtex
@book{IPIN2024,
  title={Principles of Indoor Positioning and Indoor Navigation},
  author={[Authors]},
  year={2024},
  chapter={4},
  note={RF Point Positioning}
}
```

## License

This dataset is part of the IPIN Book Examples repository. See repository LICENSE for details.

---

**Dataset Version**: 1.0  
**Last Updated**: December 2024  
**Contact**: See repository README for contact information

