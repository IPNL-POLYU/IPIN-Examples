# Ch4 RF 2D Positioning Dataset: Square Beacon Geometry

## Overview

This dataset demonstrates **RF (Radio Frequency) positioning using TOA, TDOA, and AOA measurements** with various beacon geometries. It showcases the **critical impact of geometry on DOP (Dilution of Precision)** and positioning accuracy.

**Key Learning Objective**: Understand that beacon geometry is THE most important factor in RF positioning accuracy - geometry can cause 10× variation in GDOP!

## Scenario Description

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

| Variant | Directory | Geometry | Mean GDOP: TOA / TDOA / AOA | Description |
|---------|-----------|----------|-----------------------------|-------------|
| **Baseline** | `ch4_rf_2d_square` | Square (4 corners) | 1.02 / 0.87 / 15.04 | Good geometry, low GDOP |
| **Optimal** | `ch4_rf_2d_optimal` | Diamond (evenly spaced) | 1.02 / 1.09 / 11.54 | Best AOA geometry |
| **Poor** | `ch4_rf_2d_linear` | Linear array | 1.43 / 10.36 / 9.25 | Poor for TDOA — see its README |
| **NLOS** | `ch4_rf_2d_nlos` | Square + NLOS bias | 1.02 / 0.87 / 15.04 | Good geometry but measurement bias |

DOP is reported per method because it differs by an order of magnitude between
them, and the "poor" variant is the case in point: its TOA GDOP of 1.43 is
almost as good as the square's, while its TDOA GDOP is twelve times worse.
`data/sim/ch4_rf_2d_linear/README.md` covers that geometry, including why a
healthy TOA GDOP there is still not enough to make TOA usable.

**Generate variants**:
```bash
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset baseline
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset optimal
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset poor_geometry
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset nlos
```

## Files and Data Structure

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

## Loading Example

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

initial_guess = beacons.mean(axis=0)  # centre of the anchor set
solved_tdoa = np.zeros(N, dtype=bool)

for i in range(N):
    try:
        pos_est, info = tdoa_solver.solve(tdoa_diffs[i], initial_guess=initial_guess)
    except Exception:
        estimated_pos_tdoa[i] = np.nan
        continue
    estimated_pos_tdoa[i] = pos_est
    # A solve that never left the initial guess has not solved anything, even
    # when the solver says it converged.
    stalled = np.linalg.norm(pos_est - initial_guess) < 1e-6
    solved_tdoa[i] = bool(info.get("converged", True)) and not stalled

# Report the median, not the mean: one solve landing far away makes a mean a
# property of that outlier rather than of the method.
errors_tdoa = np.linalg.norm(estimated_pos_tdoa - positions, axis=1)
good = solved_tdoa & np.isfinite(errors_tdoa) & (errors_tdoa < 100)
print(f"TDOA median error: {np.median(errors_tdoa[np.isfinite(errors_tdoa)]):.3f} m")
print(f"TDOA mean over solved: {errors_tdoa[good].mean():.3f} m")
print(f"TDOA failed to solve: {np.sum(~good)}/{N}")
```

**Expected Result**: 0.075m median, 0.081m mean over the 100 that solve, 0/100 failed.

**Note**: this is the accuracy the geometry predicts. The TDOA GDOP here is
0.873 and the range-difference noise is 0.1 m, so `sigma_position = GDOP x
sigma_range` gives 0.087 m — and a median sits a little below that, since it is
not an RMS. TDOA slightly beats TOA on this array (0.075 m against 0.095 m)
because its GDOP is lower, which is the whole point of comparing the two here.

Until recently this block printed 13.75 m with 11 of 100 fixes failing, and the
text around it explained that as the price of hyperbolic geometry. It was not:
the generator wrote each range difference as `d_ref - d_j` while
`TDOAPositioner` predicts `d_j - d_ref`, so every shipped measurement was
negated and the solver was being sent to the far branch of each hyperbola.
Missing a GDOP prediction by 158x is the kind of disagreement worth chasing
rather than narrating.

### AOA Positioning
```python
from core.rf import AOAPositioner

# Load dataset
aoa_angles = np.loadtxt(data_dir / "aoa_angles.txt")

# Initialize AOA positioner
aoa_solver = AOAPositioner(beacons)

# Estimate positions
estimated_pos_aoa = np.zeros((N, 2))

initial_guess = beacons.mean(axis=0)
solved_aoa = np.zeros(N, dtype=bool)

for i in range(N):
    try:
        pos_est, info = aoa_solver.solve(aoa_angles[i], initial_guess=initial_guess)
    except Exception:
        estimated_pos_aoa[i] = np.nan
        continue
    estimated_pos_aoa[i] = pos_est
    stalled = np.linalg.norm(pos_est - initial_guess) < 1e-6
    solved_aoa[i] = bool(info.get("converged", True)) and not stalled

errors_aoa = np.linalg.norm(estimated_pos_aoa - positions, axis=1)
good = solved_aoa & np.isfinite(errors_aoa) & (errors_aoa < 100)
print(f"AOA median error: {np.median(errors_aoa[np.isfinite(errors_aoa)]):.3f} m")
print(f"AOA mean over solved: {errors_aoa[good].mean():.3f} m")
print(f"AOA failed to solve: {np.sum(~good)}/{N}")
# Do NOT print errors_aoa.mean() -- it is about 1e15 m here.
```

**Expected Result**: 0.40m median, 0.46m mean, **0 of 100 fail**.

This used to read 36 of 100 failing, with an arithmetic mean around 1e15 m.
That was the *parameterisation*, not the geometry and not the noise. Writing
Eq. (4.64) literally and solving on `z = tan(ψ)` has two defects no starting
point repairs: `tan` has period π, so an anchor ahead and one behind give the
same measurement; and as the estimate runs to infinity every anchor tends to
the same bearing, so the residuals *shrink* — infinity is an attractor the
solver reports as convergence.

`AOAPositioner` now forms residuals as `wrap(ψ − atan2(ΔE, ΔN))`: the same
measurement model, inverted without discarding the quadrant. The book's form
is still reachable as `residual="tan"` if you want to see it misbehave.

Report the median and the failure count anyway. Both are cheap, and a mean
alone would have hidden the old defect completely.

## Visualization Example

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

## Parameter Effects and Learning Experiments

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

## Recommended Experiments

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
- **TOA**: ~0.10m, but requires clock sync
- **TDOA**: ~0.07m, and eliminates clock bias — on *this* geometry it is the
  most accurate of the three, because its GDOP is the lowest of the three
- **AOA**: ~0.40m; angle errors amplify with distance

**Code**: Use Quick Start examples for all three techniques

**Learning Point**: on a good geometry the three differ by their GDOP and
little else, so the choice is made by what the hardware can offer — clock sync
for TOA, a common time base for TDOA, an antenna array for AOA. Change the
geometry and that ranking changes completely, which is Experiment 1's job.

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

All solvers start from the centre of the anchor set. A solve counts as failed
if it raised, reported `converged: False`, never left the initial guess, or
landed more than 100 m away.

| Metric | TOA | TDOA | AOA | Notes |
|--------|-----|------|-----|-------|
| **Median Error** | 0.10m | 0.07m | 0.40m | Robust to the tail |
| **Mean over solved** | 0.10m | 0.08m | 0.46m | TDOA best, as its GDOP says |
| **Max over solved** | 0.27m | 0.23m | 1.30m | |
| **Failed to solve** | **0/100** | **0/100** | **0/100** | every fix solves on this array |
| **Mean GDOP** | 1.02 | 0.87 | 15.04 | TOA/TDOA similar |
| **Min GDOP** | 1.00 | 0.81 | 13.84 | Center of area |
| **Max GDOP** | 1.09 | 1.03 | 16.74 | Near edges |

The median and the failure count are reported rather than a mean, because a
single solve that "converges" to somewhere absurd makes a mean a property of
that outlier. AOA used to do exactly that on 36 of 100 positions.

**Key Insights**:
- TOA: 0.10m median, solves every position; requires clock sync.
- TDOA: 0.07m median, clock-free, and slightly *better* than TOA on this array
  — its GDOP is 0.87 against TOA's 1.02, and both attain `GDOP x sigma_range`.
  The cost of TDOA is not accuracy on a good geometry; it is that the geometry
  has to be good, which the collinear variant below shows.
- AOA: 0.40m median with no failures, at the price of needing bearing hardware.

**Geometry matters differently for each.** On the collinear `poor_geometry`
variant, TOA and TDOA fail on all 100 positions while AOA still solves 92 of
them to a 0.26m median. Ranges from anchors on a line leave the position
ambiguous; bearings do not. That contrast is the reason the variant exists.

## Connection to Book Equations

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

### Issue 1: A Method Misses the Accuracy Its GDOP Predicts

**Symptoms**: one method reports errors orders of magnitude worse than
`sigma_position = GDOP x sigma_range`, while the others land on it.

**Likely Cause**: a convention mismatch between the stored measurements and the
solver reading them — a sign, a reference index, a frame, or degrees against
radians. It is almost never the geometry. GDOP already *is* the geometry, so a
result that misses the GDOP bound by two orders of magnitude is saying that
something outside the geometry is wrong.

This entry used to read "TDOA Positioning Fails or Returns Large Errors —
Symptoms: TDOA gives >10m errors while TOA gives <0.5m", and it prescribed
warm-starting TDOA from the TOA fix. That was a description of a real bug in
this dataset (the range differences were negated, see the TDOA section above)
with a cause that could not produce it and a remedy that did not help —
initialisation moves the median here by less than a millimetre.

**Solution**: compare against the bound before reaching for a workaround.

```python
import json
import numpy as np

cfg = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
sigma = cfg["measurements"]["tdoa_noise_std_m"]
predicted_accuracy = np.loadtxt(data_dir / "gdop_tdoa.txt").mean() * sigma

# The measurements should match the model the solver predicts, to within noise.
# TDOAPositioner(reference_idx=0) predicts d_j - d_ref, so that is what the
# file must contain -- not d_ref - d_j.
d_ref = np.linalg.norm(positions - beacons[0], axis=1)
predicted = np.array([
    np.linalg.norm(positions - beacons[j], axis=1) - d_ref
    for j in range(1, len(beacons))
]).T
residual = np.abs(np.loadtxt(data_dir / "tdoa_diffs.txt") - predicted).mean()

print(f"GDOP predicts {predicted_accuracy:.3f} m")
print(f"measurement residual {residual:.3f} m against {sigma} m of noise")
assert residual < 3 * sigma, "the stored measurements are not what the solver expects"
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
```py
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

