# ch4_rf_2d_optimal

## Overview

Four beacons placed evenly on a circle, as the geometrically "optimal"
arrangement for the same 20 × 20 m floor used by
[`ch4_rf_2d_square`](../ch4_rf_2d_square/README.md).

**Read the numbers before assuming it wins.** Measured over the same 100 query
points, this geometry is indistinguishable from the square for TOA and slightly
*worse* for TDOA:

| Preset | Mean GDOP (TOA) | Mean GDOP (TDOA) | Mean GDOP (AOA) |
|---|---|---|---|
| `ch4_rf_2d_square` | 1.022 | 0.873 | 15.041 |
| **`ch4_rf_2d_optimal`** | **1.019** | **1.089** | **11.535** |
| `ch4_rf_2d_linear` | 1.426 | 10.355 | 9.253 |

That is the lesson this dataset actually teaches, and it is more useful than the
one its name promises: **a square already achieves near-unity GDOP across its
own interior, so there is nothing left for a cleverer arrangement to win.** The
0.3% TOA improvement is noise. Geometry optimisation pays when the geometry is
*bad* — compare the linear row, where TDOA GDOP is an order of magnitude worse.

## Scenario Description

Beacons sit on a circle of radius 10 m about the floor centre (10, 10), at the
four compass points:

```
        (10, 20)
           |
(0, 10) --- + --- (20, 10)
           |
        (10, 0)
```

100 query positions cover the interior on a grid from (2, 2) to (18, 18). Each
returns a TOA range, a TDOA range difference against beacon 0, and an AOA
azimuth, all with noise but no NLOS bias.

Against the square, whose beacons are at the corners, this arrangement moves
every beacon closer to the floor's interior. For TOA that changes the geometry
matrix very little. For AOA it helps a lot, and the mechanism is measurable:

| | mean beacon range | mean AOA GDOP |
|---|---|---|
| `ch4_rf_2d_square` | 15.049 m | 15.041 |
| `ch4_rf_2d_optimal` | 11.438 m | 11.535 |

Those columns are almost the same number, which is not a coincidence: an
angular error `dpsi` at range `r` displaces the fix by `r x dpsi`, so AOA dilution
scales with range while TOA dilution does not. Per query point the correlation
between mean beacon range and AOA GDOP is +0.87 on the square. **A circular
layout wins for AOA because it is closer, not because it is rounder.**

## Files and Data Structure

| File | Shape | Contents |
|---|---|---|
| `beacons.txt` | (4, 2) | Beacon positions `[x, y]` in metres |
| `ground_truth_positions.txt` | (100, 2) | Query positions `[x, y]` in metres |
| `toa_ranges.txt` | (100, 4) | Noisy range to each beacon, metres |
| `tdoa_diffs.txt` | (100, 3) | Range differences against beacon 0, metres |
| `aoa_angles.txt` | (100, 4) | Noisy azimuth to each beacon, radians |
| `gdop_toa.txt` | (100,) | GDOP at each query point for TOA |
| `gdop_tdoa.txt` | (100,) | GDOP for TDOA |
| `gdop_aoa.txt` | (100,) | GDOP for AOA |
| `config.json` | — | Generation parameters and seed |

The GDOP files are precomputed from the geometry alone — they contain no noise,
and are there so an experiment can compare a *predicted* error against a
measured one without re-deriving the geometry matrix.

## Loading Example

```python
import json
from pathlib import Path

import numpy as np

opt_dir = Path("data/sim/ch4_rf_2d_optimal")

opt_beacons = np.loadtxt(opt_dir / "beacons.txt")
opt_truth = np.loadtxt(opt_dir / "ground_truth_positions.txt")
opt_toa = np.loadtxt(opt_dir / "toa_ranges.txt")
opt_tdoa = np.loadtxt(opt_dir / "tdoa_diffs.txt")
opt_aoa = np.loadtxt(opt_dir / "aoa_angles.txt")
opt_gdop_toa = np.loadtxt(opt_dir / "gdop_toa.txt")
opt_config = json.load(open(opt_dir / "config.json"))

print(f"beacons:       {opt_beacons.tolist()}")
print(f"query points:  {len(opt_truth)}")
print(f"TOA ranges:    {opt_toa.shape}")
print(f"TDOA diffs:    {opt_tdoa.shape} (against beacon 0)")
print(f"AOA angles:    {opt_aoa.shape}")
print(f"GDOP (TOA):    mean {opt_gdop_toa.mean():.3f}, "
      f"{opt_gdop_toa.min():.3f} - {opt_gdop_toa.max():.3f}")
```

Expected output:

```
beacons:       [[20.0, 10.0], [10.0, 20.0], [0.0, 10.0], [10.0, 0.0]]
query points:  100
TOA ranges:    (100, 4)
TDOA diffs:    (100, 3) (against beacon 0)
AOA angles:    (100, 4)
GDOP (TOA):    mean 1.019, 1.000 - 1.069
```

## Configuration Parameters

```python
print(f"preset:        {opt_config['preset']}")
print(f"beacons:       {opt_config['geometry']['num_beacons']}")
print(f"TOA noise:     {opt_config['measurements']['toa_noise_std_m']} m")
print(f"TDOA noise:    {opt_config['measurements']['tdoa_noise_std_m']} m")
print(f"NLOS beacons:  {opt_config['nlos']['beacon_indices'] or 'none'}")
```

| Parameter | Value | Effect |
|---|---|---|
| `preset` | `optimal` | Beacons on a circle rather than at the corners |
| `measurements.toa_noise_std_m` | 0.1 | Sets the error floor: `sigma_position = GDOP x sigma_range`, so ~0.10 m here |
| `measurements.tdoa_noise_std_m` | 0.1 | TDOA differences are correlated through the shared reference beacon |
| `nlos.beacon_indices` | *(empty)* | No bias. For the biased variant see [`ch4_rf_2d_nlos`](../ch4_rf_2d_nlos/README.md) |

## Parameter Effects and Learning Experiments

| Parameter | Try | What to watch |
|---|---|---|
| Beacon layout | `optimal`, `baseline`, `poor_geometry` | GDOP barely moves between the first two, and jumps for the third. Geometry matters at the bad end, not the good end |
| `toa_noise_std_m` | 0.05, 0.1, 0.5 | Position error should scale linearly with it at fixed GDOP — that is Eq. (4.107) |
| Number of beacons | 3, 4, 6 | Three is the minimum for a 2-D fix with a clock; redundancy buys more than placement does here |
| Measurement type | TOA, TDOA, AOA | AOA is the one that prefers this layout, and the reason is range, not angle |

**Verify the DOP prediction rather than trusting it.** This is the experiment
worth doing, and the machinery is already in `core`:

```python
from core.rf.dop import compute_dop, compute_geometry_matrix

# Note the argument order: anchors first, then the position they are seen from.
opt_H = compute_geometry_matrix(opt_beacons, opt_truth[0], measurement_type="toa")
opt_dop = compute_dop(opt_H)
opt_sigma = opt_config["measurements"]["toa_noise_std_m"]

print(f"recomputed GDOP: {opt_dop['GDOP']:.6f}")
print(f"shipped GDOP:    {opt_gdop_toa[0]:.6f}")
print(f"sigma_range:     {opt_sigma} m")
print(f"predicted error: {opt_dop['GDOP'] * opt_sigma:.4f} m")
```

`VDOP` is `None` for a planar geometry — there is no vertical component to
dilute — so on this dataset `GDOP`, `PDOP` and `HDOP` are the same number.

## Visualization Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sc = ax1.scatter(opt_truth[:, 0], opt_truth[:, 1], c=opt_gdop_toa,
                 cmap="viridis", s=45)
ax1.plot(opt_beacons[:, 0], opt_beacons[:, 1], "r^", markersize=13,
         label="beacons")
fig.colorbar(sc, ax=ax1, label="GDOP (TOA)")
ax1.set_xlabel("East [m]")
ax1.set_ylabel("North [m]")
ax1.set_title("Circular layout: GDOP over the floor")
ax1.legend()
ax1.axis("equal")

ax2.hist(opt_gdop_toa, bins=20, color="steelblue", edgecolor="white")
ax2.axvline(opt_gdop_toa.mean(), color="crimson", linestyle="--",
            label=f"mean {opt_gdop_toa.mean():.3f}")
ax2.set_xlabel("GDOP (TOA)")
ax2.set_ylabel("Query points")
ax2.set_title("The spread is narrow -- there is little left to optimise")
ax2.legend()

fig.tight_layout()
print("figure built")
```

## Connection to Book Equations

| Equations | What this dataset exercises |
|---|---|
| Eqs. (4.1)–(4.3) | TOA range model, `r_i = ||x - b_i||` |
| Eqs. (4.27)–(4.33) | TDOA range differences against a reference beacon |
| Eqs. (4.63)–(4.66) | AOA azimuth model |
| Eq. (4.5) | Geometry matrix, from which DOP follows |
| Eq. (4.107) | `sigma_position = DOP x sigma_measurement` — the prediction to test |

Implementations: `core.rf.measurement_models` for the models,
`core.rf.dop.compute_geometry_matrix` and `compute_dop` for the geometry.

## Recommended Experiments

1. **Compare the three geometries on one plot.**

   ```bash
   python -m ch4_rf_point_positioning.example_comparison --compare-geometry
   ```

2. **Test the claim that "optimal" is optimal.** Solve all 100 points for both
   this dataset and `ch4_rf_2d_square`, and compare RMSE. If the difference is
   smaller than the spread across repeated noise draws, the two layouts are
   equivalent for TOA and the name is aspirational. Use enough draws to say so
   with a straight face — one draw is not a measurement.

3. **Find where a circle does help.** AOA mean GDOP is 11.5 here against 15.0
   for the square. Work out why from the geometry: AOA error grows with range,
   so plot mean beacon range per query point for both layouts and correlate it
   with the AOA GDOP.

4. **Break it.** Move one beacon onto the line joining two others and recompute
   GDOP. Collinearity is what actually destroys a solution, which is the subject
   of [`ch4_rf_2d_linear`](../ch4_rf_2d_linear/README.md).

## Generation

```bash
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset optimal
```

The `baseline`, `poor_geometry` and `nlos` presets produce the three sibling
datasets. All use the same seed, so differences between them are geometry and
bias, not noise.

## References

- Chapter 4 and Section 4.5, *Principles of Indoor Positioning and Indoor Navigation*
- [`ch4_rf_2d_square`](../ch4_rf_2d_square/README.md) — the corner-placed baseline
- [`ch4_rf_2d_linear`](../ch4_rf_2d_linear/README.md) — what bad geometry looks like
- [`ch4_rf_2d_nlos`](../ch4_rf_2d_nlos/README.md) — same geometry as the baseline, with bias
