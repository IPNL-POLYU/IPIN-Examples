# Ch4 RF 2D Positioning Dataset: Collinear Beacon Geometry

## Overview

Four beacons on a straight line, and the same 100-point grid as
`ch4_rf_2d_square`. This is the `poor_geometry` variant, but "poor" turns out
to mean something more specific than "everything is worse".

**Key Learning Objective**: A geometry can be perfectly well conditioned and
still be unusable. Collinear beacons make the range measurements ambiguous
under reflection about the beacon line — and DOP, which is a local measure,
cannot see that at all.

## Dataset Purpose

### Learning Goals

1. **Range measurements have a mirror twin.** Any position and its reflection
   across the beacon line produce identical ranges. TOA solves the ranges
   exactly and then cannot tell you which of the two you are at.
2. **DOP does not detect this.** TOA GDOP averages 1.43 here, against 1.02 for
   the square geometry — a metric saying this configuration is essentially
   fine, while half the TOA solutions are ~8.9 m wrong.
3. **Bearings break the symmetry.** Reflecting a position flips the sign of
   every azimuth, so AOA has no ambiguity and solves this geometry well.
4. **A degenerate starting point is its own failure.** The beacon centroid lies
   *on* the line of symmetry, where the range Jacobian has no across-line
   sensitivity, so TOA and TDOA cannot move off it.
5. **A median can hide a bimodal result.** TOA's overall median of 1.17 m sits
   in the empty gap between a 0.10 m cluster and an 8.92 m cluster.

### Implemented Equations

Same measurement models as the square variant:

- **Eqs. (4.1)-(4.3)**: TOA ranging
- **Eqs. (4.27)-(4.33)**: TDOA hyperbolic positioning
- **Eqs. (4.63)-(4.66)**: AOA azimuth model
- **Eq. (4.5)**: DOP from the geometry matrix

## Files

Identical layout to `ch4_rf_2d_square`:

- `beacons.txt`: Beacon positions [4×2] — all at y = 10 m
- `ground_truth_positions.txt`: Agent positions [100×2] (10×10 grid, 2–18 m)
- `toa_ranges.txt`, `tdoa_diffs.txt`, `aoa_angles.txt`: Measurements
- `gdop_toa.txt`, `gdop_tdoa.txt`, `gdop_aoa.txt`: Per-position DOP
- `config.json`: Parameters and measured performance

**Generate**:
```bash
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset poor_geometry
```

## Loading Data

```python
import numpy as np
from pathlib import Path

data_dir = Path("data/sim/ch4_rf_2d_linear")
beacons = np.loadtxt(data_dir / "beacons.txt")
positions = np.loadtxt(data_dir / "ground_truth_positions.txt")

print(f"Beacons:\n{beacons}")
print(f"All beacons on one line: {len(np.unique(beacons[:, 1])) == 1}")
print(f"Beacon centroid: {beacons.mean(axis=0)}")
print(f"Grid: {len(positions)} points, "
      f"x {positions[:, 0].min():.0f}-{positions[:, 0].max():.0f} m, "
      f"y {positions[:, 1].min():.0f}-{positions[:, 1].max():.0f} m")
print(f"Closest approach to the beacon line: "
      f"{np.abs(positions[:, 1] - 10.0).min():.2f} m")
```

## The reflection ambiguity

Ranges depend only on distance, so a target at `(x, 10 + h)` and its mirror at
`(x, 10 - h)` are exactly the same distance from every beacon on the line
`y = 10`. No amount of range data separates them.

The solver still converges — it just converges to whichever side it started
from. Seeded below the line, every one of the 100 solves ends up below it:

```python
from core.rf import TOAPositioner

toa_ranges = np.loadtxt(data_dir / "toa_ranges.txt")
LINE_Y = 10.0
seed = np.array([10.0, 3.0])  # below the beacon line

est = np.zeros_like(positions)
for i in range(len(positions)):
    est[i], _ = TOAPositioner(beacons, method="iwls").solve(
        toa_ranges[i], initial_guess=seed
    )

error = np.linalg.norm(est - positions, axis=1)
mirrored = positions.copy()
mirrored[:, 1] = 2 * LINE_Y - positions[:, 1]
error_to_mirror = np.linalg.norm(est - mirrored, axis=1)

same_side = np.sign(positions[:, 1] - LINE_Y) == np.sign(seed[1] - LINE_Y)
print(f"Ends on the seed's side of the line: "
      f"{np.sum(np.sign(est[:, 1] - LINE_Y) == np.sign(seed[1] - LINE_Y))}/100")
print(f"Targets on the seed's side:  median error {np.median(error[same_side]):.3f} m")
print(f"Targets on the other side:   median error {np.median(error[~same_side]):.3f} m")
print(f"Closer to the mirror than to the truth: "
      f"{np.sum(error_to_mirror < error)}/100")
```

**Expected**: 100/100 end on the seed's side. Targets on that side are solved
to a 0.104 m median — as good as the square geometry. Targets on the far side
come out 8.917 m off, because the solver returned their mirror image. Exactly
50 of 100 estimates are closer to the mirror than to the truth.

Note what this does to a summary statistic. The overall median is 1.17 m, which
is not a typical error for anything — the distribution is two tight clusters at
0.10 m and 8.92 m, and the median lands in the empty space between them. Report
the two modes, or report the fraction mirrored.

## Why DOP misses it

```python
gdop_toa = np.loadtxt(data_dir / "gdop_toa.txt")
distance_from_line = np.abs(positions[:, 1] - LINE_Y)

print(f"TOA GDOP: mean {gdop_toa.mean():.2f}, "
      f"min {gdop_toa.min():.2f}, max {gdop_toa.max():.2f}")
print("  (square geometry, for comparison: mean 1.02)")
print()
print("  |y - 10|     n   mean TOA GDOP")
for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]:
    band = (distance_from_line >= lo) & (distance_from_line < hi)
    print(f"  {lo:2d}-{hi:2d} m     {band.sum():3d}   {gdop_toa[band].mean():8.2f}")
```

TOA GDOP stays between 1.2 and 1.9 everywhere. That is not wrong: DOP is
computed from the linearised geometry matrix at a point, and *locally* this
configuration really does resolve position well. The ambiguity is global — two
separated minima of equal cost — and a first-order local measure has no way to
express it.

**So a healthy DOP is necessary, not sufficient.** Checking DOP would have
cleared this geometry for TOA.

## Measured performance

All solvers start from the beacon centroid `[10, 10]`, as in the square
variant. A solve counts as failed if it raised, reported `converged: False`,
never left the initial guess, or landed more than 100 m away.

| Metric | TOA | TDOA | AOA |
|--------|-----|------|-----|
| **Median error** | 6.77m | 6.77m | **0.26m** |
| **Failed to solve** | **100/100** | **100/100** | **8/100** |
| **Mean GDOP** | 1.43 | 10.36 | 9.25 |

The identical 6.77 m median for TOA and TDOA is the tell: it is the distance
from the centroid to the grid points, i.e. both solvers returned the seed
unchanged. The centroid sits on the line of symmetry, where moving across the
line changes no range to first order, so the range Jacobian is rank-deficient
in that direction and Gauss-Newton has nowhere to step.

AOA has no such problem. Reflecting a position flips the sign of every azimuth,
so the two candidate solutions are distinguishable and the centroid is an
ordinary starting point.

### Failure depends on where you start

```python
from core.rf import TDOAPositioner, AOAPositioner

tdoa_diffs = np.loadtxt(data_dir / "tdoa_diffs.txt")
aoa_angles = np.loadtxt(data_dir / "aoa_angles.txt")

def evaluate(solver, measurements, guess):
    """Median error and failure count from a given starting point."""
    estimates = np.zeros_like(positions)
    solved = np.zeros(len(positions), dtype=bool)
    for i in range(len(positions)):
        try:
            pos, info = solver.solve(measurements[i], initial_guess=guess)
        except Exception:
            estimates[i] = np.nan
            continue
        estimates[i] = pos
        stalled = np.linalg.norm(pos - guess) < 1e-6
        solved[i] = bool(info.get("converged", True)) and not stalled
    err = np.linalg.norm(estimates - positions, axis=1)
    finite = np.isfinite(err)
    failed = np.sum(~(solved & finite & (err < 100)))
    return np.median(err[finite]), failed

for label, guess in [("centroid [10, 10] (on the line)", beacons.mean(axis=0)),
                     ("off-line  [10,  3]", np.array([10.0, 3.0]))]:
    toa_med, toa_fail = evaluate(TOAPositioner(beacons, method="iwls"), toa_ranges, guess)
    aoa_med, aoa_fail = evaluate(AOAPositioner(beacons), aoa_angles, guess)
    print(f"{label}")
    print(f"   TOA median {toa_med:8.3f} m, failed {toa_fail:3d}/100")
    print(f"   AOA median {aoa_med:8.3f} m, failed {aoa_fail:3d}/100")
```

**Expected**: TOA goes from 100/100 failing on the line to 0/100 failing off
it — its problem is entirely the starting point, not the measurements. AOA goes
the other way, from 8 failures to 43: its own basin is best approached from the
middle of the array.

TDOA fails from every starting point tried. That is genuine: its GDOP averages
10.36 and reaches 111 near the line, against 1.43 for TOA on the same
geometry, because differencing collinear ranges leaves very flat hyperbolae.

## Comparison to the square variant

| | `ch4_rf_2d_square` | `ch4_rf_2d_linear` |
|---|---|---|
| TOA failed | 0/100 | 100/100 (from the centroid) |
| TOA median | 0.10m | 6.77m |
| TDOA failed | 11/100 | 100/100 |
| AOA failed | 0/100 | 8/100 |
| AOA median | 0.40m | 0.26m |
| TOA mean GDOP | 1.02 | 1.43 |

AOA is *better* here than on the square geometry (0.26 m against 0.40 m).
Spreading beacons along a line gives a wide baseline of well-separated
bearings, which is what an angle-based method wants. The lesson is not that
this geometry is bad, but that **geometry is method-specific**: the same
configuration is excellent for AOA, globally ambiguous for TOA, and genuinely
ill-conditioned for TDOA.

## Common Issues & Solutions

### Issue 1: TOA returns the initial guess unchanged

**Symptoms**: Every estimate equals the starting point; error equals the
distance from the start to the truth.

**Likely Cause**: The starting point lies on the beacon line, where the range
Jacobian is rank-deficient across the line.

**Solution**: Start off the line. Any prior about which side the target is on
also resolves the reflection:

```py
initial_guess = beacons.mean(axis=0) + np.array([0.0, 5.0])  # 5 m off the line
```

### Issue 2: Half the TOA estimates are mirrored

**Symptoms**: Errors cluster at two values, one near zero and one large; the
large ones are all on the opposite side of the beacon line.

**Likely Cause**: The reflection ambiguity. Both answers fit the ranges equally
well, so the solver returns the one in its basin.

**Solution**: This cannot be fixed with more range data or a better solver — it
is a property of the geometry. Add a measurement type that breaks the symmetry
(one bearing is enough), move a beacon off the line, or supply a side prior.

## Book Connection

### Chapter 4: RF Positioning

1. **DOP (Eq. 4.5)** — a local, first-order measure. This dataset is the
   counterexample to reading it as a sufficient quality check.
2. **TOA (Eqs. 4.1-4.3)** — the ranges are consistent and solvable; the
   difficulty is that they are consistent with two positions.
3. **TDOA (Eqs. 4.27-4.33)** — differencing collinear ranges flattens the
   hyperbolae and is genuinely ill-conditioned.
4. **AOA (Eqs. 4.63-4.66)** — bearings carry the side information that ranges
   do not.

**Formula**: a geometry is only "good" relative to a measurement type.

## Next Steps

1. Move one beacon off the line and watch the TOA ambiguity disappear
2. Fuse one AOA measurement with the TOA ranges to resolve the side
3. Compare against `ch4_rf_2d_optimal`, where the beacons surround the area
4. Check the DOP map against actual error for each method

## License

This dataset is part of the IPIN Book Examples repository. See repository LICENSE for details.

---

**Dataset Version**: 1.0
**Last Updated**: August 2026
**Contact**: See repository README for contact information
