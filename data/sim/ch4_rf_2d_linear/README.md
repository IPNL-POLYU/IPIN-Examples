# Ch4 RF 2D Positioning Dataset: Collinear Beacon Geometry

## Overview

Four beacons on a straight line, and the same 100-point grid as
`ch4_rf_2d_square`. This is the `poor_geometry` variant, but "poor" turns out
to mean something more specific than "everything is worse".

**Key Learning Objective**: A geometry can be perfectly well conditioned and
still be unusable. Collinear beacons make the range measurements ambiguous
under reflection about the beacon line — and DOP, which is a local measure,
cannot see that at all.

## Scenario Description

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

## Files and Data Structure

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

## Loading Example

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
    tdoa_med, tdoa_fail = evaluate(TDOAPositioner(beacons, reference_idx=0),
                                   tdoa_diffs, guess)
    aoa_med, aoa_fail = evaluate(AOAPositioner(beacons), aoa_angles, guess)
    print(f"{label}")
    print(f"   TOA median {toa_med:8.3f} m, failed {toa_fail:3d}/100")
    print(f"   TDOA median {tdoa_med:7.3f} m, failed {tdoa_fail:3d}/100")
    print(f"   AOA median {aoa_med:8.3f} m, failed {aoa_fail:3d}/100")
```

**Expected**: TOA goes from 100/100 failing on the line to 0/100 failing off
it — its problem is entirely the starting point, not the measurements. TDOA
goes from 100/100 to 17, and its median stays an order of magnitude worse than
TOA's (5.13 m against 1.17 m). AOA goes the other way, from 8 failures to 43:
its own basin is best approached from the middle of the array.

So the seed matters most, and the geometry still matters after it. TDOA's GDOP
here averages 10.36 and reaches 111 near the line, against 1.43 for TOA on the
same beacons, because differencing collinear ranges leaves very flat
hyperbolae — and that shows up as the residual gap once the seed is fixed.

> This paragraph used to say "TDOA fails from every starting point tried",
> which was true of the data it was written against: `tdoa_diffs.txt` shipped
> negated until the generator's argument order was corrected. Re-measured on
> the corrected file, TDOA solves 83 of 100 from `[10, 3]`. The claim about
> flat hyperbolae survives; the claim that nothing solves did not, which is why
> the loop above now evaluates all three methods rather than describing the
> third in prose.

## Comparison to the square variant

| | `ch4_rf_2d_square` | `ch4_rf_2d_linear` |
|---|---|---|
| TOA failed | 0/100 | 100/100 (from the centroid) |
| TOA median | 0.10m | 6.77m |
| TDOA failed | 0/100 | 100/100 |
| TDOA median | 0.07m | 6.77m |
| AOA failed | 0/100 | 8/100 |
| AOA median | 0.40m | 0.26m |
| TOA mean GDOP | 1.02 | 1.43 |
| TDOA mean GDOP | 0.87 | 10.36 |

TDOA is the sharpest contrast in the table: 0.07 m with no failures on the
square array, and nothing at all from the centroid here. Its GDOP moves by a
factor of twelve between the two geometries where TOA's moves by 1.4, which is
why it is the method this variant exists to break. Note that both columns are
measured from the beacon centroid, as every `config.json` is — the section
above shows how much of the collinear column is the seed rather than the
geometry.

AOA is *better* here than on the square geometry (0.26 m against 0.40 m).
Spreading beacons along a line gives a wide baseline of well-separated
bearings, which is what an angle-based method wants. The lesson is not that
this geometry is bad, but that **geometry is method-specific**: the same
configuration is excellent for AOA, globally ambiguous for TOA, and genuinely
ill-conditioned for TDOA.

## Configuration Parameters

```python
import json
import numpy as np

lin_dir = __import__("pathlib").Path("data/sim/ch4_rf_2d_linear")
lin_config = json.load(open(lin_dir / "config.json"))

print(f"preset:     {lin_config['preset']}")
print(f"geometry:   {lin_config['geometry']['type']}, "
      f"{lin_config['geometry']['num_beacons']} beacons")
print(f"TOA noise:  {lin_config['measurements']['toa_noise_std_m']} m")
print(f"AOA noise:  {lin_config['measurements']['aoa_noise_std_deg']} deg")
for lin_kind in ("toa", "tdoa", "aoa"):
    lin_d = lin_config["dop"][lin_kind]
    print(f"GDOP {lin_kind:<4}: mean {lin_d['mean']:7.3f}  "
          f"min {lin_d['min']:6.3f}  max {lin_d['max']:8.3f}")
```

Expected output:

```
preset:     poor_geometry
geometry:   linear, 4 beacons
TOA noise:  0.1 m
AOA noise:  2.0 deg
GDOP toa :  mean   1.426  min  1.016  max    3.603
GDOP tdoa:  mean  10.355  min  1.613  max  111.018
GDOP aoa :  mean   9.253  min  3.418  max   19.099
```

| Parameter | Value | Effect |
|---|---|---|
| `geometry.type` | `linear` | All four beacons on the line y = 10. This is the whole dataset |
| `geometry.num_beacons` | 4 | Adding beacons *on the same line* does not help; the ambiguity is the line, not the count |
| `measurements.toa_noise_std_m` | 0.1 | Multiply by GDOP for the error floor: 0.14 m typical for TOA, **11.1 m at the worst TDOA point** |
| `measurements.aoa_noise_std_deg` | 2.0 | AOA is the one measurement type that breaks the reflection symmetry |
| `nlos.enabled` | `false` | No bias. The difficulty here is geometric, not a corrupted measurement |

## Parameter Effects and Learning Experiments

| Parameter | Try | What to watch |
|---|---|---|
| Beacon layout | `poor_geometry`, `baseline`, `optimal` | TDOA GDOP goes 10.36 -> 0.87. TOA barely moves. **The measurement type decides how much geometry costs you** |
| Beacon spread along the line | widen or narrow | Widening improves DOP *along* the line and does nothing for the reflection. Geometry has two failure modes here and only one responds |
| Measurement type | TOA, TDOA, AOA | AOA has the best mean GDOP of the four ch4 datasets here (9.25), because a bearing distinguishes the two sides. TDOA has the worst |
| Add one off-line beacon | move beacon 3 to (10, 2) | The cheapest fix. One measurement off the line collapses the ambiguity |

**The spread, not the mean, is what a linear array does to you:**

```python
lin_gdop_tdoa = np.loadtxt(lin_dir / "gdop_tdoa.txt")

print(f"TDOA GDOP  mean {lin_gdop_tdoa.mean():.2f}   "
      f"median {np.median(lin_gdop_tdoa):.2f}   max {lin_gdop_tdoa.max():.2f}")
print(f"points above GDOP 20: {(lin_gdop_tdoa > 20).sum()} of {lin_gdop_tdoa.size}")
```

The mean is dragged by a few catastrophic points rather than describing a
typical one -- the same median-versus-mean gap that hides RSS aliasing in
Chapter 5. A single reported DOP for this geometry tells you almost nothing.

## Visualization Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lin_beacons = np.loadtxt(lin_dir / "beacons.txt")
lin_truth = np.loadtxt(lin_dir / "ground_truth_positions.txt")

lin_fig, (lin_ax1, lin_ax2) = plt.subplots(1, 2, figsize=(12, 5))

lin_sc = lin_ax1.scatter(lin_truth[:, 0], lin_truth[:, 1],
                         c=np.log10(lin_gdop_tdoa), cmap="magma", s=45)
lin_ax1.plot(lin_beacons[:, 0], lin_beacons[:, 1], "c^", markersize=13,
             label="beacons")
lin_ax1.axhline(10.0, color="cyan", linestyle="--", linewidth=1,
                label="beacon line (mirror axis)")
lin_fig.colorbar(lin_sc, ax=lin_ax1, label="log10 GDOP (TDOA)")
lin_ax1.set_xlabel("East [m]")
lin_ax1.set_ylabel("North [m]")
lin_ax1.set_title("GDOP explodes on the beacon line")
lin_ax1.legend(fontsize=8)
lin_ax1.axis("equal")

lin_ax2.hist(lin_gdop_tdoa, bins=40, color="indianred", edgecolor="white")
lin_ax2.axvline(lin_gdop_tdoa.mean(), color="black", linestyle="--",
                label=f"mean {lin_gdop_tdoa.mean():.1f}")
lin_ax2.axvline(np.median(lin_gdop_tdoa), color="steelblue", linestyle="--",
                label=f"median {np.median(lin_gdop_tdoa):.1f}")
lin_ax2.set_yscale("log")
lin_ax2.set_xlabel("GDOP (TDOA)")
lin_ax2.set_ylabel("Query points (log)")
lin_ax2.set_title("A long tail, not a shifted centre")
lin_ax2.legend()

lin_fig.tight_layout()
print("figure built")
```

Points near y = 10 sit on the mirror axis itself, where the two solutions
coincide and the geometry is at its most degenerate -- that is where the 111
comes from.

## Recommended Experiments

1. **Solve the whole grid and count how many land on the wrong side.** The
   reflection is not noise: a solver started on the wrong side converges
   confidently to a wrong answer. Compare against
   [`ch4_rf_2d_square`](../ch4_rf_2d_square/README.md) with identical noise.

2. **Predict before measuring.** `sigma_position = GDOP x sigma_range`, so the
   worst TDOA point should give about 111 x 0.1 = 11 m. Check whether a real
   solve reaches that, and whether the failures are where GDOP says they will
   be.

3. **Fix it with one beacon.** Move a single beacon off the line and regenerate.
   Watch which GDOP columns respond and which do not -- TDOA should collapse by
   an order of magnitude while TOA barely moves.

4. **Add the bearing.** AOA breaks the symmetry on its own. Compare an
   AOA-only solve against TDOA-only on the same points.

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

## Connection to Book Equations

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
