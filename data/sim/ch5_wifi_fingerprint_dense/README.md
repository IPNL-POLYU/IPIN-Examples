# ch5_wifi_fingerprint_dense

## Overview

The same 50 × 50 m three-floor building as
[`ch5_wifi_fingerprint_grid`](../ch5_wifi_fingerprint_grid/README.md), surveyed on
a **2 m** grid instead of 5 m. Same eight access points, same path-loss model,
same floors — 2,028 reference points instead of 363.

It exists to make one trade concrete. A denser survey reduces the *quantisation*
floor of nearest-neighbour matching, and costs survey effort in proportion to the
number of points:

| | sparse | grid | **dense** |
|---|---|---|---|
| Grid spacing | 10 m | 5 m | **2 m** |
| RPs per floor | 36 (6×6) | 121 (11×11) | **676 (26×26)** |
| RPs total (3 floors) | 108 | 363 | **2,028** |
| Survey effort | 1× | 3.4× | **19×** |
| Mean distance to nearest RP | 3.83 m | 1.93 m | **0.76 m** |

That row is measured, not estimated: 2,000 random positions per floor, **mean**
distance to the nearest reference point. It tracks `0.38 x spacing` to within
1% at all three densities (3.80 / 1.90 / 0.76 predicted).

**The mean is not the quantisation floor**, and the two used to share a name in
this file. The floor is the **rms** of the same distance, `sqrt(2 s^2 / 12)` =
`0.41 x spacing` — 4.08 / 2.04 / 0.82 m — and it is the one to compare an RMSE
against, which is what the experiment table further down does. The mean is the
cheaper thing to picture (how far you typically stand from the nearest surveyed
point) and the ratio between them is a constant 1.07, so no conclusion here
changes; only the label does.

The quantisation floor is the part worth understanding before running anything:
nearest-neighbour returns *a reference point*, so its error can never be smaller
than the average distance from a random position to the nearest grid node. On a
grid of spacing `s` that average is about `0.38 s`. **No amount of algorithm
work beats it** — only a denser survey does, which is what this dataset is.

## Scenario Description

Eight APs are distributed through a 50 × 50 m building of three floors, 3 m
apart vertically. At every reference point the survey records the RSS from all
eight, under a log-distance path-loss model with 4 dB of shadow fading and 15 dB
of attenuation per intervening floor.

Reference points are stacked: all three floors share the same 676 `(x, y)`
positions, and `floor_ids.npy` carries the floor. That matters for anything doing
floor classification — a nearest-neighbour search in `(x, y)` alone cannot
distinguish them, which is a defect the repository has hit before (see
`core.fingerprinting.hierarchical_localize`).

## Files and Data Structure

| File | Shape | Contents |
|---|---|---|
| `features.npy` | (2028, 8) | Mean RSS in dBm, one column per AP |
| `locations.npy` | (2028, 2) | Reference point `[x, y]` in metres |
| `floor_ids.npy` | (2028,) | Floor index, 0–2 |
| `metadata.json` | — | AP positions, grid spacing, path-loss model |

This family uses `metadata.json` where the rest of `data/sim` uses
`config.json`, but it records the same thing: the scenario parameters **and
the seed**, so it regenerates like every other dataset here.
`tests/test_datasets_reproduce_from_their_recipe.py` checks that, for all twenty.

## Loading Example

```python
import json
from pathlib import Path

import numpy as np

dense_dir = Path("data/sim/ch5_wifi_fingerprint_dense")

dense_features = np.load(dense_dir / "features.npy")
dense_locations = np.load(dense_dir / "locations.npy")
dense_floors = np.load(dense_dir / "floor_ids.npy")
dense_meta = json.load(open(dense_dir / "metadata.json"))

print(f"reference points: {len(dense_features)}")
print(f"APs:              {dense_features.shape[1]}")
print(f"RSS range:        {dense_features.min():.1f} to "
      f"{dense_features.max():.1f} dBm")
print(f"grid spacing:     {dense_meta['grid_spacing']} m")
print(f"floors:           {sorted(set(dense_floors.tolist()))}")
print(f"RPs per floor:    {np.bincount(dense_floors).tolist()}")
print(f"distinct (x, y):  {len(np.unique(dense_locations, axis=0))}")
```

Expected output:

```
reference points: 2028
APs:              8
RSS range:        -118.0 to -22.2 dBm
grid spacing:     2.0 m
floors:           [0, 1, 2]
RPs per floor:    [676, 676, 676]
distinct (x, y):  676
```

## Configuration Parameters

```python
print(f"area:           {dense_meta['area_size']} m")
print(f"floors:         {dense_meta['n_floors']} at "
      f"{dense_meta['floor_height']} m spacing")
for key, value in dense_meta["path_loss_model"].items():
    print(f"  {key}: {value}")
```

| Parameter | Value | Effect |
|---|---|---|
| `grid_spacing` | 2.0 m | The only difference from the siblings. Sets the NN quantisation floor at ~0.8 m |
| `path_loss_model.shadow_fading_std_dBm` | 4.0 | A property of the *location*, correlated over 8 m, so a denser grid samples it better rather than fighting it — see below |
| `path_loss_model.fast_fading_std_dBm` | 1.5 | Per-sample, uncorrelated, and therefore the term that actually stops density paying |
| `path_loss_model.path_loss_exponent` | 2.5 | Indoor typical. Higher means RSS falls faster, which helps discrimination |
| `path_loss_model.floor_attenuation_dB` | 15.0 | Large enough that floor classification is easy from RSS, and misleadingly so |
| `n_floors` | 3 | RPs are stacked at identical `(x, y)`, so floor must come from `floor_ids` |

**Density stops paying, but shadow fading is not the reason — and this README
used to say it was.** The claim rested on shadowing being a per-measurement
error term, which is what the generator used to make it. It is not: shadowing is
a property of the location, correlated over 8 m here, and a *denser grid samples
that field better*. What a denser survey cannot escape is the part of the
variability that is uncorrelated between visits — fast fading, at 1.5 dB.

Measured with nearest-neighbour matching, 300 queries on floor 0:

| Survey | Quantisation floor (rms) | Noiseless query | Realistic query |
|---|---|---|---|
| sparse, 10 m | 4.11 m | 4.99 m | 5.28 m |
| baseline, 5 m | 2.04 m | 3.07 m | 3.97 m |
| **dense, 2 m** | **0.82 m** | **2.51 m** | **3.42 m** |

Read the first and last columns against each other. Going from 5 m to 2 m cuts
the quantisation floor by a factor of 2.5 and the achieved error by 14%. The
grid is no longer what limits you.

**What does?** At 2 m the survey's own fast fading dominates: each reference
point was visited once, so its stored fingerprint is off by 1.5 dB in a
direction no denser grid can correct. The fix is a better survey rather than a
finer one — `ch5_wifi_fingerprint_multisamples` visits each point ten times, and
on the *baseline* 5 m grid that alone takes nearest neighbour from 3.07 m to
2.57 m against a 2.04 m floor. Ten visits at 5 m beats one visit at 2 m, for a
fifth of the reference points.

The old range-equivalence argument is still worth knowing, because it is correct
about *ranging* even though it was the wrong explanation here. Inverting the
log-distance model, `dd = dRSS x d x ln(10) / (10 n)`, so 4 dB at `n = 2.5` is
worth 0.74 m at 2 m from the AP, 3.68 m at 10 m, and 12.89 m at 35 m. That is
what shadowing costs a method that converts RSS to a *distance*. Fingerprinting
does not: it matches a pattern against a map that contains the same shadowing,
which is precisely why the field has to be in the map and in the query alike.

## Parameter Effects and Learning Experiments

| Parameter | Try | What to watch |
|---|---|---|
| Grid spacing | sparse (10 m), grid (5 m), dense (2 m) | Error should fall, then stop falling. Find where |
| `k` in k-NN | 1, 3, 5, 9 | Larger `k` interpolates between RPs, which partly substitutes for density |
| Shadow fading | regenerate at 1, 4, 8 dB | This sets the ceiling on what density can achieve |
| Method | NN, k-NN, MAP, posterior mean | NN is quantised to the grid; the interpolating methods are not |

**Compute the quantisation floor before measuring any algorithm**, so you know
what you are comparing against:

```python
rng = np.random.default_rng(42)
queries = rng.uniform(0.0, 50.0, size=(2000, 2))

floor0 = dense_locations[dense_floors == 0]
d = np.linalg.norm(queries[:, None, :] - floor0[None, :, :], axis=2)
nearest = d.min(axis=1)

print(f"grid spacing:            {dense_meta['grid_spacing']} m")
print(f"mean distance to the nearest RP: {nearest.mean():.3f} m")
print(f"0.38 x spacing (rule of thumb):  "
      f"{0.38 * dense_meta['grid_spacing']:.3f} m")
```

No nearest-neighbour localiser on this database can beat that mean, however good
its matching is, because it can only ever return one of those points.

## Visualization Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

f0 = dense_floors == 0
ax1.scatter(dense_locations[f0, 0], dense_locations[f0, 1],
            c=dense_features[f0, 0], cmap="plasma", s=12)
ap0 = dense_meta["ap_positions"][0]
ax1.plot(ap0[0], ap0[1], "w*", markersize=18, markeredgecolor="k",
         label="AP1")
ax1.set_xlabel("East [m]")
ax1.set_ylabel("North [m]")
ax1.set_title("Floor 0: RSS from AP1 across 676 RPs")
ax1.legend()
ax1.axis("equal")

ax2.hist(nearest, bins=40, color="steelblue", edgecolor="white")
ax2.axvline(nearest.mean(), color="crimson", linestyle="--",
            label=f"mean {nearest.mean():.2f} m")
ax2.set_xlabel("Distance to nearest RP [m]")
ax2.set_ylabel("Random query positions")
ax2.set_title("The floor NN cannot go below")
ax2.legend()

fig.tight_layout()
print("figure built")
```

## Connection to Book Equations

| Equations | What this dataset exercises |
|---|---|
| Eq. (5.1) | Nearest neighbour — the method whose error this dataset's density bounds |
| Eq. (5.2) | Weighted k-NN, which interpolates and so is not grid-quantised |
| Eq. (5.3) | Gaussian likelihood over reference points |
| Eq. (5.4) | MAP selection — also grid-quantised, which is why it often equals NN exactly |
| Eq. (5.5) | Posterior mean, the other way out of quantisation |

Implementations: `core.fingerprinting.deterministic` for Eqs. (5.1)–(5.2),
`core.fingerprinting.probabilistic` for Eqs. (5.3)–(5.5).

## Recommended Experiments

1. **Run the chapter comparison against all three densities** and plot RMSE
   against grid spacing. The curve should flatten; the flattening point is the
   answer to "how dense is dense enough", which is the practical question a
   deployment actually faces.

2. **Separate quantisation from matching error.** For each query, record both the
   error of the returned RP *and* the distance to the true nearest RP. The
   difference is matching error — the part an algorithm can fix. On this dataset
   it should dominate; on the sparse one it should not.

3. **Check that MAP and NN differ.** Eq. (5.4) selects a single reference point,
   as Eq. (5.1) does, and on a database like this they frequently pick the same
   one. If your MAP and NN numbers are *identical* rather than merely close, that
   is expected — but confirm it is for the right reason, by checking that the
   posterior's argmax really is the nearest fingerprint.

4. **Price the survey.** 19× the reference points of the sparse dataset buys
   exactly 5.05× on the quantisation floor (3.83 m → 0.76 m). It will buy *less*
   than that in achieved accuracy, because shadow fading does not shrink with the
   grid — measure how much less. Whether the remainder is worth 19× the survey
   effort is an engineering decision the numbers can inform but not make.

## Dataset Variants

- [`ch5_wifi_fingerprint_sparse`](../ch5_wifi_fingerprint_sparse/README.md) — 10 m grid, 108 RPs
- [`ch5_wifi_fingerprint_grid`](../ch5_wifi_fingerprint_grid/README.md) — 5 m grid, 363 RPs, the documented baseline

## Generation

```bash
python scripts/generate_ch5_wifi_fingerprint_dataset.py --grid-spacing 2.0 \
    --output data/sim/ch5_wifi_fingerprint_dense
```

`metadata.json` records the seed, so this reproduces the shipped
`features.npy` and `locations.npy` byte for byte. It was always reproducible;
until the seed was written down, nothing could say so.

## References

- Chapter 5, *Principles of Indoor Positioning and Indoor Navigation*
- `ch5_fingerprinting/README.md` — the chapter examples and their measured results
