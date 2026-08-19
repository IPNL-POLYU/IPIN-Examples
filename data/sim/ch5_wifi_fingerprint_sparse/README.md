# ch5_wifi_fingerprint_sparse

## Overview

The same 50 × 50 m three-floor building as
[`ch5_wifi_fingerprint_grid`](../ch5_wifi_fingerprint_grid/README.md), surveyed on
a **10 m** grid instead of 5 m: 108 reference points instead of 363.

This is the quick-deployment end of the survey trade-off, and it is the dataset
where **the survey grid, not the algorithm, sets the accuracy**:

| | **sparse** | grid | dense |
|---|---|---|---|
| Grid spacing | **10 m** | 5 m | 2 m |
| RPs per floor | **36 (6×6)** | 121 (11×11) | 676 (26×26) |
| RPs total (3 floors) | **108** | 363 | 2,028 |
| Survey effort | **1×** | 3.4× | 19× |
| NN quantisation floor | **3.83 m** | 1.93 m | 0.76 m |

The floor row is measured — 2,000 random positions per floor, mean distance to
the nearest reference point — and it tracks `0.38 x spacing` to within 1%.

**Nearest-neighbour on this database cannot do better than 3.83 m mean error,
however good the matching is**, because it returns one of 36 points per floor.
That makes this the dataset for demonstrating interpolation: k-NN (Eq. 5.2) and
the posterior mean (Eq. 5.5) return positions *between* reference points and are
not bound by that floor, and the gap between them and NN is widest here.

## Scenario Description

Eight APs in a 50 × 50 m building of three floors, 3 m apart vertically, with a
log-distance path-loss model, 4 dB of shadow fading and 15 dB per intervening
floor. Identical in every respect to the other two variants except the grid.

At 10 m spacing the survey is coarse relative to the shadow-fading correlation:
adjacent reference points differ enough in RSS that matching is easy, so the
error budget is dominated by *where the grid nodes happen to be* rather than by
picking the wrong node. That is the opposite regime from the dense variant, and
comparing the two is the point of shipping both.

Reference points are stacked: all three floors share the same 36 `(x, y)`
positions, with the floor in `floor_ids.npy`. A nearest-neighbour search in
`(x, y)` alone therefore cannot separate floors — a defect this repository has
been bitten by before, in `core.fingerprinting.hierarchical_localize`.

## Files and Data Structure

| File | Shape | Contents |
|---|---|---|
| `features.npy` | (108, 8) | Mean RSS in dBm, one column per AP |
| `locations.npy` | (108, 2) | Reference point `[x, y]` in metres |
| `floor_ids.npy` | (108,) | Floor index, 0–2 |
| `metadata.json` | — | AP positions, grid spacing, path-loss model |

This family uses `metadata.json` where the rest of `data/sim` uses
`config.json`. It records the scenario parameters **and the seed**, so it
regenerates like every other dataset here --
`tests/test_datasets_reproduce_from_their_recipe.py` checks that, for all twenty.

## Loading Example

```python
import json
from pathlib import Path

import numpy as np

sparse_dir = Path("data/sim/ch5_wifi_fingerprint_sparse")

sparse_features = np.load(sparse_dir / "features.npy")
sparse_locations = np.load(sparse_dir / "locations.npy")
sparse_floors = np.load(sparse_dir / "floor_ids.npy")
sparse_meta = json.load(open(sparse_dir / "metadata.json"))

print(f"reference points: {len(sparse_features)}")
print(f"APs:              {sparse_features.shape[1]}")
print(f"RSS range:        {sparse_features.min():.1f} to "
      f"{sparse_features.max():.1f} dBm")
print(f"grid spacing:     {sparse_meta['grid_spacing']} m")
print(f"RPs per floor:    {np.bincount(sparse_floors).tolist()}")
print(f"distinct (x, y):  {len(np.unique(sparse_locations, axis=0))}")
```

Expected output:

```
reference points: 108
APs:              8
RSS range:        -113.6 to -23.7 dBm
grid spacing:     10.0 m
RPs per floor:    [36, 36, 36]
distinct (x, y):  36
```

## Configuration Parameters

```python
print(f"area:           {sparse_meta['area_size']} m")
print(f"floors:         {sparse_meta['n_floors']} at "
      f"{sparse_meta['floor_height']} m spacing")
for key, value in sparse_meta["path_loss_model"].items():
    print(f"  {key}: {value}")
```

| Parameter | Value | Effect |
|---|---|---|
| `grid_spacing` | 10.0 m | The only difference from the siblings. Sets the NN floor at 3.83 m |
| `path_loss_model.shadow_fading_std_dBm` | 4.0 | Small relative to the RSS change between adjacent RPs at this spacing, so matching is easy |
| `path_loss_model.path_loss_exponent` | 2.5 | Indoor typical |
| `path_loss_model.floor_attenuation_dB` | 15.0 | Makes floor classification easy from RSS |
| `n_floors` | 3 | RPs stacked at identical `(x, y)`; floor comes from `floor_ids` |

## Parameter Effects and Learning Experiments

| Parameter | Try | What to watch |
|---|---|---|
| Method | NN vs k-NN vs posterior mean | The interpolating methods should beat NN by more here than on any other variant |
| `k` in k-NN | 1, 3, 5, 9 | With only 36 RPs per floor, `k = 9` is a quarter of the floor — watch it over-smooth |
| Grid spacing | 10 m, 5 m, 2 m | The other two variants. Plot error against spacing |
| Shadow fading | regenerate at 1, 4, 8 dB | At this spacing you have headroom before fading dominates |

**Measure the floor first, then measure a method against it:**

```python
rng = np.random.default_rng(42)
queries = rng.uniform(0.0, 50.0, size=(2000, 2))

floor0 = sparse_locations[sparse_floors == 0]
nearest = np.linalg.norm(
    queries[:, None, :] - floor0[None, :, :], axis=2
).min(axis=1)

print(f"grid spacing:                    {sparse_meta['grid_spacing']} m")
print(f"mean distance to the nearest RP: {nearest.mean():.3f} m")
print(f"0.38 x spacing (rule of thumb):  "
      f"{0.38 * sparse_meta['grid_spacing']:.3f} m")
print(f"90th percentile:                 "
      f"{np.percentile(nearest, 90):.3f} m")
```

If a reported NN error on this database is *below* that mean, something is wrong
with the evaluation — most likely the queries are drawn at reference points
rather than at random positions, which measures memorisation rather than
localisation.

## Visualization Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

f0 = sparse_floors == 0
ax1.scatter(queries[:, 0], queries[:, 1], c="lightgray", s=3,
            label="random queries")
ax1.plot(sparse_locations[f0, 0], sparse_locations[f0, 1], "bs",
         markersize=9, label="reference points")
for ap in sparse_meta["ap_positions"]:
    ax1.plot(ap[0], ap[1], "r*", markersize=13)
ax1.set_xlabel("East [m]")
ax1.set_ylabel("North [m]")
ax1.set_title("36 RPs per floor: NN can only answer with a blue square")
ax1.legend(fontsize=8)
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
| Eq. (5.1) | Nearest neighbour, bounded below by the grid at 3.83 m |
| Eq. (5.2) | Weighted k-NN — interpolates, so it is *not* bounded by the grid |
| Eq. (5.3) | Gaussian likelihood over reference points |
| Eq. (5.4) | MAP selection, which is grid-quantised like NN |
| Eq. (5.5) | Posterior mean, the probabilistic route out of quantisation |

Implementations: `core.fingerprinting.deterministic` and
`core.fingerprinting.probabilistic`.

## Recommended Experiments

1. **Show that interpolation beats selection.** Run NN, k-NN (k = 3) and the
   posterior mean on the same queries. NN and MAP are stuck at the grid floor;
   the other two are not. This dataset makes the difference largest, which is
   what it is for.

2. **Find where k stops helping.** With 36 RPs per floor, increasing `k` averages
   over an increasingly large area. Plot error against `k` from 1 to 15 and find
   the minimum — then check whether it lands where the average distance to the
   `k`-th neighbour reaches the scale of the building's RSS variation.

3. **Compare against the dense variant honestly.** Dense has 19× the reference
   points. Compute error per unit of survey effort, not just error, and decide
   which you would actually deploy.

4. **Test the floor classifier.** 15 dB per floor makes this easy, so verify the
   classifier's output *varies* before believing its accuracy — a constant
   predictor scores the base rate, 33.3% here, and that has been mistaken for a
   hard problem in this repository before.

## Dataset Variants

- [`ch5_wifi_fingerprint_grid`](../ch5_wifi_fingerprint_grid/README.md) — 5 m grid, 363 RPs, the documented baseline
- [`ch5_wifi_fingerprint_dense`](../ch5_wifi_fingerprint_dense/README.md) — 2 m grid, 2,028 RPs

## Generation

```bash
python scripts/generate_ch5_wifi_fingerprint_dataset.py --grid-spacing 10.0 \
    --output data/sim/ch5_wifi_fingerprint_sparse
```

`metadata.json` records the seed, so this reproduces the shipped
`features.npy` and `locations.npy` byte for byte.

## References

- Chapter 5, *Principles of Indoor Positioning and Indoor Navigation*
- `ch5_fingerprinting/README.md` — the chapter examples and their measured results
