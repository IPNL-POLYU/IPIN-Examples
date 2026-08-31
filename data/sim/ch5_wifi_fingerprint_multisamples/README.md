# ch5_wifi_fingerprint_multisamples

## Overview

The same 50 × 50 m three-floor building and the same 5 m grid as
[`ch5_wifi_fingerprint_grid`](../ch5_wifi_fingerprint_grid/README.md) — the same
363 reference points, the same eight APs, the same shadowing field — surveyed
**ten times at every point** instead of once.

That is the only difference, and it is the one Chapter 5's probabilistic methods
need. `features.npy` is `(363, 10, 8)` here rather than `(363, 8)`.

**Why it exists.** Eq. (5.6) models `P(z | x_i)` as a Gaussian with a mean and a
standard deviation *per reference point and per AP*. A single-sample survey
supplies the mean and nothing else, so `fit_gaussian_naive_bayes` falls back to
its `min_std` argument and uses one σ everywhere. With a globally constant σ:

```
log P(z | x_i) = const - ||z - mu_i||^2 / (2 sigma^2)
```

which is monotone in Euclidean distance. `argmax` over it is `argmin` over
distance, so **MAP (Eq. 5.4) is not merely similar to 1-NN (Eq. 5.1), it is
arithmetically identical to it** — measured at 200 of 200 queries on the
single-sample grid. Chapter 5's comparison table showed `MAP` and
`NN (Euclidean)` scoring the same to the digit for exactly this reason, and
until this dataset existed there was no database in the book where they could
differ.

## Scenario Description

Eight APs in a 50 × 50 m building of three floors, 3 m apart vertically. RSS at
a point `p` from access point `ap` is

```
rss(p, ap) = P0 - 10 n log10(d) - floor_attenuation + S_ap(p) + fast
```

The last two terms are the reason this dataset can be surveyed repeatedly and
still say something:

| Term | Std | Varies between visits? |
|---|---|---|
| `S_ap(p)`, shadow fading | 4.0 dB | **No.** A property of the location — the same wall, the same spot, every visit. A spatially correlated Gaussian field, correlation length 8 m |
| `fast`, fast fading | 1.5 dB | **Yes.** Small-scale multipath, receiver noise, people moving. The only thing ten visits average over |

So the ten samples at one reference point differ by 1.5 dB, not by 5.7 dB, and
their mean is a better estimate of the radio map than any single visit:
`1.5 / sqrt(10)` = 0.47 dB of residual error instead of 1.5 dB.

## Files and Data Structure

| File | Shape | Contents |
|---|---|---|
| `features.npy` | (363, 10, 8) | RSS in dBm: **(RP, sample, AP)** |
| `locations.npy` | (363, 2) | Reference point `[x, y]` in metres |
| `floor_ids.npy` | (363,) | Floor index, 0–2 |
| `metadata.json` | — | AP positions, grid spacing, path-loss model, shadowing field |

The three-dimensional `features` is what `FingerprintDatabase` calls the
multi-sample format; `db.has_multiple_samples` is `True`, `db.get_mean_features()`
returns the `(363, 8)` mean map and `db.get_std_features()` the per-(RP, AP)
standard deviations. Every deterministic method here already goes through
`get_mean_features()`, so `nn_localize` and `knn_localize` work unchanged.

Like its siblings this family uses `metadata.json` rather than `config.json`, and
it records the seed, so it regenerates exactly —
`tests/test_datasets_reproduce_from_their_recipe.py` checks that.

## Loading Example

```python
import json
from pathlib import Path

import numpy as np

multi_dir = Path("data/sim/ch5_wifi_fingerprint_multisamples")

multi_features = np.load(multi_dir / "features.npy")
multi_locations = np.load(multi_dir / "locations.npy")
multi_floors = np.load(multi_dir / "floor_ids.npy")
multi_meta = json.load(open(multi_dir / "metadata.json"))

print(f"features shape:   {multi_features.shape}  (RP, sample, AP)")
print(f"reference points: {len(multi_locations)}")
print(f"samples per RP:   {multi_meta['n_samples_per_rp']}")
print(f"APs:              {multi_features.shape[2]}")
print(f"RSS range:        {multi_features.min():.1f} to "
      f"{multi_features.max():.1f} dBm")
print(f"grid spacing:     {multi_meta['grid_spacing']} m")
```

Expected output:

```
features shape:   (363, 10, 8)  (RP, sample, AP)
reference points: 363
samples per RP:   10
APs:              8
RSS range:        -119.2 to -17.3 dBm
grid spacing:     5.0 m
```

## The Repeat Visits Measure Fast Fading, and Only Fast Fading

This is the experiment the dataset is for. The spread *within* a reference point
must equal `fast_fading_std_dBm`, and nothing else:

```python
within = multi_features.std(axis=1, ddof=1)

print(f"declared fast fading:  "
      f"{multi_meta['path_loss_model']['fast_fading_std_dBm']} dB")
print(f"mean within-RP spread: {within.mean():.3f} dB")
print(f"declared shadowing:    "
      f"{multi_meta['path_loss_model']['shadow_fading_std_dBm']} dB")
```

The mean within-RP spread lands on the declared fast fading, not on the
shadowing. If it ever lands on the shadowing figure instead, the generator has
gone back to redrawing the field per sample — which is exactly the defect this
dataset was added alongside fixing, and the reason
`tests/ch5_fingerprinting/test_radio_map_is_smooth.py` exists.

Note the estimator's own scatter: with S = 10 the sample standard deviation of a
1.5 dB process has a spread of about `1.5 / sqrt(2 (S - 1))` = 0.35 dB, so the
per-(RP, AP) σ values run from roughly 0.5 to 3.0 dB. **That variation is
estimation noise, not structure** — this model gives every location the same true
fast-fading std. It is enough to make MAP stop being 1-NN; it is not enough to
make MAP *better* than 1-NN, and the chapter says so rather than implying
otherwise.

## Configuration Parameters

```python
print(f"area:           {multi_meta['area_size']} m")
print(f"floors:         {multi_meta['n_floors']} at "
      f"{multi_meta['floor_height']} m spacing")
for key, value in multi_meta["path_loss_model"].items():
    print(f"  {key}: {value}")
for key, value in multi_meta["shadow_field"].items():
    print(f"  shadow_field.{key}: {value}")
```

| Parameter | Value | Effect |
|---|---|---|
| `n_samples_per_rp` | 10 | The only difference from `ch5_wifi_fingerprint_grid`. Gives Eq. (5.6) a σ to estimate |
| `grid_spacing` | 5.0 m | Same as baseline, so the comparison isolates the survey depth |
| `path_loss_model.shadow_fading_std_dBm` | 4.0 | Location, not measurement. Identical across all ten visits |
| `path_loss_model.fast_fading_std_dBm` | 1.5 | The per-visit term. What ten visits average down by `sqrt(10)` |
| `shadow_field.decorrelation_length_m` | 8.0 | Above the 5 m grid, so the survey can represent the field |
| `shadow_field.seed` | 42 | Same seed and same building as the other three variants: one radio environment, four surveys of it |

## Parameter Effects and Learning Experiments

| Parameter | Try | What to watch |
|---|---|---|
| `min_std` in `fit_gaussian_naive_bayes` | 0.5, 1.5, 2.0, 2.5 | Below the 1.5 dB fast fading the estimated σ survives and MAP differs from 1-NN; above it, the floor erases σ and MAP collapses back onto 1-NN |
| Survey depth | this vs `ch5_wifi_fingerprint_grid` | Averaging ten visits takes NN from 3.07 m to 2.57 m against a 2.04 m quantisation floor |
| Method | NN, MAP, posterior mean | MAP can now disagree with NN. Check whether disagreeing makes it *better* before assuming it does |
| `n_samples_per_rp` | regenerate at 3, 10, 30 | The σ estimate's own scatter falls as `1 / sqrt(2 (S - 1))` |

Run `python -m ch5_fingerprinting.example_comparison` for the worked version of
the first two rows; its "What a repeat survey buys" section prints the sweep.

## Visualization Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: sigma estimated for AP1 at each reference point on floor 0. If the
# per-(RP, AP) sigma carried spatial structure -- if some corners of the
# building were genuinely noisier -- this map would show it. It does not.
f0 = multi_floors == 0
sc = ax1.scatter(multi_locations[f0, 0], multi_locations[f0, 1],
                 c=within[f0, 0], cmap="viridis", s=90, marker="s")
fig.colorbar(sc, ax=ax1, label="estimated sigma, AP1 [dB]")
ax1.set_xlabel("East [m]")
ax1.set_ylabel("North [m]")
ax1.set_title("Floor 0: sigma estimated from 10 visits")
ax1.axis("equal")

# Right: all 2904 estimates against the single value they are all estimating.
declared = multi_meta["path_loss_model"]["fast_fading_std_dBm"]
ax2.hist(within.ravel(), bins=40, color="steelblue", edgecolor="white")
ax2.axvline(declared, color="crimson", linestyle="--",
            label=f"true sigma = {declared} dB, everywhere")
ax2.set_xlabel("Estimated sigma [dB]")
ax2.set_ylabel("(RP, AP) pairs")
ax2.set_title("The spread is the estimator, not the building")
ax2.legend()

fig.tight_layout()
print(f"sigma estimates: {within.size} values, "
      f"{within.min():.2f} to {within.max():.2f} dB, "
      f"mean {within.mean():.2f}")
```

The right panel is the one to read carefully. The estimates run from about 0.5
to 3.0 dB, which looks like structure and is not: with `S = 10` the sample
standard deviation of a 1.5 dB process has a spread of `1.5 / sqrt(2 (S - 1))`
= 0.35 dB, and that alone accounts for the histogram's width. The left panel is
the check -- a genuinely noisier corner would appear there as a patch, and none
does.

## Connection to Book Equations

| Equations | What this dataset exercises |
|---|---|
| Eq. (5.1) | Nearest neighbour, on the ten-visit mean map |
| Eq. (5.4) | MAP -- the one database here where it is not identical to Eq. (5.1) |
| Eq. (5.5) | Posterior mean, over a posterior that spreads across more than one RP |
| Eq. (5.6) | Gaussian likelihood with mu **and sigma** estimated from the survey, as the book assumes |

## Recommended Experiments

1. **Confirm that MAP is no longer 1-NN, and that this is a property of the
   survey rather than of the method.** Fit `fit_gaussian_naive_bayes` on this
   database and on `ch5_wifi_fingerprint_grid`, and for each query compare
   `map_localize` against `nn_localize`. On the single-sample grid they agree on
   **every** query -- `model.sigma_is_constant` is `True` and the Gaussian
   log-likelihood is monotone in Euclidean distance, so Eq. (5.4) reduces to
   Eq. (5.1) exactly. Here they disagree on about 22%. Nothing about the
   estimator changed; only the survey did.

2. **Sweep `min_std` and watch the two properties you want trade against each
   other.** Below the 1.5 dB fast fading the estimated sigma survives and MAP
   diverges from 1-NN; above it the floor erases sigma and MAP collapses back
   onto 1-NN, while the posterior width becomes honest. The reason is that the
   sigma a repeat survey measures is the spread of repeat visits *at a reference
   point*, 1.5 dB, while the spread the likelihood needs is the disagreement
   between a query and the nearest reference point, 2.09 dB -- a query stands
   *between* reference points. No amount of resampling at the RP can see the
   difference. `python -m ch5_fingerprinting.example_comparison` prints the
   sweep.

3. **Then ask whether MAP is actually better, and be ready for the answer to be
   no.** It is not, on this grid, and on *this* dataset the reason is a single
   one: estimation noise. The example's sweep ends in an `oracle` row that keeps
   everything fixed and swaps the estimated sigma for the true constant these
   samples were drawn from, 1.5 dB. MAP then differs from 1-NN on **0 of 200**
   queries and ties it at 3.16 m -- a constant sigma of any value *is* 1-NN. So
   the entire penalty in the rows above is ten visits' worth of sigma estimate
   wobbling around a flat truth, and experiment 5 below is how you watch it
   shrink.

   What caps the upside is a different term, and it is the one to carry away:
   a query stands *between* reference points, so it disagrees with the nearest
   one by the radio map's change over that gap. Measured on the shipped
   surveys that is 0.54 dB at 2 m spacing, 1.43 dB at 5 m and 2.98 dB at 10 m
   -- at 5 m already comparable to the whole 1.5 dB fast-fading budget. So
   **Eq. (5.6) pays only when the variability it models dominates the
   variability it does not.**

   A stronger claim used to stand here: that the per-visit sigma is
   **anti-correlated** with that spatial term, `corr = -0.34`, so Naive Bayes
   up-weights the APs whose unmodelled error is worst. That describes a model
   in which `sigma_fast` grows as the signal weakens -- the path-loss gradient
   `d(pathloss)/dd = -10 n / (d ln 10)` is steepest where the signal is
   strongest -- and **this dataset is not that model.** Its fast fading is one
   constant for the whole building, and measured here the same correlation is
   `+0.0156`. Measure it yourself; a constant has nothing to anti-correlate
   with.

4. **Price the repeat survey against a denser one.** Ten visits per point costs
   10x the survey effort and takes nearest neighbour from 3.07 m to 2.57 m
   against a 2.04 m quantisation floor, because averaging cuts the map's own
   fast fading by `sqrt(10)`. Compare that against spending the effort on
   `ch5_wifi_fingerprint_dense` instead: 5.6x the reference points, visited
   once, for 2.51 m. Which is the better buy depends on whether your error
   budget is dominated by the grid or by the survey, and this pair of datasets
   is how you find out.

5. **Regenerate at `--n-samples` 3, 10 and 30** and watch the sigma estimate's
   own scatter fall as `1 / sqrt(2 (S - 1))`. This is the experiment that shows
   experiment 1's 22% for what it is: disagreement driven by estimation noise,
   which shrinks with S, rather than by structure, which would not.

## Dataset Variants

- [`ch5_wifi_fingerprint_grid`](../ch5_wifi_fingerprint_grid/README.md) -- the
  same 5 m grid and the same building, visited **once** per point. The direct
  comparison for everything above.
- [`ch5_wifi_fingerprint_dense`](../ch5_wifi_fingerprint_dense/README.md) --
  2 m grid, 2,028 RPs, single visit
- [`ch5_wifi_fingerprint_sparse`](../ch5_wifi_fingerprint_sparse/README.md) --
  10 m grid, 108 RPs, single visit

## Troubleshooting / Common Student Questions

### "The estimated sigma varies a lot -- is part of the building noisier?"

No, and this is the trap the Visualization Example is drawn to spring. Every
location in this model has the same true fast-fading std. The estimates vary
because a standard deviation from ten samples is itself a random variable, with
a spread of `1.5 / sqrt(2 (S - 1))` = 0.35 dB. Plot sigma against position
before reading anything into its range: real structure would show as a patch,
and there is none.

### "MAP disagrees with NN but scores worse. Have I made a mistake?"

Probably not -- that is the documented behaviour on this grid, and experiment 3
above explains why. Check the sign of your comparison, then read that entry
rather than tuning `min_std` until the table agrees.

### "Why is `features.npy` three-dimensional?"

`(M, S, N)` is the multi-sample format: reference point, visit, access point.
`db.get_mean_features()` gives the `(M, N)` mean map every deterministic method
uses, and `db.get_std_features()` the per-(RP, AP) standard deviations. Code
written against the single-sample databases keeps working, because
`nn_localize` and `knn_localize` already go through `get_mean_features()`.

## Generation

```bash
python -m scripts.generate_ch5_wifi_fingerprint_dataset --preset multisamples
```

`metadata.json` records the seed and every parameter the generator needs,
including the shadowing field, so this reproduces the shipped arrays --
`tests/test_datasets_reproduce_from_their_recipe.py` checks it along with the
other twenty datasets.

## References

- Chapter 5, *Principles of Indoor Positioning and Indoor Navigation*
- `ch5_fingerprinting/README.md` -- the chapter examples and their measured results
- `core/fingerprinting/probabilistic.py` -- `fit_gaussian_naive_bayes`, whose
  docstring carries the `min_std` discussion in full
