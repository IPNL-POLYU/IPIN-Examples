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

## Regenerating

```bash
python -m scripts.generate_ch5_wifi_fingerprint_dataset --preset multisamples
```

## Connection to Book Equations

| Equations | What this dataset exercises |
|---|---|
| Eq. (5.1) | Nearest neighbour, on the ten-visit mean map |
| Eq. (5.4) | MAP — the one database here where it is not identical to Eq. (5.1) |
| Eq. (5.5) | Posterior mean, over a posterior that spreads across more than one RP |
| Eq. (5.6) | Gaussian likelihood with μ **and σ** estimated from the survey, as the book assumes |
