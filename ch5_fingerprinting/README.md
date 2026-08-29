# Chapter 5: Fingerprinting-based Indoor Positioning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch5_fingerprinting.ipynb)

Run this chapter in your browser — every figure below is one you can
regenerate and change. No install: [`notebooks/ch5_fingerprinting.ipynb`](../notebooks/ch5_fingerprinting.ipynb)

## Overview

This module implements fingerprinting-based positioning algorithms described in **Chapter 5** of *Principles of Indoor Positioning and Indoor Navigation*. Fingerprinting is a pattern-matching approach that compares measured radio signal strengths (RSS) against a pre-built database of reference fingerprints.

The module provides four main categories of methods:
- **Deterministic methods** (nearest-neighbor, k-nearest-neighbor)
- **Probabilistic methods** (Bayesian inference, MAP, posterior mean)
- **Pattern recognition - Regression** (linear regression)
- **Pattern recognition - Classification** (Random Forest, SVM, hierarchical coarse-to-fine)

**Key Features:**
- ✅ Multi-floor support with floor constraints
- ✅ Multi-sample database format for variance estimation
- ✅ Top-k posterior mean optimization (2.86x speedup)
- ✅ **Missing AP support** - handles signal dropout gracefully (NaN-based)

## Quick Start

```bash
# Run individual examples (use ch5_wifi_fingerprint_grid by default)
python -m ch5_fingerprinting.example_deterministic
python -m ch5_fingerprinting.example_probabilistic
python -m ch5_fingerprinting.example_pattern_recognition
python -m ch5_fingerprinting.example_classification

# Run comprehensive comparison of all methods
python -m ch5_fingerprinting.example_comparison

# Walk a user across the floor; watch the posterior aliasing (add --animate)
python -m ch5_fingerprinting.example_walk_posterior
```

## The posterior along a walk, and how it fails (Section 5.2)

| Figure | Built by | Size |
|--------|----------|------|
| `ch5_walk_posterior.{svg,pdf,png}` | `example_walk_posterior.py` | — |
| `ch5_walk_posterior.gif` | `example_walk_posterior.py --animate` | 0.30 MB |

![The posterior over a walk, and where it goes wrong](figs/ch5_walk_posterior.svg)

The intuitive animation to reach for is "watch the posterior sharpen as the
user walks." There is nothing to sharpen. With eight access points over this
grid the Gaussian Naive-Bayes posterior of Eq. (5.3) is a near-delta
everywhere — entropy 0.18 against a maximum of 4.80, peak probability
0.63–1.00 with a median of 0.99. It is already as sharp as it gets.

What *is* dynamic, and specific to fingerprinting, is how that sharp posterior
**fails**. It does not spread out and hedge under noise; it stays a confident
spike and occasionally puts the spike in the wrong place — a distant reference
point whose stored radio signature happens to resemble the current one. This is
**RSS aliasing**, and it is the characteristic failure of fingerprinting.

Measured over a 21-step L-walk, aliasing jumps beyond 10 m occur:

| Measurement noise | aliasing jumps | median error | mean error |
|---|---|---|---|
| 1 dB | 0 / 21 | 0.0 m | 0.0 m |
| 3 dB | 0 / 21 | 0.0 m | 1.4 m |
| 6 dB | **4 / 21** | **0.0 m** | **6.0 m** |

The last row is the lesson. The MAP estimate is usually exactly right — the
median stays 0 m — but at four steps it teleports to a radio-similar location up
to 35 m away, and those few jumps drag the mean to 6.0 m. **A reported mean
error hides the failure completely**; only the walk, or the median-vs-mean gap,
makes it visible. The animation shows the hot spot tracking the user, then
leaping across the floor and snapping back.

## 📂 Dataset Connection

| Example Script | Dataset | Description |
|----------------|---------|-------------|
| All examples | `data/sim/ch5_wifi_fingerprint_grid/` | Standard 5m grid, 121 RPs (default) |
| All examples | `data/sim/ch5_wifi_fingerprint_dense/` | Dense 2m grid, 676 RPs (higher accuracy) |
| All examples | `data/sim/ch5_wifi_fingerprint_sparse/` | Sparse 10m grid, 36 RPs per floor (quick deployment) |
| `example_comparison.py` | `data/sim/ch5_wifi_fingerprint_multisamples/` | Same building and 5m grid, surveyed 10x per point. The only database here where Eq. (5.6) has a sigma to estimate |

> **Note**: To use a different dataset density, edit the `db_path` variable in the example scripts.

## Usage Examples

### Nearest-Neighbor Positioning (Eq. 5.1)

```python
import numpy as np
from pathlib import Path
from core.fingerprinting import load_fingerprint_database, nn_localize

# Load database
db = load_fingerprint_database(Path("data/sim/ch5_wifi_fingerprint_grid"))

# Query fingerprint (8 RSS values from 8 APs)
query = np.array([-45, -60, -75, -80, -50, -70, -85, -90])

# Nearest-neighbor localization
pos = nn_localize(query, db, metric="euclidean", floor_id=0)
print(f"Estimated position: {pos}")
```

### k-Nearest-Neighbor (Eq. 5.2)

```python
from core.fingerprinting import knn_localize

pos = knn_localize(query, db, k=3, metric="euclidean", 
                   weighting="inverse_distance", floor_id=0)
print(f"k-NN estimate: {pos}")
```

### Bayesian MAP and Posterior Mean (Eqs. 5.3-5.5)

```python
from core.fingerprinting import fit_gaussian_naive_bayes, map_localize, posterior_mean_localize

# Fit Bayesian model (uses Gaussian likelihood Eq. 5.6)
model = fit_gaussian_naive_bayes(db, min_std=2.0)

# MAP estimate (Eq. 5.4): discrete, selects best RP
pos_map = map_localize(query, model, floor_id=0)

# Posterior mean (Eq. 5.5): continuous, weighted average over all RPs
pos_mean = posterior_mean_localize(query, model, floor_id=0)

# Top-k posterior mean: faster, typically sufficient (book guidance)
# Uses only top-k highest posterior candidates, renormalized
pos_mean_topk = posterior_mean_localize(query, model, floor_id=0, top_k=10)
```

### Classification-Based Positioning

```python
from core.fingerprinting import fit_classifier, hierarchical_localize

# Train Random Forest classifier (each RP is a class)
classifier = fit_classifier(
    db,
    classifier_type="random_forest",
    zone_type="rp",
    n_estimators=100
)

# Direct classification
pos_rf, info = classifier.predict(query)
print(f"Random Forest estimate: {pos_rf}")
print(f"Predicted class (RP): {info['predicted_class']}")

# Train SVM classifier
classifier_svm = fit_classifier(
    db,
    classifier_type="svm",
    zone_type="rp",
    kernel="rbf"
)
pos_svm, info = classifier_svm.predict(query)
```

### Hierarchical Localization (Coarse -> Fine)

```python
# Two-step hierarchical: classify floor first, then k-NN within that floor
pos_hier, info = hierarchical_localize(
    query,
    db,
    coarse_method="floor",       # Coarse: floor classification
    fine_method="knn",            # Fine: k-NN within floor
    k=5
)
print(f"Hierarchical estimate: {pos_hier}")
print(f"Classified floor: {info['coarse_floor']}")

# Alternative fine methods:
# - "nn": Nearest neighbor
# - "map": Maximum a posteriori (Bayesian)
# - "posterior_mean": Posterior mean (Bayesian)

# Alternative coarse methods:
# - "floor": Simple floor classification (default)
# - "random_forest": RF-based region classification

# Example: RF coarse + MAP fine
pos_hier_rf, info = hierarchical_localize(
    query,
    db,
    coarse_method="random_forest",
    fine_method="map"
)
```

## Expected Output

### Deterministic Methods Example

Running `python -m ch5_fingerprinting.example_deterministic` produces:

The `Time (ms)` column is shown as `~`: it is whatever the machine that last
ran this happened to manage, not a property of the method.

<!-- example-output: ch5_fingerprinting.example_deterministic -->
```
RESULTS SUMMARY
======================================================================
Method                    RMSE (m)     Median (m)   90th % (m)   Time (ms)
----------------------------------------------------------------------
NN (Euclidean)            3.80         2.52         5.30         ~
NN (Manhattan)            3.97         2.78         5.45         ~
k-NN (k=3, inv-dist)      3.26         2.06         4.95         ~
k-NN (k=5, inv-dist)      3.37         2.41         5.27         ~
k-NN (k=7, inv-dist)      3.62         2.81         5.72         ~
k-NN (k=5, uniform)       3.71         2.79         6.01         ~
```

**Visual Output:**

![Deterministic Positioning](figs/deterministic_positioning.png)

*This figure shows six subplots:*
- **Top-Left:** Reference points (blue) and test queries (red X) across the 50m x 50m area
- **Top-Center:** CDF of positioning errors comparing NN and k-NN variants
- **Top-Right:** Error distribution histogram
- **Bottom-Left:** Box plots comparing error distributions
- **Bottom-Center:** Effect of k on k-NN performance (optimal k around 5)
- **Bottom-Right:** Speed vs accuracy trade-off

### Comprehensive Comparison

Running `python -m ch5_fingerprinting.example_comparison` generates:

The example runs three noise scenarios; this is the baseline one.

<!-- example-output: ch5_fingerprinting.example_comparison -->
```
Baseline:
Method               Category             RMSE (m)     Median (m)   P90 (m)      Time (ms)
------------------------------------------------------------------------------------------
NN (Euclidean)       Deterministic        4.32         2.82         6.24         ~
k-NN (k=3)           Deterministic        3.24         2.36         4.96         ~
MAP                  Probabilistic        4.32         2.82         6.24         ~
Posterior Mean       Probabilistic        3.50         2.35         5.18         ~
Post.Mean (k=10)     Probabilistic        3.50         2.35         5.18         ~
Linear Regression    Pattern Recognition  6.13         5.22         9.16         ~
```

`MAP` and `NN (Euclidean)` agree exactly, and `Post.Mean (k=10)` agrees with the
full posterior mean. Neither is a coincidence and neither is a bug:

- **MAP is 1-NN**, arithmetically, on any single-sample database. There is no
  second sample to take a standard deviation of, so `fit_gaussian_naive_bayes`
  uses one σ everywhere, and with a constant σ the Gaussian log-likelihood is
  `const - ||z - mu_i||² / (2 σ²)` — monotone in Euclidean distance, so its
  `argmax` is the `argmin` of Eq. (5.1). Ship a survey that visits each point
  once and Eq. (5.4) cannot tell you anything Eq. (5.1) did not.
- **The posterior concentrates**, so truncating it to the top 10 changes
  nothing.

The example measures both, in its **"What a repeat survey buys"** section, using
`ch5_wifi_fingerprint_multisamples` — the same building and grid, surveyed ten
times per point instead of once. There the σ is estimated rather than assumed
and MAP does diverge from 1-NN, on 22% of queries.

**Whether diverging makes it better is a separate question, and on this grid the
answer is no.** That is a measurement, not a shortcoming of the model: making
the fast fading genuinely vary with signal strength — the physically standard
choice, RSS variance growing as SNR falls — makes MAP monotonically *worse*, and
so does substituting the true σ instead of the estimated one. The error that
decides the match is the radio map changing between the query and the nearest
reference point (1.43 dB rms here, comparable to the whole 1.5 dB fast-fading
budget), and it is **anti-correlated** with the noise a repeat survey can
measure, because the path-loss gradient is steepest exactly where the signal is
strongest. Weighting by the noise you can measure up-weights the APs whose
unmodelled error is worst.

Measured across the three shipped surveys, the penalty tracks that term
exactly — +0.04 m on the 2 m grid, +0.38 m at 5 m, +1.46 m at 10 m. So
Eq. (5.6) pays when the variability it models dominates the variability it does
not; on a 5 m grid with 8 APs it does not. The derivation is in the section's
docstring.

**Visual Output:**

![Fingerprinting Comparison](figs/comparison_all_methods.png)

*This comprehensive figure shows nine subplots comparing all fingerprinting methods:*
- **Top-Left:** RMSE across noise scenarios (baseline, moderate, high)
- **Top-Center:** Error CDF showing accuracy distribution
- **Top-Right:** Speed comparison (Linear Regression 30x faster)
- **Middle-Left:** Error distribution box plots
- **Middle-Center:** RMSE vs noise level (robustness analysis)
- **Middle-Right:** Speed vs accuracy trade-off
- **Bottom-Left:** Performance by category (Deterministic vs Probabilistic vs Pattern Recognition)
- **Bottom-Center:** Median vs P90 errors
- **Bottom-Right:** Radar chart of normalized performance metrics

### Classification and Hierarchical Methods Example

Running `python -m ch5_fingerprinting.example_classification` produces:

<!-- example-output: ch5_fingerprinting.example_classification -->
```
Test 1: Classification Accuracy
======================================================================
--- Training classifiers ---
  1. Random Forest (n_estimators=100)
  2. SVM (RBF kernel)
--- Recall on the training vectors (memorisation check) ---
  Random Forest: 100.0% (363/363)
  SVM:           100.0% (363/363)
  Expected to be 100% -- these are the training vectors themselves,
  one per class. This says the models fit; it says nothing about
  how they generalise, and is not a positioning result.
--- Held-out queries (sigma = 2.0 dBm, n = 200) ---
  Random Forest: 68.0% (136/200) exact RP
  SVM:           91.0% (182/200) exact RP
  This is the number to compare against other methods.
```

Classification reports **exact-RP accuracy**, not RMSE in metres, so these
numbers are not directly comparable with the tables above — a classifier that
picks a neighbouring reference point scores 0 here and under a metre there.
Note also that the 100% training recall is a memorisation check, printed
precisely so it is not mistaken for a positioning result.

**These two numbers went *down* when the radio map was fixed** — Random Forest
from 83% to 68%, SVM from 97% to 91% — while every positioning error in the
tables above roughly halved. Both moved for the same reason, and it is worth
sitting with:

The map used to redraw shadow fading independently at every reference point, so
each RP carried a distinctive random vector. Adjacent RPs 5 m apart differed by
**16.83 dB** in fingerprint space; they now differ by **11.23 dB**, because a
radio map that is a smooth function of position makes nearby places *look
alike*. That is harder for a classifier asked to name the exact reference point,
and easier for anything asked where you are. The old accuracy was measuring how
scrambled the map was.

So: exact-RP accuracy rewards distinctiveness, and positioning rewards spatial
structure. A dataset can be made to score well on the first by being wrong.

**Visual Output 1 - Noise Robustness:**

![Classification Noise Robustness](figs/classification_noise_robustness.png)

*This figure compares the robustness of classification-based methods (Random Forest, SVM) against k-NN as measurement noise increases:*
- **X-axis:** Noise standard deviation in dBm (1, 2, 3, 4, 5 dBm)
- **Y-axis:** Mean positioning error in meters
- **Key insight:** k-NN generally outperforms classifiers due to its weighted averaging, while RF shows better stability than SVM at high noise levels

**Visual Output 2 - Hierarchical Localization:**

![Hierarchical Localization](figs/hierarchical_localization.png)

*This figure shows four subplots evaluating hierarchical (coarse-to-fine) localization strategies:*
- **Top-Left (CDF):** Cumulative distribution of positioning errors comparing direct k-NN vs hierarchical approaches
- **Top-Right (Box Plot):** Error distribution quartiles for each method
- **Bottom-Left (RMSE):** Bar chart comparing RMSE across methods
- **Bottom-Right (Floor Accuracy):** Floor classification accuracy for hierarchical methods (typically >95%)

*Hierarchical approaches first classify the floor/region, then apply fine-grained localization only within that region. This reduces computational cost and can improve accuracy in multi-floor buildings.*

## Performance Summary

| Method | Category | RMSE | Speed | Best For |
|--------|----------|------|-------|----------|
| **NN** | Deterministic | ~5.2m | Fast | Dense reference points, simplicity |
| **k-NN (k=3)** | Deterministic | ~4.2m | Fast | Best accuracy-speed balance |
| **MAP** | Probabilistic | ~5.2m | Slow | Probabilistic interpretation needed |
| **Posterior Mean** | Probabilistic | ~4.3m | Slow | High accuracy, noisy environments |
| **Linear Regression** | Pattern Recog. | ~5.0m | Fastest | Real-time applications, sparse data |
| **Random Forest** | Classification | ~4.9m | Medium | Multi-class zones, feature importance |
| **SVM** | Classification | ~5.1m | Medium | Nonlinear boundaries |
| **Hierarchical** | Multi-stage | ~4.0m | Fast | Multi-floor buildings, large environments |

## Preprocessing Features

The module provides preprocessing utilities to improve localization robustness, as discussed in Chapter 5.

### 1. Scan Averaging

Average multiple RSS scans to reduce measurement noise (mitigates short-term fading).

**Methods:**
- `mean`: Arithmetic mean (optimal for Gaussian noise)
- `median`: Robust to outliers  
- `trimmed_mean`: Balance between robustness and efficiency

**Example:**
```python
from core.fingerprinting import average_scans
import numpy as np

# Collect multiple scans at same location
scans = np.array([
    [-50, -60, -70],  # Scan 1
    [-52, -58, -72],  # Scan 2  
    [-48, -62, -68],  # Scan 3
])

# Average scans (reduces noise by ~sqrt(N))
query = average_scans(scans, method="mean")
print(query)  # [-50, -60, -70]
```

### 2. Normalization

Normalize features to mitigate device calibration differences.

**Methods:**
- `zscore`: Standardize to zero mean, unit variance (handles systematic offsets)
- `minmax`: Scale to [0, 1] range
- `none`: No normalization

**Example:**
```python
from core.fingerprinting import (
    normalize_fingerprint,
    compute_normalization_params,
)

# Compute normalization params from database
db_features = db.get_mean_features()  # (M, N)
norm_params = compute_normalization_params(db_features, method="zscore")

# Normalize query using database statistics
query = np.array([-55, -65, -75, -85, -70, -60, -80, -90])  # +5 dBm offset
#                 8 values, one per AP in the shipped database
query_norm, _ = normalize_fingerprint(
    query,
    method="zscore",
    ref_mean=norm_params["mean"],
    ref_std=norm_params["std"]
)
```

### 3. Complete Preprocessing Pipeline

Combine averaging + normalization:

```python
from core.fingerprinting import preprocess_query

# Preprocess query: average scans + normalize
# Three scans of all 8 APs, as a phone would collect them.
scans = np.array([
    [-50, -60, -70, -80, -65, -55, -75, -85],
    [-52, -58, -72, -78, -67, -53, -77, -83],
    [-48, -62, -68, -82, -63, -57, -73, -87],
])
query_preprocessed, info = preprocess_query(
    scans,
    averaging_method="mean",
    normalization_method="zscore",
    ref_mean=norm_params["mean"],
    ref_std=norm_params["std"]
)

# Use in localization
pos = nn_localize(query_preprocessed, db, floor_id=0)
```

**Benefits:**
- **Averaging:** Reduces noise variance by ~sqrt(N) for N scans
- **Normalization:** Handles ±5 dBm device offsets (cross-device positioning)

## Missing AP Support (Signal Dropout)

The system gracefully handles missing AP readings (signal dropout), a common real-world scenario where some access points are not detectable.

**Representation:** Missing AP readings are represented as `np.nan` in fingerprint vectors.

**Behavior:**
- **Deterministic methods:** Distance computed only over overlapping (non-NaN) dimensions
- **Probabilistic methods:** Likelihood summed only over observed features
- **No overlap:** Distance = +inf, log-likelihood = -inf (RP excluded)

**Example:**
```python
from core.fingerprinting import nn_localize, fit_gaussian_naive_bayes, map_localize
import numpy as np

# Query with some APs missing (NaN)
query = np.array([-51.0, np.nan, -71.0, -81.0, np.nan, -65.0, -75.0, -85.0])
#                 AP1    AP2     AP3    AP4    AP5    AP6    AP7    AP8
#                 ✓      X       ✓      ✓      X      ✓      ✓      ✓

# Deterministic methods work with missing values
pos_nn = nn_localize(query, db, floor_id=0)  # Uses AP1, AP3, AP4, AP6

# Probabilistic methods work with missing values
model = fit_gaussian_naive_bayes(db, min_std=2.0)
pos_map = map_localize(query, model, floor_id=0)  # Likelihood from AP1, AP3, AP4, AP6
```

**Tested:** Up to 50% dropout rate, 100 queries, no crashes ✓

## Equation Reference

### Deterministic Fingerprinting

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `nn_localize()` | `core/fingerprinting/deterministic.py` | Eq. (5.1) | NN: i* = argmin_i D(z, f_i), x = x_{i*} |
| `knn_localize()` | `core/fingerprinting/deterministic.py` | Eq. (5.2) | k-NN: x = sum(w_i * x_i) / sum(w_i) |

### Probabilistic Fingerprinting (Bayesian)

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `log_likelihood()` | `core/fingerprinting/probabilistic.py` | Eq. (5.6) | Likelihood P(z\|x_i) using Gaussian model (term in Eq. 5.3) |
| `log_posterior()` | `core/fingerprinting/probabilistic.py` | Eq. (5.3) | Bayes posterior: P(x_i\|z) = P(z\|x_i)P(x_i)/P(z) |
| `map_localize()` | `core/fingerprinting/probabilistic.py` | Eq. (5.4) | MAP: i* = argmax_i p(x_i\|z) |
| `posterior_mean_localize()` | `core/fingerprinting/probabilistic.py` | Eq. (5.5) | Posterior mean: x = sum(p(x_i\|z) * x_i), supports top-k optimization |

### Pattern Recognition - Regression

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `LinearRegressionLocalizer.fit()` | `core/fingerprinting/pattern_recognition.py` | - | Train linear model x = Wz + b |
| `LinearRegressionLocalizer.predict()` | `core/fingerprinting/pattern_recognition.py` | - | Predict location from fingerprint |

### Pattern Recognition - Classification

| Function | Location | Description |
|----------|----------|-------------|
| `fit_classifier()` | `core/fingerprinting/classification.py` | Train Random Forest or SVM classifier (each RP as a class) |
| `ClassificationLocalizer.predict()` | `core/fingerprinting/classification.py` | Predict location via classification |
| `hierarchical_localize()` | `core/fingerprinting/classification.py` | Two-step: classify floor/region, then fine-grained localization |

## Architecture

Every chapter has the same shape: pick an example, it calls into `core/`,
figures land in `figs/`. The diagram and the table below are generated from
the imports themselves by `tools/chapter_dependencies.py`, so they cannot
drift from the code.

<!-- BEGIN GENERATED: architecture (tools/chapter_dependencies.py) -->

```mermaid
flowchart TB
    D["<b>optional input</b><br/>data/sim/ch5_wifi_fingerprint_grid<br/>data/sim/ch5_wifi_fingerprint_multisamples<br/><i>every example reads it</i>"]
    E["<b>ch5_fingerprinting/example_*.py</b><br/>6 runnable demos"]
    C["<b>the reusable library</b><br/>core/eval/ · core/fingerprinting/"]
    F["<b>ch5_fingerprinting/figs/</b><br/>svg + pdf + png"]
    D -. "--data" .-> E
    E ==> C
    C ==> F
```

| Example | Core modules | Optional dataset |
| --- | --- | --- |
| `example_classification` | `core.eval`, `core.fingerprinting` | `ch5_wifi_fingerprint_grid` |
| `example_comparison` | `core.eval`, `core.fingerprinting` | `ch5_wifi_fingerprint_grid`, `ch5_wifi_fingerprint_multisamples` |
| `example_deterministic` | `core.eval`, `core.fingerprinting` | `ch5_wifi_fingerprint_grid` |
| `example_pattern_recognition` | `core.eval`, `core.fingerprinting` | `ch5_wifi_fingerprint_grid` |
| `example_probabilistic` | `core.eval`, `core.fingerprinting` | `ch5_wifi_fingerprint_grid` |
| `example_walk_posterior` | `core.eval`, `core.fingerprinting` | `ch5_wifi_fingerprint_grid` |

<!-- END GENERATED: architecture -->

## File Structure

```
ch5_fingerprinting/
├── README.md                         # This file (student documentation)
├── example_deterministic.py          # NN and k-NN demo
├── example_probabilistic.py          # Bayesian methods demo
├── example_pattern_recognition.py    # Linear regression demo
├── example_classification.py         # Classification and hierarchical demo
├── example_walk_posterior.py         # Sec. 5.2: posterior along a walk
├── example_comparison.py             # Compare all methods
├── figs/                             # Generated figures
│   ├── deterministic_positioning.png
│   ├── comparison_all_methods.png
│   ├── classification_noise_robustness.png
│   └── hierarchical_localization.png

core/fingerprinting/
├── deterministic.py                  # NN, k-NN algorithms (with missing AP support)
├── probabilistic.py                  # Bayesian methods (with missing AP support)
├── pattern_recognition.py            # Linear regression
├── classification.py                 # Classification methods (RF, SVM, hierarchical)
├── preprocess.py                     # Preprocessing utilities (averaging, normalization)
├── types.py                          # FingerprintDatabase class (allows NaN)
└── dataset.py                        # Load/save utilities

data/sim/
├── ch5_wifi_fingerprint_grid/        # Default dataset (5m grid, 121 RPs)
│   ├── features.npy                  # RSS per RP, (n_rp, n_ap) in dBm
│   ├── locations.npy                 # RP coordinates, (n_rp, 2)
│   ├── floor_ids.npy                 # Floor label per RP, (n_rp,)
│   └── metadata.json                 # AP ids/positions, grid spacing, path-loss model
├── ch5_wifi_fingerprint_dense/       # Dense dataset (2m grid, 676 RPs)
│   ├── features.npy
│   ├── locations.npy
│   ├── floor_ids.npy
│   └── metadata.json
├── ch5_wifi_fingerprint_sparse/      # Sparse dataset (10m grid, 36 RPs/floor)
│   ├── features.npy
│   ├── locations.npy
│   ├── floor_ids.npy
│   └── metadata.json
└── ch5_wifi_fingerprint_multisamples/ # 10 visits per RP, features.npy is (363, 10, 8)
    ├── features.npy
    ├── locations.npy
    ├── floor_ids.npy
    └── metadata.json

tests/core/fingerprinting/
├── test_deterministic.py             # Deterministic method tests
├── test_probabilistic.py             # Probabilistic method tests
├── test_preprocess.py                # Preprocessing tests (32 tests)
├── test_classification.py            # Classification tests (17 tests)
├── test_missing_aps.py               # Missing AP handling tests (28 tests)
├── test_multisamples.py              # Multi-sample database tests
├── test_topk_posterior_mean.py       # Top-k optimization tests
└── ...
```

## Figure Gallery

All figures are generated by the example scripts and stored in the `figs/` directory.

| Figure | Source Script | Description |
|--------|--------------|-------------|
| `deterministic_positioning.png` | `example_deterministic.py` | 6-panel comparison of NN and k-NN methods: RP layout, error CDF, histogram, box plots, k-sensitivity, speed-accuracy trade-off |
| `comparison_all_methods.png` | `example_comparison.py` | 9-panel comprehensive comparison: RMSE by noise, error CDF, speed, box plots, robustness, trade-offs, category analysis, median/P90, radar chart |
| `classification_noise_robustness.png` | `example_classification.py` | Noise robustness comparison: Random Forest vs SVM vs k-NN performance as measurement noise increases |
| `hierarchical_localization.png` | `example_classification.py` | 4-panel hierarchical evaluation: error CDF, box plots, RMSE bars, floor classification accuracy |

**Regenerating Figures:**

```bash
# Generate all figures
python -m ch5_fingerprinting.example_deterministic
python -m ch5_fingerprinting.example_comparison
python -m ch5_fingerprinting.example_classification
```

## References

- **Chapter 5**: Indoor Positioning Using Feature Matching Methods
  - Section 5.1: Fundamentals of Fingerprinting (deterministic + probabilistic)
  - Section 5.2: Pattern Recognition Approaches (classification + regression)
  - Section 5.3: Deep Learning-Based Approaches (not yet implemented)

