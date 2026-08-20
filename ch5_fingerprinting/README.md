# Chapter 5: Fingerprinting-based Indoor Positioning

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

The intuitive animation to reach for is "watch the posterior sharpen as the
user walks." There is nothing to sharpen. With eight access points over this
grid the Gaussian Naive-Bayes posterior of Eq. (5.3) is a near-delta
everywhere — entropy ~0.1 against a maximum of 4.8, peak probability 0.92–1.00.
It is already as sharp as it gets.

What *is* dynamic, and specific to fingerprinting, is how that sharp posterior
**fails**. It does not spread out and hedge under noise; it stays a confident
spike and occasionally puts the spike in the wrong place — a distant reference
point whose stored radio signature happens to resemble the current one. This is
**RSS aliasing**, and it is the characteristic failure of fingerprinting.

Measured over a 21-step L-walk, aliasing jumps beyond 10 m occur:

| Measurement noise | aliasing jumps | median error | mean error |
|---|---|---|---|
| 1 dB | 0 / 21 | 0.0 m | 0.0 m |
| 3 dB | 0 / 21 | 0.0 m | 0.2 m |
| 6 dB | **6 / 21** | **0.0 m** | **6.6 m** |

The last row is the lesson. The MAP estimate is usually exactly right — the
median stays 0 m — but at six steps it teleports to a radio-similar location up
to 25 m away, and those few jumps drag the mean to 6.6 m. **A reported mean
error hides the failure completely**; only the walk, or the median-vs-mean gap,
makes it visible. The animation shows the hot spot tracking the user, then
leaping across the floor and snapping back.

## 📂 Dataset Connection

| Example Script | Dataset | Description |
|----------------|---------|-------------|
| All examples | `data/sim/ch5_wifi_fingerprint_grid/` | Standard 5m grid, 121 RPs (default) |
| All examples | `data/sim/ch5_wifi_fingerprint_dense/` | Dense 2m grid, 676 RPs (higher accuracy) |
| All examples | `data/sim/ch5_wifi_fingerprint_sparse/` | Sparse 10m grid, 36 RPs per floor (quick deployment) |

> **Note**: To use a different dataset density, edit the `db_path` variable in the example scripts.

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
NN (Euclidean)            5.90         3.28         9.74         ~
NN (Manhattan)            6.46         3.46         10.27        ~
k-NN (k=3, inv-dist)      4.73         3.24         7.23         ~
k-NN (k=5, inv-dist)      4.70         3.57         7.24         ~
k-NN (k=7, inv-dist)      4.83         3.84         7.06         ~
k-NN (k=5, uniform)       5.08         3.80         7.92         ~
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
NN (Euclidean)       Deterministic        10.05        8.22         16.18        ~
k-NN (k=3)           Deterministic        7.55         6.01         12.05        ~
MAP                  Probabilistic        10.05        8.22         16.18        ~
Posterior Mean       Probabilistic        9.22         7.56         15.21        ~
Post.Mean (k=10)     Probabilistic        9.22         7.56         15.21        ~
Linear Regression    Pattern Recognition  7.74         6.66         11.83        ~
```

`MAP` and `NN (Euclidean)` agree exactly, and `Post.Mean (k=10)` agrees with
the full posterior mean, because both pairs select the same reference points on
this database — the example prints the reasoning under Key Insights.

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
  Random Forest: 83.0% (166/200) exact RP
  SVM:           97.0% (194/200) exact RP
  This is the number to compare against other methods.
```

Classification reports **exact-RP accuracy**, not RMSE in metres, so these
numbers are not directly comparable with the tables above — a classifier that
picks a neighbouring reference point scores 0 here and under a metre there.
Note also that the 100% training recall is a memorisation check, printed
precisely so it is not mistaken for a positioning result.

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

## Architecture Diagrams

For a visual understanding of the chapter's implementation, refer to the following diagrams:

### Component Architecture

![Component Architecture](../docs/architecture/ipin_ch5_component_clean.svg)

This diagram shows:
- **Example Scripts**: Five demonstration scripts (`example_deterministic.py`, `example_probabilistic.py`, `example_pattern_recognition.py`, `example_classification.py`, `example_comparison.py`)
- **Core Modules**: Reusable fingerprinting implementations in `core/fingerprinting/` (dataset, preprocessing, deterministic, probabilistic, pattern recognition, classification, types)
- **Datasets**: Three WiFi fingerprint databases in `data/sim/`:
  - `ch5_wifi_fingerprint_grid/` (default, 5m grid, 121 RPs)
  - `ch5_wifi_fingerprint_dense/` (2m grid, 676 RPs)
  - `ch5_wifi_fingerprint_sparse/` (10m grid, 36 RPs per floor)
- **Output**: Generated figures saved to `figs/` subdirectory

**Source**: PlantUML source available at [`docs/architecture/ipin_ch5_component_overview.puml`](../docs/architecture/ipin_ch5_component_overview.puml)

### Execution Flow

![Execution Flow](../docs/architecture/ipin_ch5_flow_clean.svg)

This diagram illustrates the execution pipeline for each example script:

1. **`example_deterministic.py`** (Nearest-Neighbor):
   - Load fingerprint DB (default: `ch5_wifi_fingerprint_grid/`)
   - Generate test queries (floor + RSS noise)
   - Run deterministic methods: NN (Eq. 5.1) and k-NN (Eq. 5.2)
   - Compute errors + runtime (RMSE / median / P90)
   - Plot + save `figs/deterministic_positioning.png`

2. **`example_probabilistic.py`** (Bayesian Methods):
   - Load fingerprint DB
   - Fit Gaussian Naive Bayes (Eq. 5.6)
   - Generate test queries (+ RSS noise)
   - Compute posterior P(x_i|z) (Eq. 5.3)
   - Estimate location: MAP (Eq. 5.4) and Posterior mean (Eq. 5.5)
   - Evaluate errors + runtime
   - Plot posterior + save `figs/probabilistic_positioning.png`

3. **`example_pattern_recognition.py`** (Linear Regression):
   - Load fingerprint DB
   - Split train/test (per floor)
   - Train linear regression localizer: x̂ = Wz + b (ridge, λ ∈ {0, 0.1, 1, 10})
   - Evaluate RMSE + R² + time/query
   - Plot weights + error stats, save `figs/pattern_recognition_positioning.png`

4. **`example_classification.py`** (Classification + Hierarchical):
   - Create synthetic multi-floor DB (3 floors, multiple APs)
   - Train Random Forest + SVM classifiers
   - Test accuracy + noise robustness
   - Hierarchical localization: floor classification → k-NN / Bayes refinement
   - Save `figs/classification_noise_robustness.png` and `figs/hierarchical_localization.png`

5. **`example_comparison.py`** (Method Comparison):
   - Load fingerprint DB
   - Define scenarios: Baseline / Moderate / High noise
   - Run method suite: Deterministic + Probabilistic + Linear Regression
   - Aggregate metrics (RMSE / median / P90 / time)
   - Plot + save `figs/comparison_all_methods.png`

**Source**: PlantUML source available at [`docs/architecture/ipin_ch5_activity_flow.puml`](../docs/architecture/ipin_ch5_activity_flow.puml)

---

## File Structure

```
ch5_fingerprinting/
├── README.md                         # This file (student documentation)
├── example_deterministic.py          # NN and k-NN demo
├── example_probabilistic.py          # Bayesian methods demo
├── example_pattern_recognition.py    # Linear regression demo
├── example_classification.py         # Classification and hierarchical demo
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

docs/architecture/
├── ipin_ch5_component_overview.puml  # Component diagram source (PlantUML)
├── ipin_ch5_component_clean.svg      # Component diagram (rendered)
├── ipin_ch5_activity_flow.puml       # Execution flow source (PlantUML)
└── ipin_ch5_flow_clean.svg           # Execution flow diagram (rendered)

data/sim/
├── ch5_wifi_fingerprint_grid/        # Default dataset (5m grid, 121 RPs)
│   ├── fingerprints.csv              # Reference fingerprints
│   └── metadata.json                 # Dataset configuration
├── ch5_wifi_fingerprint_dense/       # Dense dataset (2m grid, 676 RPs)
│   ├── fingerprints.csv
│   └── metadata.json
└── ch5_wifi_fingerprint_sparse/      # Sparse dataset (10m grid, 36 RPs/floor)
    ├── fingerprints.csv
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

