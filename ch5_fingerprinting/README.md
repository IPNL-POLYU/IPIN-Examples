# Chapter 5: Fingerprinting-based Indoor Positioning

## Overview

This module implements fingerprinting-based positioning algorithms described in **Chapter 5** of *Principles of Indoor Positioning and Indoor Navigation*. Fingerprinting is a pattern-matching approach that compares measured radio signal strengths (RSS) against a pre-built database of reference fingerprints.

The module provides simulation-based examples of three main categories of fingerprinting methods:
- **Deterministic methods** (nearest-neighbor, k-nearest-neighbor)
- **Probabilistic methods** (Bayesian inference, MAP, posterior mean)
- **Pattern recognition** (linear regression, supervised learning)

## Equation Mapping: Code ↔ Book

The following tables map the implemented functions to their corresponding equations in Chapter 5 of the book:

### Deterministic Fingerprinting

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `distance()` | `core/fingerprinting/deterministic.py` | - | ✓ | Euclidean or Manhattan distance D(z, f) |
| `pairwise_distances()` | `core/fingerprinting/deterministic.py` | - | ✓ | Vectorized distance computation for all RPs |
| `nn_localize()` | `core/fingerprinting/deterministic.py` | **Eq. (5.1)** | ✓ | NN: i* = argmin_i D(z, f_i), x̂ = x_{i*} |
| `knn_localize()` | `core/fingerprinting/deterministic.py` | **Eq. (5.2)** | ✓ | k-NN: x̂ = Σ w_i x_i / Σ w_i |

### Probabilistic Fingerprinting (Bayesian)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `fit_gaussian_naive_bayes()` | `core/fingerprinting/probabilistic.py` | - | ✓ | Fit Gaussian model per RP per AP |
| `log_likelihood()` | `core/fingerprinting/probabilistic.py` | **Eq. (5.3)** | ✓ | Log p(z\|x_i) under Gaussian Naive Bayes |
| `log_posterior()` | `core/fingerprinting/probabilistic.py` | - | ✓ | Log p(x_i\|z) via Bayes' rule |
| `map_localize()` | `core/fingerprinting/probabilistic.py` | **Eq. (5.4)** | ✓ | MAP: i* = argmax_i p(x_i\|z), x̂ = x_{i*} |
| `posterior_mean_localize()` | `core/fingerprinting/probabilistic.py` | **Eq. (5.5)** | ✓ | Posterior mean: x̂ = Σ p(x_i\|z) x_i |

### Pattern Recognition (Supervised Learning)

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `LinearRegressionLocalizer.fit()` | `core/fingerprinting/pattern_recognition.py` | - | ✓ | Train linear model x̂ = Wz + b (ridge regression) |
| `LinearRegressionLocalizer.predict()` | `core/fingerprinting/pattern_recognition.py` | - | ✓ | Predict location from fingerprint |
| `LinearRegressionLocalizer.score()` | `core/fingerprinting/pattern_recognition.py` | - | ✓ | Compute R² coefficient on test set |

### Data Structures and Utilities

| Function | Location | Equation | Status | Description |
|----------|----------|----------|--------|-------------|
| `FingerprintDatabase` | `core/fingerprinting/types.py` | - | ✓ | Multi-floor database with locations, features, floor_ids |
| `load_fingerprint_database()` | `core/fingerprinting/dataset.py` | - | ✓ | Load database from NPZ format |
| `save_fingerprint_database()` | `core/fingerprinting/dataset.py` | - | ✓ | Save database to NPZ format |
| `validate_database()` | `core/fingerprinting/dataset.py` | - | ✓ | Quality checks and validation |

**Legend:**
- ✓ Implemented and tested (125 unit tests, 100% pass rate)
- ⚠️ Planned / To be implemented
- ✗ Not implemented (out of scope)

## Implementation Notes

### ✓ Fully Implemented

#### 1. **Deterministic Methods (Eqs. 5.1-5.2)**

**Nearest-Neighbor (NN)**
- Decision rule: i* = argmin_i D(z, f_i)
- Returns location of nearest reference point
- Supports Euclidean and Manhattan distance metrics
- Fast: O(M) for M reference points
- Provides discrete position estimates
- Best for dense reference point grids

**k-Nearest-Neighbor (k-NN)**
- Weighted average: x̂ = Σ w_i x_i / Σ w_i
- Two weighting schemes:
  - Inverse distance: w_i = 1/(D(z, f_i) + ε)
  - Uniform: w_i = 1 (simple average)
- Smoother estimates than NN
- Optimal k depends on RP density and noise
- Typically k=3-7 for indoor scenarios

**Multi-floor Support**
- Floor-constrained search (optional `floor_id` parameter)
- Uses `FingerprintDatabase.get_floor_mask()` for filtering
- If no floor constraint: searches across all floors

#### 2. **Probabilistic Methods (Eqs. 5.3-5.5)**

**Gaussian Naive Bayes Model**
- Assumes Gaussian distribution per feature per RP
- Parameters: μ_ij (mean RSS), σ_ij (std)
- Naive Bayes: features conditionally independent
- Likelihood: p(z|x_i) = ∏_j N(z_j; μ_ij, σ_ij²)
- Log-likelihood (Eq. 5.3): log p(z|x_i) = Σ_j log N(z_j; μ_ij, σ_ij²)

**MAP Estimation (Eq. 5.4)**
- Maximum A Posteriori: i* = argmax_i p(x_i|z)
- Bayes' rule: p(x_i|z) ∝ p(z|x_i) p(x_i)
- Returns discrete estimate (one of the RPs)
- Fast: single argmax operation
- Sensitive to model parameters (σ)

**Posterior Mean (Eq. 5.5)**
- Expected value: x̂ = Σ_i p(x_i|z) x_i
- Continuous estimate (weighted average)
- Smoother than MAP
- Better for uncertain posteriors
- More robust to outliers

**Model Parameters**
- `min_std`: Minimum standard deviation (default 2.0 dBm)
- `prior`: Prior distribution (currently uniform: p(x_i) = 1/M)
- Larger σ → smoother posteriors, more averaging
- Smaller σ → sharper posteriors, closer to NN

#### 3. **Pattern Recognition (Linear Regression)**

**Linear Model**
- Learns direct mapping: x̂ = Wz + b
- W: weight matrix, shape (d, N) for d-dimensional position, N features
- b: bias vector, shape (d,)
- Training: Ridge regression with regularization parameter λ

**Training Algorithm**
- Solves: min_{W,b} Σ_i ||x_i - (Wz_i + b)||² + λ||W||²_F
- Closed-form solution using normal equations
- Bias trick: augment features with 1's
- Uses `np.linalg.lstsq` for numerical stability
- Regularization prevents overfitting (λ > 0)

**Advantages**
- Very fast prediction: single matrix multiplication
- Continuous smooth estimates
- Good generalization with proper λ
- Works well for sparse RP grids

**Limitations**
- Assumes linear relationship (RSS-to-location)
- Requires sufficient training data (M ≥ N)
- May underperform with strong multipath/nonlinearity
- Needs per-floor training for multi-floor

## Dataset

### Synthetic Wi-Fi Fingerprint Database

Located in: `data/sim/wifi_fingerprint_grid/`

**Specifications:**
- **Coverage**: 50m × 50m area per floor
- **Grid**: 11×11 RPs, 5m spacing (121 RPs per floor)
- **Floors**: 3 (floor 0, 1, 2)
- **Total RPs**: 363 (121 × 3)
- **Access Points**: 8 APs strategically positioned
- **RSS Model**: Log-distance path-loss + shadow fading
  - P(d) = P₀ - 10n log₁₀(d/d₀) + X_σ
  - P₀ = -30 dBm (reference power at 1m)
  - n = 2.5 (indoor path-loss exponent)
  - σ = 4 dBm (shadow fading std)
  - Floor attenuation: 15 dB per floor

**Generation Script:**
```bash
python scripts/generate_wifi_fingerprint_dataset.py
```

**Data Format:**
- Saved as NPZ file (NumPy compressed)
- Contains: locations (363×2), features (363×8), floor_ids (363,), metadata
- Compatible with `load_fingerprint_database()` and `save_fingerprint_database()`

## Examples

### Individual Method Examples

Each example script demonstrates a specific category of methods with detailed analysis and visualization.

#### Example 1: Deterministic Fingerprinting

```bash
python ch5_fingerprinting/example_deterministic.py
```

**Demonstrates:**
- NN with Euclidean and Manhattan distance (Eq. 5.1)
- k-NN with varying k and weighting schemes (Eq. 5.2)
- Effect of k on positioning accuracy
- Decision boundaries and smoothing

**Key Findings:**
- NN is fastest but has discrete jumps
- k-NN smooths estimates, optimal k≈3-5
- Inverse distance weighting > uniform
- Manhattan can be faster than Euclidean

**Generates:** `deterministic_positioning.png`

#### Example 2: Probabilistic Fingerprinting

```bash
python ch5_fingerprinting/example_probabilistic.py
```

**Demonstrates:**
- Gaussian Naive Bayes model fitting
- Log-likelihood computation (Eq. 5.3)
- MAP estimation (Eq. 5.4)
- Posterior mean estimation (Eq. 5.5)
- Effect of model uncertainty (σ parameter)

**Key Findings:**
- MAP provides discrete estimates
- Posterior mean provides smooth estimates
- Model σ controls uncertainty/smoothness trade-off
- Larger σ → more averaging, lower accuracy but more robust

**Generates:** `probabilistic_positioning.png`

#### Example 3: Pattern Recognition

```bash
python ch5_fingerprinting/example_pattern_recognition.py
```

**Demonstrates:**
- Linear regression model training (x̂ = Wz + b)
- Ridge regression with regularization (λ)
- Train/test split evaluation
- R² coefficient analysis
- Overfitting analysis

**Key Findings:**
- Linear regression is very fast (<0.1ms per query)
- Regularization prevents overfitting (optimal λ≈1.0)
- Good generalization with proper λ
- Works well for interpolation between RPs

**Generates:** `pattern_recognition_positioning.png`

### Comprehensive Comparison

```bash
python ch5_fingerprinting/example_comparison.py
```

**Compares all methods across multiple scenarios:**
1. **Baseline**: σ=1 dBm noise, floor 0
2. **Moderate Noise**: σ=2 dBm noise
3. **High Noise**: σ=5 dBm noise

**Methods Evaluated:**
- NN (Euclidean)
- k-NN (k=3, inverse distance)
- MAP (σ=2 dBm)
- Posterior Mean (σ=2 dBm)
- Linear Regression (λ=1.0)

**Metrics:**
- RMSE (Root Mean Square Error)
- Median error
- 90th percentile error
- Computation time per query

**Generates:** `comparison_all_methods.png`

## Results Summary

### Performance Comparison (Baseline Scenario)

| Method | RMSE (m) | Median (m) | P90 (m) | Time (ms) | Category |
|--------|----------|------------|---------|-----------|----------|
| NN (Euclidean) | ~2.5 | ~2.0 | ~4.5 | ~0.5 | Deterministic |
| k-NN (k=3) | ~2.0 | ~1.5 | ~3.8 | ~0.8 | Deterministic |
| MAP | ~2.2 | ~1.7 | ~4.0 | ~1.5 | Probabilistic |
| Posterior Mean | ~2.0 | ~1.5 | ~3.7 | ~1.6 | Probabilistic |
| Linear Regression | ~2.3 | ~1.8 | ~4.2 | ~0.05 | Pattern Recognition |

*Note: Values are approximate and depend on dataset characteristics and noise level.*

### Key Insights

**Speed Ranking** (fastest → slowest):
1. **Linear Regression**: 0.05ms (50× faster, just matrix mult)
2. **NN**: 0.5ms (simple distance computation)
3. **k-NN**: 0.8ms (k distance computations + averaging)
4. **MAP**: 1.5ms (likelihood + posterior computation)
5. **Posterior Mean**: 1.6ms (likelihood + posterior + averaging)

**Accuracy Ranking** (most → least accurate, baseline):
1. **k-NN & Posterior Mean**: ~2.0m RMSE (tie)
2. **MAP**: ~2.2m RMSE
3. **Linear Regression**: ~2.3m RMSE
4. **NN**: ~2.5m RMSE

**Robustness to Noise** (most → least robust):
1. **Posterior Mean**: Smooth averaging over posterior
2. **k-NN**: Averaging over k neighbors
3. **MAP**: Model uncertainty helps
4. **Linear Regression**: Depends on training data
5. **NN**: Most sensitive (no averaging)

## Recommendations

### Application-Specific Guidance

**Real-time Applications** (speed critical):
- Use **Linear Regression** or **NN**
- Linear Reg: Train offline, ultra-fast online
- NN: No training, very fast lookup

**High Accuracy Required** (accuracy critical):
- Use **k-NN (k=3-5)** or **Posterior Mean**
- Both provide smooth estimates
- k-NN simpler, Posterior Mean more principled

**Noisy Environments** (robustness critical):
- Use **k-NN** with moderate k (3-7)
- Or **Posterior Mean** with appropriate σ (2-4 dBm)
- Avoid NN (too sensitive)

**Dense Reference Points**:
- **NN** sufficient (RPs every 2-3m)
- Simple and fast

**Sparse Reference Points**:
- **k-NN** or **Linear Regression**
- Interpolation between RPs essential
- Avoid NN (large jumps)

**Multi-floor Buildings**:
- Use floor-constrained search
- Or train separate model per floor (Linear Reg)
- Consider floor identification as separate step

## Usage Examples

### Basic Usage

```python
from pathlib import Path
import numpy as np
from core.fingerprinting import (
    load_fingerprint_database,
    nn_localize,
    knn_localize,
    fit_gaussian_naive_bayes,
    map_localize,
    LinearRegressionLocalizer,
)

# Load database
db = load_fingerprint_database(Path("data/sim/wifi_fingerprint_grid"))

# Query fingerprint (8 RSS values from 8 APs)
query = np.array([-45, -60, -75, -80, -50, -70, -85, -90])

# Method 1: Nearest-Neighbor (Eq. 5.1)
pos_nn = nn_localize(query, db, metric="euclidean", floor_id=0)
print(f"NN estimate: {pos_nn}")

# Method 2: k-Nearest-Neighbor (Eq. 5.2)
pos_knn = knn_localize(query, db, k=3, metric="euclidean", 
                       weighting="inverse_distance", floor_id=0)
print(f"k-NN estimate: {pos_knn}")

# Method 3: Bayesian MAP (Eqs. 5.3-5.4)
model_bayes = fit_gaussian_naive_bayes(db, min_std=2.0)
pos_map = map_localize(query, model_bayes, floor_id=0)
print(f"MAP estimate: {pos_map}")

# Method 4: Linear Regression
model_lr = LinearRegressionLocalizer.fit(db, floor_id=0, regularization=1.0)
pos_lr = model_lr.predict(query)
print(f"Linear Reg estimate: {pos_lr}")
```

### Advanced: Multi-floor Localization

```python
# Option 1: Search all floors (finds best match globally)
pos_global = nn_localize(query, db, floor_id=None)

# Option 2: Per-floor models (train separate model per floor)
models = {}
for floor_id in db.floor_list:
    models[floor_id] = LinearRegressionLocalizer.fit(
        db, floor_id=floor_id, regularization=1.0
    )

# Predict on each floor, select best
best_floor = None
best_error = float('inf')
for floor_id, model in models.items():
    pos = model.predict(query)
    # Use some metric to score (e.g., likelihood, residual, etc.)
    # ...
```

## Technical Details

### Coordinate System
- **2D positioning**: (x, y) in meters
- **Origin**: (0, 0) at corner of coverage area
- **Floors**: Identified by integer floor_id (0, 1, 2, ...)
- **Height**: Assumed constant per floor (not estimated)

### RSS Characteristics
- **Range**: Typically -30 to -110 dBm
- **Mean**: ~-80 dBm (varies by distance and floor)
- **Std**: ~4 dBm (shadow fading)
- **Units**: dBm (decibels relative to 1 milliwatt)

### Computational Complexity

| Method | Training | Query | Space |
|--------|----------|-------|-------|
| NN | O(1) | O(MN) | O(MN) |
| k-NN | O(1) | O(MN) | O(MN) |
| MAP | O(MN) | O(MN) | O(MN) |
| Posterior Mean | O(MN) | O(MN) | O(MN) |
| Linear Reg | O(N³) | O(N) | O(Nd) |

Where:
- M = number of reference points
- N = number of features (APs)
- d = location dimension (2 for 2D)

## Testing

All fingerprinting methods have comprehensive unit tests:

```bash
# Run all fingerprinting tests
python -m pytest tests/core/fingerprinting/ -v

# 125 tests total:
#   - 35 tests: Data structures (types, dataset)
#   - 33 tests: Deterministic methods
#   - 29 tests: Probabilistic methods
#   - 28 tests: Pattern recognition
```

**Test Coverage:**
- 100% pass rate
- All core functions tested
- Edge cases and error conditions
- Multi-floor scenarios
- Integration tests

## References

### Book Equations
- **Eq. (5.1)**: NN decision rule i* = argmin_i D(z, f_i)
- **Eq. (5.2)**: k-NN weighted average x̂ = Σ w_i x_i / Σ w_i
- **Eq. (5.3)**: Gaussian log-likelihood log p(z|x_i)
- **Eq. (5.4)**: MAP estimate i* = argmax_i p(x_i|z)
- **Eq. (5.5)**: Posterior mean x̂ = Σ p(x_i|z) x_i

### Related Chapters
- **Chapter 3**: State estimation (filters used for tracking)
- **Chapter 4**: RF positioning (ranging methods)
- **Chapter 6**: Sensor fusion (combining fingerprinting with IMU)
- **Chapter 7**: Map-aided positioning (constraints from building layout)

## Future Enhancements

Potential extensions (not currently implemented):
- **Advanced ML models**: Neural networks, Gaussian processes
- **Hybrid methods**: Combine fingerprinting with ranging (TOA/TDOA)
- **Online learning**: Update database with new measurements
- **Missing data**: Handle partial fingerprints (missing APs)
- **Floor identification**: Separate classifier for floor detection
- **Weighted metrics**: Mahalanobis distance with covariance
- **Ray-tracing simulation**: Physics-based RSS prediction

## Troubleshooting

**Issue**: Poor accuracy (RMSE > 5m)
- Check RP density (should be ≤ 5m spacing)
- Verify RSS quality (avoid NaN, check range)
- Try different methods (k-NN, Posterior Mean)
- Increase regularization (Linear Reg)

**Issue**: Slow computation
- Use Linear Regression for fastest queries
- Reduce database size (subsample RPs)
- Use floor constraints (if floor known)

**Issue**: Floor confusion (wrong floor)
- Use floor-constrained search
- Train separate models per floor
- Add floor classification step

**Issue**: Overfitting (Linear Reg)
- Increase regularization parameter λ
- Use more training data
- Reduce number of features

## Contact

For questions or issues related to this implementation:
- **Author**: Navigation Engineer
- **Date**: December 2024
- **Repository**: IPIN_Book_Examples

---

**Note**: This implementation is for educational purposes, demonstrating the concepts from Chapter 5 of the book. For production systems, consider additional factors such as: computational constraints, real-time requirements, sensor characteristics, building layout, and user motion patterns.

