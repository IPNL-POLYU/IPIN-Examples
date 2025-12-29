# Chapter 5: Missing AP (Dropout) Support Implementation Summary

**Date:** December 24, 2024  
**Task:** P1 — Support missing AP readings (dropout) in distance + likelihood computations

## Summary

Successfully implemented comprehensive support for missing AP readings (signal dropout) in the fingerprinting system. This enables the system to handle real-world scenarios where some access points may not be detectable due to signal loss, obstruction, or device limitations.

## Book Reference

**Chapter 5, Section 5.1:**

The book discusses handling missing or unavailable AP measurements as a practical consideration in fingerprinting-based positioning. When some APs are not visible or their signals drop out, the system should compute distances and likelihoods using only the overlapping (observed) features.

## Implementation Approach

### Design Decision: NaN as Missing Value Sentinel

- **Representation:** Missing AP readings are represented as `np.nan` (Not a Number)
- **Rationale:** 
  - Natural representation for missing data in NumPy
  - Distinguishable from valid RSS values (which are always finite)
  - Supported by NumPy's `nanmean`, `nanstd`, `nansum` functions
  - No need for separate mask arrays

### Core Changes

#### 1. Database Type (`core/fingerprinting/types.py`)

**Updated validation:**
```python
# Old: Rejected NaN in features
if np.any(np.isnan(self.features)):
    raise ValueError("features contain NaN values (not supported)")

# New: Allow NaN in features (missing APs)
# Note: NaN values in features are allowed (represent missing AP readings)
```

**Updated statistics methods:**
```python
def get_mean_features(self) -> np.ndarray:
    """Compute mean features, ignoring NaN values."""
    if self.has_multiple_samples:
        return np.nanmean(self.features, axis=1)  # Ignore NaN
    else:
        return self.features

def get_std_features(self, min_std: float = 0.0) -> np.ndarray:
    """Compute std features, ignoring NaN values."""
    if self.has_multiple_samples:
        stds = np.nanstd(self.features, axis=1, ddof=1)  # Ignore NaN
        stds = np.maximum(stds, min_std)
        # Replace NaN stds (all samples missing) with min_std
        stds = np.where(np.isnan(stds), min_std, stds)
        return stds
    else:
        return np.full((self.n_reference_points, self.n_features), min_std)
```

#### 2. Deterministic Methods (`core/fingerprinting/deterministic.py`)

**Updated `distance()` function:**
```python
def distance(z: np.ndarray, f: np.ndarray, metric: str = "euclidean") -> float:
    """
    Compute distance D(z, f) between query and reference fingerprint.
    
    **Missing AP Handling:**
    If either z or f contains NaN values, the distance is computed only
    over dimensions where both values are present. If no overlapping
    dimensions exist, returns +inf.
    """
    # Find valid (non-NaN) dimensions in both z and f
    valid_mask = ~(np.isnan(z) | np.isnan(f))
    n_valid = np.sum(valid_mask)
    
    # If no overlapping valid dimensions, return infinity
    if n_valid == 0:
        return np.inf
    
    # Extract valid dimensions
    z_valid = z[valid_mask]
    f_valid = f[valid_mask]
    
    # Compute distance using only valid dimensions
    if metric == "euclidean":
        return float(np.linalg.norm(z_valid - f_valid))
    elif metric == "manhattan":
        return float(np.sum(np.abs(z_valid - f_valid)))
```

**Updated `pairwise_distances()` function:**
```python
def pairwise_distances(z: np.ndarray, F: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute distances D(z, f_i) for all fingerprints f_i in F.
    
    **Missing AP Handling:**
    If z or any row of F contains NaN values, distances are computed only
    over dimensions where both values are present.
    """
    # Check if there are any NaN values
    has_missing = np.any(np.isnan(z)) or np.any(np.isnan(F))
    
    if not has_missing:
        # Fast path: no missing values, use vectorized computation
        if metric == "euclidean":
            return np.linalg.norm(F - z, axis=1)
        elif metric == "manhattan":
            return np.sum(np.abs(F - z), axis=1)
    
    # Slow path: handle missing values per RP
    distances = np.zeros(M)
    for i in range(M):
        distances[i] = distance(z, F[i], metric=metric)
    
    return distances
```

#### 3. Probabilistic Methods (`core/fingerprinting/probabilistic.py`)

**Updated `log_likelihood()` function:**
```python
def log_likelihood(
    z: Fingerprint,
    model: NaiveBayesFingerprintModel,
    floor_id: Optional[int] = None,
) -> np.ndarray:
    """
    Compute log-likelihood log P(z | x_i) for all reference points.
    
    **Missing AP Handling:**
    If z contains NaN values, the sum includes only terms for observed
    (non-NaN) features. If no observed features exist, returns -inf.
    """
    # ... compute log_gaussian for all features ...
    
    # Handle missing values: identify observed (non-NaN) features
    observed_mask = ~np.isnan(z)  # Shape: (N,)
    n_observed = np.sum(observed_mask)
    
    if n_observed == 0:
        # No observed features: return -inf for all RPs
        log_likelihoods = np.full(M, -np.inf)
    else:
        # Sum only over observed features
        # Use nansum to ignore NaN contributions in log_gaussian
        log_likelihoods = np.nansum(log_gaussian, axis=1)
    
    return log_likelihoods
```

## Test Coverage

Created comprehensive test suite: `tests/core/fingerprinting/test_missing_aps.py`

### Test Classes

1. **TestDistanceWithMissingAPs** (8 tests)
   - No missing values (baseline)
   - Partial missing in query
   - Partial missing in reference
   - Partial missing in both (overlapping dims)
   - No overlap → infinity
   - All NaN query → infinity
   - All NaN reference → infinity
   - Shape mismatch error

2. **TestPairwiseDistancesWithMissingAPs** (4 tests)
   - No missing values (fast path)
   - Query with missing values
   - Reference with missing values
   - Manhattan metric with missing

3. **TestNNLocalizationWithMissingAPs** (2 tests)
   - NN with missing query
   - NN with missing database

4. **TestKNNLocalizationWithMissingAPs** (2 tests)
   - k-NN with missing query
   - k-NN with 20% dropout rate

5. **TestLogLikelihoodWithMissingAPs** (4 tests)
   - No missing (baseline)
   - Partial missing
   - All missing → -inf
   - 20% dropout rate

6. **TestMAPLocalizationWithMissingAPs** (1 test)
   - MAP with missing query

7. **TestPosteriorMeanLocalizationWithMissingAPs** (2 tests)
   - Posterior mean with missing query
   - Posterior mean with top_k and missing

8. **TestDatabaseWithMissingAPsInSamples** (3 tests)
   - Database creation with NaN
   - Mean/std computation with NaN
   - Fit model with NaN samples

9. **TestAcceptanceCriteria** (2 tests)
   - **20% dropout, 100 queries, no crash** ✓
   - **Varying dropout rates (0%, 10%, 30%, 50%)** ✓

### Test Results

```
tests/core/fingerprinting/test_missing_aps.py::
  28 passed in 1.84s
```

**All fingerprinting tests (160 total):**
```
tests/core/fingerprinting/
  160 passed in 4.00s
```

## Acceptance Criteria

✅ **If 20% of AP readings are dropped from queries, localization still runs and doesn't crash**
- Tested with 100 queries, each with 20% random dropout
- All deterministic methods (NN, k-NN) work correctly
- All probabilistic methods (MAP, Posterior Mean) work correctly

✅ **Unit tests cover NaN/masked paths**
- 28 dedicated tests for missing AP handling
- Tests cover edge cases (no overlap, all missing, varying dropout rates)
- Tests verify both deterministic and probabilistic methods

## Backward Compatibility

✅ **All existing tests pass** (160/160)
- No breaking changes to existing functionality
- Fast path for complete fingerprints (no performance regression)
- Slow path only activated when NaN values are detected

## Performance Considerations

### Fast Path (No Missing Values)

When no NaN values are present, the system uses vectorized NumPy operations:
```python
# Fast vectorized distance computation
return np.linalg.norm(F - z, axis=1)
```

### Slow Path (Missing Values Present)

When NaN values are detected, the system falls back to per-RP computation:
```python
# Per-RP distance computation with masking
for i in range(M):
    distances[i] = distance(z, F[i], metric=metric)
```

**Overhead:** Minimal for typical dropout rates (10-30%)

## Usage Examples

### Example 1: Query with Missing APs

```python
from core.fingerprinting import (
    load_fingerprint_database,
    nn_localize,
    fit_gaussian_naive_bayes,
    map_localize,
)
import numpy as np

# Load database
db = load_fingerprint_database('data/sim/ch5_wifi_fingerprint_grid')

# Query with some APs missing (NaN)
query = np.array([-51.0, np.nan, -71.0, -81.0, np.nan, -65.0])
#                 AP1    AP2     AP3    AP4    AP5    AP6
#                 ✓      X       ✓      ✓      X      ✓

# Deterministic methods work with missing values
pos_nn = nn_localize(query, db, floor_id=0)
print(f"NN position: {pos_nn}")  # Uses AP1, AP3, AP4, AP6

# Probabilistic methods work with missing values
model = fit_gaussian_naive_bayes(db, min_std=2.0)
pos_map = map_localize(query, model, floor_id=0)
print(f"MAP position: {pos_map}")  # Likelihood computed using AP1, AP3, AP4, AP6
```

### Example 2: Database with Missing APs

```python
# Create database where some RPs have missing AP readings
locations = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
features = np.array([
    [-50, -60, -70],      # RP0: All APs visible
    [-60, np.nan, -80],   # RP1: AP2 not visible
    [np.nan, -80, -50],   # RP2: AP1 not visible
], dtype=float)
floor_ids = np.array([0, 0, 0])

db = FingerprintDatabase(
    locations=locations,
    features=features,
    floor_ids=floor_ids,
    meta={"ap_ids": ["AP1", "AP2", "AP3"], "unit": "dBm"}
)

# Localization works normally
query = np.array([-55.0, -65.0, -75.0])
pos = nn_localize(query, db, floor_id=0)
# Distance to RP0: uses all 3 APs
# Distance to RP1: uses only AP1 and AP3
# Distance to RP2: uses only AP2 and AP3
```

### Example 3: Multi-Sample Database with Missing Values

```python
# Database with multiple samples per RP, some samples have missing APs
locations = np.array([[0, 0]], dtype=float)
features = np.array([
    [[-50, -60], [-52, np.nan], [-48, -62], [np.nan, -58], [-51, -59]],
])  # Shape: (1, 5, 2) - 1 RP, 5 samples, 2 APs

db = FingerprintDatabase(
    locations=locations,
    features=features,
    floor_ids=np.array([0]),
    meta={"ap_ids": ["AP1", "AP2"], "unit": "dBm"}
)

# Compute statistics (ignores NaN)
means = db.get_mean_features()
# AP1: mean of [-50, -52, -48, -51] = -50.25 (ignoring NaN)
# AP2: mean of [-60, -62, -58, -59] = -59.75 (ignoring NaN)

stds = db.get_std_features(min_std=0.5)
# AP1: std of [-50, -52, -48, -51] ≈ 1.71
# AP2: std of [-60, -62, -58, -59] ≈ 1.71
```

## Edge Cases Handled

1. **No Overlapping Features:**
   - Distance returns `+inf`
   - RP is effectively excluded from consideration

2. **All Query APs Missing:**
   - Log-likelihood returns `-inf` for all RPs
   - Localization falls back to prior (uniform → center of RPs)

3. **Partial Overlap:**
   - Distance/likelihood computed using only overlapping dimensions
   - System gracefully degrades with reduced information

4. **High Dropout Rates:**
   - Tested up to 50% dropout
   - System remains stable and produces valid estimates

## Implementation Notes

### Why NaN Instead of Mask Arrays?

**Advantages of NaN:**
- Single array representation (no separate mask needed)
- Natural NumPy support (`nanmean`, `nanstd`, `nansum`)
- Clear semantic meaning (missing/unavailable)
- Efficient memory usage

**Disadvantages:**
- Requires careful handling in arithmetic operations
- Need to distinguish NaN from invalid data

### Performance Optimization

The implementation uses a **two-path strategy:**

1. **Fast path:** When no NaN values are present, use vectorized operations
2. **Slow path:** When NaN values are detected, use per-element masking

This ensures:
- No performance regression for existing complete datasets
- Graceful handling of missing values when needed

## Future Enhancements (Optional)

1. **Weighted Dropout Handling:**
   - Weight features by their reliability/availability
   - Downweight frequently-missing APs

2. **Imputation Strategies:**
   - Predict missing AP values using spatial correlation
   - Use neighboring RPs to estimate missing values

3. **Adaptive Thresholding:**
   - Reject RPs with too few overlapping features
   - Require minimum overlap (e.g., 50% of APs)

4. **Dropout-Aware Model Training:**
   - Train models specifically for high-dropout scenarios
   - Learn which AP subsets are most informative

## Conclusion

The implementation successfully adds robust missing AP support to the fingerprinting system while maintaining full backward compatibility. The system now handles real-world scenarios where signal dropout occurs, making it more practical for deployment in challenging indoor environments.

**Key Achievements:**
- ✅ NaN-based missing value representation
- ✅ Deterministic methods handle missing APs
- ✅ Probabilistic methods handle missing APs
- ✅ Database statistics handle missing APs
- ✅ Comprehensive test coverage (28 new tests)
- ✅ 100% backward compatibility (160/160 tests pass)
- ✅ Acceptance criteria met (20% dropout, no crash)









