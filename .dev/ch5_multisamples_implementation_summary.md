# Chapter 5: Multi-Sample Fingerprinting Implementation Summary

**Date:** December 24, 2024  
**Task:** P0 — Make probabilistic fingerprinting actually estimate μ and σ from survey samples

## Summary

Successfully implemented **Option A (preferred)**: Extended database format to support multiple samples per RP, enabling proper μ and σ estimation from survey data as described in the book's Eq. (5.6).

## Implementation Approach

### Option A: Extended Database Format
- Extended `FingerprintDatabase` to support both single-sample (M, N) and multi-sample (M, S, N) formats
- Backward compatible with existing single-sample datasets
- Proper variance estimation when multiple samples available

## Changes Made

### 1. core/fingerprinting/types.py

**Extended FingerprintDatabase class:**
- `features` attribute now supports two formats:
  - Single sample: shape (M, N) - one fingerprint per RP
  - Multiple samples: shape (M, S, N) - S samples per RP
  
**New properties:**
```python
@property
def n_samples_per_rp(self) -> Optional[int]:
    """Number of samples per RP (None for single-sample format)"""
    
@property
def has_multiple_samples(self) -> bool:
    """True if database contains multiple samples per RP"""
```

**New methods:**
```python
def get_mean_features(self) -> np.ndarray:
    """Get mean features across samples (shape: M, N)"""
    
def get_std_features(self, min_std: float = 0.0) -> np.ndarray:
    """Get std features across samples (shape: M, N)"""
```

**Validation updates:**
- Features can be 2D or 3D
- `n_features` property uses last dimension (always N)
- Updated `__repr__` to show samples_per_rp when applicable

### 2. core/fingerprinting/probabilistic.py

**Updated fit_gaussian_naive_bayes():**

**Before:**
```python
# Always used min_std everywhere (no actual variance estimation)
means = db.features.copy()
stds = np.full((M, N), min_std)
```

**After:**
```python
# Computes actual μ and σ from samples when available
means = db.get_mean_features()  # Handles both formats
stds = db.get_std_features(min_std=min_std)  # Actual variance + floor
```

**Behavior:**
- Single-sample DB: Sets σ_ij = min_std everywhere (backward compatible)
- Multi-sample DB: Computes actual μ and σ from S samples, applies min_std as floor
- Aligns with book's Eq. (5.6): P(z | x_i) = ∏_j N(z_j; μ_ij, σ_ij²)

**Updated docstring:**
- Clarifies behavior for both database formats
- References book Section 5.1.3 and Eq. (5.6)
- Notes that min_std is a numerical floor, not the primary source

### 3. core/fingerprinting/deterministic.py

**Updated nn_localize() and knn_localize():**
- Now use `db.get_mean_features()` instead of `db.features` directly
- Ensures deterministic methods work with both database formats
- Uses mean across samples for distance computation

### 4. core/fingerprinting/dataset.py

**Updated validate_database():**
- Uses `get_mean_features()` for variance checks
- Adds `has_multiple_samples` and `n_samples_per_rp` to stats
- Handles duplicate locations appropriately for multi-sample DBs

**Updated print_database_summary():**
- Shows samples_per_rp when applicable
- Displays within-RP variability statistics for multi-sample DBs
- Uses mean features for cross-RP statistics

### 5. scripts/generate_ch5_wifi_fingerprint_dataset.py

**Added n_samples_per_rp parameter:**
```python
def generate_wifi_fingerprint_database(
    ...
    n_samples_per_rp: int = 1,  # NEW PARAMETER
    ...
) -> FingerprintDatabase:
```

**Implementation:**
- Collects S independent samples at each RP
- Each sample gets independent shadow fading (realistic variance)
- Returns shape (M, N) if n_samples=1, (M, S, N) if n_samples>1

**New preset:**
- `multisamples`: Standard grid with 10 samples/RP for proper μ/σ estimation
- Saves to `data/sim/ch5_wifi_fingerprint_multisamples/`

**CLI argument:**
```bash
--n-samples INT    Number of RSS samples per RP (default: 1)
```

### 6. ch5_fingerprinting/test_multisamples.py

**Created comprehensive test script:**
- Test 1: Single-sample DB backward compatibility
- Test 2: Multi-sample DB with proper μ and σ estimation
- Test 3: Localization behavior changes with varying σ

**Test Results:**
```
[OK] Created single-sample DB: FingerprintDatabase(n_rps=4, n_features=3, ...)
  All stds = min_std? True

[OK] Created multi-sample DB: FingerprintDatabase(n_rps=3, n_features=2, samples_per_rp=5, ...)
  Model uses actual variance from samples: True
  Std range: [0.71, 6.32] dBm
  
  Average std per RP:
    RP0 (low var):    0.76 dBm
    RP1 (medium var): 2.36 dBm
    RP2 (high var):   4.96 dBm

ALL TESTS PASSED OK
```

## Acceptance Criteria

✅ **For a DB with repeated samples per RP, model.stds varies by RP and/or feature**
- Test 2 demonstrates stds ranging from 0.71 to 6.32 dBm
- Different RPs have different average stds (0.76, 2.36, 4.96 dBm)

✅ **Probabilistic localization behavior changes when σ differs**
- Test 3 creates two models with same means but different variances
- High variance at RP reduces its posterior probability influence
- Demonstrates book's narrative about variability affecting localization

✅ **Existing synthetic datasets continue to load**
- Test 1 validates backward compatibility with single-sample format
- All existing ch5 example scripts work without modification
- Deterministic methods handle both formats transparently

## Usage Examples

### Generate Multi-Sample Database

```bash
# Using preset
python scripts/generate_ch5_wifi_fingerprint_dataset.py --preset multisamples

# Custom parameters
python scripts/generate_ch5_wifi_fingerprint_dataset.py \
    --n-samples 10 \
    --output data/sim/my_multisamples
```

### Use in Probabilistic Fingerprinting

```python
from core.fingerprinting import (
    load_fingerprint_database,
    fit_gaussian_naive_bayes,
    map_localize,
)

# Load multi-sample database
db = load_fingerprint_database('data/sim/ch5_wifi_fingerprint_multisamples')
print(f"Database: {db}")
# Output: FingerprintDatabase(n_rps=363, n_features=8, samples_per_rp=10, ...)

# Fit model - automatically computes μ and σ from samples
model = fit_gaussian_naive_bayes(db, min_std=1.0)
print(f"Std range: [{model.stds.min():.2f}, {model.stds.max():.2f}] dBm")
# Output: Std range: [1.23, 5.67] dBm (varies by RP and AP!)

# Localization works as before
query = np.array([-50, -60, -70, -80, -55, -65, -75, -85])
position = map_localize(query, model, floor_id=0)
```

### Backward Compatibility

```python
# Single-sample databases work exactly as before
db_single = load_fingerprint_database('data/sim/ch5_wifi_fingerprint_grid')
print(f"Database: {db_single}")
# Output: FingerprintDatabase(n_rps=363, n_features=8, location_dim=2, ...)

model_single = fit_gaussian_naive_bayes(db_single, min_std=2.0)
# All stds will be 2.0 dBm (uniform, as before)
```

## Key Implementation Details

### Database Format Detection

The `FingerprintDatabase` class automatically detects format:
```python
if db.has_multiple_samples:
    # features.shape = (M, S, N)
    # Use actual variance from S samples
    stds = np.std(features, axis=1, ddof=1)
else:
    # features.shape = (M, N)
    # Use min_std everywhere (backward compatible)
    stds = np.full((M, N), min_std)
```

### Variance Estimation

For multi-sample format:
```python
# Sample standard deviation (ddof=1 for unbiased estimator)
stds = np.std(db.features, axis=1, ddof=1)  # Shape: (M, N)

# Apply minimum floor for numerical stability
stds = np.maximum(stds, min_std)
```

### Shadow Fading Independence

Each sample at an RP gets independent shadow fading:
```python
for sample_idx in range(n_samples_per_rp):
    # Each call generates new random shadow fading
    rss = log_distance_path_loss(
        distance_3d,
        P0=-30.0,
        n=2.5,
        sigma=4.0,  # Independent per sample
    )
```

This creates realistic within-RP variance that reflects temporal/spatial RSS fluctuations.

## Book Alignment

### Eq. (5.6): Gaussian Likelihood Model

**Book:** P(z | x_i) = ∏_j N(z_j; μ_ij, σ_ij²)

**Implementation:**
- μ_ij: Computed from sample mean at RP i, feature j
- σ_ij: Computed from sample std at RP i, feature j
- min_std acts as numerical floor (prevents σ=0)

### Section 5.1.3: Probabilistic Fingerprinting

**Book assumption:** "During the offline phase, one can build statistical models or empirical histograms of Z_i for each reference location."

**Implementation:** Multi-sample format (M, S, N) provides the S samples needed to build these statistical models (Gaussian with empirical μ and σ).

## Testing

Run validation tests:
```bash
python ch5_fingerprinting/test_multisamples.py
```

Expected output:
- Test 1: Backward compatibility ✓
- Test 2: Multi-sample variance estimation ✓
- Test 3: Behavior changes with varying σ ✓

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `core/fingerprinting/types.py` | Extended DB format, new methods | ~60 |
| `core/fingerprinting/probabilistic.py` | Updated fit function | ~30 |
| `core/fingerprinting/deterministic.py` | Use mean features | ~10 |
| `core/fingerprinting/dataset.py` | Validation updates | ~20 |
| `scripts/generate_ch5_wifi_fingerprint_dataset.py` | Multi-sample generation | ~50 |
| `ch5_fingerprinting/test_multisamples.py` | Validation tests | 280 (new) |

## Conclusion

The implementation successfully:
1. ✅ Extends database format to support multiple samples per RP
2. ✅ Computes proper μ and σ from survey samples (Eq. 5.6)
3. ✅ Changes probabilistic localization behavior with varying σ
4. ✅ Maintains backward compatibility with single-sample datasets
5. ✅ Aligns with book's theoretical framework for probabilistic fingerprinting

The code now properly implements the book's assumption that "sufficient survey samples are available to estimate P(z|x_i)" rather than using a constant variance everywhere.













