# Prompt 4 Summary: Allan Deviation Equation References (6.56-6.58)

**Author:** Li-Ta Hsu  
**Date:** December 2025  
**Status:** ✅ COMPLETED

## Objective

Fix Allan deviation equation references to match the book's equations (6.56-6.58):
- **Eq. (6.56)**: σ(τ) = ARW/√τ (ARW extraction from Allan deviation slope)
- **Eq. (6.57)**: log(σ(τ)) = log(ARW) - (1/2)log(τ) (log-linear form)
- **Eq. (6.58)**: σ_ω = ARW × √Δt (ARW to per-sample noise conversion)

## Problem Summary

The original code **incorrectly** labeled:
- Eq. (6.56) as "Cluster averages (binning)" ❌
- Eq. (6.57) as "Allan variance definition" ❌
- Eq. (6.58) as "Allan deviation (square root of variance)" ❌

These are **WRONG**! The book's equations 6.56-6.58 describe how to **extract ARW** from Allan deviation curves and **convert** it to per-sample noise, NOT the computational algorithm.

## Changes Made

### 1. Module Docstring (`core/sensors/calibration.py`)

**Before:**
```python
References:
    Eq. (6.56): Cluster averages (binning)
    Eq. (6.57): Allan variance definition
    Eq. (6.58): Allan deviation (square root of variance)
    IEEE Std 952-1997: Allan variance standard
```

**After:**
```python
References:
    Eq. (6.56): ARW extraction from Allan deviation: σ(τ) = ARW/√τ
    Eq. (6.57): Log-linear form: log(σ(τ)) = log(ARW) - (1/2)log(τ)
    Eq. (6.58): ARW to per-sample noise: σ_ω = ARW × √Δt
    IEEE Std 952-1997: Allan variance computational algorithm
```

### 2. `allan_variance()` Function Docstring

**Before:**
```python
Implements Eqs. (6.56)-(6.58) in Chapter 6:
    1. Cluster/bin the data into averaging intervals tau (Eq. 6.56)
    2. Compute Allan variance: σ²(τ) = (1/2) E[(θ̄_{k+1} - θ̄_k)²] (Eq. 6.57)
    3. Allan deviation: σ(τ) = √(σ²(τ)) (Eq. 6.58)
```

**After:**
```python
Implements the standard Allan variance computational algorithm (IEEE Std 952-1997):
    1. Cluster/bin the data into averaging intervals tau
    2. Compute Allan variance: σ²(τ) = (1/2) E[(θ̄_{k+1} - θ̄_k)²]
    3. Allan deviation: σ(τ) = √(σ²(τ))

The resulting Allan deviation curve reveals noise characteristics:
    - Quantization noise: slope -1 on log-log plot
    - Angle/velocity random walk: slope -1/2 (extract ARW via Eq. 6.56)
    - Bias instability: flat region (minimum of curve)
    - Rate random walk: slope +1/2
    - Rate ramp: slope +1
```

Removed incorrect equation citations and clarified that Eq. 6.56 is used to **extract** ARW from the curve, not to compute it.

### 3. `identify_random_walk()` Function Docstring

**Before:**
```python
def identify_random_walk(...) -> float:
    """
    Identify angle/velocity random walk coefficient from Allan deviation.

    Angle random walk (ARW) or velocity random walk (VRW) appears as a
    slope of -1/2 on the log-log Allan deviation plot at short averaging
    times. It characterizes the white noise in the sensor output.

    The random walk coefficient N is found from:
        σ(τ=1s) = N
    ...
    """
```

**After:**
```python
def identify_random_walk(...) -> float:
    """
    Identify angle/velocity random walk coefficient from Allan deviation.

    Implements the book's Eq. (6.56) extraction method:
        σ(τ) = ARW / √τ

    Angle random walk (ARW) or velocity random walk (VRW) appears as a
    slope of -1/2 on the log-log Allan deviation plot at short averaging
    times. It characterizes the white noise in the sensor output.

    The random walk coefficient is found by reading σ(τ) at τ=1s on the
    -1/2 slope region (book's Eq. 6.56 and 6.57):
        ARW = σ(τ=1s)
    ...
    - To convert ARW to per-sample noise σ_ω, use arw_to_noise_std() (Eq. 6.58).
    ...
    """
```

Added explicit reference to Eq. (6.56) and connection to Eq. (6.58) conversion.

### 4. New Functions Implementing Eq. (6.58)

Added two new helper functions to implement the book's Eq. (6.58):

#### `arw_to_noise_std(arw, dt)` - Forward Conversion

```python
def arw_to_noise_std(arw: float, dt: float) -> float:
    """
    Convert angle/velocity random walk to per-sample noise standard deviation.

    Implements the book's Eq. (6.58):
        σ_ω = ARW × √Δt

    This converts the angle random walk (ARW) coefficient (typically extracted
    from Allan deviation analysis) to the per-sample noise standard deviation
    needed for IMU simulation and filtering.

    Args:
        arw: Angle random walk coefficient (ARW for gyro, VRW for accel).
             Units: rad/√s (gyro) or m/s^(3/2) (accel).
        dt: Sampling interval (seconds).

    Returns:
        Per-sample noise standard deviation σ_ω.
        Units: rad/s (gyro) or m/s² (accel).
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if arw < 0:
        raise ValueError(f"arw must be non-negative, got {arw}")

    # Eq. (6.58): σ_ω = ARW × √Δt
    sigma_omega = arw * np.sqrt(dt)

    return sigma_omega
```

#### `noise_std_to_arw(sigma, dt)` - Inverse Conversion

```python
def noise_std_to_arw(sigma: float, dt: float) -> float:
    """
    Convert per-sample noise standard deviation to angle/velocity random walk.

    Inverse of the book's Eq. (6.58):
        ARW = σ_ω / √Δt

    Args:
        sigma: Per-sample noise standard deviation σ_ω (rad/s or m/s²).
        dt: Sampling interval (seconds).

    Returns:
        Angle random walk coefficient (rad/√s or m/s^(3/2)).
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    # Inverse of Eq. (6.58): ARW = σ_ω / √Δt
    arw = sigma / np.sqrt(dt)

    return arw
```

### 5. Tests Added

Added 11 new unit tests in `tests/core/sensors/test_sensors_calibration.py`:

1. `test_arw_to_noise_std_basic` - Basic forward conversion
2. `test_noise_std_to_arw_basic` - Basic inverse conversion
3. `test_arw_noise_roundtrip` - Roundtrip consistency
4. `test_arw_to_noise_different_sample_rates` - Different sampling rates
5. `test_arw_to_noise_realistic_gyro` - Realistic gyro parameters
6. `test_arw_to_noise_realistic_accel` - Realistic accel parameters
7. `test_arw_to_noise_invalid_inputs` - Input validation
8. `test_noise_to_arw_invalid_inputs` - Input validation
9. `test_arw_noise_consistency_with_allan` - Integration with Allan variance
10. `test_arw_to_noise_zero_arw` - Edge case (noiseless)
11. `test_noise_to_arw_zero_noise` - Edge case (noiseless)

**Result:** All 36 tests pass ✅

## Verification

Created `.dev/ch6_verify_prompt4_allan_equations.py` to verify:

1. ✅ **Eq. (6.58) implementation**: Correctly implements σ_ω = ARW × √Δt
2. ✅ **Eq. (6.56) ARW extraction**: `identify_random_walk()` correctly extracts ARW
3. ✅ **Documentation references**: All equation numbers match the book
4. ✅ **Units consistency**: ARW units (rad/√s, m/s^(3/2)) match the book
5. ✅ **Function signatures**: New functions have correct parameters
6. ✅ **Tests coverage**: All 11 new tests pass

**Result:** All 6 acceptance checks PASSED ✅

## Usage Example

### Before (Missing Eq. 6.58):
```python
# Extract ARW from Allan variance
taus, adev = allan_variance(gyro_data, fs=100.0)
arw = identify_random_walk(taus, adev, tau_target=1.0)
# Result: ARW in rad/√s

# ❌ No way to convert to per-sample noise!
# Users had to manually implement: sigma = arw * sqrt(dt)
```

### After (With Eq. 6.58):
```python
# Step 1: Extract ARW from Allan variance (Eq. 6.56)
taus, adev = allan_variance(gyro_data, fs=100.0)
arw = identify_random_walk(taus, adev, tau_target=1.0)
# Result: ARW = 0.01 rad/√s

# Step 2: Convert to per-sample noise (Eq. 6.58) ✅
dt = 1.0 / 100.0  # 100 Hz
sigma_omega = arw_to_noise_std(arw, dt)
# Result: σ_ω = 0.001 rad/s per sample

# Step 3: Use in simulation
gyro_noise = np.random.randn(N) * sigma_omega
```

## Files Modified

1. **Core:**
   - `core/sensors/calibration.py` (added 2 functions, fixed docstrings)

2. **Tests:**
   - `tests/core/sensors/test_sensors_calibration.py` (+11 tests, 36 total)

3. **Documentation:**
   - `.dev/ch6_allan_eq_analysis.md` (analysis document)
   - `.dev/ch6_verify_prompt4_allan_equations.py` (acceptance script)
   - `.dev/ch6_prompt4_allan_equations_summary.md` (this file)

## Key Improvements

1. **Book Alignment**: Equation numbers now match the book exactly
2. **Eq. (6.58) Implemented**: Missing conversion formula now available
3. **Clear Separation**: Computational algorithm (IEEE) vs. extraction/conversion (book)
4. **Units Clarity**: ARW units (rad/√s) explicitly documented
5. **Complete Workflow**: Allan → ARW → per-sample noise fully supported

## Acceptance Criteria Status

✅ **Criterion 1:** Equation numbering in code matches the book's numbering
   - Eq. (6.56): ARW extraction from Allan deviation slope
   - Eq. (6.57): Log-linear form for fitting
   - Eq. (6.58): ARW to per-sample noise conversion

✅ **Criterion 2:** "ARW" returned by code is clearly in the same units implied by the book
   - Gyro: rad/√s (or deg/√hr)
   - Accel: m/s^(3/2) or m/(s^(3/2))
   - Per-sample noise: rad/s (gyro) or m/s² (accel)

## Next Steps

Prompt 4 is **COMPLETE** ✅. Ready for next prompt or integration verification.

---

*This prompt corrects a fundamental misunderstanding in the original code: the book's equations 6.56-6.58 describe how to USE Allan variance results, not how to COMPUTE them. The code now correctly implements both the standard Allan variance algorithm (IEEE Std 952-1997) AND the book's extraction/conversion formulas (Eqs. 6.56-6.58).*









