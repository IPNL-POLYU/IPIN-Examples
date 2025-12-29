# Allan Deviation Equation Mismatch: Book vs. Code

**Author:** Li-Ta Hsu  
**Date:** December 2025

## Book's Equations (6.56-6.58)

From `references/ch6.txt`, lines 283-288:

### Eq. (6.56) - ARW from Allan Deviation Slope
```
σ(τ) = ARW / √τ
```
This describes the relationship between Allan deviation σ(τ) and angle random walk (ARW) on the -1/2 slope region.

### Eq. (6.57) - Log-Linear Form
```
log(σ(τ)) = log(ARW) - 1/2 log(τ)
```
This is the log-linear form used for fitting the Allan deviation curve to extract ARW.

### Eq. (6.58) - ARW to Per-Sample Noise
```
σ_ω = ARW × √Δt
```
Where:
- `σ_ω` is the standard deviation of the white noise per sample
- `ARW` is the angle random walk coefficient (rad/√s or deg/√hr)
- `Δt` is the sampling interval (s)

## Current Code Issues

### Issue 1: Incorrect Equation Labels

**Current code (lines 21-23 in `calibration.py`):**
```python
Eq. (6.56): Cluster averages (binning)
Eq. (6.57): Allan variance definition
Eq. (6.58): Allan deviation (square root of variance)
```

**These are WRONG!** The book's equations 6.56-6.58 have nothing to do with the computational algorithm for Allan variance. They describe:
- How to extract ARW from the slope
- The log-linear relationship
- The conversion from ARW to per-sample noise

### Issue 2: Missing Implementation of Eq. (6.58)

The code does NOT implement the book's Eq. (6.58) relationship: `σ_ω = ARW × √Δt`.

This is a critical conversion that users need when going from ARW (which is what Allan variance gives you) to the per-sample noise standard deviation (which is what you need for simulation and filtering).

### Issue 3: Confusion About What is Computed

The current `allan_variance()` function computes the **standard Allan variance** using the IEEE Std 952-1997 algorithm:
1. Cluster data into bins of size m
2. Compute differences between adjacent cluster averages
3. Compute variance: σ²(τ) = (1/2) E[(θ̄_{k+1} - θ̄_k)²]
4. Take square root to get Allan deviation

This is **correct**, but it's NOT what equations 6.56-6.58 describe!

## What Needs to Be Fixed

### 1. Update Equation References

Remove the incorrect equation references from the main `allan_variance()` function docstring. Instead:
- State that it computes "standard Allan deviation computation (IEEE Std 952-1997)"
- Mention that the result can be used to extract ARW via the book's Eq. (6.56)

### 2. Fix `identify_random_walk()` Documentation

This function DOES implement the book's approach (finding σ(τ) at τ=1s on the -1/2 slope). Update its docstring to:
- Cite Eq. (6.56): σ(τ) = ARW / √τ
- Note that ARW = σ(τ=1s) for the -1/2 slope region

### 3. Implement Eq. (6.58) Helper

Add a new helper function:
```python
def arw_to_noise_std(arw: float, dt: float) -> float:
    """
    Convert angle/velocity random walk to per-sample noise standard deviation.
    
    Implements Eq. (6.58): σ_ω = ARW × √Δt
    
    Args:
        arw: Angle random walk coefficient (rad/√s) or velocity random walk (m/s^(3/2))
        dt: Sampling interval (s)
    
    Returns:
        Per-sample noise standard deviation (rad/s for gyro, m/s² for accel)
    """
    return arw * np.sqrt(dt)
```

### 4. Implement Inverse (Noise to ARW)

Also add the inverse:
```python
def noise_std_to_arw(sigma: float, dt: float) -> float:
    """
    Convert per-sample noise standard deviation to angle/velocity random walk.
    
    Inverse of Eq. (6.58): ARW = σ_ω / √Δt
    
    Args:
        sigma: Per-sample noise standard deviation (rad/s for gyro, m/s² for accel)
        dt: Sampling interval (s)
    
    Returns:
        Angle random walk coefficient (rad/√s) or velocity random walk (m/s^(3/2))
    """
    return sigma / np.sqrt(dt)
```

## Correct Usage Pattern

After fixing:

```python
# Step 1: Compute Allan deviation (standard algorithm, NOT book's Eq. 6.56-6.58)
taus, adev = allan_variance(gyro_data, fs=100.0)

# Step 2: Extract ARW from the -1/2 slope region (book's Eq. 6.56)
arw = identify_random_walk(taus, adev, tau_target=1.0)
# Result: ARW in rad/√s

# Step 3: Convert ARW to per-sample noise (book's Eq. 6.58)
dt = 1.0 / fs
sigma_omega = arw_to_noise_std(arw, dt)
# Result: σ_ω in rad/s (per-sample standard deviation)
```

## Units Consistency

### Gyroscope
- **Allan deviation**: rad/s (or deg/s)
- **ARW (from Eq. 6.56)**: rad/√s (or deg/√hr)
- **Per-sample noise (Eq. 6.58)**: rad/s (or deg/s)

### Accelerometer
- **Allan deviation**: m/s²
- **VRW (from Eq. 6.56)**: m/s^(3/2) or m/(s^(3/2))
- **Per-sample noise (Eq. 6.58)**: m/s²

## Summary

The code's Allan variance computation is **correct**, but the equation numbering is **completely wrong**. The book's equations 6.56-6.58 describe:
1. The relationship σ(τ) = ARW/√τ
2. The log-linear form for fitting
3. The conversion ARW → per-sample noise

The code needs to:
1. Remove incorrect equation citations from the Allan variance algorithm
2. Add correct citations to the ARW extraction function
3. Implement the missing Eq. (6.58) helper functions





