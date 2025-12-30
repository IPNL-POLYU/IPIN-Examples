# Chapter 7 Prompt 3: NDT Math Alignment (Eqs 7.12-7.16) + 2D vs 3D Decision

**Date**: December 2025  
**Author**: Li-Ta Hsu  
**Status**: ✅ COMPLETE

## Task Description

Align the NDT implementation with the book's Eqs (7.12)–(7.16), including proper covariance definition and equation references.

### Book References:
- **Eq. (7.9)**: LiDAR point cloud is 3D: p_i = [x_i, y_i, z_i]^T
- **Eq. (7.12)**: Mean p̄_{k,t-1} = (1/n_k) Σ_{i=1}^{n_k} p_{i,t-1}
- **Eq. (7.13)**: Covariance Σ_{k,t-1} = 1/(n_k-1) Σ (p - p̄)(p - p̄)^T  **[Uses n_k-1, NOT n_k]**
- **Eq. (7.14)**: Likelihood for ONE voxel k
- **Eq. (7.15)**: Joint likelihood across ALL voxels
- **Eq. (7.16)**: MLE objective T̂ = argmin (1/2) Σ_k Σ_j ||T p_j - p̄_k||²_Σ

## Decision: Keep 2D (Decision B - Pedagogical)

**Rationale:**
- Book uses 3D LiDAR (Eq. 7.9), but this codebase focuses on 2D SLAM for educational clarity
- 2D is consistent with other examples (ICP, pose graph) which all use SE(2)
- Mathematical principles are identical in 2D and 3D (mean, covariance, likelihood)
- Explicitly documented the 2D restriction with clear note in docstrings

**Documentation Added:**
> "**Note on 2D vs 3D**: The book presents NDT for 3D LiDAR point clouds (Eq. 7.9).
> This implementation restricts to 2D (x, y) for educational clarity and consistency
> with other 2D SLAM examples. The mathematical principles (mean, covariance, likelihood)
> are identical in 2D and 3D."

## Critical Issue Fixed: Covariance Denominator

### **WRONG (Before)**:
```python
cov = (centered.T @ centered) / len(voxel_points)  # Biased estimator!
```

### **CORRECT (After)**:
```python
n_k = len(voxel_points)
cov = (centered.T @ centered) / (n_k - 1)  # Unbiased estimator per Eq. 7.13
```

**Why this matters:**
- Eq. (7.13) explicitly uses 1/(n_k - 1), not 1/n_k
- This is the unbiased covariance estimator
- Using wrong denominator produces systematically biased covariance matrices
- Affects likelihood computation and alignment quality

## Equation Reference Fixes

### Module Header Fixed

**Before:**
```python
Section 7.2.2: Normal Distributions Transform (NDT)  # WRONG SECTION
Eq. (7.12): NDT score function  # WRONG! Eq. 7.12 is MEAN, not score
Eq. (7.13): Probability density per voxel  # WRONG! Eq. 7.13 is COVARIANCE
Eq. (7.14): Negative log-likelihood formulation  # Imprecise
```

**After:**
```python
Section 7.3.2: Feature-based LiDAR SLAM - NDT  # CORRECT SECTION
Eq. (7.12): Voxel mean p̄_{k,t-1} = (1/n_k) Σ p_{i,t-1}  # CORRECT
Eq. (7.13): Voxel covariance Σ_{k,t-1} = 1/(n_k-1) Σ (p-p̄)(p-p̄)^T  # CORRECT
Eq. (7.14): Likelihood for one voxel k  # CORRECT
Eq. (7.15): Joint likelihood across all voxels  # CORRECT
Eq. (7.16): MLE objective (minimize 0.5 Σ ||T p_j - p̄_k||²_Σ)  # CORRECT
```

### Function Docstring Fixes

#### `build_ndt_map()`

**Changes:**
- Added explicit Eq. (7.12) formula for mean
- Added explicit Eq. (7.13) formula for covariance with (n_k-1) denominator
- Added note about 2D restriction vs book's 3D
- Fixed section reference: 7.2.2 → 7.3.2

**Key addition:**
```python
"""
For each voxel k with n_k points:
    - Mean (Eq. 7.12): p̄_k = (1/n_k) Σ_{i=1}^{n_k} p_i
    - Covariance (Eq. 7.13): Σ_k = 1/(n_k-1) Σ_{i=1}^{n_k} (p_i - p̄_k)(p_i - p̄_k)^T

Notes:
    - Implements Eqs. (7.12)-(7.13) from Section 7.3.2.
    - Uses (n_k - 1) denominator for unbiased covariance estimate (Eq. 7.13).
```

#### `ndt_score()`

**Changes:**
- Was claiming "Eqs. (7.12)-(7.14)" but those are mean, covariance, and single voxel likelihood
- Now correctly references the objective function from Eq. (7.16)
- Added detailed explanation of Eq. (7.14), (7.15), and (7.16)
- Clarified what the function actually computes

**Key addition:**
```python
"""
The likelihood for a single voxel k (Eq. 7.14):
    likelihood_k(T) = ∏_{j=1}^{N} exp( -||T p_j - p̄_k||²_Σ / 2 )

The joint likelihood across all voxels (Eq. 7.15):
    likelihood(T) = ∏_{k=1}^{N_voxel} ∏_{j=1}^{N} exp( -||T p_j - p̄_k||²_Σ / 2 )

The MLE objective to minimize (Eq. 7.16):
    T̂ = argmin_T  (1/2) Σ_k Σ_j ||T p_j - p̄_k||²_Σ

Notes:
    - Implements the MLE objective from Eq. (7.16), Section 7.3.2.
```

#### `ndt_gradient()`

**Changes:**
- Fixed docstring to say "gradient of Eq. (7.16)" not "Eq. (7.15)"
- Eq. (7.15) is the joint likelihood, not the gradient

#### `ndt_align()`

**Changes:**
- Fixed section reference: 7.2.2 → 7.3.2
- Added note about 2D restriction
- Clarified that it uses Eqs. (7.12)-(7.13) for map building
- Clarified that it minimizes Eq. (7.16) objective

#### `ndt_covariance()`

**Changes:**
- Fixed reference to say "Hessian of Eq. (7.16)" not wrong equation

## Changes Made

### 1. ✅ Fixed Module Header (`core/slam/ndt.py`)
- Updated section: 7.2.2 → 7.3.2
- Fixed all equation descriptions to match book
- Added 2D vs 3D note

### 2. ✅ Fixed Covariance Calculation
**File**: `core/slam/ndt.py`, line ~109

**Changed from:**
```python
cov = (centered.T @ centered) / len(voxel_points)  # WRONG: biased estimator
```

**Changed to:**
```python
n_k = len(voxel_points)
# Compute covariance (Eq. 7.13): Σ_k = 1/(n_k-1) Σ (p_i - p̄_k)(p_i - p̄_k)^T
# Note: For n_k=1, use biased estimator (divide by n_k) to avoid division by zero
if n_k > 1:
    cov = (centered.T @ centered) / (n_k - 1)  # Unbiased estimator (Eq. 7.13)
else:
    # Single point: use biased estimator to avoid division by zero
    cov = (centered.T @ centered) / n_k
```

**Edge case handling**: When `n_k=1`, we use the biased estimator to avoid division by zero. This is acceptable since a single-point voxel has zero variance anyway (and we add regularization).

### 3. ✅ Fixed All Function Docstrings
- `build_ndt_map()`: Added Eq. 7.12-7.13 formulas, 2D note
- `ndt_score()`: Fixed to reference Eq. 7.16 (not 7.12-7.14)
- `ndt_gradient()`: Fixed to reference gradient of Eq. 7.16
- `ndt_align()`: Added complete equation references
- `ndt_covariance()`: Fixed Hessian reference

### 4. ✅ Updated Section References Throughout
All references changed from "Section 7.2.2" to "Section 7.3.2"

## Verification

### ✅ Covariance Uses (n-1) Denominator
```python
# Lines ~120-126 in core/slam/ndt.py
if n_k > 1:
    cov = (centered.T @ centered) / (n_k - 1)  ✓ (Eq. 7.13)
else:
    cov = (centered.T @ centered) / n_k  ✓ (edge case: avoid division by zero)
```

### ✅ Equation References Correct
- Eq. (7.12) → Mean calculation ✓
- Eq. (7.13) → Covariance with (n_k-1) ✓  
- Eq. (7.14) → Single voxel likelihood ✓
- Eq. (7.15) → Joint likelihood ✓
- Eq. (7.16) → MLE objective ✓

### ✅ Section References Correct
All references now say "Section 7.3.2" ✓

### ✅ README Points to Correct File
- `core/slam/ndt.py` ✓ (not scan_matching.py)

### ✅ No Linter Errors
```
core/slam/ndt.py: ✓ No errors
```

### ✅ All Unit Tests Pass
```
tests/core/slam/test_ndt.py: 31 passed ✓
tests/core/slam/test_ndt_voxel_stats.py: 5 passed, 1 skipped ✓
```

### ✅ 2D Restriction Documented
Explicit note added about 2D vs 3D throughout ✓

## Summary of Equation Mapping

| Function | What It Computes | Book Equation | Fixed? |
|----------|------------------|---------------|--------|
| `build_ndt_map()` - mean | Voxel mean p̄_k | Eq. (7.12) | ✅ |
| `build_ndt_map()` - cov | Voxel covariance Σ_k | Eq. (7.13) with (n_k-1) | ✅ |
| `ndt_score()` | Negative log-likelihood | Eq. (7.16) objective | ✅ |
| `ndt_gradient()` | Gradient of objective | ∇ Eq. (7.16) | ✅ |
| `ndt_align()` | Full NDT alignment | Eqs. (7.12)-(7.16) | ✅ |

## Files Modified

1. `core/slam/ndt.py` - Fixed all math, equations, and docstrings
2. `.dev/ch7_prompt3_ndt_equations_fix_summary.md` - This summary

## Acceptance Criteria: ALL MET ✅

✅ `build_ndt_map` computes μ and Σ consistent with Eq. (7.12)/(7.13)  
✅ Covariance uses (n_k-1) factor per Eq. (7.13)  
✅ `ndt_score/ndt_align` documentation references correct equations (7.14-7.16)  
✅ README correctly points to `core/slam/ndt.py` for `ndt_align()`  
✅ All section references updated to 7.3.2  
✅ 2D restriction explicitly documented  
✅ No factually wrong equation claims  

## Next Steps

Ready for Prompt 4 which will address additional equation fixes and implementation details.

