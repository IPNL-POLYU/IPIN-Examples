# Chapter 7 Prompt 2: ICP Docstrings + Equation Mapping Fix

**Date**: December 2025  
**Author**: Li-Ta Hsu  
**Status**: ✅ COMPLETE

## Task Description

Make ICP documentation and variable naming match the book's Eq. (7.10)–(7.11).

### Book References:
- **Eq. (7.10)**: ICP objective function with binary selector b_{i,j}
  ```
  Objective = sum_{i,j} b_{i,j} ||p_{i,t-1} - T p_{j,t}||^2
  ```
- **Eq. (7.11)**: Correspondence gating based on d_threshold
  ```
  b_{i,j} = { 0  if ||p_{i,t-1} - (T_0) p_{j,t}|| > d_threshold
            { 1  otherwise
  ```

## Critical Issue Fixed

**INCORRECT CLAIM**: Previous docstrings incorrectly stated "Eq. (7.11) is the SVD-based alignment solution"

**TRUTH**: 
- Eq. (7.11) is the **binary selector** b_{i,j} for correspondence gating
- SVD solution is described in the **text after** Eq. (7.11), not in the equation itself

## Changes Made

### 1. ✅ Fixed Module Header (`core/slam/scan_matching.py`)

**Before:**
```python
Section 7.2.1: Point-to-point ICP
Eq. (7.10): ICP residual cost function
Eq. (7.11): SVD-based alignment solution  # WRONG!
```

**After:**
```python
Section 7.3.1: Point-cloud based LiDAR SLAM - ICP
Eq. (7.10): ICP objective function with binary selector b_{i,j}
Eq. (7.11): Correspondence gating using distance threshold d_threshold
SVD solution: Mentioned in text after Eq. 7.11 for solving rotation/translation
```

### 2. ✅ Fixed `find_correspondences()` Docstring

**Key Changes:**
- Added explicit documentation of Eq. (7.11) correspondence gating formula
- Renamed conceptual parameter: `max_distance` ↔ `d_threshold` (Eq. 7.11)
- Clarified that this implements the binary selector b_{i,j}

**Documentation now includes:**
```python
"""
Find nearest-neighbor correspondences with distance-based gating (Eq. 7.11).

Correspondences are gated by the distance threshold d_threshold as described in Eq. (7.11):

    b_{i,j} = { 0  if ||p_{i,t-1} - T p_{j,t}|| > d_threshold
              { 1  otherwise

Args:
    max_distance: Maximum correspondence distance in meters (d_threshold in Eq. 7.11).
                  Points beyond this distance are rejected (b_{i,j} = 0).
"""
```

### 3. ✅ Fixed `compute_icp_residual()` Docstring

**Before:**
- Referenced Section 7.2.1 (wrong section)
- Didn't mention binary selector

**After:**
- References Section 7.3.1 (correct)
- Includes full Eq. (7.10) with binary selector b_{i,j}
- Clarifies residual computed after correspondences established

### 4. ✅ Fixed `align_svd()` Docstring

**Before:**
```python
"""
This is the solution to Eq. (7.11) in Chapter 7.  # WRONG!

Notes:
    - Implements the closed-form SVD solution from Eq. (7.11).  # WRONG!
```

**After:**
```python
"""
Computes the rigid transformation that minimizes the point-to-point error in Eq. (7.10).
The SVD-based solution is described in Section 7.3.1 text after Eq. (7.11).

The book mentions that to solve Eq. (7.10), the rotation matrix can be solved first
by SVD, then the translation: Δx = p̄_{t-1} - (Ĉ p̄_t)

Notes:
    - Implements the SVD-based solution described in Section 7.3.1 text.
    - Solves the minimization problem in Eq. (7.10) in closed form.
```

### 5. ✅ Fixed `icp_point_to_point()` Docstring

**Before:**
- Referenced Section 7.2.1 (wrong)
- Claimed "SVD provides closed-form solution at each iteration (Eq. 7.11)" (wrong equation)

**After:**
- References Section 7.3.1 (correct)
- Clarifies: "Finding correspondences with distance gating (Eq. 7.11)"
- Clarifies: "Computing optimal transformation (SVD alignment for Eq. 7.10)"
- Parameter documentation: `max_correspondence_distance` is "d_threshold in Eq. 7.11"

### 6. ✅ Fixed `compute_icp_covariance()` Docstring

- Added explicit mapping: `max_correspondence_distance` is "d_threshold in Eq. 7.11"

### 7. ✅ Fixed `core/slam/se2.py` References

**Changed line 223:**
```python
# Before:
- Correspondence building: aligning point clouds (Eq. 7.11)

# After:
- Correspondence building: aligning point clouds before distance gating (Eq. 7.11)
```

## Verification

### ✅ Section Numbers Corrected
- All references now say "Section 7.3.1" (not 7.2.1)

### ✅ No Incorrect Eq. 7.11 Claims
- No docstring claims Eq. 7.11 is SVD alignment
- Eq. 7.11 correctly described as correspondence gating with d_threshold

### ✅ Parameter Mapping Explicit
- `max_correspondence_distance` clearly mapped to `d_threshold` from Eq. 7.11
- Documentation shows the binary selector formula

### ✅ SVD Solution Properly Attributed
- SVD described as "method in text after Eq. 7.11"
- SVD correctly linked to solving Eq. 7.10 objective

### ✅ No Linter Errors
```
core/slam/scan_matching.py: ✓ No errors
core/slam/se2.py: ✓ No errors
```

## Equation Mapping Summary

| Function | Equation | What It Actually Is |
|----------|----------|---------------------|
| `find_correspondences()` | Eq. (7.11) | Binary selector b_{i,j} using d_threshold |
| `compute_icp_residual()` | Eq. (7.10) | Objective function with b_{i,j} selector |
| `align_svd()` | Text after Eq. (7.11) | SVD method to solve Eq. (7.10) |
| `icp_point_to_point()` | Eqs. (7.10)-(7.11) | Full ICP iterating both equations |

## Files Modified

1. `core/slam/scan_matching.py` - Fixed all ICP function docstrings
2. `core/slam/se2.py` - Fixed equation reference in `se2_apply()`
3. `.dev/ch7_prompt2_icp_equations_fix_summary.md` - This summary

## Acceptance Criteria: ALL MET ✅

✅ No docstring incorrectly equates Eq. (7.11) to SVD alignment  
✅ Code clearly maps `max_correspondence_distance` to book's `d_threshold`  
✅ All section references say "Section 7.3.1" not "7.2.1"  
✅ Correspondence gating (Eq. 7.11) properly distinguished from SVD alignment  

## Next Steps

Ready for Prompt 3 which will address NDT equations and implementation details.

