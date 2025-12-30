# Chapter 7 Prompt 4: Fix Camera Distortion Model (Eqs 7.40-7.43)

**Date**: December 2025  
**Author**: Li-Ta Hsu  
**Status**: ✅ COMPLETE

## Task Description

Fix the camera intrinsic and distortion model to match the book's Equations (7.40)-(7.43), including adding the k3 parameter and correcting all equation references.

### Book References:
- **Eq. (7.40)**: Intrinsic matrix model s p^pixel = K p^C
- **Eq. (7.41)**: Distortion model with k1, k2, k3 (radial) and d1, d2 (tangential, called p1, p2 in formula)
- **Eqs. (7.42)-(7.43)**: Pixel projection after distortion: u = fx*x̂ + cx, v = fy*ŷ + cy

## Critical Issue Fixed: Missing k3 Parameter

### **The Problem**:
The distortion model was missing the **third radial distortion coefficient k3** (the r⁶ term) specified in Eq. (7.41).

**Book's Eq. (7.41)**:
```
x̂ = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
ŷ = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p2*x*y + p1*(r² + 2*y²)
```

**Before (WRONG)**:
```python
# CameraIntrinsics dataclass - MISSING k3
k1: float = 0.0
k2: float = 0.0  # <-- No k3!
p1: float = 0.0
p2: float = 0.0

# distort_normalized function - MISSING k3*r^6 term
radial_distortion = 1.0 + k1 * r_squared + k2 * r_squared**2  # <-- No k3*r^6!
```

**After (CORRECT)**:
```python
# CameraIntrinsics dataclass - k3 added
k1: float = 0.0  # Radial distortion (Eq. 7.41)
k2: float = 0.0  # Radial distortion (Eq. 7.41)
k3: float = 0.0  # Radial distortion (Eq. 7.41) - 3rd order term
p1: float = 0.0  # Tangential distortion (Eq. 7.41, book calls it d1)
p2: float = 0.0  # Tangential distortion (Eq. 7.41, book calls it d2)

# distort_normalized function - k3*r^6 term included
radial_distortion = 1.0 + k1 * r_squared + k2 * r_squared**2 + k3 * r_squared**3  ✓
```

### Why k3 Matters:
- **Higher-order correction**: k3 provides additional correction for large distortions far from the image center
- **Standard in OpenCV**: Modern camera calibration uses k3 routinely
- **Book specifies it**: Eq. (7.41) explicitly includes the k3*r⁶ term
- **Default value 0**: Backward compatible - setting k3=0 recovers the old 2-parameter radial model

## Equation Reference Fixes

### Before (WRONG References):

| File | What Was Wrong |
|------|----------------|
| `types.py` | Claimed "Eq. (7.43)-(7.46)" for distortion, but those equations don't exist! |
| `camera.py` | `distort_normalized()` referenced "Eqs. (7.43)-(7.46)" - WRONG! |
| `camera.py` | `project_point()` referenced "Eqs. (7.43)-(7.46)" for distortion - WRONG! |
| `factors.py` | `create_reprojection_factor()` said "Eqs. 7.43-7.46, 7.40" - WRONG! |
| `test_camera.py` | Tests claimed "Eqs. 7.43-7.46" - WRONG! |

### After (CORRECT References):

| Component | Correct Equation | Description |
|-----------|------------------|-------------|
| Intrinsic matrix K | **Eq. (7.40)** | s p^pixel = K p^C |
| Distortion model | **Eq. (7.41)** | Radial (k1, k2, k3) + Tangential (p1, p2) |
| Pixel u coordinate | **Eq. (7.42)** | u = fx*x̂ + cx |
| Pixel v coordinate | **Eq. (7.43)** | v = fy*ŷ + cy |

**Note**: The book mentions "(k1, k2, k3) and (d1, d2)" for the coefficients but then uses p1, p2 in the formula. This follows OpenCV convention where tangential distortion uses p1, p2. We keep p1, p2 and document that the book calls them d1, d2.

## Changes Made

### 1. ✅ Extended CameraIntrinsics Dataclass (`core/slam/types.py`)

**Added k3 parameter**:
```python
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0  # Radial distortion (Eq. 7.41)
    k2: float = 0.0  # Radial distortion (Eq. 7.41)
    k3: float = 0.0  # Radial distortion (Eq. 7.41) - NEW!
    p1: float = 0.0  # Tangential distortion (Eq. 7.41, book calls it d1)
    p2: float = 0.0  # Tangential distortion (Eq. 7.41, book calls it d2)
```

**Updated methods**:
- `has_distortion()`: Now checks k3 in addition to k1, k2, p1, p2
- `__repr__()`: Now displays k3 in the distortion parameters

**Fixed docstrings**:
- Corrected all equation references (7.40, 7.41, 7.42-7.43)
- Added note about d1/d2 vs p1/p2 naming convention
- Listed all 5 distortion coefficients with correct descriptions

### 2. ✅ Updated distort_normalized() (`core/slam/camera.py`)

**Added k3 parameter and term**:
```python
def distort_normalized(
    xy_normalized: np.ndarray,
    k1: float,
    k2: float,
    k3: float,  # NEW!
    p1: float,
    p2: float,
) -> np.ndarray:
    """
    Apply radial and tangential distortion (Eq. 7.41).
    
    x̂ = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
    ŷ = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p2*x*y + p1*(r² + 2*y²)
    """
    # ...
    radial_distortion = 1.0 + k1 * r_squared + k2 * r_squared**2 + k3 * r_squared**3  ✓
```

**Fixed docstrings**:
- References Eq. (7.41) instead of non-existent "Eqs. (7.43)-(7.46)"
- Shows complete distortion formula from the book
- Documents k3 as third radial distortion coefficient

### 3. ✅ Updated undistort_normalized() (`core/slam/camera.py`)

**Added k3 parameter**:
```python
def undistort_normalized(
    xy_distorted: np.ndarray,
    k1: float,
    k2: float,
    k3: float,  # NEW!
    p1: float,
    p2: float,
    ...
) -> np.ndarray:
    """Remove distortion (inverse of Eq. 7.41)."""
    # ... iterative Newton-Raphson with k3
```

### 4. ✅ Updated project_point() (`core/slam/camera.py`)

**Passes k3 to distortion**:
```python
def project_point(
    intrinsics: CameraIntrinsics,
    point_camera: np.ndarray,
) -> np.ndarray:
    """
    Project 3D point to pixel coordinates.
    
    Complete model:
        - Eq. (7.40): Intrinsic matrix K
        - Eq. (7.41): Distortion model with k1, k2, k3, p1, p2
        - Eqs. (7.42)-(7.43): Pixel coordinates u, v
    """
    # ...
    xy_distorted = distort_normalized(
        xy_normalized,
        intrinsics.k1,
        intrinsics.k2,
        intrinsics.k3,  # NEW!
        intrinsics.p1,
        intrinsics.p2,
    )
```

**Fixed docstrings**:
- Now references Eq. (7.40), (7.41), and (7.42)-(7.43) correctly
- Explains the complete projection pipeline
- Documents all 5 distortion coefficients

### 5. ✅ Updated unproject_pixel() (`core/slam/camera.py`)

**Passes k3 to undistortion**:
```python
xy_normalized = undistort_normalized(
    xy_distorted,
    intrinsics.k1,
    intrinsics.k2,
    intrinsics.k3,  # NEW!
    intrinsics.p1,
    intrinsics.p2,
)
```

### 6. ✅ Fixed create_reprojection_factor() Docstrings (`core/slam/factors.py`)

**Before**:
```python
"""
Residual:
    r = h(pose, landmark) - observed_pixel
where h is the camera projection function (Eqs. 7.43-7.46, 7.40).  # WRONG!
"""
```

**After**:
```python
"""
Residual:
    r = h(pose, landmark) - observed_pixel
where h is the camera projection function:
    - Eq. (7.40): Intrinsic matrix K
    - Eq. (7.41): Distortion model with k1, k2, k3, p1, p2
    - Eqs. (7.42)-(7.43): Pixel coordinates u, v
"""
```

### 7. ✅ Updated README (`ch7_slam/README.md`)

**Before**:
```markdown
| `project_point()` | `core/slam/camera.py` | Eq. (7.40) | Pinhole camera projection |
| `Camera` distortion methods | `core/slam/camera.py` | Eq. (7.41)-(7.43) | Radial and tangential distortion |
| Bundle adjustment factors | `core/slam/factors.py` | Eq. (7.70) | Reprojection error minimization |
```

**After**:
```markdown
| `project_point()` | `core/slam/camera.py` | Eq. (7.40), (7.41), (7.42)-(7.43) | Full camera projection + distortion |
| `distort_normalized()` | `core/slam/camera.py` | Eq. (7.41) | Distortion model (k1,k2,k3,p1,p2) |
| `create_reprojection_factor()` | `core/slam/factors.py` | Eq. (7.68)-(7.70) | Bundle adjustment reprojection error |
```

### 8. ✅ Fixed All Unit Tests (`tests/core/slam/test_camera.py`)

**Updated all test calls to include k3**:
- 9 calls to `distort_normalized()` - all now include `k3=0` or `k3=0.001`
- 2 calls to `undistort_normalized()` - all now include `k3=0` or `k3=0.001`
- Updated test docstring references from "Eqs. 7.43-7.46" to "Eq. 7.41"
- Updated test date to December 2025

**All 28 tests pass** ✅

## Verification

### ✅ k3 Parameter Added
```python
# core/slam/types.py
k3: float = 0.0  ✓

# core/slam/camera.py - distort_normalized signature
def distort_normalized(xy, k1, k2, k3, p1, p2):  ✓

# core/slam/camera.py - distortion formula
radial_distortion = 1.0 + k1*r² + k2*r⁴ + k3*r⁶  ✓
```

### ✅ All Equation References Correct

| Function | Equation Reference | Status |
|----------|-------------------|--------|
| `CameraIntrinsics` | Eq. (7.40), (7.41), (7.42)-(7.43) | ✅ |
| `distort_normalized()` | Eq. (7.41) | ✅ |
| `undistort_normalized()` | Inverse of Eq. (7.41) | ✅ |
| `project_point()` | Eq. (7.40), (7.41), (7.42)-(7.43) | ✅ |
| `create_reprojection_factor()` | Eq. (7.40), (7.41), (7.42)-(7.43), (7.68)-(7.70) | ✅ |

### ✅ All Tests Pass
```
tests/core/slam/test_camera.py: 28 passed ✓
```

### ✅ No Linter Errors
```
core/slam/types.py: ✓
core/slam/camera.py: ✓
core/slam/factors.py: ✓
tests/core/slam/test_camera.py: ✓
ch7_slam/README.md: ✓
```

### ✅ Backward Compatibility
- k3 defaults to 0.0, so existing code works without changes
- All existing tests updated and passing

## Summary of Equation Mapping

| Equation | What It Defines | Implemented In |
|----------|-----------------|----------------|
| **Eq. (7.40)** | Intrinsic matrix K: s p^pixel = K p^C | `CameraIntrinsics.to_matrix()`, `project_point()` |
| **Eq. (7.41)** | Distortion: x̂, ŷ with k1, k2, k3, p1, p2 | `distort_normalized()` |
| **Eq. (7.42)** | Pixel u: u = fx*x̂ + cx | `project_point()` |
| **Eq. (7.43)** | Pixel v: v = fy*ŷ + cy | `project_point()` |

## Files Modified

1. `core/slam/types.py` - Added k3, fixed docstrings, updated methods
2. `core/slam/camera.py` - Added k3 to all functions, fixed equation references
3. `core/slam/factors.py` - Fixed reprojection factor docstrings
4. `ch7_slam/README.md` - Updated equation reference table
5. `tests/core/slam/test_camera.py` - Added k3 to all test calls, fixed references
6. `.dev/ch7_prompt4_camera_distortion_fix_summary.md` - This summary

## Acceptance Criteria: ALL MET ✅

✅ Distortion model includes k3 parameter (can be 0 by default)  
✅ k3*r⁶ term correctly implemented in `distort_normalized()`  
✅ All camera-related docstrings cite correct equation numbers  
✅ `CameraIntrinsics` dataclass extended with k3  
✅ `project_point()` references Eq. (7.40), (7.41), (7.42)-(7.43)  
✅ `create_reprojection_factor()` references correct equations  
✅ README updated with accurate equation mapping  
✅ All unit tests pass (28/28)  
✅ No linter errors  
✅ Backward compatible (k3 defaults to 0.0)  

## Next Steps

**Prompt 4 is COMPLETE!** The camera distortion model now fully aligns with the book's Equations (7.40)-(7.43), including the k3 parameter and all correct equation references.

Ready for Prompt 5 (if any).

