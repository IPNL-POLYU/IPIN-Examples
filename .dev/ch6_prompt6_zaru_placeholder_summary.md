# Prompt 6 Summary: ZARU Marked as Intentionally Incomplete

**Author:** Li-Ta Hsu  
**Date:** December 2025  
**Status:** ✅ COMPLETED

## Objective

Either implement ZARU properly or mark it as intentionally incomplete, ensuring no "stub" class claims to implement a numbered equation without actually doing so.

## Decision: Mark as Incomplete Placeholder

After analyzing the book's Eq. (6.60) and the current implementation, **we chose to mark ZARU as an incomplete placeholder** rather than attempt a proper implementation, because:

1. **Interface Limitation**: The standard EKF measurement model interface `h(x)` only receives the state vector `x`, but ZARU requires the current gyro measurement `ω_meas` as well.
2. **State Vector Issue**: Angular velocity `ω_b` is NOT in the state vector `[p, v, q, b_g, b_a]` - it's an external input.
3. **Breaking Change**: Proper implementation would require modifying the measurement model interface, affecting all existing code.

## Problem Analysis

### Book's Eq. (6.60)

From `references/ch6.txt`, lines 408-410:

```
z_zaru = 0, h(x) = ω_b
```

Where:
- **z_zaru = 0**: Measurement (zero angular rate)
- **h(x) = ω_b**: Predicted measurement (body angular rate)
- **ω_b**: Angular rate of the body relative to inertial frame (as measured by gyro)

### The Fundamental Issue

**ω_b is NOT in the state vector!**

The EKF state is: `x = [p (3), v (3), q (4), b_g (3), b_a (3)]` (16 elements)

The corrected angular rate is: `ω_b = ω_meas - b_g`

Where:
- `ω_meas`: Current gyro measurement (INPUT, not state)
- `b_g`: Gyro bias (IN STATE at indices 10:13)

### Current Implementation Problems

**Before (claimed to implement Eq. 6.60):**
```python
class ZaruMeasurementModel:
    """
    Zero angular rate update (ZARU) pseudo-measurement model.
    
    Implements Eq. (6.60) in Chapter 6:  # ❌ FALSE CLAIM!
        z_ZARU = 0 (expected measurement when not rotating)
        h(x) = ω (angular velocity, if estimated in state)
    """
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement function: h(x) = corrected angular velocity (Eq. 6.60)."""
        return np.zeros(3)  # ❌ WRONG! Should return ω_meas - b_g
```

Problems:
1. ✗ Claims to implement Eq. (6.60) but doesn't
2. ✗ `h(x)` returns zeros instead of `ω_meas - b_g`
3. ✗ No way to pass `ω_meas` to `h(x)`
4. ✗ Misleading docstring with equation reference

## Changes Made

### 1. Renamed Class

**Old:** `ZaruMeasurementModel`  
**New:** `ZaruMeasurementModelPlaceholder`

This immediately signals to developers that this is incomplete.

### 2. Updated Class Docstring

**New docstring clearly states:**
```python
class ZaruMeasurementModelPlaceholder:
    """
    PLACEHOLDER: Zero angular rate update (ZARU) pseudo-measurement model.

    **INCOMPLETE IMPLEMENTATION** - This class does NOT properly implement
    the book's Eq. (6.60) and is provided as a placeholder for future work.

    Why this implementation is incomplete:
    ------------------------------------
    The book's ZARU (Eq. 6.60) defines:
        z_zaru = 0 (measurement: zero angular rate)
        h(x) = ω_b (predicted: body angular rate from gyro)

    However, ω_b (angular rate) is NOT in the EKF state vector
    x = [p, v, q, b_g, b_a]. The angular rate is an *input* (gyro measurement
    ω_meas), not a state variable. The corrected angular rate is:
        ω_b = ω_meas - b_g

    For proper ZARU implementation, h(x) must compute ω_meas - b_g, but ω_meas
    is not available in the h(x) interface (which only receives state x).

    Current implementation limitations:
    -----------------------------------
    1. h(x) returns zeros (incorrect - should return predicted angular rate)
    2. H has -I at gyro bias indices (partially correct but not justified)
    3. No way to pass current gyro measurement ω_meas to h(x)
    4. Does NOT match the book's Eq. (6.60) formulation

    What would be needed for proper implementation:
    -----------------------------------------------
    - Extend measurement model interface to accept external measurements
    - Pass current gyro reading ω_meas to h(x)
    - Compute h(x) = ω_meas - b_g[x]
    - Jacobian: ∂h/∂b_g = -I (already implemented)
    ```
```

Key points:
- ✓ **Honest about limitations**
- ✓ **Explains WHY it's incomplete**
- ✓ **No false equation claims**
- ✓ **Provides guidance for future implementation**

### 3. Updated Method Docstrings

#### h(x) Method
**New docstring:**
```python
def h(self, x: np.ndarray) -> np.ndarray:
    """
    Measurement function (INCOMPLETE PLACEHOLDER).

    **LIMITATION**: This returns zeros, but the book's Eq. (6.60) requires
    h(x) = ω_b = ω_meas - b_g, where ω_meas is the current gyro reading.
    Since ω_meas is not available in this interface, this implementation
    is fundamentally incomplete.
    """
    # INCOMPLETE: Should return ω_meas - b_g, but ω_meas not available
    return np.zeros(3)
```

#### H(x) Method
**New docstring:**
```python
def H(self, x: np.ndarray) -> np.ndarray:
    """
    Measurement Jacobian (partially correct).

    If h(x) = ω_meas - b_g, then ∂h/∂b_g = -I.
    This part is implemented correctly, but h(x) itself is not.
    """
```

#### R(x) Method
**New docstring:**
```python
def R(self, x: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Measurement noise covariance (correctly implemented).
    """
```

### 4. Updated Module Docstring

**core/sensors/constraints.py:**
```python
"""
Drift correction constraints for INS (Chapter 6).

This module implements constraint-based drift reduction techniques:
    - ZUPT: Zero velocity update (Eq. (6.44)-(6.45)) [FULLY IMPLEMENTED]
    - ZARU: Zero angular rate update [PLACEHOLDER - see ZaruMeasurementModelPlaceholder]
    - NHC: Nonholonomic constraint (Eq. (6.61)) [FULLY IMPLEMENTED]

Implementation Status:
    - ZuptMeasurementModel: Fully implements Eq. (6.45) ✓
    - ZaruMeasurementModelPlaceholder: Incomplete (interface limitation) ⚠
    - NhcMeasurementModel: Fully implements Eq. (6.61) ✓

References:
    Chapter 6, Section 6.6.2: Motion constraints for drift mitigation
    Eq. (6.44): ZUPT detector (stationary detection)
    Eq. (6.45): ZUPT pseudo-measurement (v = 0)
    Eq. (6.60): ZARU concept (ω = 0) - NOT fully implemented here
    Eq. (6.61): NHC pseudo-measurement (lateral/vertical velocity = 0)
"""
```

Key changes:
- ✓ Explicit status markers: ✓ ⚠
- ✓ Softened Eq. (6.60) reference: "ZARU concept" instead of "implements"
- ✓ Clear note: "NOT fully implemented here"

### 5. Updated Imports and Exports

**core/sensors/__init__.py:**
```python
from .constraints import (
    # ...
    ZaruMeasurementModelPlaceholder,  # Changed from ZaruMeasurementModel
    # ...
)

__all__ = [
    # ...
    "ZaruMeasurementModelPlaceholder",  # Changed from ZaruMeasurementModel
    # ...
]
```

Module docstring updated:
```python
"""
constraints: ZUPT/ZARU/NHC detectors and pseudo-measurements

Measurement Models:
    ZuptMeasurementModel: Zero velocity update (Eq. 6.45)
    ZaruMeasurementModelPlaceholder: Zero angular rate update (INCOMPLETE PLACEHOLDER)
    NhcMeasurementModel: Nonholonomic constraint (Eq. 6.61)
"""
```

### 6. Updated Tests

**tests/core/sensors/test_sensors_constraints.py:**
- Renamed class: `TestZaruMeasurementModel` → `TestZaruMeasurementModelPlaceholder`
- Updated test method names: `test_zaru_*` → `test_zaru_placeholder_*`
- Updated docstrings to reflect placeholder status

**tests/core/test_ins_state_ordering.py:**
- Updated import: `ZaruMeasurementModel` → `ZaruMeasurementModelPlaceholder`
- Updated test method: `test_zaru_jacobian_*` → `test_zaru_placeholder_jacobian_*`

### 7. Updated Documentation

**ch6_dead_reckoning/README.md:**

Before:
```markdown
| `ZaruMeasurementModel.h()` | `core/sensors/constraints.py` | Eq. (6.60) | ZARU pseudo-measurement |
```

After:
```markdown
| `ZaruMeasurementModelPlaceholder.h()` | `core/sensors/constraints.py` | ⚠️ INCOMPLETE | ZARU placeholder (see class docs) |
```

Changes:
- ✓ New class name
- ✓ Warning emoji ⚠️
- ✓ "INCOMPLETE" instead of equation number
- ✓ Refers to class documentation for details

## Files Modified

1. **core/sensors/constraints.py**
   - Renamed `ZaruMeasurementModel` → `ZaruMeasurementModelPlaceholder`
   - Updated class docstring (comprehensive explanation)
   - Updated method docstrings (honest about limitations)
   - Updated module docstring (status markers)

2. **core/sensors/__init__.py**
   - Updated import statement
   - Updated `__all__` export
   - Updated module docstring

3. **tests/core/sensors/test_sensors_constraints.py**
   - Updated import
   - Renamed test class and methods
   - Updated test docstrings

4. **tests/core/test_ins_state_ordering.py**
   - Updated import
   - Updated test method name

5. **ch6_dead_reckoning/README.md**
   - Updated equation mapping table
   - Added warning marker

6. **.dev/ch6_prompt6_zaru_placeholder_summary.md** (this file)
   - Complete documentation of changes

7. **.dev/ch6_verify_prompt6_zaru_placeholder.py**
   - Acceptance verification script

## Comparison: ZUPT (Correct) vs. ZARU (Incomplete)

### ZUPT (Fully Implemented) ✓

**Book's Eq. (6.45):**
```
z_zupt = 0, h(x) = v
```

**Why it works:**
- Velocity `v` IS in the state vector at indices 3:6
- `h(x)` can directly extract `v` from state: `h(x) = x[3:6]`
- No external measurements needed

**Implementation:**
```python
class ZuptMeasurementModel:
    def h(self, x: np.ndarray) -> np.ndarray:
        """Extract velocity from state."""
        return x[3:6]  # ✓ Works perfectly!
    
    def H(self, x: np.ndarray) -> np.ndarray:
        """Jacobian: ∂v/∂x = [0, I, 0, 0, 0]"""
        H = np.zeros((3, len(x)))
        H[:, 3:6] = np.eye(3)
        return H
```

### ZARU (Incomplete) ⚠

**Book's Eq. (6.60):**
```
z_zaru = 0, h(x) = ω_b
```

**Why it DOESN'T work:**
- Angular velocity `ω_b` is NOT in the state vector
- `ω_b = ω_meas - b_g` requires external measurement `ω_meas`
- Standard interface `h(x)` only receives state `x`

**Current Placeholder:**
```python
class ZaruMeasurementModelPlaceholder:
    def h(self, x: np.ndarray) -> np.ndarray:
        """INCOMPLETE: Should return ω_meas - b_g, but can't."""
        return np.zeros(3)  # ⚠ Wrong, but interface doesn't allow correct implementation
    
    def H(self, x: np.ndarray) -> np.ndarray:
        """Jacobian: ∂(ω_meas - b_g)/∂x = [0, 0, 0, -I, 0]"""
        H = np.zeros((3, len(x)))
        H[:, 10:13] = -np.eye(3)  # ✓ This part is correct
        return H
```

**What would be needed:**
```python
# HYPOTHETICAL proper implementation (requires interface change)
class ZaruMeasurementModelProper:
    def h(self, x: np.ndarray, omega_meas: np.ndarray) -> np.ndarray:
        """Proper implementation with external measurement."""
        b_g = x[10:13]  # Extract gyro bias from state
        return omega_meas - b_g  # ✓ Correct!
```

## Acceptance Criteria

✅ **No "stub" class claims to implement a numbered equation**

Before:
- ❌ `ZaruMeasurementModel` claimed to implement Eq. (6.60)
- ❌ But actually didn't match the equation

After:
- ✅ `ZaruMeasurementModelPlaceholder` does NOT claim to implement Eq. (6.60)
- ✅ Clearly states it's incomplete
- ✅ Explains exactly why it's incomplete
- ✅ Provides guidance for future proper implementation
- ✅ References softened: "ZARU concept" not "implements Eq. (6.60)"

## Future Work: Proper ZARU Implementation

To properly implement ZARU in the future, one would need to:

### Option 1: Extend Measurement Model Interface

```python
class MeasurementModelWithExternalInputs:
    """Extended interface allowing external measurements."""
    
    def h(self, x: np.ndarray, external_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Measurement function with external inputs."""
        pass
    
    def H(self, x: np.ndarray, external_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Jacobian with respect to state."""
        pass

class ZaruMeasurementModelProper(MeasurementModelWithExternalInputs):
    def h(self, x: np.ndarray, external_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        omega_meas = external_inputs['omega_meas']  # Current gyro reading
        b_g = x[10:13]  # Gyro bias from state
        return omega_meas - b_g  # ✓ Correct!
```

### Option 2: Pass Gyro Measurement at Update Time

```python
# In EKF update
if zaru_detected:
    # Create measurement that includes current gyro reading
    z = omega_meas  # Instead of zeros
    # Modify h(x) to predict omega_meas from state (= b_g when stationary)
```

### Option 3: Augment State Vector

Add angular velocity to the state vector (computationally expensive):
```python
x = [p, v, q, omega, b_g, b_a]  # 19 elements instead of 16
```

This would make ZARU work like ZUPT, but requires significant refactoring.

## Key Takeaway

**Honesty in documentation is crucial.** Rather than leaving a misleading "stub" that claims to implement an equation but doesn't, we've clearly marked ZARU as incomplete and explained exactly why. This:

1. ✓ Prevents confusion for students/developers
2. ✓ Satisfies the acceptance criterion
3. ✓ Provides a clear path for future proper implementation
4. ✓ Maintains scientific integrity

The placeholder remains available for experimental use, but with full disclosure of its limitations.

---

*This prompt ensures code claims match reality, preventing students from assuming incomplete implementations are correct.*







