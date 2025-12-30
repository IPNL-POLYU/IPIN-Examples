# Prompt 1: EKF State Ordering Fix - Summary

**Author:** Li-Ta Hsu  
**Date:** December 2025  
**Status:** ✅ COMPLETE

## Overview

Fixed the EKF state vector ordering to match Equation (6.16) from the book exactly. The previous implementation used `[q, v, p, b_g, b_a]` ordering, but the book clearly specifies `[p, v, q, b_g, b_a]`.

## Problem Statement

**Book's Equation 6.16 (line 102 of references/ch6.txt):**
```
x_k = [p_k, v_k, q_k, b_{G,k}^B, b_{A,k}^B]^T
```

**Previous (Incorrect) Implementation:**
```python
# core/sensors/ins_ekf.py (OLD)
x = [q, v, p, b_g, b_a]  # WRONG ORDER!
```

**Impact:**
- Developers couldn't directly map book equations to code without mental translation
- Jacobian matrices had wrong block indices
- Measurement models (ZUPT, ZARU, NHC) selected wrong state components
- Documentation claimed to follow Eq. 6.16 but didn't
- NavStateQPVPBias claimed dimension was 13 (should be 16)

## Changes Made

### 1. **Core State Refactoring** (`core/sensors/ins_ekf.py`)

**Before:**
```python
@dataclass
class INSState:
    q: np.ndarray  # (4,)
    v: np.ndarray  # (3,)
    p: np.ndarray  # (3,)
    b_g: np.ndarray  # (3,)
    b_a: np.ndarray  # (3,)
    P: np.ndarray  # (16, 16)
    
    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.q, self.v, self.p, self.b_g, self.b_a])
    
    @classmethod
    def from_vector(cls, x: np.ndarray, P: np.ndarray) -> "INSState":
        return cls(
            q=x[0:4],
            v=x[4:7],
            p=x[7:10],
            b_g=x[10:13],
            b_a=x[13:16],
            P=P
        )
```

**After:**
```python
@dataclass
class INSState:
    p: np.ndarray  # (3,)  # POSITION FIRST
    v: np.ndarray  # (3,)  # VELOCITY SECOND
    q: np.ndarray  # (4,)  # QUATERNION THIRD
    b_g: np.ndarray  # (3,)
    b_a: np.ndarray  # (3,)
    P: np.ndarray  # (16, 16)
    
    def to_vector(self) -> np.ndarray:
        """State vector following Eq. (6.16): [p, v, q, b_g, b_a]^T"""
        return np.concatenate([self.p, self.v, self.q, self.b_g, self.b_a])
    
    @classmethod
    def from_vector(cls, x: np.ndarray, P: np.ndarray) -> "INSState":
        """Unpack following Eq. (6.16): [p, v, q, b_g, b_a]^T"""
        return cls(
            p=x[0:3],    # Position at 0:3
            v=x[3:6],    # Velocity at 3:6
            q=x[6:10],   # Quaternion at 6:10
            b_g=x[10:13],  # Gyro bias at 10:13
            b_a=x[13:16],  # Accel bias at 13:16
            P=P
        )
```

### 2. **ZUPT Measurement Jacobian** (`core/sensors/constraints.py`)

**Before:**
```python
def H(self, x: np.ndarray) -> np.ndarray:
    H = np.zeros((3, 16))
    H[:, 4:7] = np.eye(3)  # WRONG! Velocity not at 4:7
    return H
```

**After:**
```python
def H(self, x: np.ndarray) -> np.ndarray:
    """
    Measurement Jacobian (Eq. 6.45): H = [0_3x3, I_3, 0_3x4, 0_3x3, 0_3x3]
    State ordering: x = [p (3), v (3), q (4), b_g (3), b_a (3)]
    """
    H = np.zeros((3, 16))
    H[:, 3:6] = np.eye(3)  # CORRECT! Velocity at 3:6
    return H
```

### 3. **ZARU Measurement Jacobian** (`core/sensors/constraints.py`)

**Before:**
```python
def H(self, x: np.ndarray) -> np.ndarray:
    """For state x = [q, v, p, b_g, b_a], ..."""  # WRONG DOC
    H = np.zeros((3, 16))
    if n >= 13:
        H[:, 10:13] = -np.eye(3)  # Happens to be correct by luck!
    return H
```

**After:**
```python
def H(self, x: np.ndarray) -> np.ndarray:
    """
    Measurement Jacobian (Eq. 6.60): H = [0, 0, 0, -I_3, 0]
    State ordering: x = [p (3), v (3), q (4), b_g (3), b_a (3)]
    """
    H = np.zeros((3, 16))
    H[:, 10:13] = -np.eye(3)  # Gyro bias still at 10:13 (lucky!)
    return H
```

### 4. **NHC Measurement Model** (`core/sensors/constraints.py`)

**Before:**
```python
def h(self, x: np.ndarray) -> np.ndarray:
    """State x = [q (4), v_map (3), p (3), ...]"""  # WRONG
    q = x[0:4]    # WRONG indices
    v_map = x[4:7]  # WRONG indices
    # ...
```

**After:**
```python
def h(self, x: np.ndarray) -> np.ndarray:
    """
    State x = [p (3), v_map (3), q (4), ...] (Eq. 6.16)
    """
    v_map = x[3:6]   # CORRECT indices
    q = x[6:10]      # CORRECT indices
    # ...
```

### 5. **Covariance Initialization** (`core/sensors/ins_ekf.py`)

**Before:**
```python
P0 = np.eye(16) * 1e-6
P0[0:4, 0:4] *= 1e-4    # Attitude (WRONG position)
P0[4:7, 4:7] *= 0.1**2  # Velocity (WRONG position)
P0[7:10, 7:10] *= 0.1**2  # Position (WRONG position)
```

**After:**
```python
P0 = np.eye(16) * 1e-6
P0[0:3, 0:3] *= 0.1**2    # Position (CORRECT)
P0[3:6, 3:6] *= 0.1**2    # Velocity (CORRECT)
P0[6:10, 6:10] *= 1e-4    # Attitude (CORRECT)
```

### 6. **Process Noise Ordering** (`core/sensors/ins_ekf.py`)

**Before:**
```python
Q[0:4, 0:4] = ...    # Attitude noise (WRONG block)
Q[4:7, 4:7] = ...    # Velocity noise (WRONG block)
Q[7:10, 7:10] = ...  # Position noise (WRONG block)
```

**After:**
```python
Q[0:3, 0:3] = ...    # Position noise (CORRECT)
Q[3:6, 3:6] = ...    # Velocity noise (CORRECT)
Q[6:10, 6:10] = ...  # Attitude noise (CORRECT)
```

### 7. **Documentation Fixes** (`core/sensors/types.py`)

**Before:**
```python
class NavStateQPVPBias:
    """
    Notes:
        - Full state dimension is 13 (4 + 3 + 3 + 3).  # WRONG!
        ...
        - Eq. (6.16): State vector [q, v, p, b_g, b_a]  # WRONG ORDER!
    """
```

**After:**
```python
class NavStateQPVPBias:
    """
    Notes:
        - Full state dimension is 16 (3 + 3 + 4 + 3 + 3).  # CORRECT!
        ...
        - Eq. (6.16): State vector [p, v, q, b_g, b_a]  # CORRECT ORDER!
    """
```

### 8. **Example Script Updates** (`ch6_dead_reckoning/example_zupt.py`)

**Before:**
```python
state = ekf.initialize(
    q0=initial_state.q.copy(),
    v0=initial_state.v.copy(),
    p0=initial_state.p.copy()
)
```

**After:**
```python
state = ekf.initialize(
    p0=initial_state.p.copy(),  # Position first
    v0=initial_state.v.copy(),  # Velocity second
    q0=initial_state.q.copy()   # Quaternion third
)
```

## New Test Coverage

Created `tests/core/test_ins_state_ordering.py` with 10 comprehensive tests:

1. **Test state dimension is 16** (not 13!)
2. **Test `to_vector()` produces [p, v, q, b_g, b_a] ordering**
3. **Test `from_vector()` unpacks correctly**
4. **Test round-trip consistency**
5. **Test ZUPT Jacobian selects velocity at indices [3:6]**
6. **Test ZUPT `h(x)` extracts velocity from indices [3:6]**
7. **Test ZARU Jacobian selects gyro bias at indices [10:13]**
8. **Test NHC Jacobian uses correct velocity indices [3:6]**
9. **Test NHC Jacobian uses correct quaternion indices [6:10]**
10. **Test overall Eq. (6.16) consistency**

**All 10 tests PASS ✅**

## Acceptance Criteria Verification

### Criterion 1: Block Indices Match Book

| Component | Book Definition | Code Indices | Status |
|-----------|----------------|--------------|--------|
| Position (p) | [p_k] | `x[0:3]` | ✅ PASS |
| Velocity (v) | [v_k] | `x[3:6]` | ✅ PASS |
| Quaternion (q) | [q_k] | `x[6:10]` | ✅ PASS |
| Gyro bias (b_g) | [b_{G,k}^B] | `x[10:13]` | ✅ PASS |
| Accel bias (b_a) | [b_{A,k}^B] | `x[13:16]` | ✅ PASS |

**ZUPT Jacobian (Eq. 6.45):**
```
H = [0_3x3, I_3, 0_3x4, 0_3x3, 0_3x3]
```
- Non-zero block at indices [3:6]: ✅ PASS

**ZARU Jacobian (Eq. 6.60):**
```
H = [0_3x3, 0_3x3, 0_3x4, -I_3, 0_3x3]
```
- Non-zero block at indices [10:13]: ✅ PASS

### Criterion 2: Unit Tests Confirm Ordering

All 10 unit tests in `test_ins_state_ordering.py`: ✅ PASS

## Files Modified

1. `core/sensors/ins_ekf.py` - State refactoring, covariance, process noise
2. `core/sensors/constraints.py` - ZUPT, ZARU, NHC measurement models
3. `core/sensors/types.py` - Documentation fixes (dimension 13→16)
4. `ch6_dead_reckoning/example_zupt.py` - Initialize call argument order

## Files Created

1. `tests/core/test_ins_state_ordering.py` - Comprehensive unit tests
2. `.dev/ch6_verify_prompt1_state_ordering.py` - Acceptance verification script
3. `.dev/ch6_prompt1_state_ordering_summary.md` - This document

## Impact

**Before (BROKEN):**
- Developer reads Eq. (6.16): `x = [p, v, q, ...]`
- Looks at code: `x = [q, v, p, ...]`
- Mental translation required ❌
- Jacobian blocks don't match book ❌
- ZUPT selects wrong indices ❌

**After (FIXED):**
- Developer reads Eq. (6.16): `x = [p, v, q, ...]`
- Looks at code: `x = [p, v, q, ...]`
- Direct correspondence ✅
- Jacobian blocks match book ✅
- All measurement models correct ✅

## Verification Results

```
================================================================================
[PASS] ALL ACCEPTANCE CRITERIA MET FOR PROMPT 1

State ordering now matches Eq. (6.16) exactly:
  - Position at [0:3]
  - Velocity at [3:6]
  - Quaternion at [6:10]
  - Gyro bias at [10:13]
  - Accel bias at [13:16]

Developers can now directly map book equations to code!
================================================================================
```

## Next Steps

State ordering is now correct and validated. Developers can:
1. Read book equations (e.g., Eq. 6.28 for process Jacobian)
2. Directly implement block matrices at correct indices
3. No mental translation needed
4. Jacobians match book structure exactly

**Prompt 1: COMPLETE ✅**







