# Prompt 2: Eq. (6.7) Notation Reconciliation - Summary

**Author:** Li-Ta Hsu  
**Date:** December 2025  
**Status:** ✅ COMPLETE

## Overview

Reconciled the apparent sign discrepancy between the book's Equation (6.7) and the code implementation for velocity update. The formulas are **algebraically equivalent** but use different notation conventions.

## Problem Statement

**Book's Equation 6.7** (references/ch6.txt, line 58):
```
v_k = v_{k-1} + (C_B^M @ a_B - g^M) * dt
```
where `g^M = [0, 0, g]^T` (line 59)

**Code Implementation** (core/sensors/strapdown.py, line 399):
```python
a_M = C_B_M @ f_b + g_M
```
where `g_M = [0, 0, -g]` for ENU

**Question:** Why does the book SUBTRACT `g^M` while the code ADDS `g_M`?

## Solution: Different Notation Conventions

Both formulations are **algebraically equivalent**. The difference is purely notational:

### Book's Convention
```
g_M (book) = [0, 0, +g]  (upward magnitude to subtract)
```
- The book writes `g` as a positive scalar in the upward (+z) direction for ENU
- This represents the "gravity compensation term" that needs to be subtracted

### Code's Convention
```
g_M (code) = [0, 0, -g]  (actual downward gravity vector)
```
- The code uses the physical gravity vector (pointing downward in ENU)
- This represents actual gravitational acceleration

### Algebraic Equivalence Proof

For ENU frame:

**Book:**
```
a_M = C_B^M @ a_B - [0, 0, +g]
    = C_B^M @ a_B + [0, 0, -g]
```

**Code:**
```
a_M = C_B^M @ f_B + [0, 0, -g]
```

**They're identical!** (where `a_B` (book) = `f_B` (code) = specific force)

### Relationship
```
g_M_book = -g_M_code
```

More explicitly:
- Book: `g_M = [0, 0, +9.81]` used with SUBTRACTION
- Code: `g_M = [0, 0, -9.81]` used with ADDITION

## Accelerometer Convention Established

Both book and code use **standard specific force** convention:

**Specific force** = what accelerometer measures = non-gravitational force

### For Stationary Accelerometer in ENU:
- True kinematic acceleration: `a_kin = 0` (not moving)
- Gravitational acceleration: `g_vec = [0, 0, -9.81]` (downward)
- Accelerometer measures: `f = a_kin - g_vec = 0 - (-9.81) = +9.81` (upward reaction)

```
f_B = [0, 0, +9.81]  (upward reaction force from table/ground)
```

### Physical Verification

**Test: Stationary accelerometer in ENU**
```
f_B = [0, 0, +9.81]  (what sensor reads)
g_M = [0, 0, -9.81]  (gravity vector)
v_dot = C @ f_B + g_M
      = [0, 0, +9.81] + [0, 0, -9.81]  (for aligned device)
      = [0, 0, 0]  ✓ Correct!
```

No velocity change for stationary device - as expected!

## Changes Made

### 1. Updated `vel_update()` Docstring (`core/sensors/strapdown.py`)

**Added comprehensive explanation:**
```python
"""
Velocity update with gravity compensation (Eq. 6.7).

Implements Eq. (6.7) in Chapter 6 using standard specific force convention.

CODE FORMULATION (what this function implements):
    v_k^M = v_{k-1}^M + (C_B^M(q) @ f_b + g_M) * Δt

BOOK'S EQ. (6.7) FORMULATION:
    v_k^M = v_{k-1}^M + (C_B^M(q) @ a_B - g_M_book) * Δt

ALGEBRAIC EQUIVALENCE:
    These are identical! The difference is notation:
    - f_b (code) = a_B (book) = specific force (accelerometer reading)
    - g_M (code) = -g_M_book
    
    For ENU:
      g_M (code) = [0, 0, -9.81]  (physical gravity vector, downward)
      g_M (book) = [0, 0, +9.81]  (magnitude to subtract, upward)
    
    Proof: C @ a_B - [0,0,+g] = C @ a_B + [0,0,-g] = C @ f_b + g_M ✓

PHYSICAL MEANING:
    - Accelerometer measures specific force f_b (reaction force, NOT gravity)
    - For stationary in ENU: f_b = [0, 0, +9.81] (upward reaction from ground)
    - Gravity vector: g_M = [0, 0, -9.81] (downward in ENU)
    - True kinematic accel: a_M = f_b + g_M = [0,0,0] for stationary ✓
...
"""
```

### 2. Updated `gravity_vector()` Docstring (`core/sensors/strapdown.py`)

**Added notation clarification:**
```python
"""
Gravity vector in map frame (physical gravity, pointing downward).

Implements Eq. (6.8) in Chapter 6 with standard physics convention:
    g_M = [0, 0, -g]^T    (for ENU: gravity points downward = negative z)
    g_M = [0, 0, +g]^T    (for NED: gravity points downward = positive z)

NOTATION NOTE:
    The book writes g^M = [0, 0, g]^T in Eq. (6.7) context, which can be 
    ambiguous. We interpret this as the MAGNITUDE in the z-direction.
    
    Book's convention: g^M = [0, 0, +g] (upward) used with SUBTRACTION
    Code's convention: g_M = [0, 0, -g] (downward) used with ADDITION
    
    These are equivalent: (a_B - g_book) = (f_B + g_code) where g_book = -g_code

PHYSICAL MEANING:
    This function returns the actual gravitational acceleration vector:
    - ENU: [0, 0, -9.81] m/s² (gravity pulls downward = negative z)
    - NED: [0, 0, +9.81] m/s² (gravity pulls downward = positive z in z-down frame)
...
"""
```

### 3. Verified Forward Model Consistency

Checked `core/sim/imu_from_trajectory.py`:

**Forward model (acceleration → measurement):**
```python
f_b = C_M_B @ (accel_map - g_M)
```

**This is the correct inverse of Eq. (6.7):**
```
Eq. 6.7:  a_M = C_B^M @ f_b + g_M  (measurement → acceleration)
Forward:  f_b = C_M^B @ (a_M - g_M)  (acceleration → measurement)
```

**Round-trip verification:**
```
a_M = C_B^M @ [C_M^B @ (a_M - g_M)] + g_M
    = C_B^M @ C_M^B @ (a_M - g_M) + g_M
    = I @ (a_M - g_M) + g_M
    = a_M - g_M + g_M
    = a_M  ✓
```

## Acceptance Criteria Verification

### ✅ Criterion 1: No Irreconcilable Formula

**Test Results:**
1. ✅ Stationary ENU: `v_dot = 0` (zero velocity change)
2. ✅ Stationary NED: `v_dot = 0` (zero velocity change)  
3. ✅ Forward/inverse round-trip: error < 1e-10 m/s²

**Conclusion:** Book's formula and code formula are algebraically equivalent!

### ✅ Criterion 2: Gravity Unambiguous

**Test Results:**
1. ✅ ENU gravity: `[0, 0, -9.81]` (downward)
2. ✅ NED gravity: `[0, 0, +9.81]` (downward in z-down)
3. ✅ ENU `gravity_direction = -1` (correct)
4. ✅ NED `gravity_direction = +1` (correct)
5. ✅ Documentation explains book/code equivalence

**Conclusion:** Gravity convention is explicit and unambiguous for both ENU and NED!

## Convention Established

### **Final Decision: Use Standard Physics Convention**

✅ **Accelerometer:**
- Variable name: `f_B` (specific force in body frame)
- Physical meaning: What accelerometer actually measures (reaction force)
- For stationary in ENU: `f_B = [0, 0, +9.81]` (upward)

✅ **Gravity:**
- Variable name: `g_M` (gravity vector in map frame)
- Physical meaning: Actual gravitational acceleration (downward)
- For ENU: `g_M = [0, 0, -9.81]` (downward)
- For NED: `g_M = [0, 0, +9.81]` (downward in z-down frame)

✅ **Equation:**
```
a_M = C_B^M @ f_B + g_M
```

✅ **Equivalence to Book:**
- Book: `a_M = C_B^M @ a_B - g_M_book`
- Code: `a_M = C_B^M @ f_B + g_M_code`
- Where: `g_M_book = -g_M_code`

## Documentation Created

1. **`.dev/ch6_eq67_notation_analysis.md`** - Detailed analysis of the notation difference
2. **`.dev/ch6_verify_prompt2_eq67_notation.py`** - Acceptance verification script
3. **`.dev/ch6_prompt2_eq67_notation_summary.md`** - This summary document

## Files Modified

1. `core/sensors/strapdown.py` - Updated docstrings for `vel_update()` and `gravity_vector()`

## Files Verified (No Changes Needed)

1. `core/sim/imu_from_trajectory.py` - Forward model already correct
2. `ch6_dead_reckoning/example_imu_strapdown.py` - Uses correct convention

## Impact

**Before:** Developers confused by apparent sign discrepancy between book and code.

**After:** 
- ✅ Clear documentation of algebraic equivalence
- ✅ Explicit statement of conventions used
- ✅ Both notations validated as correct
- ✅ Unambiguous gravity handling for ENU/NED

## Key Takeaways

1. **Both formulations are correct!** Book and code just use different notation.

2. **Notation relationship:**
   ```
   g_M_book = [0, 0, +g] (to subtract)
   g_M_code = [0, 0, -g] (to add)
   g_M_book = -g_M_code
   ```

3. **Physical interpretation:**
   - Code uses actual gravity vector (physically intuitive)
   - Book uses magnitude to subtract (mathematically equivalent)

4. **Forward model is inverse of Eq. 6.7:**
   ```
   Eq. 6.7:  a_M = C_B^M @ f_B + g_M
   Forward:  f_B = C_M^B @ (a_M - g_M)
   ```

5. **Everything is self-consistent** and produces correct physics!

## Verification Command

```bash
python .dev/ch6_verify_prompt2_eq67_notation.py
```

**Result:** ALL TESTS PASS ✅

---

✨ **Prompt 2 Complete!** The velocity update equation now has **explicit documentation** showing its algebraic equivalence to the book's Eq. (6.7), with unambiguous gravity conventions for both ENU and NED frames.







