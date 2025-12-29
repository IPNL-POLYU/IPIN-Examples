# Eq. (6.7) Notation Analysis: Book vs. Code

**Author:** Li-Ta Hsu  
**Date:** December 2025

## The Issue

The book's Equation (6.7) and the code implementation appear to use different signs for gravity:

**Book (references/ch6.txt, line 58):**
```
v_k = v_{k-1} + ((C_M^B)^T · a_B - g^M) · Δt
```
where `g^M = [0, 0, g]^T` (line 59)

**Code (core/sensors/strapdown.py, line 399):**
```python
a_M = C_B_M @ f_b + g_M
```
where `g_M = [0, 0, -g]` for ENU

**Question:** Why does the book SUBTRACT `g^M` while the code ADDS `g_M`?

## Answer: Different Notation Conventions (Both Correct!)

The book and code use **algebraically equivalent but notational different** conventions for the gravity term.

### Book's Convention

The book defines (line 59):
```
g^M = [0, 0, g]^T
```

where `g` is the **magnitude** (9.81 m/s², always positive). In ENU, this vector points **upward** (in the +z direction).

**Physical meaning:** The book's `g^M` represents the "gravity compensation term to subtract" from the accelerometer reading.

### Code's Convention

The code defines:
```python
g_M = [0, 0, -g]  # for ENU, where g = 9.81
```

**Physical meaning:** The code's `g_M` represents the actual **gravity vector** (pointing downward in ENU).

### Algebraic Equivalence

Let's prove these are the same. For ENU frame:

**Book notation:**
```
a_M = C_B^M @ a_B - g_M_book
    = C_B^M @ a_B - [0, 0, +g]
    = C_B^M @ a_B + [0, 0, -g]
```

**Code notation:**
```
a_M = C_B^M @ f_B + g_M_code
    = C_B^M @ f_B + [0, 0, -g]
```

**They're identical!** (assuming `a_B` (book) = `f_B` (code), which they are - both are accelerometer readings)

### Notation Relationship

```
g_M_book = -g_M_code
```

Or more explicitly for ENU:
```
g_M (book) = [0, 0, +9.81]  (upward magnitude to subtract)
g_M (code) = [0, 0, -9.81]  (downward gravity to add)
```

## Physical Verification: Stationary Accelerometer

Let's verify with a stationary accelerometer in ENU:

**Physical reality:**
- Accelerometer reads: `f_B = [0, 0, +9.81]` (upward reaction force from table)
- Device is stationary: kinematic acceleration = 0
- Expected result: `v_dot = 0` (no velocity change)

**Book's equation:**
```
v_dot = C_B^M @ a_B - g_M
      = C_B^M @ [0, 0, +9.81] - [0, 0, +9.81]  (for aligned device, C = I)
      = [0, 0, +9.81] - [0, 0, +9.81]
      = [0, 0, 0]  ✓ Correct!
```

**Code's equation:**
```
v_dot = C_B^M @ f_B + g_M
      = C_B^M @ [0, 0, +9.81] + [0, 0, -9.81]
      = [0, 0, +9.81] + [0, 0, -9.81]
      = [0, 0, 0]  ✓ Correct!
```

**Both give the correct answer!**

## Accelerometer Convention

Both book and code use **standard specific force convention:**

**Specific force** = what accelerometer measures = non-gravitational force

For a stationary object:
- True kinematic acceleration: `a_kin = 0`
- Gravitational acceleration: `a_grav = [0, 0, -9.81]` (downward in ENU)
- Accelerometer measures: `f = a_kin - a_grav = 0 - (-9.81) = +9.81` (upward reaction)

So for stationary in ENU:
```
f_B = [0, 0, +9.81]  (upward, what sensor actually reads)
```

The book's notation `a_B` and code's notation `f_B` both refer to this same quantity.

## Reconciliation Strategy

To maintain consistency with the book while keeping physically intuitive code:

1. **Use standard specific force `f_B`** in code (what accelerometer measures)
2. **Use physical gravity vector `g_M`** in code (actual gravity, downward in ENU)
3. **Equation: `a_M = C_B^M @ f_B + g_M`** (add gravity)
4. **Document the equivalence** to book's notation

### Documentation Template

```python
"""
Implements Eq. (6.7) using standard specific force convention.

Book's Eq. (6.7): v_k = v_{k-1} + (C_B^M @ a_B - g_M_book) * dt
where g_M_book = [0, 0, +g] (upward magnitude to subtract)

Code equivalent: v_k = v_{k-1} + (C_B^M @ f_B + g_M_code) * dt
where g_M_code = [0, 0, -g] (downward gravity to add)

These are algebraically identical:
  a_B (book) = f_B (code) = specific force
  g_M_book = -g_M_code
  
For ENU: g_M_book = [0,0,+9.81], g_M_code = [0,0,-9.81]
"""
```

## Conclusion

**The current implementation is CORRECT!**

The apparent sign difference is purely notational:
- Book uses "magnitude to subtract" convention
- Code uses "physical vector to add" convention

Both are valid and produce identical results. The code's convention is more physically intuitive:
- Specific force `f_B` points upward for stationary device
- Gravity `g_M` points downward (ENU) or upward in z-down (NED)
- Adding them gives true kinematic acceleration

## Recommendation

✅ **Keep current implementation**  
✅ **Add detailed documentation explaining equivalence**  
✅ **No code changes needed**  

The implementation is already correct and follows standard aerospace conventions for specific force and gravity.





