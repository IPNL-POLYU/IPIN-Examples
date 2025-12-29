# Eq. (6.11) Analysis: Book vs. Current Implementation

**Author:** Li-Ta Hsu  
**Date:** December 2025

## Book's Equation (6.11)

From references/ch6.txt, line 87:
```
v^A = C_S^A · v^S - [ω^A]_× · l^A
```

where (line 86):
```
v^S = [0, v, 0]^T  (velocity in y component, forward direction)
```

## Speed Frame Convention (Book)

From line 82-86, the book defines:
- **S-frame**: Speed sensor frame (wheel encoder frame)
- **Axis convention** (implied by NHC): x=right, y=forward, z=up
- **Velocity**: v^S = [0, v_forward, 0]^T

The forward speed is in the **y component** because of the non-holonomic constraint assumption.

## Current Implementation Issues

### Issue 1: Missing C_S^A Rotation

**Current code (line 160):**
```python
v_a = v_s - omega_skew @ lever_arm_b
```

**Should be:**
```python
v_a = C_S_A @ v_s - omega_skew @ lever_arm
```

The C_S^A rotation matrix is completely absent!

### Issue 2: Wrong Speed Frame Convention

**Current docs (line 114):**
```python
# Typically v_s = [v_forward, 0, 0] for forward vehicle motion.
```

**Should be (book line 86):**
```python
# Book convention: v_s = [0, v_forward, 0] for forward vehicle motion.
# Speed frame axes: x=right, y=forward, z=up
```

The velocity component is in the wrong axis!

### Issue 3: Inconsistent Frame Labels

**Current code:**
- Uses `omega_b` (body frame)
- Uses `lever_arm_b` (body frame)

**Book uses:**
- `ω^A` (attitude frame)
- `l^A` (attitude frame)

The book assumes A-frame and B-frame are close/aligned, but the code mixes them inconsistently.

## Required Changes

### 1. Add C_S^A Parameter

```python
def wheel_speed_to_attitude_velocity(
    v_s: np.ndarray,
    omega_a: np.ndarray,
    lever_arm_a: np.ndarray,
    C_S_A: Optional[np.ndarray] = None,  # NEW
) -> np.ndarray:
    """
    Implements Eq. (6.11): v^A = C_S^A @ v^S - [ω^A ×] l^A
    
    Args:
        v_s: Velocity in speed frame S, shape (3,).
             Book convention: v_s = [0, v_forward, 0] 
             (x=right, y=forward, z=up)
        omega_a: Angular velocity in attitude frame A, shape (3,).
        lever_arm_a: Lever arm in attitude frame A, shape (3,).
        C_S_A: Rotation matrix from S to A, shape (3,3).
               Default: None (uses identity, aligned frames).
    """
    if C_S_A is None:
        C_S_A = np.eye(3)
    
    # Eq. (6.11)
    v_a = C_S_A @ v_s - skew(omega_a) @ lever_arm_a
    return v_a
```

### 2. Update Documentation

All docstrings must clarify:
- Speed frame convention: x=right, y=forward, z=up
- Forward velocity goes in y component: [0, v, 0]
- C_S^A is identity for aligned frames
- Non-aligned frames require explicit C_S^A

### 3. Update Tests

Need tests for:
1. **Aligned frames** (C_S^A = I): verify v^A = v^S - [ω×]l
2. **Misaligned frames** (C_S^A ≠ I): verify full Eq. (6.11)
3. **Book convention**: v_s = [0, v, 0] produces expected results
4. **Lever arm effect**: non-zero l produces correct compensation

## Backward Compatibility

To avoid breaking existing code, we can:
1. Make C_S^A optional (default to identity)
2. Keep existing parameter names but add deprecation warnings
3. Provide helper to convert old convention to new

Or just update it directly since this is a book-aligned repository.

## Decision

**Use book's convention exactly:**
- Speed frame: x=right, y=forward, z=up
- Velocity: v^S = [0, v_forward, 0]
- Full Eq. (6.11) with C_S^A rotation
- Default C_S^A = I for aligned frames
- Update all docs and tests

This ensures the code matches the book without translation.






