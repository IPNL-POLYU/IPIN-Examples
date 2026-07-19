# Chapter 2: Quick Reference Guide

## Equation → Code Mapping

> Numbers follow the **final book**. Convention: roll→Y, pitch→X, yaw→Z;
> `C` is the passive transform `x_new = C @ x_old`. See
> `docs/ch2_equation_mapping.md` for the full note.

### Section 2.1 — Coordinate Transformations

| Equation | Description | Function | File |
|----------|-------------|----------|------|
| **Eq. (2.3)** | Map → Body (yaw) | `map_to_body()` / `body_to_map()` | `core/coords/transforms.py` |
| **Eq. (2.5)** | ENU ↔ NED | `enu_to_ned()` / `ned_to_enu()` | `core/coords/transforms.py` |
| **Eq. (2.6)** | ENU → Body | `enu_to_body()` | `core/coords/transforms.py` |
| **Eq. (2.7)** | Body → ENU | `body_to_enu()` | `core/coords/transforms.py` |
| **Eq. (2.9)** | LLH → ECEF | `llh_to_ecef()` (inv: `ecef_to_llh()`) | `core/coords/transforms.py` |
| **Eq. (2.10)** | ECEF → ENU | `ecef_to_enu()` (inv: `enu_to_ecef()`) | `core/coords/transforms.py` |

### Section 2.2 — Attitude Representations

| Equation | Description | Function | File |
|----------|-------------|----------|------|
| **Eq. (2.14–2.17)** | Euler → Rotation Matrix | `euler_to_rotation_matrix()` (inv: `rotation_matrix_to_euler()`) | `core/coords/rotations.py` |
| **Eq. (2.21)** | Quaternion → Rotation Matrix | `quat_to_rotation_matrix()` (inv: `rotation_matrix_to_quat()`, Shepperd) | `core/coords/rotations.py` |
| **Eq. (2.22)** | Quaternion → Euler | `quat_to_euler()` | `core/coords/rotations.py` |
| **Eq. (2.23)** | Euler → Quaternion | `euler_to_quat()` | `core/coords/rotations.py` |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Implemented equations (indexed + verified)** | 14 |
| **Coordinate/rotation functions** | 16 |
| **Tests (`tests/core/coords/`)** | 55 passing |
| **Index + verification gate** | `check_equation_index.py --strict` |

---

## Quick Commands

```bash
# Run all Chapter 2 tests
pytest tests/core/coords/ -v

# Run examples
python ch2_coords/example_coordinate_transforms.py

# Search for equation
grep -r "Eq. (2.1)" core/ ch2_coords/

# View documentation
cat ch2_coords/README.md
cat docs/ch2_equation_mapping.md
```

---

## Documentation Files

1. **`ch2_coords/README.md`** - User guide with examples
2. **`docs/equation_index.yml`** - Machine-readable mapping
3. **`docs/ch2_equation_mapping.md`** - Detailed technical reference
4. **`CHAPTER_2_MAPPING_SUMMARY.md`** - Executive summary

---

**Last Updated**: December 11, 2025



