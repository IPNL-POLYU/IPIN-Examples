# Chapter 2: Quick Reference Guide

## Equation → Code Mapping

### Coordinate Transformations

| Equation | Description | Function | File | Tests Pass |
|----------|-------------|----------|------|------------|
| **Eq. (2.1)** | LLH → ECEF | `llh_to_ecef()` | `core/coords/transforms.py` | ✅ 5/5 |
| **Eq. (2.2)** | ECEF → LLH | `ecef_to_llh()` | `core/coords/transforms.py` | ✅ 8/8 |
| **Eq. (2.3)** | ECEF → ENU | `ecef_to_enu()` | `core/coords/transforms.py` | ✅ 9/9 |
| **Eq. (2.4)** | ENU → ECEF | `enu_to_ecef()` | `core/coords/transforms.py` | ✅ 7/7 |

### Rotation Representations

| Equation | Description | Function | File | Tests Pass |
|----------|-------------|----------|------|------------|
| **Eq. (2.5)** | Euler → Rotation Matrix | `euler_to_rotation_matrix()` | `core/coords/rotations.py` | ✅ 11/11 |
| **Eq. (2.6)** | Rotation Matrix → Euler | `rotation_matrix_to_euler()` | `core/coords/rotations.py` | ✅ 11/11 |
| **Eq. (2.7)** | Euler → Quaternion | `euler_to_quat()` | `core/coords/rotations.py` | ✅ 10/10 |
| **Eq. (2.8)** | Quaternion → Euler | `quat_to_euler()` | `core/coords/rotations.py` | ✅ 9/9 |
| **Eq. (2.9)** | Quaternion → Rotation Matrix | `quat_to_rotation_matrix()` | `core/coords/rotations.py` | ✅ 9/9 |
| **Eq. (2.10)** | Rotation Matrix → Quaternion | `rotation_matrix_to_quat()` | `core/coords/rotations.py` | ✅ 10/10 |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Equations Mapped** | 10 |
| **Functions Implemented** | 10 |
| **Test Cases** | 47 |
| **Subtests** | 27 |
| **Pass Rate** | 100% ✅ |
| **Documentation Files** | 4 |

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



