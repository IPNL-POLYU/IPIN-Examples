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

## Where the authoritative counts live

This section used to be a table of numbers. Every one of them was a claim
nothing recomputed, and they rotted at different rates: "55 passing tests" was
68 by the time anyone looked, and "16 functions" was 17 from the day
`enu_to_llh_offset` was added. A count in prose goes stale silently — there is
no failing run to notice — so what follows names the source instead.

| What | Where it is counted |
|------|---------------------|
| **Implemented equations** | the Section 2.1 and 2.2 tables above; `docs/equation_index.yml` is the machine-readable form |
| **Coordinate/rotation functions** | `core.coords.__all__`, defined by `core/coords/transforms.py` and `core/coords/rotations.py` |
| **Tests** | `tests/core/coords/` — `pytest tests/core/coords/` reports the number |
| **Index + verification gate** | `python tools/check_equation_index.py --strict` |

---

## Quick Commands

```bash
# Run all Chapter 2 tests
pytest tests/core/coords/ -v

# Run examples
python -m ch2_coords.example_coordinate_transforms

# Search for equation. Pick a two-digit number: "Eq. (2.1)" is a substring of
# "Eq. (2.10)" through "Eq. (2.17)" and matches all of them.
grep -r "Eq. (2.17)" core/ ch2_coords/

# View documentation
cat ch2_coords/README.md
cat docs/ch2_equation_mapping.md
```

---

## Documentation Files

1. **`ch2_coords/README.md`** - User guide with examples
2. **`docs/equation_index.yml`** - Machine-readable mapping
3. **`docs/ch2_equation_mapping.md`** - Detailed technical reference

(A fourth entry here pointed at `CHAPTER_2_MAPPING_SUMMARY.md`, which exists
nowhere in the repository and has no predecessor under another name. It is
removed rather than renamed; `docs/ch2_equation_mapping.md` above is the
detailed reference a reader was being sent to look for.)

---

**Last updated**: `git log -1 -- docs/CH2_QUICK_REFERENCE.md`. A hand-written
date here is the same rotting claim as the counts above — it read
"December 11, 2025" through several edits that did not touch it.



