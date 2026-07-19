# Chapter 2: Equation Mapping (Book → Code)

Mapping between the equations in **Chapter 2** of *Principles of Indoor
Positioning and Indoor Navigation* and their implementations in this repo.

**Status**: verified against the final book text (Section 2.1 Coordinate Systems
and Transformations; Section 2.2 Attitude). Every implemented equation is backed
by a conformance/round-trip test and is cross-checked by
`tools/check_equation_index.py`.

> **Audit note (correction).** An earlier version of this document mapped the ten
> code functions to Eqs. (2.1)–(2.10) sequentially. That numbering was wrong: in
> the book, Eqs. (2.1)–(2.10) are the coordinate-system transforms and
> (2.11)–(2.23) are the attitude equations. The numbers below follow the actual
> book. See the convention note before relying on the attitude functions.

---

## Conventions (from the book, Section 2.2)

Body frame is **X-right, Y-forward, Z-up**. In this frame the book defines:

- **roll = φ about the Y-axis** (Eq. (2.15))
- **pitch = θ about the X-axis** (Eq. (2.16))
- **yaw = ψ about the Z-axis** (Eq. (2.14))

This differs from the common aerospace convention (roll→X, pitch→Y). The code
follows the **book** so that running it reproduces the book's equations.

The rotation matrix `C` is the **passive coordinate transform** `x_new = C @ x_old`
(Eqs. (2.11), (2.17)); the active (vector-rotating) matrix is its transpose.
Quaternions are scalar-first `q = [q0, q1, q2, q3]` (Eq. (2.20)). Euler
composition is `C = Rx(pitch) @ Ry(roll) @ Rz(yaw)` (Eq. (2.17)).

---

## Section 2.1 — Coordinate Systems and Transformations

| Book Eq. | Operation | Implementation | Status |
|---|---|---|---|
| (2.1) | Local body vector `x_BODY` | *definition — no function* | — |
| (2.2) | Local map vector `x_MAP` | *definition — no function* | — |
| **(2.3)** | Map → body (yaw `Rz`) | `transforms.py::map_to_body` / `body_to_map` | ✅ |
| (2.4) | ENU vector `x_ENU` | *definition — no function* | — |
| **(2.5)** | ENU ↔ NED | `transforms.py::enu_to_ned` / `ned_to_enu` | ✅ |
| **(2.6)** | ENU → body | `transforms.py::enu_to_body` | ✅ |
| **(2.7)** | Body → ENU | `transforms.py::body_to_enu` | ✅ |
| (2.8) | Geodetic vector `x_LLH` | *definition — no function* | — |
| **(2.9)** | LLH → ECEF (closed form) | `transforms.py::llh_to_ecef` | ✅ |
| **(2.10)** | ECEF → ENU | `transforms.py::ecef_to_enu` | ✅ |

Inverses without an explicit book equation:

- `ecef_to_llh` — iterative inverse of (2.9). The book states the geodetic
  transform is done via ECEF with an iteration method and refers to Kaplan &
  Hegarty [2]; no closed form is given.
- `enu_to_ecef` — inverse of (2.10) (transpose of the rotation plus the
  reference offset).

## Section 2.2 — Attitude: Definition and Representation

| Book Eq. | Operation | Implementation | Status |
|---|---|---|---|
| (2.11) | `x_new = C x_old` relation | `rotations.py::euler_to_rotation_matrix` | ✅ |
| (2.12) | Orthogonality `C^-1 = C^T` | *property — used implicitly* | — |
| (2.13) | Yaw example | *worked example* | — |
| **(2.14)** | Yaw `Rz(ψ)` | composed in `euler_to_rotation_matrix` | ✅ |
| **(2.15)** | Roll `Ry(φ)` (about Y) | composed in `euler_to_rotation_matrix` | ✅ |
| **(2.16)** | Pitch `Rx(θ)` (about X) | composed in `euler_to_rotation_matrix` | ✅ |
| **(2.17)** | Euler → `C` | `rotations.py::euler_to_rotation_matrix` (inverse: `rotation_matrix_to_euler`) | ✅ |
| (2.18) | Rotation + translation | *general relation* | — |
| (2.19) | Gimbal-lock example | *worked example* | — |
| (2.20) | Quaternion `q = [q0..q3]` | scalar-first convention in `rotations.py` | — |
| **(2.21)** | Quaternion → `C` | `rotations.py::quat_to_rotation_matrix` (inverse: `rotation_matrix_to_quat`, Shepperd) | ✅ |
| **(2.22)** | Quaternion → Euler | `rotations.py::quat_to_euler` | ✅ |
| **(2.23)** | Euler → quaternion | `rotations.py::euler_to_quat` | ✅ |

`rotation_matrix_to_quat` uses Shepperd's method and has **no explicit book
equation**; it is the numerically-stable inverse of (2.21).

---

## Verification

The book's own attitude equations are mutually self-consistent: composing
`(2.23) → (2.21)` reproduces `(2.17)`, and the Euler round-trip via `(2.22)`
closes, to machine precision. The tests encode this:

- `tests/core/coords/test_rotations.py::TestEulerToRotationMatrix::test_matches_book_eq_2_17`
  asserts `euler_to_rotation_matrix` equals the closed form printed in Eq. (2.17).
- Round-trip and cross-conversion tests cover Euler ↔ matrix ↔ quaternion.
- `tests/core/coords/test_transforms.py` covers the four frame transforms
  (map↔body, ENU↔NED, ENU↔body) plus the geodetic chain.

Run:

```bash
pytest tests/core/coords/ -v
python tools/check_equation_index.py --strict   # index + verification gate
```

`check_equation_index.py` now confirms every Chapter-2 code reference is indexed
**and** backed by a resolvable `verified_by` test (not just that file paths
exist).

---

## References

1. Chapter 2, *Principles of Indoor Positioning and Indoor Navigation* — Section
   2.1 (Coordinate Systems and Transformations) and Section 2.2 (Attitude).
2. Kaplan, E. D., and C. Hegarty (eds.), *Understanding GPS/GNSS: Principles and
   Applications*, Artech House, 2017.
3. Shepperd, S. W. (1978). "Quaternion from Rotation Matrix." *Journal of
   Guidance and Control*, 1(3), 223–224.
