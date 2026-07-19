# Book Errata

Errors found in the published book *Principles of Indoor Positioning and Indoor
Navigation* (Artech House, 2026) while auditing this companion code against the
text. Each entry records the printed equation, the correct form, and how the code
handles it. This list is intended both to keep the code honest and to feed
corrections back to the publisher.

Format: **E-NN** | chapter/eq | printed | correct | code status.

> **Note on the source PDF.** The audited PDF still contains `{AU: ...}` copyedit
> query markers (e.g., Ch. 2 "last part of this sentence is unclear", Ch. 3
> "missing eq 3.31?", UKF "cite ref [3] in order"), so it appears to be a proof /
> galley rather than the final print. Some items below may already be tracked in
> those proof queries.

---

## E-01 — Ch. 3, Eqs. (3.19) and (3.20): linear Kalman-filter covariance update

**Printed (both (3.19) and (3.20)):**

```
Σ_{x_k} = P_{k|k-1} − F_k K_k H_k P_{k|k-1}
```

**Correct:**

```
Σ_{x_k} = (I − K_k H_k) P_{k|k-1}          (= P_{k|k-1} − K_k H_k P_{k|k-1})
```

**Problem.** The printed update carries a spurious state-transition matrix `F_k`.
That factor does not belong in the measurement-update covariance (the `F_k` is
already consumed by the prediction step). With `F_k ≠ I` the printed formula
returns a matrix that is **non-symmetric and can have negative diagonal
entries**, i.e. not a valid covariance.

**Verification.** For `F = [[1,1],[0,1]]`, `H = [[1,0]]`, the Monte-Carlo
empirical posterior covariance of `x_true − x̂` (N = 4e5) matches `(I − KH)P`
to ~6e-4, while the printed `P − FKHP` differs by ~0.39 and is non-symmetric.
See the note in `core/estimators/kalman_filter.py`.

**Correct in the book's own EKF section.** Eq. (3.23) (EKF update) already prints
the correct `P_k = (I − K_k H_k) P_k^-`, so only the linear-KF (3.19)/(3.20) are
affected.

**Code status.** Correct. `KalmanFilter.update()` uses the numerically-stable
Joseph form `(I − KH) P (I − KH)^T + K R K^T`, which equals `(I − KH)P` at the
optimal gain. The code deliberately deviates from the printed (3.19)/(3.20).

---

## E-02 — Ch. 3, Eq. (3.27): UKF reuses predicted sigma points (omits Q)

**Printed.** The UKF measurement step (3.27) sets `Z_i = h(χ_i^-)`, reusing the
predicted sigma points `χ_i^-` from (3.25). Those points have spread `P_pred`
(the transformed prior), which **excludes the process noise `Q`** added in (3.26).
Yet (3.26) and (3.30) use `P_k^- = P_pred + Q`.

**Problem.** Because the measurement sigma points carry spread `P_pred` while the
covariance bookkeeping uses `P_k^- = P_pred + Q`, the algorithm is internally
inconsistent: on a **linear** system the resulting UKF does **not** reduce to the
Kalman filter (it under-counts `Q` in the innovation covariance).

**Correct.** Re-draw sigma points from the predicted `(x̂_k^-, P_k^-)` (with `Q`)
before applying `h`. This is the standard (van der Merwe) additive-noise UKF.

**Verification.** With the redraw, the UKF matches the linear KF exactly (state
diff 0, covariance diff ~4e-16). See
`tests/core/estimators/test_unscented_kalman_filter.py::test_ukf_matches_kf_on_linear_system`.

**Code status.** Correct. `UnscentedKalmanFilter.update()` re-draws sigma points
from `(x̂_k^-, P_k^-)`, so it deviates from the literal (3.27) and reduces to the
KF on linear systems. (This is a variant choice, not an invalid formula like
E-01; flag for author review.)

---

## E-03 — Ch. 3, Section 3.3: equation (3.31) is missing

**Printed.** The particle-filter section jumps from Eq. (3.30) to Eq. (3.32); the
book's own proof markup flags "au: missing eq 3.31?" at the initialization step.

**Correct.** Equation (3.31) should be the SIR initialization: draw
`x_0^(i) ~ p(x_0)` and set equal weights `w_0^(i) = 1/N`. Either add it as (3.31)
or renumber (3.32)-(3.56).

**Code status.** Implemented. `ParticleFilter.__init__` draws particles from the
initial distribution and sets uniform weights `1/N`.

---

## E-04 — Ch. 3, Algorithm 3.2 (Levenberg-Marquardt): wrong cross-reference

**Printed.** Algorithm 3.2, line 4: "Calculate gain ratio g by **(2.56)**".

**Correct.** The gain-ratio equation is **(3.56)**, not (2.56). Simple
cross-reference typo (Eq. (2.56) does not exist; Chapter 2 ends at (2.23)).

**Code status.** Not applicable (documentation only). The solver in
`core/estimators/nonlinear_least_squares.py` computes the gain ratio per the
correct Eq. (3.56).
