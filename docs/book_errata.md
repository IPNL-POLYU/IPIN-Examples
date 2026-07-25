# Book Errata

Errors found in *Principles of Indoor Positioning and Indoor Navigation* (Artech
House, 2026) while auditing this companion code equation-by-equation against the
text. Each entry records the printed form, the correct form, why it matters, how
it was verified, and how the code handles it. This list keeps the code honest and
is the source for corrections fed back to the publisher.

Format: **E-NN** | chapter/eq | printed | correct | code status.

## Summary, most consequential first

| # | Location | Issue | Consequence for a reader who implements the printed form |
|---|---|---|---|
| **E-01** | Ch. 3, Eqs. (3.19), (3.20) | Spurious `F_k` in the linear-KF covariance update | **Broken filter.** The result is non-symmetric with a **negative variance** (measured −0.667 on a standard 2-state example): not a valid covariance at all |
| **E-05** | Ch. 8, Eq. (8.7) | Robust covariance scaling is inverted | **Silently wrong behaviour.** Multiplying `R` by a Table 3.1 weight *shrinks* it, so the filter trusts an outlier **~15× more** instead of less — the opposite of the stated intent |
| **E-02** | Ch. 3, Eq. (3.27) | UKF reuses predicted sigma points, omitting `Q` | Degraded accuracy. The UKF **fails to reduce to the Kalman filter** on a linear system (covariance error 4.6e-2 versus 1.7e-11 when corrected) — the standard sanity check for any UKF |
| **E-03** | Ch. 3, §3.3 | Equation (3.31) is missing | Reader cannot implement the particle-filter initialisation step. Numbering jumps (3.30) → (3.32) |
| **E-04** | Ch. 3, Algorithm 3.2, line 4 | Cites "(2.56)" for the gain ratio | Reader is sent to an equation that does not exist. Chapter 2 ends at (2.23) |
| **Q-01** | Ch. 4, Eq. (4.106) | Unanswered `{AU:}` proof query | Still open in the proof: *"Is this the correct equation? Do you mean (4.77)?"* Needs an author decision |

All five errata were re-verified independently on 2026-07-26; the numbers quoted
above are from that run, not from the original audit.

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

**Verification.** For `F = [[1,1],[0,1]]`, `H = [[1,0]]`, `R = 0.25`, the
Monte-Carlo empirical posterior covariance of `x_true − x̂` (N = 400,000) matches
`(I − KH)P` to **1.5e-3**, while the printed `P − F K H P` differs by **0.89**.
The printed result is

```
[[-0.667, -0.332],
 [ 0.111,  0.568]]
```

which is **not symmetric** and whose leading entry is a **negative variance**.
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

**Verification.** On a linear system, re-drawing the sigma points reproduces the
Kalman filter to machine precision (state error **1.7e-11**, covariance error
**6.3e-14**), whereas reusing the predicted points as printed leaves a state
error of **5.1e-3** and a covariance error of **4.6e-2**. Reducing to the KF on a
linear system is the defining sanity check for a UKF. See
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

**Correct.** The gain-ratio equation is **(3.56)**, not (2.56) — it is defined a
few lines earlier in the same section. Confirmed by scanning the text: Chapter 2
contains exactly 23 numbered equations, (2.1) through (2.23), and the single
occurrence of the string "(2.56)" anywhere in the book *is* this citation.

**Code status.** Not applicable (documentation only). The solver in
`core/estimators/nonlinear_least_squares.py` computes the gain ratio per the
correct Eq. (3.56).

---

## E-05 — Ch. 8, Eq. (8.7): robust covariance scaling is inverted

**Printed.** `R_k = w(ỹ_k) · R_k`, described as "where `w(ỹ_k)` is the weight
output by the robust function based on the current innovation."

**Problem.** The weights "output by the robust function" are the IRLS weights of
the book's own **Table 3.1**, and all of them are bounded by 1:

| loss | weight (Table 3.1) | value at large \|r\| |
|---|---|---|
| Huber | `w = 1` if `|r| ≤ c`, else `c/|r|` | → 0 |
| Cauchy | `w = 1/(1 + (r/c)²)` | → 0 |
| Geman-McClure | `w = 1/(1 + (r/c)²)²` | → 0 |

Multiplying `R_k` by `w ≤ 1` therefore **shrinks** the measurement covariance as
the innovation grows, making the filter trust an outlier *more* — the opposite of
the intent. The same page states the intent plainly two paragraphs later: "If a
residual is large, they scale up its covariance (down-weight that measurement)."

**Correct.** Either invert the factor or state that `w` is not the Table 3.1
weight:

- `R_k = R_k / w(ỹ_k)`, with `w` the Table 3.1 weight (≤ 1); or
- `R_k = w_R(ỹ_k) · R_k`, with `w_R := 1/w ≥ 1` defined as a covariance
  *inflation* factor.

For Huber this gives `w_R = max(1, |r|/δ)`; for Cauchy `w_R = 1 + (r/c)²`.

**Code status.** Correct; the code takes the second form. `core/fusion/tuning.py`
provides `huber_R_scale` / `cauchy_R_scale` returning `w_R ≥ 1`, and
`scale_measurement_covariance` rejects any factor below 1. The IRLS-weight
spellings `huber_weight` / `cauchy_weight` are retained for Table 3.1 use but
emit a `DeprecationWarning` steering callers to the `_R_scale` forms for Eq.
(8.7) — this erratum is exactly the confusion that warning exists to prevent.
Guard test: `tests/core/fusion/test_fusion_tuning.py::TestEq87CovarianceInflation`.

---

# Open author queries

Not errors — proof queries addressed to the authors that appear still unanswered
in the audited PDF, and that only an author can resolve.

## Q-01 — Ch. 4, Eq. (4.106): unresolved `{AU:}` query

**Printed.** Immediately before (4.106), which names the components
`κ_xexe … κ_xuxu` of `(H_aᵀH_a)⁻¹`, the proof carries the copyedit query:

> `{AU: Is this the correct equation? Do you mean (4.77)?}`

**Assessment.** From the surrounding derivation, (4.106) as printed is
*consistent*: (4.105) produces `σ = sqrt(tr((HᵀH)⁻¹))·σ_z`, (4.106) names the
diagonal entries of that same inverse, and (4.107) then writes GDOP as
`sqrt(κ_xexe + κ_xnxn + κ_xuxu)·σ_z`. The chain reads correctly with (4.106)
where it stands, so the query looks answerable with "yes, (4.106) is correct" —
but that is an author's call, not a reviewer's, and it should not ship with the
query still open.

**Code status.** Not applicable. `core/rf/dop.py` implements (4.104)–(4.108) as
printed and is verified against them by
`tests/core/rf/test_dop.py::TestBookDOPFormulas`.

---

# Clarification worth considering

## C-01 — Ch. 6, Eq. (6.60): ZARU needs the gyro measurement, not just the state

**Printed.** The zero-angular-rate update sets `z = 0` with `h(x) = ω_b`.

**Problem.** This is not an error in the mathematics, but it cannot be
implemented through the interface the rest of the chapter uses. Angular rate is
an EKF **input** (`ω_meas`), not a member of the state
`x = [p, v, q, b_g, b_a]`; the corrected rate is `ω_b = ω_meas − b_g`. A
measurement model of the form `h(x)` therefore has no access to `ω_meas` and
cannot form the prediction. Implementing (6.60) requires `h(x, u)`.

**Suggestion.** A sentence noting that the ZARU pseudo-measurement needs the
current gyro reading — i.e. `h(x, u)` rather than `h(x)` — would save every
reader the same discovery.

**Code status.** **Known gap, and the only conformance gap left in this
repository.** `core/sensors/constraints.py` provides
`ZaruMeasurementModelPlaceholder`, which documents that it does *not* implement
(6.60) and returns zeros instead of the predicted angular rate. Its tests pin the
placeholder's current behaviour; they do not demonstrate conformance. Recorded as
a KNOWN GAP against Eq. (6.60) in `docs/equation_index.yml`.
