# Book Errata

Errors found in the published book *Principles of Indoor Positioning and Indoor
Navigation* (Artech House, 2026) while auditing this companion code against the
text. Each entry records the printed equation, the correct form, and how the code
handles it. This list is intended both to keep the code honest and to feed
corrections back to the publisher.

Format: **E-NN** | chapter/eq | printed | correct | code status.

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
