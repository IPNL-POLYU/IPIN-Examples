# Chapter 5 Equation Numbering and Section Reference Fixes

**Date:** December 24, 2024  
**Task:** P0 — Fix equation numbering and book section references

## Summary

Fixed all equation references and section numbers in Chapter 5 fingerprinting code to align with the book's actual structure from `references/ch5.txt`.

## Book Ground Truth

### Equations
- **Eq. (5.1):** NN argmin distance `i* = argmin D(z, f_i)`
- **Eq. (5.2):** k-NN weighted average `x̂ = Σ(w_i * x_i) / Σ(w_i)`
- **Eq. (5.3):** Bayes posterior `P(x_i|z) = P(z|x_i)P(x_i)/P(z)`
- **Eq. (5.4):** MAP estimate `i* = argmax_i P(x_i|z)`
- **Eq. (5.5):** Posterior mean `x̂ = Σ P(x_i|z) x_i`
- **Eq. (5.6):** Gaussian likelihood example (for computing P(z|x_i))

### Section Structure
- **Section 5.1:** Fundamentals of Fingerprinting (deterministic + probabilistic)
- **Section 5.2:** Pattern Recognition Approaches (classification + regression)
- **Section 5.3:** Deep Learning-Based Approaches

## Changes Made

### 1. ch5_fingerprinting/README.md

#### Equation Reference Table
**Before:**
```
| `log_likelihood()` | ... | Eq. (5.3) | Log p(z|x_i) under Gaussian Naive Bayes |
```

**After:**
```
| `log_likelihood()` | ... | Eq. (5.6) | Likelihood P(z|x_i) using Gaussian model (term in Eq. 5.3) |
| `log_posterior()` | ... | Eq. (5.3) | Bayes posterior: P(x_i|z) = P(z|x_i)P(x_i)/P(z) |
```

#### Section References
**Before:**
```
- Section 5.1: Deterministic Methods (NN, k-NN)
- Section 5.2: Probabilistic Methods (Bayesian)
- Section 5.3: Pattern Recognition (Regression)
```

**After:**
```
- Section 5.1: Fundamentals of Fingerprinting (deterministic + probabilistic)
- Section 5.2: Pattern Recognition Approaches (classification + regression)
- Section 5.3: Deep Learning-Based Approaches (not yet implemented)
```

#### Usage Example Comments
**Before:**
```python
# Fit Bayesian model
model = fit_gaussian_naive_bayes(db, min_std=2.0)
```

**After:**
```python
# Fit Bayesian model (uses Gaussian likelihood Eq. 5.6)
model = fit_gaussian_naive_bayes(db, min_std=2.0)

# MAP estimate (Eq. 5.4): discrete, selects best RP
pos_map = map_localize(query, model, floor_id=0)

# Posterior mean (Eq. 5.5): continuous, weighted average
pos_mean = posterior_mean_localize(query, model, floor_id=0)
```

### 2. core/fingerprinting/probabilistic.py

#### Module Header
**Before:**
```python
"""Probabilistic fingerprinting methods for Chapter 5.

This module implements Bayesian fingerprinting using Gaussian Naive Bayes
models, as described in Section 5.2 of the book.

Key equations:
    - Eq. (5.3): Log-likelihood log p(z | x_i)
    - Eq. (5.4): MAP estimate i* = argmax_i p(x_i | z)
    - Eq. (5.5): Posterior mean estimate x̂ = Σ p(x_i | z) x_i
```

**After:**
```python
"""Probabilistic fingerprinting methods for Chapter 5.

This module implements Bayesian fingerprinting using Gaussian Naive Bayes
models, as described in Section 5.1.3 (Probabilistic Fingerprinting) of the book.

Key equations:
    - Eq. (5.3): Bayes posterior P(x_i | z) = P(z | x_i) P(x_i) / P(z)
    - Eq. (5.4): MAP estimate i* = argmax_i P(x_i | z)
    - Eq. (5.5): Posterior mean estimate x̂ = Σ P(x_i | z) x_i
    - Eq. (5.6): Gaussian likelihood P(z | x_i) = N(z; μ_i, Σ_i)
```

#### `log_likelihood()` Function
**Before:**
```python
"""
Compute log-likelihood log p(z | x_i) for all reference points.

Implements Eq. (5.3) in Chapter 5. Under Gaussian Naive Bayes:
    log p(z | x_i) = Σ_j log N(z_j; μ_ij, σ_ij²)
    ...

References:
    Chapter 5, Eq. (5.3): Log-likelihood under Gaussian Naive Bayes.
"""
```

**After:**
```python
"""
Compute log-likelihood log P(z | x_i) for all reference points.

This function computes the likelihood term P(z | x_i) that appears in
Bayes' rule (Eq. 5.3). Under Gaussian Naive Bayes (Eq. 5.6):
    log P(z | x_i) = Σ_j log N(z_j; μ_ij, σ_ij²)
    ...

References:
    Chapter 5, Eq. (5.6): Gaussian likelihood model.
    Chapter 5, Eq. (5.3): Bayes posterior uses this likelihood term.
"""
```

#### `log_posterior()` Function
**Before:**
```python
"""
Compute log-posterior log p(x_i | z) for all reference points.

Uses Bayes' rule:
    p(x_i | z) ∝ p(z | x_i) p(x_i)
    ...

References:
    Chapter 5, Eqs. (5.4)-(5.5): Bayesian inference for localization.
"""
```

**After:**
```python
"""
Compute log-posterior log P(x_i | z) for all reference points.

Implements Eq. (5.3) using Bayes' rule:
    P(x_i | z) = P(z | x_i) P(x_i) / P(z)
    ...

References:
    Chapter 5, Eq. (5.3): Bayes posterior P(x_i | z) = P(z | x_i) P(x_i) / P(z).
    Chapter 5, Eqs. (5.4)-(5.5): Used in MAP and posterior mean estimation.
"""
```

#### Code Comments
**Before:**
```python
# Implements: Σ_j log N(z_j; μ_ij, σ_ij²) from Eq. (5.3)
```

**After:**
```python
# Implements: Σ_j log N(z_j; μ_ij, σ_ij²) from Eq. (5.6)
```

### 3. ch5_fingerprinting/example_probabilistic.py

#### File Header
**Before:**
```python
"""
Example: Probabilistic Fingerprinting (Bayesian Methods)

Demonstrates Bayesian fingerprinting using Gaussian Naive Bayes model
from Chapter 5.

Implements:
    - Gaussian Naive Bayes model fitting
    - Log-likelihood computation (Eq. 5.3): log p(z|x_i)
    - MAP estimation (Eq. 5.4): i* = argmax_i p(x_i|z)
    - Posterior mean estimation (Eq. 5.5): x̂ = Σ p(x_i|z) x_i
```

**After:**
```python
"""
Example: Probabilistic Fingerprinting (Bayesian Methods)

Demonstrates Bayesian fingerprinting using Gaussian Naive Bayes model
from Chapter 5, Section 5.1.3.

Implements:
    - Gaussian Naive Bayes model fitting (Eq. 5.6)
    - Bayes posterior computation (Eq. 5.3): P(x_i|z) = P(z|x_i)P(x_i)/P(z)
    - MAP estimation (Eq. 5.4): i* = argmax_i P(x_i|z)
    - Posterior mean estimation (Eq. 5.5): x̂ = Σ P(x_i|z) x_i
```

#### Print Statements
**Before:**
```python
print("   (Equations 5.3, 5.4, 5.5 from Chapter 5)")
```

**After:**
```python
print("   (Eqs. 5.3-5.6 from Chapter 5, Section 5.1.3)")
```

**Before:**
```python
print("\nReferences:")
print("  - Equation 5.3: Log-likelihood log p(z|x_i)")
print("  - Equation 5.4: MAP estimate")
print("  - Equation 5.5: Posterior mean estimate")
```

**After:**
```python
print("\nReferences:")
print("  - Equation 5.3: Bayes posterior P(x_i|z) = P(z|x_i)P(x_i)/P(z)")
print("  - Equation 5.4: MAP estimate i* = argmax_i P(x_i|z)")
print("  - Equation 5.5: Posterior mean x̂ = Σ P(x_i|z) x_i")
print("  - Equation 5.6: Gaussian likelihood P(z|x_i) = N(z; μ_i, Σ_i)")
```

## Verification

### No Files Claim "Eq (5.3) = log-likelihood"
✅ All references to Eq. (5.3) now correctly identify it as Bayes posterior

### README Section Numbering Matches Book
✅ Section 5.1: Fundamentals (both deterministic + probabilistic)  
✅ Section 5.2: Pattern Recognition  
✅ Section 5.3: Deep Learning (noted as not implemented)

### Example Outputs Reference Right Equations
✅ All print statements updated with correct equation descriptions  
✅ Comments clarify that log_likelihood() computes P(z|x_i) (Eq. 5.6), which is used in Eq. 5.3

## Acceptance Criteria

✅ **No file claims "Eq (5.3) = log-likelihood"**  
   - All references corrected to Eq. (5.6) for Gaussian likelihood
   - Eq. (5.3) consistently refers to Bayes posterior

✅ **README section numbering matches book's Chapter 5 structure**  
   - Section references aligned with book organization
   - Subsections properly noted (e.g., 5.1.3 for probabilistic)

✅ **Example outputs reference the right equations**  
   - Print statements updated with full equation descriptions
   - Comments clarify the role of each equation

## Linter Status

All modified files pass linting with no errors:
- `ch5_fingerprinting/README.md`
- `core/fingerprinting/probabilistic.py`
- `ch5_fingerprinting/example_probabilistic.py`

## Conclusion

All equation numbering and section references in Chapter 5 fingerprinting code now correctly align with the book's structure as documented in `references/ch5.txt`. The distinction between:
- **Eq. (5.3):** Bayes posterior formula
- **Eq. (5.6):** Gaussian likelihood (used to compute P(z|x_i) in Eq. 5.3)

is now clear throughout the codebase.









