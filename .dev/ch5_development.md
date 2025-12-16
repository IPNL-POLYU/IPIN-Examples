# Chapter 5: Development Notes

> **Note:** This document contains implementation details, design decisions, and development notes for Chapter 5. For student-facing documentation, see [ch5_fingerprinting/README.md](../ch5_fingerprinting/README.md).

## Implementation Status

| Feature | Status | Test Cases | Notes |
|---------|--------|------------|-------|
| Nearest-Neighbor (NN) | Complete | 10 | Eq. (5.1) |
| k-Nearest-Neighbor (k-NN) | Complete | 12 | Eq. (5.2) |
| Gaussian Naive Bayes | Complete | 8 | Model fitting |
| MAP Estimation | Complete | 8 | Eq. (5.4) |
| Posterior Mean | Complete | 8 | Eq. (5.5) |
| Linear Regression | Complete | 12 | Ridge regression |
| FingerprintDatabase | Complete | 20 | Data structures |
| Multi-floor Support | Complete | 15 | Floor constraints |

**Total Tests:** 125 unit tests, 100% pass rate

## Implementation Notes

### Deterministic Methods (Eqs. 5.1-5.2)

**k-Nearest-Neighbor (k-NN)**
- Weighted average: x = sum(w_i * x_i) / sum(w_i)
- Two weighting schemes: Inverse distance, Uniform
- Optimal k depends on RP density and noise (typically k=3-7)

### Probabilistic Methods (Eqs. 5.3-5.5)

**Gaussian Naive Bayes Model**
- Assumes Gaussian distribution per feature per RP
- Parameters: mu_ij (mean RSS), sigma_ij (std)

**MAP vs Posterior Mean**
- MAP returns discrete estimate
- Posterior Mean returns continuous estimate (more robust)

## Performance Characteristics

| Scenario | NN | k-NN | MAP | Posterior Mean | Linear Reg |
|----------|-----|------|-----|----------------|------------|
| Baseline (1 dBm) | 5.2m | 4.2m | 5.2m | 4.3m | 5.0m |
| Moderate (2 dBm) | 6.9m | 5.6m | 6.9m | 5.8m | 6.3m |
| High Noise (5 dBm) | 11.1m | 9.0m | 11.1m | 10.6m | 10.2m |

## Recommendations

- **Real-time Applications:** Use Linear Regression or NN
- **High Accuracy:** Use k-NN (k=3-5) or Posterior Mean
- **Noisy Environments:** Use k-NN with moderate k (3-7)

---

**Test Coverage:** 125 test cases, 100% pass rate  
**Last Updated:** December 2025


