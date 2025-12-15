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

**Nearest-Neighbor (NN)**
- Decision rule: i* = argmin_i D(z, f_i)
- Supports Euclidean and Manhattan distance metrics
- Fast: O(M) for M reference points
- Provides discrete position estimates

**k-Nearest-Neighbor (k-NN)**
- Weighted average: x = sum(w_i * x_i) / sum(w_i)
- Two weighting schemes:
  - Inverse distance: w_i = 1/(D(z, f_i) + epsilon)
  - Uniform: w_i = 1 (simple average)
- Optimal k depends on RP density and noise (typically k=3-7)

### Probabilistic Methods (Eqs. 5.3-5.5)

**Gaussian Naive Bayes Model**
- Assumes Gaussian distribution per feature per RP
- Parameters: mu_ij (mean RSS), sigma_ij (std)
- Log-likelihood (Eq. 5.3): log p(z|x_i) = sum_j log N(z_j; mu_ij, sigma_ij^2)

**MAP Estimation (Eq. 5.4)**
- Maximum A Posteriori: i* = argmax_i p(x_i|z)
- Returns discrete estimate (one of the RPs)
- Sensitive to model parameters (sigma)

**Posterior Mean (Eq. 5.5)**
- Expected value: x = sum_i p(x_i|z) x_i
- Continuous estimate (weighted average)
- More robust to outliers

### Pattern Recognition (Linear Regression)

**Linear Model**
- Learns direct mapping: x = Wz + b
- Training: Ridge regression with regularization parameter lambda
- Closed-form solution using normal equations

## Dataset Specifications

**Synthetic Wi-Fi Fingerprint Database** (`data/sim/wifi_fingerprint_grid/`)
- Coverage: 50m x 50m area per floor
- Grid: 11x11 RPs, 5m spacing (121 RPs per floor)
- Floors: 3 (floor 0, 1, 2)
- Total RPs: 363 (121 x 3)
- Access Points: 8 APs strategically positioned
- RSS Model: Log-distance path-loss + shadow fading
  - P0 = -30 dBm (reference power at 1m)
  - n = 2.5 (indoor path-loss exponent)
  - sigma = 4 dBm (shadow fading std)
  - Floor attenuation: 15 dB per floor

## Performance Characteristics

### Computational Complexity

| Method | Training | Query | Space |
|--------|----------|-------|-------|
| NN | O(1) | O(MN) | O(MN) |
| k-NN | O(1) | O(MN) | O(MN) |
| MAP | O(MN) | O(MN) | O(MN) |
| Posterior Mean | O(MN) | O(MN) | O(MN) |
| Linear Reg | O(N^3) | O(N) | O(Nd) |

Where: M = reference points, N = features (APs), d = location dimension

### Typical Performance

| Scenario | NN | k-NN | MAP | Posterior Mean | Linear Reg |
|----------|-----|------|-----|----------------|------------|
| Baseline (1 dBm) | 5.2m | 4.2m | 5.2m | 4.3m | 5.0m |
| Moderate (2 dBm) | 6.9m | 5.6m | 6.9m | 5.8m | 6.3m |
| High Noise (5 dBm) | 11.1m | 9.0m | 11.1m | 10.6m | 10.2m |

### Speed Ranking (fastest to slowest)
1. Linear Regression: 0.03ms (30x faster)
2. NN: 0.5ms
3. k-NN: 0.5ms
4. MAP: 1.1ms
5. Posterior Mean: 1.1ms

## Recommendations

### Application-Specific Guidance

**Real-time Applications (speed critical):**
- Use Linear Regression or NN
- Linear Reg: Train offline, ultra-fast online

**High Accuracy Required:**
- Use k-NN (k=3-5) or Posterior Mean
- Both provide smooth estimates

**Noisy Environments:**
- Use k-NN with moderate k (3-7)
- Or Posterior Mean with appropriate sigma (2-4 dBm)

**Dense Reference Points:**
- NN sufficient (RPs every 2-3m)

**Sparse Reference Points:**
- k-NN or Linear Regression for interpolation

## Future Enhancements

Potential extensions (not currently implemented):
- Neural network models
- Gaussian processes
- Hybrid methods (fingerprinting + ranging)
- Online learning
- Missing data handling
- Floor identification classifier

## Troubleshooting

**Poor accuracy (RMSE > 5m):**
- Check RP density (should be <= 5m spacing)
- Verify RSS quality
- Try k-NN or Posterior Mean
- Increase regularization for Linear Reg

**Slow computation:**
- Use Linear Regression
- Reduce database size
- Use floor constraints

**Floor confusion:**
- Use floor-constrained search
- Train separate models per floor

---

**Test Coverage:** 125 test cases, 100% pass rate  
**Last Updated:** December 2025

