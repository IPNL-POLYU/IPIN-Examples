# Chapter 8: Development Notes

> **Note:** This document contains implementation details, design decisions, and development notes for Chapter 8. For student-facing documentation, see [ch8_sensor_fusion/README.md](../ch8_sensor_fusion/README.md).

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| TC IMU + UWB Fusion | Complete | 5D state EKF |
| LC IMU + UWB Fusion | Complete | WLS + EKF |
| Chi-Square Gating | Complete | Eq. (8.9) |
| Innovation Monitoring | Complete | Eqs. (8.5)-(8.6) |
| Robust Down-weighting | Complete | Eq. (8.7), Huber/Cauchy |
| Observability Demo | Complete | Translation unobservability |
| Temporal Calibration | Complete | TimeSyncModel |
| LC vs TC Comparison | Complete | Side-by-side analysis |

## Implementation Notes

### State Representation

5D state vector: `[px, py, vx, vy, yaw]`

**Note:** This simplified model does not include IMU bias estimation. Production systems would use a 15D state (Chapter 6 full strapdown).

### Tightly Coupled Fusion

- Fuses raw UWB range measurements directly
- One EKF update per anchor per epoch
- Chi-square gating with 1 DOF (scalar range)
- Better observability and dropout handling

### Loosely Coupled Fusion

- First computes position fix using WLS (Chapter 4)
- Then fuses position in EKF
- One EKF update per epoch
- Chi-square gating with 2 DOF (position vector)
- Simpler but requires ≥3 valid ranges

### Chi-Square Gating (Eq. 8.9)

```python
# Accept measurement if:
d² = y' * S⁻¹ * y  # Mahalanobis distance squared
d² < χ²(α, m)      # Chi-square threshold
```

Where:
- y = innovation (z - h(x))
- S = innovation covariance
- α = significance level (default 0.05)
- m = measurement dimension (1 for TC, 2 for LC)

### Robust Down-weighting (Eq. 8.7)

Three loss functions implemented:
- **Huber:** Linear tail for outliers
- **Cauchy:** Bounded influence function
- **L2 (baseline):** Standard quadratic loss

## Performance Characteristics

### Expected Results (Baseline Dataset)

| Metric | TC Fusion | LC Fusion |
|--------|-----------|-----------|
| RMSE 2D | ~12.4 m | ~12.9 m |
| Acceptance Rate | ~33% | ~30% |
| Updates per Epoch | 4 | 1 |

**Why ~12m RMSE?**

This is expected for the simplified 5D model without IMU bias estimation. The demo prioritizes clarity over accuracy to illustrate:
- How TC vs LC fusion works architecturally
- How chi-square gating operates
- How to monitor NIS consistency

### Dataset Variants

| Dataset | Purpose |
|---------|---------|
| `fusion_2d_imu_uwb/` | Baseline demos |
| `fusion_2d_imu_uwb_nlos/` | +0.8m NLOS bias on anchors 1,2 |
| `fusion_2d_imu_uwb_timeoffset/` | 50ms offset + 100ppm drift |

## Design Decisions

### Why Both TC and LC?

**Educational Value:**
- TC demonstrates raw sensor fusion principles
- LC demonstrates architectural simplification
- Both use same dataset for direct comparison

**Practical Guidance:**
- TC: Better for accuracy, dropout robustness
- LC: Simpler, good when position solver exists

### Observability Analysis

Key insight demonstrated:
- Odometry alone cannot observe absolute translation
- Adding occasional absolute fixes enables full observability

### Temporal Calibration

TimeSyncModel: `t_fusion = (1 + drift) * t_sensor + offset`

Demo shows:
- How small time offsets (~50ms) degrade fusion
- How proper calibration recovers accuracy

## Future Extensions

- 15D state with IMU bias estimation
- Additional sensor modalities (WiFi, BLE)
- Real sensor data integration
- Online calibration

## Related Chapters

- **Chapter 3:** EKF implementation
- **Chapter 4:** UWB positioning (WLS solver)
- **Chapter 6:** IMU strapdown integration

---

**Last Updated:** December 2025


