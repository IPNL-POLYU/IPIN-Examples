# IMU + UWB Fusion Dataset (NLOS Variant)

## Overview

**Purpose**: Test fusion robustness under Non-Line-of-Sight (NLOS) conditions where UWB ranges are systematically biased due to signal obstruction and multipath.

**Learning Objectives**:
- Understand NLOS effects on ranging measurements (Ch4, Section 4.5)
- Learn chi-square gating for outlier detection and rejection (Ch8, Eqs. 8.8-8.9)
- Compare performance with vs. without gating
- Study robust loss functions (Huber, Cauchy) as alternatives
- Observe filter behavior when measurements contradict predictions

**Related Chapter**: Chapter 8 - Sensor Fusion (Section 8.3: Robust Estimation)

**Related Book Equations**: Eqs. (8.8)-(8.9) for chi-square gating, Eq. (8.10) for robust losses

---

## Scenario Description

**Identical to baseline EXCEPT**:
- **Anchors 1 and 2 have NLOS bias**: +0.8 meters added to all ranges
- This simulates signal obstruction (walls, furniture) causing longer propagation paths

**NLOS Characteristics**:
- **Positive bias only**: NLOS always increases measured range (signal takes longer path)
- **Anchor 1** (20.0, 0.0): Bottom-right corner - NLOS
- **Anchor 2** (20.0, 15.0): Top-right corner - NLOS
- **Anchors 0 and 3**: Line-of-sight (clean)

**Physical Interpretation**:
- Right side of building may have metal structure, dense walls, or equipment
- Left side has clear propagation paths
- This asymmetry is realistic in indoor environments

**All other parameters** (trajectory, IMU, other UWB settings) identical to baseline. See `fusion_2d_imu_uwb/README.md` for full details.

---

## Files and Data Structure

Same structure as baseline - see `fusion_2d_imu_uwb/README.md`:
- `truth.npz`: Ground truth (t, p_xy, v_xy, yaw)
- `imu.npz`: IMU measurements  
- `uwb_ranges.npz`: UWB ranges (but Anchors 1,2 biased +0.8m)
- `uwb_anchors.npy`: Anchor positions
- `config.json`: Configuration with `nlos_anchors: [1, 2]` and `nlos_bias_m: 0.8`

---

## Loading Example

```python
import numpy as np
import json

# Load dataset
dataset_path = 'data/sim/ch8_fusion_2d_imu_uwb_nlos'
truth = np.load(f'{dataset_path}/truth.npz')
uwb = np.load(f'{dataset_path}/uwb_ranges.npz')
anchors = np.load(f'{dataset_path}/uwb_anchors.npy')

with open(f'{dataset_path}/config.json') as f:
    config = json.load(f)

# Check NLOS configuration
nlos_anchors = config['uwb']['nlos_anchors']
nlos_bias = config['uwb']['nlos_bias_m']
print(f"NLOS anchors: {nlos_anchors}")
print(f"NLOS bias: +{nlos_bias} meters")

# Compute true vs. measured range statistics
t_uwb = uwb['t']
ranges = uwb['ranges']
p_xy = truth['p_xy']

# Interpolate truth to UWB timestamps
p_xy_uwb = np.column_stack([
    np.interp(t_uwb, truth['t'], p_xy[:, 0]),
    np.interp(t_uwb, truth['t'], p_xy[:, 1])
])

# Compute true ranges
ranges_true = np.array([
    np.linalg.norm(p_xy_uwb - anchor, axis=1) 
    for anchor in anchors
]).T

# Analyze errors per anchor
for i in range(4):
    valid = ~np.isnan(ranges[:, i])
    errors = ranges[valid, i] - ranges_true[valid, i]
    print(f"\nAnchor {i} range errors:")
    print(f"  Mean: {np.mean(errors):.3f} m")
    print(f"  Std: {np.std(errors):.3f} m")
    print(f"  Expected: {'+0.8m bias' if i in nlos_anchors else '~0m (noise only)'}")
```

---

## Configuration Parameters

**Key differences from baseline**:

### UWB Parameters

- `nlos_anchors`: [1, 2] **(Changed from [])**
- `nlos_bias_m`: 0.8 **(Changed from 0.0)**

All other parameters identical to baseline:
- IMU: accel_noise = 0.1 m/s², gyro_noise = 0.01 rad/s
- UWB: range_noise = 0.05 m, rate = 10 Hz, dropout = 5%
- Trajectory: 20m × 15m rectangle, 1 m/s, 60s

---

## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on Gating Performance | Learning Objective |
|-----------|---------|------------------|------------------------------|-------------------|
| `nlos_bias` | 0.8 | 0.2-2.0 | Larger bias → higher NIS values → more rejections | Observe gating effectiveness vs. outlier magnitude |
| `nlos_anchors` | [1,2] | [0], [1], [1,2], [0,1,2] | More NLOS → fewer valid updates → larger uncertainty | Understand redundancy importance |
| `range_noise_std` | 0.05 | 0.01-0.2 | Higher noise → harder to distinguish NLOS from noise | Learn threshold tuning (Ch8, Eq. 8.9) |
| Gating threshold (α) | 0.05 | 0.01-0.20 | Stricter (α↓) → more rejections, fewer false accepts | Trade-off between robustness and availability |

**Key Insight**: Chi-square gating (Ch8, Eqs. 8.8-8.9) provides principled outlier rejection based on innovation statistics. With 0.8m bias on 0.05m-noise ranges, expect 60-80% rejection rate on NLOS anchors while keeping ~95% of clean measurements.

---

## Dataset Variants

This is the **NLOS variant**. Related datasets:

**1. fusion_2d_imu_uwb/** - Baseline (Clean)
   - No NLOS, all anchors clean
   - Use for: Establishing performance baseline
   - Comparison: NLOS variant should show degraded performance without gating, similar performance with gating

**2. fusion_2d_imu_uwb_timeoffset/** - Temporal Calibration
   - Different challenge: time synchronization instead of outliers
   - Use for: Temporal calibration experiments

---

## Visualization Example

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data
dataset_path = 'data/sim/ch8_fusion_2d_imu_uwb_nlos'
truth = np.load(f'{dataset_path}/truth.npz')
uwb = np.load(f'{dataset_path}/uwb_ranges.npz')
anchors = np.load(f'{dataset_path}/uwb_anchors.npy')

# Compute range errors
t_uwb = uwb['t']
ranges = uwb['ranges']
p_xy_uwb = np.column_stack([
    np.interp(t_uwb, truth['t'], truth['p_xy'][:, 0]),
    np.interp(t_uwb, truth['t'], truth['p_xy'][:, 1])
])

ranges_true = np.array([
    np.linalg.norm(p_xy_uwb - anchor, axis=1) 
    for anchor in anchors
]).T

# Plot range errors per anchor
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i in range(4):
    ax = axes[i // 2, i % 2]
    valid = ~np.isnan(ranges[:, i])
    errors = ranges[valid, i] - ranges_true[valid, i]
    
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='k', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(np.mean(errors), color='r', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(errors):.3f}m')
    
    title_suffix = ' (NLOS +0.8m bias)' if i in [1, 2] else ' (Clean)'
    ax.set_xlabel('Range Error (m)')
    ax.set_ylabel('Count')
    ax.set_title(f'Anchor {i}{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fusion_nlos_range_errors.svg')

# Plot trajectory with NLOS anchors highlighted
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1], 
        'k-', label='Ground Truth', linewidth=2, alpha=0.7)

# Highlight clean vs. NLOS anchors
clean_anchors = anchors[[0, 3]]
nlos_anchors = anchors[[1, 2]]

ax.plot(clean_anchors[:, 0], clean_anchors[:, 1], 
        'g^', markersize=15, label='Clean Anchors (0, 3)')
ax.plot(nlos_anchors[:, 0], nlos_anchors[:, 1], 
        'r^', markersize=15, label='NLOS Anchors (1, 2)')

for i in range(4):
    color = 'red' if i in [1, 2] else 'green'
    ax.text(anchors[i, 0], anchors[i, 1] + 1, f'A{i}', 
            ha='center', fontsize=12, fontweight='bold', color=color)

ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_title('IMU + UWB Fusion (NLOS Variant)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.savefig('fusion_nlos_scenario.svg')
```

---

## Connection to Book Equations

**Chi-Square Gating** (Ch8, Eqs. 8.8-8.9):

Innovation: y_k = z_k - h(x̂_k)

Innovation Covariance: S_k = H_k P_k H_k' + R

NIS (Normalized Innovation Squared):
```
d² = y_k' S_k⁻¹ y_k
```

Gating decision:
```
Accept measurement if d² < χ²_α(m)
Reject otherwise
```

For m=1 range (single anchor), α=0.05: χ²_{0.05}(1) = 3.84

**Expected Behavior**:

Clean anchor (e.g., Anchor 0):
- Innovation y ≈ noise (σ = 0.05m)
- d² typically < 3.84
- ~5% false rejections (false alarms)

NLOS anchor (e.g., Anchor 1):
- Innovation y ≈ bias + noise (0.8m + 0.05m)
- d² >> 3.84 (bias dominates)
- ~60-80% correct rejections

**Mathematical Insight**: For bias b and noise σ:
```
d² ≈ (b/σ)² ~ (0.8/0.05)² = 256 >> 3.84
```

So NLOS measurements should be easily detected!

---

## Recommended Experiments

### Experiment 1: Gating Effectiveness Study

**Objective**: Quantify chi-square gating performance under NLOS conditions.

**Setup**:
```bash
# Run fusion without gating (corrupted by NLOS)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/ch8_fusion_2d_imu_uwb_nlos \
    --no-gating \
    --output results_no_gating.json

# Run fusion with chi-square gating (α=0.05)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/ch8_fusion_2d_imu_uwb_nlos \
    --alpha 0.05 \
    --output results_gating_0.05.json

# Run with stricter gating (α=0.01)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/ch8_fusion_2d_imu_uwb_nlos \
    --alpha 0.01 \
    --output results_gating_0.01.json
```

**Expected Observations**:

| Configuration | Position RMSE | Anchors 1,2 Rejection Rate | Anchors 0,3 Rejection Rate |
|---------------|---------------|----------------------------|----------------------------|
| No gating | 0.50-0.70 m | 0% (all accepted) | 0% |
| α = 0.05 (95%) | 0.10-0.15 m | 60-80% | ~5% (false alarms) |
| α = 0.01 (99%) | 0.08-0.12 m | 75-90% | ~1% |

**Analysis**:
1. Plot position error over time (all 3 cases)
2. Visualize accepted vs. rejected measurements on trajectory
3. Compute per-anchor rejection statistics
4. Plot NIS values with χ² threshold lines

**Key Insight**: Without gating, NLOS bias causes systematic position error (~0.4-0.6m toward biased anchors). With gating, performance recovers to near-baseline levels by automatically rejecting outliers.

---

### Experiment 2: NLOS Bias Severity Sweep

**Objective**: Understand how gating effectiveness scales with outlier magnitude.

**Setup**:
```bash
# Generate datasets with varying NLOS bias
for bias in 0.2 0.5 1.0 2.0; do
    python scripts/generate_fusion_2d_imu_uwb_dataset.py \
        --nlos-anchors 1 2 \
        --nlos-bias $bias \
        --output data/sim/fusion_nlos_bias_${bias}
done

# Run gated fusion on each
for bias in 0.2 0.5 1.0 2.0; do
    python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
        --data data/sim/fusion_nlos_bias_${bias} \
        --alpha 0.05 \
        --output results_bias_${bias}.json
done
```

**Expected Observations**:

| NLOS Bias | Rejection Rate | Position RMSE | Gating Effectiveness |
|-----------|----------------|---------------|----------------------|
| 0.2 m | ~20-30% | 0.15-0.20 m | Moderate (some leakage) |
| 0.5 m | ~40-60% | 0.12-0.15 m | Good |
| 1.0 m | ~70-85% | 0.10-0.12 m | Excellent |
| 2.0 m | ~85-95% | 0.08-0.10 m | Nearly perfect |

**Analysis**:
- Plot rejection rate vs. bias (should be increasing, sigmoid-like)
- Plot RMSE vs. bias (U-shaped: small bias leaks through, large bias rejected)
- Verify theoretical prediction: rejection rate ≈ P(χ²(λ) > threshold) where λ = (bias/σ)²

**Key Insight**: Gating becomes more effective as bias-to-noise ratio increases. For bias < 3σ, some NLOS measurements may be accepted (type II error). For bias > 5σ, nearly all are rejected.

---

### Experiment 3: Robust Loss Functions Comparison

**Objective**: Compare chi-square gating (hard threshold) with robust loss functions (soft downweighting).

**Setup**:
Modify fusion to support different loss functions:
- **Squared loss** (standard): ρ(r) = r²
- **Huber loss**: ρ(r) = r² if |r| < k, else 2k|r| - k²
- **Cauchy loss**: ρ(r) = (k²/2) log(1 + (r/k)²)
- **Chi-square gating**: Hard rejection if d² > threshold

**Expected Observations**:
- Squared loss: Poor performance (NLOS dominates)
- Huber loss: Moderate improvement (downweights but doesn't reject)
- Cauchy loss: Good improvement (aggressive downweighting)
- Chi-square gating: Best for this scenario (complete rejection)

**Analysis**:
- Plot effective weight vs. innovation magnitude for each loss
- Understand trade-off: soft (keeps information) vs. hard (conservative)

**Key Insight**: Hard gating (Ch8, Eqs. 8.8-8.9) is simple and effective when outliers are clearly separable. Robust losses (Ch8, Eq. 8.10) are better when outlier/inlier boundary is fuzzy.

---

## Troubleshooting / Common Student Questions

**Q: Why not just use all 4 anchors and average the errors?**
A: NLOS bias is systematic, not random. Averaging with clean ranges still yields biased result. Must detect and reject (or downweight) NLOS measurements.

**Q: Can I detect which anchors are NLOS and exclude them permanently?**
A: NLOS can be time-varying (depends on user location, occlusions). Better to use adaptive methods (chi-square gating, robust losses) that work per-measurement rather than per-anchor.

**Q: My gating rejects too many clean measurements. What should I do?**
A: Causes:
1. Under-estimated R → innovations appear large → tune R upward
2. Poor prediction → large prediction errors → check IMU integration
3. Too strict α (e.g., α=0.001) → relax to α=0.05 or α=0.10
4. Check NIS plot for clean baseline first

**Q: What if all anchors are NLOS?**
A: With all measurements rejected, filter becomes IMU-only (unbounded drift). Solutions:
1. Relax gating threshold temporarily
2. Use robust loss instead of hard gating
3. Combine with other modalities (Wi-Fi, vision, maps)
4. Conservative mode: increase covariance when all rejected

**Q: How does this compare to real indoor NLOS?**
A: This simulates constant bias. Real NLOS can be:
- Time-varying (moving people/doors)
- Multiple reflections (harder to model)
- Ranging algorithm-dependent (some detect/mitigate NLOS)
Still, constant bias is good first approximation for testing robustness.

---

## Generation

This dataset was generated using:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --nlos-anchors 1 2 \
    --nlos-bias 0.8 \
    --output data/sim/ch8_fusion_2d_imu_uwb_nlos
```

**Or using preset**:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py --preset nlos_severe
```

**Generate all 3 standard variants**:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py --all-variants
```

**Custom NLOS experiments**:
```bash
# Very severe NLOS (2m bias on 3 anchors)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --nlos-anchors 0 1 2 \
    --nlos-bias 2.0 \
    --output data/sim/fusion_nlos_extreme

# Mild NLOS (0.2m bias, harder to detect)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --nlos-anchors 1 \
    --nlos-bias 0.2 \
    --output data/sim/fusion_nlos_mild
```

---

## References

- **Book Chapter**: Chapter 8, Section 8.3 (Robust Estimation)
- **Key Equations**: 
  - Chi-square gating: Eqs. (8.8)-(8.9)
  - Robust losses: Eq. (8.10)
  - Innovation monitoring: Eqs. (8.5)-(8.7)
- **NLOS Background**: Chapter 4, Section 4.5 (Multipath and NLOS)
- **Related Examples**:
  - `ch8_sensor_fusion/tc_uwb_imu_ekf.py` - Main fusion with gating support
  - `ch8_sensor_fusion/example_robust_losses.py` - Robust loss comparison
  - `ch8_sensor_fusion/example_innovation_monitoring.py` - NIS analysis
- **Generation Script**: `scripts/generate_fusion_2d_imu_uwb_dataset.py`
- **Baseline**: See `fusion_2d_imu_uwb/README.md` for clean reference


