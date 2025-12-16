# Data Simulation Guide: Theory to Practice

This guide bridges theoretical concepts from the IPIN book to practical simulation parameters, helping students understand how mathematical models translate into experimental data.

## Purpose

This guide enables students to:

1. **Connect Theory to Simulation**: Understand which simulation parameters correspond to which book equations
2. **Predict Behavior**: Anticipate algorithm performance based on parameter choices
3. **Design Experiments**: Systematically explore parameter space to test hypotheses
4. **Interpret Results**: Understand why algorithms behave as they do under different conditions

---

## Table of Contents

- [Theory-to-Simulation Mappings](#theory-to-simulation-mappings)
  - [IMU Error Models (Ch6)](#imu-error-models-ch6)
  - [RF Measurement Models (Ch4)](#rf-measurement-models-ch4)
  - [RSS Path Loss (Ch5)](#rss-path-loss-ch5)
- [Step-by-Step Experiment Guides](#step-by-step-experiment-guides)
  - [Experiment 1: IMU Drift Characterization](#experiment-1-imu-drift-characterization)
  - [Experiment 2: Filter Tuning Sensitivity](#experiment-2-filter-tuning-sensitivity)
  - [Experiment 3: NLOS Detection and Rejection](#experiment-3-nlos-detection-and-rejection)
- [Parameter Sensitivity Reference](#parameter-sensitivity-reference)
- [Common Student Questions](#common-student-questions)

---

## Theory-to-Simulation Mappings

### IMU Error Models (Ch6)

#### Book Equations

**Gyroscope Error Model** (Ch6, Eq. 6.5):
```
ω̃ = ω + b_g + n_g
```

Where:
- ω̃: measured angular velocity
- ω: true angular velocity
- b_g: gyroscope bias (slowly varying)
- n_g: white noise

**Accelerometer Error Model** (Ch6, similar to Eq. 6.9):
```
f̃ = f + b_a + n_a
```

Where:
- f̃: measured specific force
- f: true specific force
- b_a: accelerometer bias
- n_a: white noise

#### Simulation Parameters

| Book Term | Simulation Parameter | Typical Values | Units |
|-----------|---------------------|----------------|-------|
| n_g (gyro noise) | `gyro_noise_std` | 0.001-0.1 | rad/s |
| b_g (gyro bias) | `gyro_bias` | 0.0-0.01 | rad/s |
| n_a (accel noise) | `accel_noise_std` | 0.01-1.0 | m/s² |
| b_a (accel bias) | `accel_bias` | [0, 0, 0]-[0.5, 0.5, 0.5] | m/s² |

#### Behavior Predictions

**Gyro Noise Effect**:
- Random walk in heading: σ_heading(t) = σ_gyro × √t
- Example: σ_gyro = 0.01 rad/s → heading drift ~0.6° after 10s, ~6° after 1000s
- Visible as: Oscillating heading error, bounded by √t growth

**Gyro Bias Effect**:
- Linear heading drift: Δheading(t) = b_g × t
- Example: b_g = 0.01 rad/s → 57°/s drift, reaches 90° in ~1.5s
- Visible as: Systematic, unbounded heading error (linear growth)

**Accel Noise Effect**:
- Random walk in velocity: σ_v(t) = σ_a × √t
- Random walk in position: σ_p(t) = (σ_a / √2) × t^(3/2)
- Example: σ_a = 0.1 m/s² → position error ~3m after 100s, ~95m after 1000s
- Visible as: Quadratic-ish position drift

**Accel Bias Effect**:
- Quadratic position drift: Δp(t) = ½ b_a t²
- Example: b_a = 0.1 m/s² → 50m error after 10s, 5000m after 100s
- Visible as: Severe, systematic divergence

#### Example Configuration

```json
// Consumer-grade IMU (typical smartphone)
{
  "gyro_noise_std_rad_s": 0.01,    // ~0.57°/s noise
  "gyro_bias_rad_s": 0.001,        // ~0.057°/s drift
  "accel_noise_std_m_s2": 0.1,     // 0.1 m/s² noise
  "accel_bias_m_s2": [0.05, 0.05, 0.05]  // 50 mg bias
}
```

**Expected 60s trajectory behavior**:
- Heading error: ~5-10° (dominated by noise, not bias)
- Velocity error: ~0.8 m/s RMS
- Position error: ~30-50 m (unbounded, growing)

---

### RF Measurement Models (Ch4)

#### TOA Ranging (Ch4, Eq. 4.1)

**Book Equation**:
```
d̃ = c × Δt + ε_timing
```

Where:
- d̃: measured range
- c: speed of light (~3×10⁸ m/s)
- Δt: time of flight
- ε_timing: timing error

**Simulation Parameters**:

| Book Term | Simulation Parameter | Typical Values | Units |
|-----------|---------------------|----------------|-------|
| ε_timing | `timing_noise_std` | 1-50 | nanoseconds |
| — | `range_noise_std` | 0.01-0.5 | meters |

**Conversion**:
- 1 ns timing error = 0.3 m range error (c = 3×10⁸ m/s)
- 10 ns = 3 m
- 50 ns = 15 m

**Behavior Prediction**:
- Range noise propagates through geometry via DOP
- Position error ≈ range_noise × GDOP
- Example: 0.1m range noise, GDOP=2 → ~0.2m position error

#### NLOS Bias (Ch4, Section 4.5)

**Physical Model**:
- Signal travels longer path (reflects/diffracts)
- Measured range > true range (always positive bias)
- Bias magnitude: 0-several meters depending on environment

**Simulation Parameters**:
- `nlos_anchors`: list of affected beacon/anchor indices
- `nlos_bias`: positive bias added to ranges (meters)

**Behavior Prediction**:
- Unmitigated NLOS: systematic position bias toward affected anchors
- With chi-square gating: measurements exceeding threshold rejected
- Rejection rate increases with bias magnitude

---

### RSS Path Loss (Ch5)

#### Log-Distance Path Loss Model (Ch5)

**Book Equation**:
```
P(d) = P₀ - 10n log₁₀(d/d₀) + X_σ
```

Where:
- P(d): received power at distance d (dBm)
- P₀: reference power at distance d₀
- n: path loss exponent (2-4 for indoor)
- X_σ: shadow fading (log-normal, σ = 4-8 dBm)

**Simulation Parameters**:

| Book Term | Simulation Parameter | Typical Values | Units |
|-----------|---------------------|----------------|-------|
| P₀ | `P0_dBm` | -30 to -40 | dBm |
| d₀ | `d0_m` | 1.0 | meters |
| n | `path_loss_exponent` | 2.0-4.0 | — |
| σ | `shadow_fading_std_dBm` | 2-8 | dBm |

**Behavior Prediction**:
- Higher n: faster signal decay with distance
- Higher σ: more variable RSS, harder fingerprint matching
- Example: n=2.5, σ=4 dBm → k-NN matching ~3-5m accuracy at 5m grid spacing

---

## Step-by-Step Experiment Guides

### Experiment 1: IMU Drift Characterization

**Objective**: Understand how IMU noise and bias cause unbounded position drift.

**Duration**: 30 minutes

**Prerequisites**:
- Chapter 6 readings (Sections 6.1-6.2)
- Understanding of strapdown integration

#### Step 1: Generate Datasets (5 min)

```bash
# Perfect IMU (no noise, no bias) - control
python scripts/generate_ch6_strapdown_dataset.py \
    --accel-noise 0.0 --gyro-noise 0.0 \
    --accel-bias 0.0 --gyro-bias 0.0 \
    --duration 120 \
    --output data/sim/strapdown_perfect

# Consumer-grade IMU (realistic)
python scripts/generate_ch6_strapdown_dataset.py \
    --imu-grade consumer \
    --duration 120 \
    --output data/sim/strapdown_consumer

# High noise IMU
python scripts/generate_ch6_strapdown_dataset.py \
    --accel-noise 0.5 --gyro-noise 0.05 \
    --duration 120 \
    --output data/sim/strapdown_high_noise

# With bias (to isolate bias effect)
python scripts/generate_ch6_strapdown_dataset.py \
    --accel-noise 0.0 --gyro-noise 0.0 \
    --accel-bias 0.1 --gyro-bias 0.01 \
    --duration 120 \
    --output data/sim/strapdown_biased
```

#### Step 2: Run Strapdown Integration (5 min)

```bash
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/strapdown_perfect
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/strapdown_consumer
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/strapdown_high_noise
python -m ch6_dead_reckoning.example_imu_strapdown --data data/sim/strapdown_biased
```

#### Step 3: Analyze Results (10 min)

**Metrics to Extract**:
1. Position RMSE over time (plot)
2. Final position error
3. Drift rate (m/s) from linear fit to error
4. Error growth pattern (linear? quadratic?)

**Analysis Questions**:
- Q1: Does perfect IMU have zero drift? Why or why not?
- Q2: How does position error grow with consumer IMU? Is it linear, quadratic, or something else?
- Q3: Which is worse: noise or bias? Compare consumer vs biased datasets.
- Q4: Can you predict the bias from the observed drift rate?

**Expected Observations**:
- Perfect: Minimal error (only integration/discretization errors)
- Consumer: Quadratic-ish growth, ~50m error after 120s
- High noise: Severe quadratic growth, ~200m error after 120s
- Biased: Extreme quadratic growth (from accel bias), linear heading drift (from gyro bias)

#### Step 4: Theoretical Verification (10 min)

**Verify noise predictions**:

For consumer IMU (σ_a = 0.1 m/s²):
```
Predicted position error at t=120s:
σ_p = (σ_a / √2) × t^(3/2)
    = (0.1 / √2) × 120^1.5
    ≈ 93 m

Observed: ~50m (close, depends on trajectory specifics)
```

**Verify bias predictions**:

For biased IMU (b_a = 0.1 m/s²):
```
Predicted position error at t=120s:
Δp = ½ b_a t²
   = 0.5 × 0.1 × 120²
   = 720 m

Observed: Should be close to prediction (systematic)
```

**Key Takeaways**:
1. IMU drift is **unbounded** without external corrections
2. Noise causes √t (heading) or t^(3/2) (position) growth
3. Bias causes linear (heading) or quadratic (position) growth
4. Bias is worse than noise for same magnitude
5. This motivates constraints (ZUPT), fusion, and calibration

---

### Experiment 2: Filter Tuning Sensitivity

**Objective**: Understand how measurement covariance R affects EKF performance.

**Duration**: 45 minutes

**Prerequisites**:
- Chapter 3 (EKF theory)
- Chapter 8 (Innovation monitoring, Eqs. 8.5-8.6)

#### Step 1: Generate Baseline Dataset (5 min)

```bash
# Use standard fusion dataset
python scripts/generate_fusion_2d_imu_uwb_dataset.py
# Uses default: range_noise_std = 0.05 m
```

#### Step 2: Run Fusion with Different R Scaling (10 min)

Modify the fusion script to scale R by factors: 0.1, 0.5, 1.0, 2.0, 5.0

```bash
# Under-estimated R (overconfident)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/fusion_2d_imu_uwb \
    --r-scale 0.1 \
    --output results_r_scale_0.1.json

# Correct R
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/fusion_2d_imu_uwb \
    --r-scale 1.0 \
    --output results_r_scale_1.0.json

# Over-estimated R (conservative)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/fusion_2d_imu_uwb \
    --r-scale 5.0 \
    --output results_r_scale_5.0.json
```

#### Step 3: Analyze Innovation Statistics (15 min)

**Metrics to compute**:
1. Position RMSE
2. NIS (Normalized Innovation Squared) statistics
3. NIS consistency (% within χ² bounds)
4. Covariance trace over time

**Analysis Framework**:

| R Scaling | Expected NIS | Position RMSE | Filter Behavior |
|-----------|--------------|---------------|-----------------|
| 0.1 (too small) | << χ² threshold | May diverge | Overconfident, ignores prediction uncertainty |
| 1.0 (correct) | ~χ² threshold (95%) | Optimal | Balanced |
| 5.0 (too large) | >> χ² threshold | Suboptimal | Conservative, underutilizes measurements |

**Plot NIS over time**:
- Correct R: Most NIS values fall within χ²(α=0.05) bounds
- Under-estimated R: Many NIS values exceed bounds (filter inconsistent)
- Over-estimated R: Almost all NIS values below bounds (wasting information)

#### Step 4: Interpret Results (15 min)

**Key Equations**:

Innovation (Ch8, Eq. 8.5):
```
y = z - h(x̂)
```

Innovation Covariance (Ch8, Eq. 8.6):
```
S = H P H' + R
```

NIS (Ch8, Eq. 8.8):
```
d² = y' S⁻¹ y
```

**When R is too small**:
- S is too small → S⁻¹ is large
- Innovation y appears statistically significant even when normal
- Filter over-trusts measurements, ignores prediction
- Can lead to divergence if noise is underestimated

**When R is too large**:
- S is too large → S⁻¹ is small
- Even significant innovations appear statistically normal
- Filter under-trusts measurements, relies on prediction
- Suboptimal but stable

**Key Takeaway**: Proper covariance tuning is critical for optimal, consistent filtering. NIS monitoring (Ch8) detects mistuning.

---

### Experiment 3: NLOS Detection and Rejection

**Objective**: Observe chi-square gating in action rejecting outlier measurements.

**Duration**: 40 minutes

#### Step 1: Generate NLOS Dataset (5 min)

```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --nlos-anchors 1 2 \
    --nlos-bias 1.0 \
    --output data/sim/fusion_nlos_1m
```

#### Step 2: Run with Different Gating Strategies (10 min)

```bash
# No gating (baseline - corrupted by NLOS)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/fusion_nlos_1m \
    --no-gating \
    --output results_no_gating.json

# Chi-square gating (α=0.05, 95% confidence)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/fusion_nlos_1m \
    --alpha 0.05 \
    --output results_gating_0.05.json

# Stricter gating (α=0.01, 99% confidence)
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/fusion_nlos_1m \
    --alpha 0.01 \
    --output results_gating_0.01.json
```

#### Step 3: Analyze Rejection Statistics (15 min)

**Metrics**:
1. Position RMSE (overall accuracy)
2. Per-anchor rejection rates
3. NIS values for accepted vs rejected measurements
4. Trajectory visualization showing measurement acceptance

**Expected Results** (1m NLOS bias):

| Anchor | NLOS? | Rejection Rate (α=0.05) | NIS typical |
|--------|-------|------------------------|-------------|
| 0 | No | ~5% (false alarms) | < 3.84 (χ²₀.₀₅,₁) |
| 1 | Yes | ~60-80% | >> 3.84 |
| 2 | Yes | ~60-80% | >> 3.84 |
| 3 | No | ~5% | < 3.84 |

**Analysis Questions**:
- Q1: Why aren't 100% of NLOS measurements rejected?
- Q2: Why are some clean measurements rejected (~5%)?
- Q3: How does stricter gating (α=0.01) change rejection rates?
- Q4: Is there a trade-off between robustness and availability?

#### Step 4: Visualize Gating Decision Boundary (10 min)

Plot innovation vs. NIS:
- X-axis: Time
- Y-axis: NIS value (d²)
- Horizontal line: χ² threshold
- Color: Accepted (green) vs Rejected (red)
- Per-anchor subplots

**Expected Visualization**:
- Clean anchors (0,3): Green points mostly below threshold
- NLOS anchors (1,2): Many red points above threshold
- Clear separation between clean and NLOS distributions

**Key Takeaway**: Chi-square gating (Ch8, Eqs. 8.8-8.9) provides principled outlier rejection based on statistical consistency, significantly improving robustness.

---

## Parameter Sensitivity Reference

### IMU Parameters

| Parameter | Low | Medium | High | Effect |
|-----------|-----|--------|------|--------|
| `accel_noise_std` | 0.01 | 0.1 | 0.5 | Velocity/position drift rate |
| `gyro_noise_std` | 0.001 | 0.01 | 0.05 | Heading drift rate |
| `accel_bias` | 0.0 | 0.05 | 0.2 | Systematic position drift |
| `gyro_bias` | 0.0 | 0.001 | 0.01 | Systematic heading drift |

**Rule of Thumb**:
- Tactical-grade: noise × 0.1, bias × 0.1
- Consumer-grade: noise × 1.0, bias × 1.0  
- MEMS-grade: noise × 5.0, bias × 5.0

### RF Parameters

| Parameter | Low | Medium | High | Effect |
|-----------|-----|--------|------|--------|
| `range_noise_std` | 0.01 | 0.05 | 0.5 | Position uncertainty |
| `nlos_bias` | 0.2 | 1.0 | 2.0 | Outlier severity |
| `dropout_rate` | 0.0 | 0.05 | 0.3 | Measurement availability |

**Trade-offs**:
- Lower noise → better accuracy, may be unrealistic
- Higher dropout → tests robustness, longer IMU-only intervals

### Fingerprinting Parameters

| Parameter | Sparse | Medium | Dense | Effect |
|-----------|--------|--------|-------|--------|
| `n_aps` | 4 | 8 | 16 | Uniqueness, disambiguation |
| `grid_spacing` | 10m | 5m | 2m | Resolution limit |
| `shadow_fading_std` | 2 | 4 | 8 | RSS variability |

---

## Common Student Questions

**Q: How do I choose realistic parameters for my experiment?**
A: Start with "medium" values from the sensitivity tables. For specific sensor types, consult datasheets or Allan variance plots (Ch6, Eqs. 6.56-6.58).

**Q: My filter diverges. What's wrong?**
A: Common causes:
1. Under-estimated R (measurement covariance) → check NIS plot
2. Over-estimated Q (process covariance) → increase Q
3. Poor initialization → check initial P₀
4. Severe outliers without gating → enable chi-square gating

**Q: How long should my dataset be to see meaningful drift?**
A: For IMU drift: >60s. For filter convergence: >30s. For statistical analysis: >100 samples after convergence.

**Q: Can I use these parameters for real hardware?**
A: These are typical ranges. Real sensors vary significantly. Always characterize your specific hardware (Allan variance, static tests).

**Q: Why doesn't my theoretical prediction match simulation exactly?**
A: Several factors:
- Discretization effects (time step dt)
- Trajectory-specific geometry (DOP)
- Nonlinear effects in large-error regimes
- Interactions between error sources

Expect agreement within 2× for order-of-magnitude predictions.

---

## References

### Book Sections
- Chapter 3: State Estimation (EKF theory)
- Chapter 4: RF Positioning (TOA/TDOA/RSS models)
- Chapter 5: Fingerprinting (RSS path loss)
- Chapter 6: Dead Reckoning (IMU error models, Eqs. 6.2-6.10)
- Chapter 8: Sensor Fusion (Innovation monitoring, Eqs. 8.5-8.9)

### Related Documentation
- `data/sim/README.md` - Dataset catalog
- `scripts/README.md` - Generation script usage
- `references/design_doc.md` - Section 5.3 (Dataset standards)

### Tools
- `tools/plot_dataset_overview.py` - Quick visualization
- `tools/compare_dataset_variants.py` - Side-by-side comparison
- `tools/validate_dataset.py` - Format validation

