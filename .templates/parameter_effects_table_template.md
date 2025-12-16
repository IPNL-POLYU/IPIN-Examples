# Parameter Effects Table Template

Use this template in dataset READMEs to document how parameters affect algorithm behavior.

## Template

```markdown
## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on [Algorithm/System] | Learning Objective |
|-----------|---------|------------------|------------------------------|-------------------|
| `[param1]` | [value] | [min]-[max] | [Clear description of observable effect] | [What students learn from varying this] |
| `[param2]` | [value] | [min]-[max] | [Clear description of observable effect] | [What students learn from varying this] |
| `[param3]` | [value] | [min]-[max] | [Clear description of observable effect] | [What students learn from varying this] |
```

---

## Examples

### IMU Parameters (Ch6 Dead Reckoning)

```markdown
## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on Strapdown Integration | Learning Objective |
|-----------|---------|------------------|--------------------------------|-------------------|
| `accel_noise_std` | 0.1 | 0.01-0.5 | Higher → faster velocity drift → t^(3/2) position error growth | Understand noise propagation through double integration |
| `gyro_noise_std` | 0.01 | 0.001-0.05 | Higher → faster heading drift → √t growth rate | Observe random walk in orientation |
| `accel_bias` | 0.0 | 0-0.2 | Non-zero → systematic drift → quadratic position error (½bt²) | Distinguish bias from noise effects |
| `gyro_bias` | 0.0 | 0-0.01 | Non-zero → linear heading drift (bt) → unbounded error | Motivates need for calibration |
```

**Key Insight**: IMU drift is unbounded without external corrections (Ch6 core concept).

---

### Fusion Parameters (Ch8 Sensor Fusion)

```markdown
## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on Fusion Performance | Learning Objective |
|-----------|---------|------------------|------------------------------|-------------------|
| `range_noise_std` | 0.05 | 0.01-0.5 | Higher → noisier UWB fixes → EKF trusts IMU prediction more | Understand measurement uncertainty balance |
| `nlos_anchors` | [] | [0], [1,2] | Biased anchors → systematic position error if not rejected | Learn importance of outlier detection |
| `nlos_bias` | 0.5 | 0.2-2.0 | Larger bias → higher chi-square gating rejection rate | Observe gating effectiveness (Ch8, Eqs. 8.8-8.9) |
| `dropout_rate` | 0.05 | 0.1-0.3 | More dropouts → longer IMU-only intervals → larger prediction uncertainty | Understand multi-rate fusion challenges |
| `time_offset_sec` | 0.0 | -0.1 to 0.1 | Non-zero → systematic innovation residuals → degraded accuracy | Learn temporal calibration importance |
```

**Key Insight**: Proper tuning, gating, and synchronization are critical for practical fusion (Ch8 theme).

---

### Fingerprinting Parameters (Ch5)

```markdown
## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on k-NN Positioning | Learning Objective |
|-----------|---------|------------------|---------------------------|-------------------|
| `n_aps` | 8 | 4-16 | More APs → better uniqueness → improved accuracy (diminishing returns) | Understand feature space dimensionality |
| `grid_spacing` | 5.0 | 2-10 | Finer grid → better resolution but more survey effort | Learn trade-off between accuracy and cost |
| `shadow_fading_std` | 4.0 | 2-8 | Higher → more variable RSS → harder matching → lower accuracy | Observe environment impact on RSS |
| `path_loss_exponent` | 2.5 | 2.0-4.0 | Higher → faster signal decay → less discriminative far from AP | Understand indoor propagation models |
```

**Key Insight**: Fingerprinting accuracy depends on both database quality and RF environment (Ch5).

---

### RF Positioning Parameters (Ch4)

```markdown
## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on TOA/TDOA Positioning | Learning Objective |
|-----------|---------|------------------|---------------------------------|-------------------|
| `timing_noise_std` | 10 | 1-50 | Higher → larger range errors (1ns = 0.3m) → amplified by DOP | Understand timing-to-range-to-position error chain |
| `n_beacons` | 4 | 4-12 | More beacons → lower DOP → better conditioning | Learn geometry impact on accuracy |
| `beacon_layout` | distributed | clustered/linear | Poor geometry → high DOP → errors amplified 10-100× | Observe DOP sensitivity to placement |
| `nlos_bias` | 0.0 | 0-5.0 | Positive bias on ranges → systematic position offset | Understand NLOS multipath effects |
```

**Key Insight**: Beacon geometry (DOP) is as important as measurement noise (Ch4, Section 4.1).

---

## Guidelines for Writing Parameter Tables

### "Parameter" Column
- Use exact parameter name from `config.json` (in code format: `parameter_name`)
- Must match generation script CLI arguments

### "Default" Column
- Show the value used in the baseline/standard dataset
- Include units implicitly or explicitly

### "Experiment Range" Column
- Show realistic range for experimentation
- Format: `min-max` for continuous, `[option1/option2]` for discrete
- Range should span meaningful algorithmic behaviors

### "Effect on [System]" Column
- Be specific and observable
- Include directionality: "Higher → effect"
- Quantify when possible: "2× increase → 4× error"
- Reference equations when relevant: "(Ch6, Eq. 6.5)"

### "Learning Objective" Column
- Frame as what students learn/observe
- Connect to theoretical concepts
- Use action verbs: "Understand", "Observe", "Learn", "Distinguish"
- Keep concise (1-2 lines)

### Additional Tips

1. **Group related parameters** (e.g., all IMU params together)
2. **Order by importance** (most impactful first)
3. **Add Key Insight** after table summarizing main takeaway
4. **Link to book sections** for theoretical grounding
5. **Reference specific equations** where parameters map to theory

---

## Validation Checklist

- [ ] All parameters in table match `config.json` exactly
- [ ] Default values match baseline dataset configuration
- [ ] Experiment ranges are realistic and meaningful
- [ ] Effects are observable and testable
- [ ] Learning objectives connect to book concepts
- [ ] Key insight provided after table
- [ ] At least 3 parameters documented
- [ ] Table properly formatted (Markdown)


