# Dataset Generation Scripts

This folder contains scripts for generating simulation datasets used throughout the IPIN book examples. These scripts enable students to create custom datasets with different parameters to explore algorithm behavior systematically.

## Purpose

Generation scripts serve two main purposes:

1. **Reproducibility**: Regenerate the exact datasets used in book examples
2. **Experimentation**: Create custom variants to study parameter sensitivity

## Quick Start

```bash
# Generate default datasets (all chapters)
python scripts/generate_fusion_2d_imu_uwb_dataset.py
python scripts/generate_wifi_fingerprint_dataset.py
# ... (see script inventory below)

# Generate with custom parameters
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.5 \
    --output data/sim/my_experiment

# Use preset configurations
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --preset high_dropout \
    --output data/sim/high_dropout_test
```

---

## Generation Scripts Inventory

### Chapter 8: Sensor Fusion

**`generate_fusion_2d_imu_uwb_dataset.py`** - IMU + UWB Fusion Dataset

Generates 2D walking trajectory with high-rate IMU and low-rate UWB ranging.

**Key Parameters**:
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--duration` | 60.0 | 10-300 | Trajectory duration (seconds) |
| `--speed` | 1.0 | 0.5-2.0 | Walking speed (m/s) |
| `--accel-noise` | 0.1 | 0.01-1.0 | Accelerometer noise σ (m/s²) |
| `--gyro-noise` | 0.01 | 0.001-0.1 | Gyroscope noise σ (rad/s) |
| `--range-noise` | 0.05 | 0.01-0.5 | UWB range noise σ (meters) |
| `--nlos-anchors` | [] | [0-3] | List of NLOS anchor indices |
| `--nlos-bias` | 0.5 | 0-2.0 | NLOS positive bias (meters) |
| `--dropout-rate` | 0.05 | 0-0.5 | Measurement dropout probability |
| `--time-offset` | 0.0 | -0.5 to 0.5 | Sensor time offset (seconds) |

**Presets**: `baseline`, `nlos_severe`, `high_dropout`, `degraded_imu`, `time_offset_50ms`

**Examples**: See "Chapter 8 Experimentation Scenarios" section below.

---

### Chapter 6: Dead Reckoning

**`generate_ch6_strapdown_dataset.py`** - IMU Strapdown Integration

Generates pure IMU data to demonstrate drift characteristics.

**Key Parameters**:
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--imu-grade` | consumer | tactical/consumer/mems | IMU quality preset |
| `--accel-noise` | 0.1 | 0.01-1.0 | Accel noise σ (m/s²) |
| `--gyro-noise` | 0.01 | 0.001-0.1 | Gyro noise σ (rad/s) |
| `--accel-bias` | 0.0 | 0-0.5 | Constant accel bias (m/s²) |
| `--gyro-bias` | 0.0 | 0-0.1 | Constant gyro bias (rad/s) |

**Presets**: 
- `tactical`: Low noise (0.01 m/s², 0.001 rad/s)
- `consumer`: Medium noise (0.1 m/s², 0.01 rad/s) - default
- `mems`: High noise (0.5 m/s², 0.05 rad/s)

**`generate_ch6_wheel_odom_dataset.py`** - Vehicle Wheel Odometry

**`generate_ch6_zupt_walk_dataset.py`** - Foot-Mounted IMU with ZUPT

**`generate_ch6_pdr_corridor_dataset.py`** - Pedestrian Dead Reckoning

**`generate_ch6_env_sensors_dataset.py`** - Magnetometer + Barometer

---

### Chapter 5: Fingerprinting

**`generate_wifi_fingerprint_dataset.py`** - Wi-Fi RSS Fingerprint Database

Generates multi-floor fingerprint database with log-distance path-loss model.

**Key Parameters**:
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--area-size` | 50×50 | 20-100 | Floor dimensions (meters) |
| `--grid-spacing` | 5.0 | 1-10 | Reference point spacing (meters) |
| `--n-floors` | 3 | 1-5 | Number of floors |
| `--n-aps` | 8 | 4-20 | Number of access points |
| `--path-loss-exp` | 2.5 | 2.0-4.0 | Path loss exponent |
| `--shadow-fading` | 4.0 | 2-8 | Shadow fading σ (dBm) |
| `--floor-atten` | 15.0 | 10-20 | Floor attenuation (dB) |

**Examples**: See "Chapter 5 Experimentation Scenarios" section below.

---

### Chapter 4: RF Point Positioning

**`generate_rf_2d_floor_dataset.py`** - RF Beacon/Anchor Dataset

Generates TOA/TDOA/AOA/RSS measurements from beacons.

**Key Parameters**:
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--beacon-layout` | distributed | clustered/distributed/linear | Beacon geometry |
| `--n-beacons` | 4 | 4-12 | Number of beacons |
| `--timing-noise` | 10 | 1-50 | Timing noise (nanoseconds) |
| `--rss-sigma` | 4.0 | 2-10 | RSS noise σ (dBm) |
| `--nlos-beacons` | [] | varies | NLOS beacon indices |

---

### Chapter 7: SLAM

**`generate_slam_lidar2d_dataset.py`** - 2D LiDAR SLAM

**`generate_slam_visual_dataset.py`** - Visual SLAM / Bundle Adjustment

---

## Common CLI Patterns

All generation scripts follow consistent patterns:

### Basic Usage

```bash
# Generate with defaults (outputs to standard location)
python scripts/generate_[dataset]_dataset.py

# Specify output directory
python scripts/generate_[dataset]_dataset.py --output data/sim/my_custom_name

# Use preset configuration
python scripts/generate_[dataset]_dataset.py --preset [preset_name]

# Set random seed for reproducibility
python scripts/generate_[dataset]_dataset.py --seed 12345
```

### Getting Help

```bash
# Show all parameters and examples
python scripts/generate_[dataset]_dataset.py --help

# List available presets
python scripts/generate_[dataset]_dataset.py --help | grep "presets:"
```

---

## Experimentation Scenarios

### Chapter 8: Sensor Fusion

#### Scenario 1: Effect of IMU Noise on Drift

**Learning Objective**: Understand how IMU noise propagates to position error over time.

**Setup**:
```bash
# Low noise (tactical-grade IMU)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.01 --gyro-noise 0.001 \
    --output data/sim/fusion_low_noise

# Medium noise (consumer-grade IMU) - baseline
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.1 --gyro-noise 0.01 \
    --output data/sim/fusion_med_noise

# High noise (degraded IMU)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.5 --gyro-noise 0.05 \
    --output data/sim/fusion_high_noise
```

**Run Experiments**:
```bash
# Test on all three datasets
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/fusion_low_noise
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/fusion_med_noise
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/fusion_high_noise

# Compare results
python tools/compare_dataset_variants.py \
    data/sim/fusion_low_noise \
    data/sim/fusion_med_noise \
    data/sim/fusion_high_noise
```

**Expected Observations**:
- Low noise: RMSE ~5-8m, slow drift accumulation
- Medium noise: RMSE ~12-15m, moderate drift
- High noise: RMSE ~25-35m, rapid drift, fusion relies heavily on UWB

**Key Insight**: Higher IMU noise → faster drift → fusion becomes more dependent on absolute measurements (UWB).

---

#### Scenario 2: NLOS Severity Study

**Learning Objective**: Observe how chi-square gating rejects NLOS-corrupted measurements.

**Setup**:
```bash
# Generate datasets with varying NLOS bias
for bias in 0.2 0.5 1.0 2.0; do
    python scripts/generate_fusion_2d_imu_uwb_dataset.py \
        --nlos-anchors 1 2 \
        --nlos-bias $bias \
        --output data/sim/fusion_nlos_bias_${bias}
done
```

**Run Experiments**:
```bash
# Test with and without gating
for bias in 0.2 0.5 1.0 2.0; do
    echo "Testing bias: $bias"
    
    # Without gating
    python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
        --data data/sim/fusion_nlos_bias_${bias} \
        --no-gating \
        --output results_no_gate_${bias}.json
    
    # With gating (default)
    python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
        --data data/sim/fusion_nlos_bias_${bias} \
        --output results_with_gate_${bias}.json
done
```

**Expected Observations**:
- Bias 0.2m: Minimal impact, most measurements accepted
- Bias 0.5m: ~20-30% rejection rate, moderate improvement
- Bias 1.0m: ~40-60% rejection rate, significant improvement
- Bias 2.0m: ~70-80% rejection rate, dramatic improvement

**Key Insight**: Chi-square gating (Ch8, Eqs. 8.8-8.9) effectively detects and rejects NLOS outliers, preventing filter corruption.

---

#### Scenario 3: Temporal Calibration Impact

**Learning Objective**: Understand importance of temporal synchronization between sensors.

**Setup**:
```bash
# Generate dataset with time offset
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --time-offset -0.05 \
    --clock-drift 0.0001 \
    --output data/sim/fusion_time_offset_50ms
```

**Run Experiments**:
```bash
# Without correction (degraded performance)
python -m ch8_sensor_fusion.temporal_calibration_demo \
    --data data/sim/fusion_time_offset_50ms \
    --no-correction

# With correction (recovered performance)
python -m ch8_sensor_fusion.temporal_calibration_demo \
    --data data/sim/fusion_time_offset_50ms
```

**Expected Observations**:
- Without correction: RMSE increases by 50-100%, systematic residuals visible
- With correction: RMSE returns to baseline performance

**Key Insight**: Even small time offsets (50ms) significantly degrade fusion performance. Proper temporal calibration is critical.

---

### Chapter 6: Dead Reckoning

#### Scenario 4: IMU Grade Comparison

**Learning Objective**: Quantify drift rates for different IMU qualities.

**Setup**:
```bash
# Generate datasets for different IMU grades
python scripts/generate_ch6_strapdown_dataset.py --imu-grade tactical
python scripts/generate_ch6_strapdown_dataset.py --imu-grade consumer
python scripts/generate_ch6_strapdown_dataset.py --imu-grade mems
```

**Run Experiments**:
```bash
python -m ch6_dead_reckoning.example_imu_strapdown --data ch6_strapdown_tactical
python -m ch6_dead_reckoning.example_imu_strapdown --data ch6_strapdown_consumer
python -m ch6_dead_reckoning.example_imu_strapdown --data ch6_strapdown_mems
```

**Expected Observations**:
- Tactical: ~1% of distance traveled (typical)
- Consumer: ~5-10% of distance
- MEMS: ~20-50% of distance (unusable without corrections)

**Key Insight**: IMU-only positioning is unbounded without corrections (Ch6 fundamental finding).

---

#### Scenario 5: ZUPT Effectiveness

**Learning Objective**: Demonstrate dramatic drift reduction from zero-velocity updates.

**Setup**:
```bash
# Generate foot-mounted IMU data with stance phases
python scripts/generate_ch6_zupt_walk_dataset.py --duration 120
```

**Run Experiments**:
```bash
# Pure IMU integration (no ZUPT)
python -m ch6_dead_reckoning.example_imu_strapdown --data ch6_foot_zupt_walk

# With ZUPT corrections
python -m ch6_dead_reckoning.example_zupt --data ch6_foot_zupt_walk
```

**Expected Observations**:
- Without ZUPT: Position RMSE > 1000% of distance (unusable)
- With ZUPT: Position RMSE ~2-5% of distance (excellent)

**Key Insight**: ZUPT (Ch6, Eqs. 6.44-6.45) transforms IMU from unbounded to bounded drift.

---

### Chapter 5: Fingerprinting

#### Scenario 6: AP Density Impact on k-NN Accuracy

**Learning Objective**: Understand relationship between AP coverage and positioning accuracy.

**Setup**:
```bash
# Sparse AP coverage
python scripts/generate_wifi_fingerprint_dataset.py \
    --n-aps 4 --grid-spacing 5.0 \
    --output data/sim/wifi_sparse_4ap

# Medium AP coverage (baseline)
python scripts/generate_wifi_fingerprint_dataset.py \
    --n-aps 8 --grid-spacing 5.0 \
    --output data/sim/wifi_medium_8ap

# Dense AP coverage
python scripts/generate_wifi_fingerprint_dataset.py \
    --n-aps 16 --grid-spacing 5.0 \
    --output data/sim/wifi_dense_16ap
```

**Run Experiments**:
```bash
python -m ch5_fingerprinting.example_deterministic --data data/sim/wifi_sparse_4ap
python -m ch5_fingerprinting.example_deterministic --data data/sim/wifi_medium_8ap
python -m ch5_fingerprinting.example_deterministic --data data/sim/wifi_dense_16ap
```

**Expected Observations**:
- 4 APs: RMSE ~8-12m (poor uniqueness, ambiguous matches)
- 8 APs: RMSE ~4-6m (good performance)
- 16 APs: RMSE ~2-4m (excellent, diminishing returns)

**Key Insight**: More APs improve uniqueness up to a point, but with diminishing returns.

---

#### Scenario 7: Grid Resolution vs. Accuracy

**Learning Objective**: Study trade-off between survey effort and accuracy.

**Setup**:
```bash
# Coarse grid (less survey effort)
python scripts/generate_wifi_fingerprint_dataset.py \
    --grid-spacing 10.0 --output data/sim/wifi_coarse_grid

# Fine grid (more survey effort)  
python scripts/generate_wifi_fingerprint_dataset.py \
    --grid-spacing 2.0 --output data/sim/wifi_fine_grid
```

**Expected Observations**:
- Coarse grid (10m): RMSE ~8-10m, limited resolution
- Fine grid (2m): RMSE ~2-3m, excellent resolution but 25× more survey points

**Key Insight**: Grid spacing creates fundamental resolution limit for NN/k-NN methods (Ch5, Eqs. 5.1-5.2).

---

## Parameter Validation

All scripts validate parameters before generation:

```bash
# Example: Invalid duration
python scripts/generate_fusion_2d_imu_uwb_dataset.py --duration -10
# Error: Duration must be positive

# Example: Invalid noise range
python scripts/generate_fusion_2d_imu_uwb_dataset.py --accel-noise 10.0
# Warning: Unusually high noise (>1.0), are you sure? [y/N]
```

---

## Troubleshooting

### Common Issues

**Q: Script fails with "ModuleNotFoundError"?**
A: Make sure you've installed the package: `pip install -e .`

**Q: Generation is very slow?**
A: Large durations or high sample rates take time. Start with shorter durations (30s) for testing.

**Q: Output file already exists?**
A: Scripts won't overwrite by default. Use `--force` or delete the existing directory first.

**Q: How do I know if my parameters are realistic?**
A: Check the "Range" column in parameter tables. Values outside these ranges may not be physically meaningful.

**Q: Can I generate real-time data?**
A: No, these scripts generate offline datasets. For real-time simulation, see `core/sim/` utilities.

### Getting Help

```bash
# Script-specific help
python scripts/generate_[dataset]_dataset.py --help

# Report issues
# Check: references/design_doc.md Section 5.3
# Contact: [your support contact]
```

---

## Adding New Generation Scripts

If you're creating a new generation script:

1. ✓ Copy `templates/generation_script_CLI_template.py`
2. ✓ Implement data generation functions
3. ✓ Add full CLI with argparse
4. ✓ Include at least 3 preset configurations
5. ✓ Add parameter validation
6. ✓ Document in this README (add to inventory table)
7. ✓ Provide 2+ experimentation scenarios
8. ✓ Test all CLI options
9. ✓ Add output dataset README using template

See `templates/` and `references/design_doc.md` Section 5.3 for requirements.

---

## References

- **Design Document**: `references/design_doc.md` Section 5.3
- **Dataset Catalog**: `data/sim/README.md`
- **Learning Guide**: `docs/data_simulation_guide.md`
- **Templates**: `templates/` folder
- **Tools**: `tools/` (visualization, validation, comparison)

