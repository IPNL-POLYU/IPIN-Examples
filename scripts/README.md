# Dataset Generation Scripts

This folder contains scripts for generating simulation datasets used throughout the IPIN book examples. These scripts enable students to create custom datasets with different parameters to explore algorithm behavior systematically.

## Purpose

Generation scripts serve two main purposes:

1. **Reproducibility**: Regenerate the exact datasets used in book examples
2. **Experimentation**: Create custom variants to study parameter sensitivity

## Quick Start

```bash
# Generate default datasets (all chapters)
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py
python scripts/generate_ch5_wifi_fingerprint_dataset.py
# ... (see script inventory below)

# Generate with custom parameters
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.5 \
    --output data/sim/my_experiment

# Use preset configurations
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py \
    --preset high_dropout \
    --output data/sim/high_dropout_test
```

---

## Generation Scripts Inventory

Eleven generators, one per dataset family. **`--help` is the authority on
parameters** -- the tables below name the ones worth reaching for first, not
every flag. Each generator also takes `--preset`, `--output` and `--seed`, and
`--output` wins over a preset's own directory when both are given.

### Chapter 2: Coordinate Systems

**`generate_ch2_coordinate_transforms_dataset.py`** - LLH / ENU / NED transforms

Samples points across a building footprint and stores each in several frames.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--latitude` | 37.7749 | Center latitude (degrees) |
| `--building-size` | 50.0 | Building footprint size (meters) |

**Presets**: `san_francisco`, `tokyo`, `london`

---

### Chapter 3: State Estimation

**`generate_ch3_estimator_comparison_dataset.py`** - Range/bearing estimator comparison

Range and bearing measurements to beacons along a trajectory, for comparing
least squares, EKF, UKF and particle filters.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--duration` | 30.0 | Duration (seconds) |
| `--dt` | 0.1 | Time step (seconds) |
| `--n-beacons` | 4 | Number of beacons |
| `--range-noise` | 0.5 | Range noise σ (meters) |
| `--bearing-noise` | 5.0 | Bearing noise σ (degrees) |

**Presets**: `linear`, `nonlinear`, `high_nonlinearity`, `outliers`

---

### Chapter 4: RF Point Positioning

**`generate_ch4_rf_2d_positioning_dataset.py`** - TOA / TDOA / AOA measurements

Anchor geometry plus per-point measurements, for point positioning and DOP.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-points` | 100 | Number of evaluation points |
| `--toa-noise` | 0.1 | TOA noise σ (meters) |
| `--tdoa-noise` | 0.1 | TDOA noise σ (meters) |
| `--aoa-noise` | 2.0 | AOA noise σ (degrees) |

**Presets**: `baseline`, `optimal`, `poor_geometry`, `nlos`

---

### Chapter 5: Fingerprinting

**`generate_ch5_wifi_fingerprint_dataset.py`** - Wi-Fi RSS fingerprint database

Multi-floor fingerprint database from a log-distance path-loss model.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--area-width` | 50.0 | Area width (meters) |
| `--area-height` | 50.0 | Area height (meters) |
| `--grid-spacing` | 5.0 | Reference point spacing (meters) |
| `--n-floors` | 3 | Number of floors |
| `--floor-height` | 3.0 | Floor height (meters) |
| `--n-aps` | 8 | Number of access points |

**Presets**: `baseline`, `dense`, `sparse`, `few_aps`, `multisamples`

---

### Chapter 6: Dead Reckoning

**`generate_ch6_strapdown_dataset.py`** - IMU strapdown integration

Circular trajectory with body-frame IMU samples.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--radius` | 10.0 | Circle radius (meters) |
| `--speed` | 1.0 | Constant speed (m/s) |
| `--duration` | 60.0 | Duration (seconds) |
| `--dt` | 0.01 | Time step (seconds, 100 Hz) |
| `--accel-noise` | 0.1 | Accelerometer noise σ (m/s²) |
| `--gyro-noise` | 0.01 | Gyroscope noise σ (rad/s) |

**Presets**: `tactical`, `consumer`, `mems`, `biased_consumer`

Biases are per axis: `--accel-bias-x`, `--accel-bias-y`, `--gyro-bias`.

---

**`generate_ch6_zupt_dataset.py`** - Foot-mounted IMU with ZUPT

Stepping trajectory with stance phases a zero-velocity detector can find.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-steps` | 20 | Number of steps |
| `--step-length` | 0.7 | Distance per step (meters) |
| `--step-duration` | 0.6 | Time per step (seconds) |
| `--stance-duration` | 0.2 | Stationary time per step (seconds) |
| `--accel-noise` | 0.1 | Accelerometer noise σ (m/s²) |
| `--gyro-noise` | 0.01 | Gyroscope noise σ (rad/s) |

**Presets**: `baseline`, `fast_walk`, `slow_walk`, `noisy_imu`

---

**`generate_ch6_pdr_dataset.py`** - Pedestrian dead reckoning

Corridor walk with step events, for step detection and heading.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-legs` | 4 | Number of corridor legs |
| `--leg-length` | 30.0 | Length of each leg (meters) |
| `--step-freq` | 2.0 | Step frequency (Hz) |
| `--height` | 1.75 | Pedestrian height (meters) |

**Presets**: `baseline`, `noisy`, `poor_gyro`, `poor_mag`

---

**`generate_ch6_wheel_odom_dataset.py`** - Vehicle wheel odometry

Square laps with encoder and gyro, optional wheel slip.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--side-length` | 20.0 | Square side length (meters) |
| `--speed` | 5.0 | Forward speed (m/s) |
| `--num-laps` | 2 | Number of laps |
| `--encoder-noise` | 0.05 | Encoder noise σ (m/s) |
| `--gyro-noise` | 0.001 | Gyro noise σ (rad/s) |
| `--wheel-bias` | 0.01 | Wheel speed bias (m/s) |
| `--gyro-bias` | 0.0005 | Gyro bias (rad/s) |

**Presets**: `baseline`, `noisy`, `slip`, `poor`

---

**`generate_ch6_env_sensors_dataset.py`** - Magnetometer and barometer

Building walk across floors, with optional magnetic disturbances.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--duration` | 180.0 | Duration (seconds) |
| `--floor-height` | 3.5 | Height of each floor (meters) |
| `--mag-noise` | 2.0 | Magnetometer noise (microTesla) |
| `--pressure-noise` | 10.0 | Pressure noise (Pa) |

**Presets**: `baseline`, `noisy`, `disturbances`, `poor`

---

### Chapter 7: SLAM

**`generate_ch7_slam_2d_dataset.py`** - 2D LiDAR SLAM

Odometry, scans and landmarks around a closed trajectory.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--size` | 20.0 | Trajectory size (meters) |
| `--n-poses-per-side` | 10 | Poses per segment |
| `--n-landmarks` | 50 | Number of landmarks |
| `--max-range` | 15.0 | Sensor max range (meters) |
| `--translation-noise` | 0.1 | Odometry translation noise σ (m) |
| `--rotation-noise` | 0.02 | Odometry rotation noise σ (rad) |

**Presets**: `baseline`, `low_drift`, `high_drift`, `figure8`

Visual SLAM and bundle adjustment are covered by
`ch7_slam/example_bundle_adjustment.py`, which builds its own scene inline --
there is no visual dataset generator.

---

### Chapter 8: Sensor Fusion

**`generate_ch8_fusion_2d_imu_uwb_dataset.py`** - IMU + UWB fusion dataset

Generates 2D walking trajectory with high-rate IMU and low-rate UWB ranging.

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--duration` | 60.0 | Trajectory duration (seconds) |
| `--speed` | 1.0 | Walking speed (m/s) |
| `--accel-noise` | 0.1 | Accelerometer noise σ (m/s²) |
| `--gyro-noise` | 0.01 | Gyroscope noise σ (rad/s) |
| `--range-noise` | 0.05 | UWB range noise σ (meters) |
| `--nlos-anchors` | [] | List of NLOS anchor indices |
| `--nlos-bias` | 0.5 | NLOS positive bias (meters) |
| `--dropout-rate` | 0.05 | Measurement dropout probability |
| `--time-offset` | 0.0 | Sensor time offset (seconds) |

**Presets**: `baseline`, `nlos_severe`, `high_dropout`, `degraded_imu`,
`time_offset_50ms`, `tactical_imu`

`--all-variants` writes every preset in one run.

**Examples**: See "Chapter 8 Experimentation Scenarios" section below.

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
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.01 --gyro-noise 0.001 \
    --output data/sim/fusion_low_noise

# Medium noise (consumer-grade IMU) - baseline
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.1 --gyro-noise 0.01 \
    --output data/sim/fusion_med_noise

# High noise (degraded IMU)
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.5 --gyro-noise 0.05 \
    --output data/sim/fusion_high_noise
```

**Run Experiments**:
```bash
# Test on all three datasets
python -m ch8_sensor_fusion.example_tc_fusion --data data/sim/fusion_low_noise
python -m ch8_sensor_fusion.example_tc_fusion --data data/sim/fusion_med_noise
python -m ch8_sensor_fusion.example_tc_fusion --data data/sim/fusion_high_noise

# Compare results
python tools/compare_fusion_variants.py \
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
    python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py \
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
    
    # Without gating. There is no --output: the run prints its RMSE and
    # acceptance counts, and --save takes a figure path, not a JSON one.
    python -m ch8_sensor_fusion.example_tc_fusion \
        --data data/sim/fusion_nlos_bias_${bias} \
        --no-gating

    # With gating (default)
    python -m ch8_sensor_fusion.example_tc_fusion \
        --data data/sim/fusion_nlos_bias_${bias}
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
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py \
    --time-offset -0.05 \
    --clock-drift 0.0001 \
    --output data/sim/fusion_time_offset_50ms
```

**Run Experiments**:
```bash
# One run does both. The demo fuses the dataset twice -- once ignoring the
# offset, once correcting it with TimeSyncModel -- and prints the pair, so
# there is nothing to switch off and no --no-correction to do it with.
python -m ch8_sensor_fusion.example_temporal_calibration \
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
python scripts/generate_ch6_strapdown_dataset.py --preset tactical
python scripts/generate_ch6_strapdown_dataset.py --preset consumer
python scripts/generate_ch6_strapdown_dataset.py --preset mems
```

**Run Experiments**:
```bash
# The example builds its own trajectory and takes no dataset argument. The
# three presets differ only in their IMU noise and bias, which config.json
# records, so compare those and integrate the datasets with the block in
# data/sim/ch6_strapdown_basic/README.md.
for d in ch6_strapdown_tactical ch6_strapdown_basic ch6_strapdown_mems; do
    python -c "import json;print('$d',json.load(open('data/sim/$d/config.json'))['imu'])"
done
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
python scripts/generate_ch6_zupt_dataset.py --num-steps 40
```

**Run Experiments**:
```bash
# Neither takes a dataset; both build their own trajectory and show the
# contrast on it.
python -m ch6_dead_reckoning.example_imu_strapdown   # unbounded drift
python -m ch6_dead_reckoning.example_zupt            # bounded by ZUPT
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
python scripts/generate_ch5_wifi_fingerprint_dataset.py \
    --n-aps 4 --grid-spacing 5.0 \
    --output data/sim/wifi_sparse_4ap

# Medium AP coverage (baseline)
python scripts/generate_ch5_wifi_fingerprint_dataset.py \
    --n-aps 8 --grid-spacing 5.0 \
    --output data/sim/wifi_medium_8ap

# Dense AP coverage
python scripts/generate_ch5_wifi_fingerprint_dataset.py \
    --n-aps 16 --grid-spacing 5.0 \
    --output data/sim/wifi_dense_16ap
```

**Run Experiments**:
```bash
# example_deterministic reads data/sim/ch5_wifi_fingerprint_grid and declares
# no flags, so it cannot be pointed at these. ch5_fingerprinting's README shows
# the NN and k-NN calls directly against a loaded database -- load each variant
# there instead, which is also less machinery than a CLI would be.
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
python scripts/generate_ch5_wifi_fingerprint_dataset.py \
    --grid-spacing 10.0 --output data/sim/wifi_coarse_grid

# Fine grid (more survey effort)  
python scripts/generate_ch5_wifi_fingerprint_dataset.py \
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
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py --duration -10
# Error: Duration must be positive

# Example: Invalid noise range
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py --accel-noise 10.0
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
# Check: .templates/dataset_README_template.md
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

See `.templates/dataset_README_template.md` for requirements.

---

## References

- **Dataset README standard**: `.templates/dataset_README_template.md`
- **Dataset Catalog**: `data/sim/README.md`
- **Learning Guide**: `docs/data_simulation_guide.md`
- **Templates**: `templates/` folder
- **Tools**: `tools/` (visualization, validation, comparison)


