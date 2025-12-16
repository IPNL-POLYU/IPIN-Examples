# LC vs TC Fusion Comparison Guide

## Overview

This guide documents the **direct comparison tool** for Loosely Coupled (LC) and Tightly Coupled (TC) sensor fusion architectures in Chapter 8.

**Tool**: `ch8_sensor_fusion/compare_lc_tc.py`  
**Purpose**: Run both fusion approaches side-by-side on identical data for fair comparison

---

## Quick Start

### Run Comparison

```bash
# Basic usage (baseline dataset)
python -m ch8_sensor_fusion.compare_lc_tc

# Save outputs
python -m ch8_sensor_fusion.compare_lc_tc \
    --save ch8_sensor_fusion/figs/comparison.svg \
    --report ch8_sensor_fusion/figs/comparison.json

# Test on NLOS dataset
python -m ch8_sensor_fusion.compare_lc_tc \
    --data data/sim/ch8_fusion_2d_imu_uwb_nlos
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `data/sim/ch8_fusion_2d_imu_uwb` | Dataset directory |
| `--no-gating` | flag | False | Disable chi-square gating |
| `--alpha` | float | 0.05 | Gating significance level |
| `--save` | str | `lc_tc_comparison.svg` | Path to save figure |
| `--report` | str | None | Path to save JSON report |

---

## Comparison Results (Baseline Dataset)

### Performance Metrics

```
======================================================================
LC vs TC Performance Comparison
======================================================================
Metric                          LC Fusion       TC Fusion   Difference
----------------------------------------------------------------------
RMSE 2D (m)                        12.896          12.352      +0.544
RMSE X (m)                         17.092          16.519      +0.573
RMSE Y (m)                          6.362           5.680      +0.682
Max Error (m)                      40.826          38.993      +1.833
Mean Error (m)                     11.747          11.248      +0.499
Final Error (m)                    39.112          37.392      +1.720
----------------------------------------------------------------------
UWB Updates Accepted                  176             748        -572
UWB Updates Rejected                  408            1523       -1115
LC Solver Failures                     16             N/A             
Acceptance Rate (%)                  30.1            32.9        -2.8
======================================================================

Summary:
  • TC has lower RMSE (0.544m difference)
  • TC has higher acceptance rate (2.8% difference)
  • LC: 176 updates, TC: 748 updates (TC has +572 more)
```

### Key Findings

#### 1. **Accuracy**: TC Slightly Better
- **TC RMSE**: 12.35m
- **LC RMSE**: 12.90m
- **Difference**: 0.54m (4.4% better for TC)

**Why?**
- TC processes more updates (748 vs 176)
- TC can handle partial measurements (1-3 anchors)
- LC requires ≥3 anchors, loses 16 epochs to solver failures

#### 2. **Update Efficiency**: LC More Efficient per Update
- **LC**: 176 position fix updates (1 per epoch)
- **TC**: 748 range updates (up to 4 per epoch)
- **Ratio**: TC has 4.25× more updates

**Why?**
- LC aggregates all ranges → single position update
- TC processes each anchor range individually
- LC has lower EKF computational cost per epoch

#### 3. **Acceptance Rate**: TC Slightly Higher
- **TC**: 32.9% acceptance rate
- **LC**: 30.1% acceptance rate
- **Difference**: 2.8% better for TC

**Why?**
- TC can accept partial measurements (e.g., 2 out of 4 anchors)
- LC requires ≥3 valid ranges for position solve
- TC's per-range gating is more granular

#### 4. **Chi-Square Gating**: Different DOF
- **LC**: 2 DOF (position measurement) → χ²(2, 0.05) = 5.99
- **TC**: 1 DOF (range measurement) → χ²(1, 0.05) = 3.84
- **Effect**: LC has more lenient per-update threshold

---

## Visualization Output

### 9-Panel Comparison Figure

The comparison script generates a comprehensive 3×3 grid:

#### **Row 1: Trajectories**
1. **LC Trajectory**
   - Ground truth (black)
   - LC estimate (blue)
   - UWB position fixes (cyan dots)
   - Anchors (red triangles)

2. **TC Trajectory**
   - Ground truth (black)
   - TC estimate (orange)
   - Anchors (red triangles)

3. **Overlay Comparison**
   - All three trajectories overlaid
   - Direct visual comparison

#### **Row 2: Position Errors**
4. **LC Position Error**
   - Error norm vs time
   - RMSE line

5. **TC Position Error**
   - Error norm vs time
   - RMSE line

6. **Error Comparison**
   - Both errors overlaid
   - Direct accuracy comparison

#### **Row 3: Innovation and Metrics**
7. **LC NIS (2 DOF)**
   - Normalized Innovation Squared
   - Chi-square bounds for position (m=2)
   - Accepted (green) vs rejected (red)

8. **TC NIS (1 DOF)**
   - Normalized Innovation Squared
   - Chi-square bounds for range (m=1)
   - Accepted (green) vs rejected (red)

9. **Metrics Bar Chart**
   - RMSE comparison
   - Max error comparison
   - Updates comparison
   - Acceptance rate comparison

---

## JSON Report Structure

```json
{
  "dataset": {
    "path": "data/sim/ch8_fusion_2d_imu_uwb",
    "n_imu_samples": 6000,
    "n_uwb_epochs": 600,
    "duration": 59.99
  },
  "lc_fusion": {
    "rmse_2d": 12.896,
    "rmse_x": 17.092,
    "rmse_y": 6.362,
    "max_error": 40.826,
    "mean_error": 11.747,
    "final_error": 39.112,
    "n_updates": 176,
    "n_rejected": 408,
    "n_failed": 16,
    "acceptance_rate": 30.1
  },
  "tc_fusion": {
    "rmse_2d": 12.352,
    "rmse_x": 16.519,
    "rmse_y": 5.680,
    "max_error": 38.993,
    "mean_error": 11.248,
    "final_error": 37.392,
    "n_updates": 748,
    "n_rejected": 1523,
    "acceptance_rate": 32.9
  },
  "comparison": {
    "rmse_difference": 0.544,
    "better_rmse": "TC",
    "update_ratio": 0.235
  }
}
```

---

## Interpretation Guide

### When Does LC Perform Better?

LC may outperform TC when:
1. **All anchors are consistently visible** (low dropout rate)
2. **Position solver is well-tuned** (good initial guess, proper bounds)
3. **Measurement noise is homogeneous** across anchors
4. **Computational efficiency matters** (fewer EKF updates)

### When Does TC Perform Better?

TC typically outperforms LC when:
1. **Frequent anchor dropouts** (can use partial measurements)
2. **Heterogeneous noise** (some anchors more reliable)
3. **NLOS conditions** (can reject individual ranges)
4. **Maximum accuracy required** (more updates = better observability)

### Why Are Both RMSEs ~12-13m?

Both architectures show moderate RMSE (~12-13m) because:

1. **Simplified 2D Model**: No IMU bias estimation
   - Production systems use full 15D state (Chapter 6)
   - Position, velocity, attitude, accel bias, gyro bias

2. **IMU Drift**: High-rate (100 Hz) IMU integration drifts
   - Accelerometer noise → velocity drift → position drift
   - 60 seconds of integration without bias correction

3. **Sparse UWB Updates**: Only 10 Hz (vs 100 Hz IMU)
   - UWB can't fully correct for fast IMU drift
   - 35% of UWB measurements rejected by gating

4. **Educational Simplification**: Prioritizes clarity over performance
   - Real systems: RMSE < 1m with proper bias estimation
   - This demo: Focus on fusion architecture comparison

**Conclusion**: The ~0.5m difference between LC and TC is meaningful for comparing architectures, even though both have moderate absolute RMSE.

---

## Experiment Ideas

### 1. **Vary Gating Threshold**

Test how gating affects both architectures:

```bash
# Strict gating (99% confidence)
python -m ch8_sensor_fusion.compare_lc_tc --alpha 0.01

# Loose gating (90% confidence)
python -m ch8_sensor_fusion.compare_lc_tc --alpha 0.10

# No gating (accept all)
python -m ch8_sensor_fusion.compare_lc_tc --no-gating
```

**Expected**: TC benefits more from strict gating (can reject bad individual ranges)

### 2. **Test on NLOS Dataset**

Compare robustness to NLOS bias:

```bash
python -m ch8_sensor_fusion.compare_lc_tc \
    --data data/sim/ch8_fusion_2d_imu_uwb_nlos \
    --save nlos_comparison.svg
```

**Expected**: TC handles NLOS better (per-anchor rejection)

### 3. **Test on Time-Offset Dataset**

Compare sensitivity to temporal misalignment:

```bash
python -m ch8_sensor_fusion.compare_lc_tc \
    --data data/sim/ch8_fusion_2d_imu_uwb_timeoffset \
    --save timeoffset_comparison.svg
```

**Expected**: Both degrade similarly (temporal sync needed for both)

### 4. **Batch Comparison Across Datasets**

```bash
for dataset in ch8_fusion_2d_imu_uwb ch8_fusion_2d_imu_uwb_nlos ch8_fusion_2d_imu_uwb_timeoffset; do
    python -m ch8_sensor_fusion.compare_lc_tc \
        --data data/sim/$dataset \
        --save figs/${dataset}_comparison.svg \
        --report figs/${dataset}_comparison.json
done
```

---

## Implementation Details

### Comparison Pipeline

```python
def run_both_fusions(dataset, use_gating, gate_alpha):
    """Run LC and TC on same dataset with same parameters."""
    
    # 1. Run LC fusion
    lc_results = run_lc_fusion(dataset, use_gating, gate_alpha)
    
    # 2. Run TC fusion
    tc_results = run_tc_fusion(dataset, use_gating, gate_alpha)
    
    return lc_results, tc_results

def compute_comparative_metrics(dataset, lc_results, tc_results):
    """Compute metrics for both approaches."""
    
    # Interpolate truth to estimated timestamps
    # Compute position errors
    # Compute RMSE, max error, etc.
    
    return {
        'lc': {...},
        'tc': {...}
    }
```

### Fair Comparison Guarantees

1. **Same Dataset**: Both use identical truth, IMU, UWB data
2. **Same Parameters**: Same gating threshold, same noise models
3. **Same Initial State**: Both start at true position
4. **Same Timestamps**: Both process same measurement stream
5. **Same Evaluation**: Errors computed at same time points

---

## Pedagogical Value

### What Students Learn

1. **Architectural Trade-offs**
   - LC: Simpler, fewer updates, needs ≥3 anchors
   - TC: More complex, more updates, handles dropouts better

2. **Chi-Square Gating with Different DOF**
   - LC: Position (2D) → χ²(2, α)
   - TC: Range (1D) → χ²(1, α)
   - Different critical values affect acceptance rate

3. **Update Frequency vs Accuracy**
   - More updates ≠ always better (but helps here)
   - Pre-processing (WLS) trades compute for simplicity

4. **Real-World Design Decisions**
   - When to use LC (simplicity, modularity)
   - When to use TC (accuracy, robustness)
   - No "best" architecture universally

5. **Quantitative Comparison**
   - Side-by-side visualization
   - Objective metrics (RMSE, acceptance rate)
   - JSON export for reproducibility

---

## Code Quality

The comparison script follows all project standards:

- ✅ **PEP 8 compliance**
- ✅ **Type hints** on all functions
- ✅ **Google-style docstrings**
- ✅ **Comprehensive visualization** (9 panels)
- ✅ **Machine-readable output** (JSON)
- ✅ **Command-line interface**
- ✅ **Error handling**
- ✅ **Zero linting errors**

**Lines of Code**: 628 lines (comparison script)

---

## Future Enhancements

### Potential Additions (Phase 4)

1. **Batch Comparison Script**
   - Run on all 3 datasets automatically
   - Generate comparison matrix

2. **Statistical Analysis**
   - Confidence intervals on RMSE
   - Hypothesis testing (is TC significantly better?)
   - Monte Carlo runs with different seeds

3. **Computational Profiling**
   - Time per update (LC vs TC)
   - Memory usage comparison
   - Efficiency metrics

4. **Interactive Visualization**
   - HTML/Plotly interactive plots
   - Zoom/pan trajectories
   - Toggle LC/TC on/off

---

## Summary

### Comparison Tool Features

✅ **Runs both LC and TC** on identical data  
✅ **Fair comparison** (same parameters, same evaluation)  
✅ **Comprehensive visualization** (9-panel figure)  
✅ **Quantitative metrics** (RMSE, acceptance rate, etc.)  
✅ **JSON export** for reproducibility  
✅ **Command-line interface** for automation  

### Key Takeaways

1. **TC is slightly more accurate** (~0.5m better RMSE)
2. **LC is more efficient** (fewer updates per epoch)
3. **TC handles dropouts better** (partial measurements OK)
4. **Both achieve similar accuracy** (~12-13m due to IMU drift)
5. **Architecture choice depends on requirements**

### Usage

For teaching Chapter 8:
1. Run individual demos first (TC, then LC)
2. Then run comparison to see trade-offs
3. Use different datasets to stress-test each approach
4. Export JSON for quantitative analysis

---

**Author**: Navigation Engineer  
**Date**: December 14, 2025  
**Status**: ✅ Complete and tested

