# Chapter 7: 2D Pose Graph SLAM - Quick Start Guide

## Overview

This chapter demonstrates a **complete observation-driven 2D SLAM pipeline**:

| Stage | Description | Performance |
|-------|-------------|-------------|
| **1. Front-End** | Prediction → Scan-to-map ICP → Map update | ~37% local improvement |
| **2. Loop Closure** | Observation-based detection + ICP verification | 147 closures (3-lap square) |
| **3. Back-End** | Pose graph optimization | ~64% full improvement |

**Default:** Square trajectory (8m x 8m, 3 laps) with asymmetric room features.

---

## Quick Start

### Run the Main Example

```bash
# Full SLAM pipeline (default: square trajectory, 3 laps)
python -m ch7_slam.example_pose_graph_slam

# With corridor trajectory (legacy)
python -m ch7_slam.example_pose_graph_slam --trajectory corridor

# With 2 laps (faster)
python -m ch7_slam.example_pose_graph_slam --laps 2

# With pre-generated dataset
python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_square
python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_high_drift
```

### Run the Frontend Demo

```bash
# Standalone frontend demo showing scan-to-map ICP
python -m ch7_slam.example_slam_frontend
```

---

## What You'll Learn

### 1. SLAM Front-End (Prediction → Correction → Update)

**File:** `core/slam/frontend_2d.py`

```py
from core.slam import SlamFrontend2D

frontend = SlamFrontend2D(submap_voxel_size=0.1)

for i, (odom_delta, scan) in enumerate(trajectory):
    result = frontend.step(i, odom_delta, scan)
    
    # Predicted pose (odometry only)
    pose_pred = result['pose_pred']
    
    # Estimated pose (after scan-to-map ICP)
    pose_est = result['pose_est']
    
    # Match quality metrics
    match_quality = result['match_quality']
    print(f"Step {i}: residual={match_quality.residual:.4f}, "
          f"converged={match_quality.converged}")
```

**Key concepts:**
- Odometry provides prediction (with drift)
- Scan-to-map ICP corrects drift locally
- Submap accumulates corrected scans

### 2. Loop Closure Detection (Descriptor Similarity + Verification)

**File:** `core/slam/loop_closure_2d.py`

```py
from core.slam import LoopClosureDetector2D

detector = LoopClosureDetector2D(
    min_descriptor_similarity=0.65,  # PRIMARY filter
    max_distance=15.0,                # SECONDARY filter (optional)
)

loop_closures = detector.detect(scans, poses)

for lc in loop_closures:
    print(f"Loop: {lc.j} -> {lc.i}")
    print(f"  Descriptor similarity: {lc.descriptor_similarity:.3f}")
    print(f"  ICP residual: {lc.icp_residual:.4f}")
```

**Key concepts:**
- Scan descriptors (range histograms) for place recognition
- Descriptor similarity as primary candidate filter
- ICP verification for geometric consistency
- Finds 2-3x more loop closures than oracle methods

### 3. Pose Graph Optimization (Global Consistency)

**File:** `core/slam/factors.py` (existing)

```py
from core.slam import create_pose_graph

# Build graph
graph = create_pose_graph(
    poses=initial_trajectory,
    odometry_measurements=odometry_factors,
    loop_closures=loop_closure_factors,
    odometry_information=odom_info,
    loop_information=loop_info,
)

# Optimize
optimized_vars, error_history = graph.optimize(
    method="gauss_newton",
    max_iterations=50,
    tol=1e-6
)
```

**Key concepts:**
- Factor graph representation
- Prior, odometry, and loop closure factors
- Gauss-Newton optimization
- Information matrices (inverse covariances)

---

## Example Output

### Square Dataset Results

Running `python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_square` produces:

<!-- example-output: ch7_slam.example_pose_graph_slam --data ch7_slam_2d_square -->
```
Dataset Info:
  Trajectory: square
  Poses: 41
  Landmarks: 50
  Loop closures: 2

  Initial drift: 0.000 m
  Final drift (without SLAM): 0.546 m
...
  Frontend converged ratio: 26.8%
  Frontend avg residual: 0.0725 m
...
  Detected 1 loop closures (observation-based)
...
Building pose graph...
  Graph initial: frontend_poses (scan-to-map corrected trajectory)
  Pose graph: 41 variables, 42 factors
  Factors: 1 prior + 40 odometry + 1 loop closures
...
Optimizing pose graph...
  Initial error: 8.350582
  Final error: 0.003522
  Iterations: 3
  Error reduction: 99.96%
...
Results:
  Odometry RMSE: 0.3281 m (baseline)
  Frontend RMSE: 0.3404 m (scan-to-map corrected)
  Optimized RMSE: 0.2184 m (backend with 1 loop closures)
  Frontend improvement: -3.74%
  Full pipeline improvement: +33.44%
  Final loop closure error: 0.0360 m
...
   Building map point clouds...
   Map before (front-end): 537 points
   Map after (backend):    481 points
```

This dataset has only 41 poses and one lap, so the front-end has little overlap
to align against — its RMSE is slightly *worse* than raw odometry here, and the
whole improvement comes from the single loop closure the backend finds. The
default inline run (145 poses, 3 laps) behaves differently; see the Performance
Summary below.

### Visualization Output

The script generates a comprehensive figure showing:

**Layout (1x3 grid):**
1. **Left:** Trajectories (ground truth, odometry, optimized) + loop closures
2. **Middle-top:** Map before optimization (red points from odometry poses)
3. **Middle-bottom:** Map after optimization (blue points from optimized poses)
4. **Right:** Position error over time

**Key Visual Features:**
- Map "tightening" is clearly visible: 537 → 481 points (~10% reduction)
- Red map shows odometry drift and misalignment
- Blue map shows optimized alignment and consistency
- Loop closure connections shown in magenta

**File:** Saved to `ch7_slam/figs/slam_with_maps.png`

---

## Performance Summary (Default: Square, 3 laps)

This is the **inline mode** default — `python -m ch7_slam.example_pose_graph_slam`
with no `--data` flag (145 poses over 3 laps), not the dataset run shown above.

| Stage | RMSE | Improvement | Notes |
|-------|------|-------------|-------|
| **Odometry** | 0.85 m | baseline | Raw sensor integration |
| **Front-end** | 0.53 m | **+37%** ✅ | Scan-to-map ICP |
| **Full SLAM** | 0.30 m | **+64%** | + 147 loop closures |

**Verification Checks:**
- `Detected 147 loop closures` ✅
- `Frontend RMSE <= Odometry RMSE` ✅
- `Optimized RMSE <= 0.95 * Odometry RMSE` ✅ (>5% improvement)

---

## Key Features

### ✅ Observation-Driven
- No oracle position information for loop closure
- Scan descriptor similarity as primary filter
- All constraints from sensor measurements

### ✅ Complete Pipeline
- Front-end: Scan-to-map alignment
- Loop closure: Descriptor matching + ICP verification
- Back-end: Global pose graph optimization

### ✅ Robust
- Graceful fallback when ICP fails
- Quality checks on all loop closures
- Individual covariances per constraint

### ✅ Well-Tested
- 76 unit tests (100% pass rate)
- 4 example scripts/demos
- Comprehensive documentation

---

## Module Reference

### Core Modules

- **`core.slam.submap_2d`**: Local map accumulation with downsampling
- **`core.slam.frontend_2d`**: SLAM front-end (predict-correct-update)
- **`core.slam.scan_descriptor_2d`**: Range histogram descriptors
- **`core.slam.loop_closure_2d`**: Observation-based loop detection
- **`core.slam.se2`**: 2D transformations (existing)
- **`core.slam.scan_matching`**: ICP/NDT algorithms (existing)
- **`core.slam.factors`**: Pose graph factors (existing)

### Example Scripts

- **`ch7_slam/example_pose_graph_slam.py`**: Main SLAM pipeline
- **`ch7_slam/example_slam_frontend.py`**: Frontend-only demo
- **`ch7_slam/example_bundle_adjustment.py`**: Visual SLAM (existing)

---

## Troubleshooting

### No Loop Closures Detected

**Possible causes:**
- Trajectory too short (need ≥ `min_time_separation` + few poses)
- Scans too different (adjust `min_descriptor_similarity`)
- ICP failing (increase `max_icp_residual`)

**Solutions:**
- Use longer trajectories (≥20 poses)
- Lower descriptor threshold (e.g., 0.60)
- Check scan quality (need ≥5 points per scan)

### Poor SLAM Performance

**Possible causes:**
- Few loop closures detected
- High odometry drift
- Poor scan quality

**Solutions:**
- Tune detector parameters (see `LoopClosureDetector2D.__init__`)
- Use better sensors (lower noise)
- Ensure environment has features for scan matching

### Frontend Making Results Worse

**Cause:** Not enough scan overlap for ICP to help. This is expected on the
41-pose, single-lap square dataset (see Example Output above) — the front-end
has too little overlap between consecutive scans to extract a correction bigger
than the noise, so its RMSE lands slightly above raw odometry. It is not a
broken front-end: the same code improves the 145-pose, 3-lap inline default by
~37% (see Performance Summary).

**Solution:** Use a trajectory with more revisit/overlap (more laps, or the
inline default) if you need the front-end stage itself to show improvement.

---

## References

### Book Sections
- **Section 7.2:** Pose Graph SLAM Formulation
- **Section 7.3:** Factor Graph Optimization
- **Section 7.3.5:** Close-Loop Constraints (Eq. 7.22)

### Papers
- Lu & Milios (1997): Globally Consistent Range Scan Alignment
- Olson et al. (2006): Fast Iterative Alignment of Pose Graphs
- Grisetti et al. (2010): g2o: A General Framework for Graph Optimization

---

## Support

### Run Tests
```bash
# All SLAM tests
python -m unittest tests.core.slam.test_submap_2d \
                  tests.core.slam.test_frontend_2d \
                  tests.core.slam.test_scan_descriptor_2d \
                  tests.core.slam.test_loop_closure_2d -v
```

### Check Implementation
```bash
# Verify no ground truth contamination
grep -n "true_poses\[i\], true_poses\[i+1\]" ch7_slam/example_pose_graph_slam.py
# Should return empty ✅
```

---

**Author:** Li-Ta Hsu  
**Date:** December 2025  
**Version:** 2.0 (Observation-driven)
