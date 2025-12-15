# Chapter 7: SLAM (Simultaneous Localization and Mapping)

## Overview

This module implements SLAM algorithms described in **Chapter 7** of *Principles of Indoor Positioning and Indoor Navigation*.

SLAM addresses the chicken-and-egg problem:
- **Localization** requires a map
- **Mapping** requires knowing the robot's location
- **SLAM** solves both simultaneously

The chapter implements:
- **Scan matching** (ICP, NDT) for relative pose estimation
- **Factor graph optimization** for trajectory correction
- **Loop closure detection** for drift reduction
- **Visual SLAM** with camera models and bundle adjustment

## Quick Start

```bash
# Run LiDAR SLAM example
python -m ch7_slam.example_pose_graph_slam

# Run Visual SLAM example
python -m ch7_slam.example_bundle_adjustment
```

## Equation Reference

### Scan Matching

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `icp_point_to_point()` | `core/slam/scan_matching.py` | Eq. (7.10)-(7.11) | ICP alignment with SVD |
| `ndt_align()` | `core/slam/scan_matching.py` | Eq. (7.14)-(7.16) | NDT probabilistic alignment |

### Pose Graph Optimization

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `create_odometry_factor()` | `core/slam/pose_graph.py` | Section 7.3 | Connect consecutive poses |
| `create_loop_closure_factor()` | `core/slam/pose_graph.py` | Section 7.3 | Connect non-consecutive poses |
| `create_prior_factor()` | `core/slam/pose_graph.py` | Section 7.3 | Anchor first pose |

### Visual SLAM

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `project_point()` | `core/slam/camera_model.py` | Eq. (7.40) | Pinhole camera projection |
| `distort_normalized()` | `core/slam/camera_model.py` | Eq. (7.43)-(7.46) | Brown-Conrady distortion |
| `create_reprojection_factor()` | `core/slam/bundle_adjustment.py` | Eq. (7.68)-(7.70) | Bundle adjustment factor |

## Expected Output

### LiDAR Pose Graph SLAM

Running `python -m ch7_slam.example_pose_graph_slam` produces:

```
====================================================================================
CHAPTER 7: 2D POSE GRAPH SLAM EXAMPLE
====================================================================================

1. Generating square trajectory...
   Generated 22 poses in square loop

2. Generating environment landmarks...
   Generated 60 landmarks

3. Generating LiDAR scans...
   Generated 22 scans (avg 25.4 points/scan)

4. Simulating odometry with drift...
   Final drift (without SLAM): 1.234 m

5. Detecting loop closures...
   Detected 1 loop closures

6. Building pose graph...
   Factors: 1 prior + 21 odometry + 1 loop closures

7. Optimizing pose graph...
   Error reduction: 99.73%

8. Evaluating results...
   Odometry RMSE: 0.8234 m
   Optimized RMSE: 0.0567 m
   Improvement: 93.11%
```

**Visual Output:**

![Pose Graph SLAM Results](figs/pose_graph_slam_results.png)

*This figure shows two plots:*
- **Left:** Trajectories comparing ground truth (green), odometry with drift (red dashed), and optimized SLAM (blue)
- **Right:** Position error over time showing how loop closure corrects accumulated drift

### Visual Bundle Adjustment

Running `python -m ch7_slam.example_bundle_adjustment` produces:

```
================================================================================
CHAPTER 7: VISUAL BUNDLE ADJUSTMENT EXAMPLE
================================================================================

1. Setting up camera parameters...
   Camera: fx=500.0, fy=500.0

2. Generating ground truth...
   Generated 8 camera poses (circular trajectory)
   Generated 15 3D landmarks

3. Simulating camera observations...
   Generated 117 observations

4. Creating noisy initial estimates...
   Initial pose RMSE: 0.0696 m
   Initial landmark RMSE: 0.1708 m

5. Running bundle adjustment optimization...
   Error reduction: 98.61%
```

**Visual Output:**

![Bundle Adjustment Results](figs/bundle_adjustment_results.png)

*This figure shows three plots:*
- **Left:** Camera trajectory and 3D landmarks (top view)
- **Middle:** Position errors before and after optimization
- **Right:** Optimization convergence curve

## Performance Summary

| Method | Input | RMSE | Improvement |
|--------|-------|------|-------------|
| **Odometry only** | Wheel encoders | ~1-2 m | Baseline |
| **Pose Graph SLAM** | + Loop closures | ~0.05 m | 93-95% |
| **Bundle Adjustment** | Camera images | ~0.01 m | 98%+ |

## Key Concepts

### Scan Matching (ICP/NDT)

- **ICP**: Finds correspondences, computes optimal transform via SVD
- **NDT**: Represents target as Gaussian distributions, gradient-based optimization
- **Used for**: Loop closure detection and relative pose estimation

### Pose Graph Structure

```
pose_0 --odom--> pose_1 --odom--> ... --odom--> pose_N
  ^                                                  |
  +------------------- loop closure -----------------+
```

### Bundle Adjustment

- **Variables**: Camera poses + 3D landmarks
- **Objective**: Minimize sum of squared reprojection errors
- **Result**: Globally consistent reconstruction

## File Structure

```
ch7_slam/
├── README.md                       # This file (student documentation)
├── example_pose_graph_slam.py      # LiDAR SLAM pipeline demo
├── example_bundle_adjustment.py    # Visual bundle adjustment demo
└── figs/                           # Generated figures
    ├── pose_graph_slam_results.png
    └── bundle_adjustment_results.png

core/slam/
├── scan_matching.py                # ICP, NDT algorithms
├── pose_graph.py                   # Pose graph construction
├── camera_model.py                 # Camera projection, distortion
└── bundle_adjustment.py            # Visual SLAM optimization
```

## References

- **Chapter 7**: SLAM
  - Section 7.2: Scan Matching (ICP, NDT)
  - Section 7.3: Pose Graph Optimization
  - Section 7.4: Visual SLAM and Bundle Adjustment

---

For implementation details and development notes, see [docs/ch7_development.md](../docs/ch7_development.md).
