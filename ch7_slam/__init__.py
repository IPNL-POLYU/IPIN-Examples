"""Chapter 7: SLAM (Simultaneous Localization and Mapping) Examples.

This package provides reference implementations and examples for SLAM algorithms
from Chapter 7 of Principles of Indoor Positioning and Indoor Navigation.

Examples:
    - example_pose_graph_slam.py: Complete 2D pose graph SLAM with ICP/NDT
    - (Future: example_loop_closure.py, example_landmark_slam.py)

Key Concepts Demonstrated:
    - Scan matching: ICP and NDT alignment
    - Odometry integration: Building trajectory from dead-reckoning
    - Loop closure detection: Recognizing previously visited places
    - Pose graph optimization: Correcting drift with factor graphs
    - Visualization: Trajectory plots and error analysis

Dependencies:
    - core.slam: SE(2) operations, ICP, NDT, factors
    - core.estimators: Factor graph optimization
    - matplotlib: Visualization
    - numpy: Numerical operations

Author: Li-Ta Hsu
Date: 2024
"""

__version__ = "0.1.0"

__all__ = []


