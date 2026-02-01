"""Demo script showing Submap2D usage for scan-to-map alignment.

This demonstrates how Submap2D will be used in the SLAM front-end for
building local maps incrementally.

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
from core.slam import Submap2D, se2_compose


def main():
    """Demonstrate Submap2D functionality."""
    print("=" * 70)
    print("SUBMAP2D DEMO: Building Local Map from Scans")
    print("=" * 70)
    print()
    
    # Initialize empty submap
    submap = Submap2D()
    print("1. Created empty submap")
    print(f"   Points: {len(submap)}, Scans: {submap.n_scans}")
    print()
    
    # Simulate robot moving along trajectory, collecting scans
    print("2. Simulating robot trajectory (square path)...")
    print()
    
    poses = [
        np.array([0.0, 0.0, 0.0]),           # Start at origin, facing east
        np.array([1.0, 0.0, 0.0]),           # Move 1m east
        np.array([1.0, 1.0, np.pi / 2]),     # Turn north, move 1m
        np.array([0.0, 1.0, np.pi]),         # Turn west, move 1m
        np.array([0.0, 0.0, -np.pi / 2]),    # Turn south, move 1m (back to start)
    ]
    
    # Generate synthetic scans (wall observations)
    np.random.seed(42)
    for i, pose in enumerate(poses):
        # Simulate observing walls at different distances
        scan = np.random.rand(10, 2) * 2.0 + np.array([3.0, 0.0])
        
        # Add scan to submap
        submap.add_scan(pose, scan)
        
        print(f"   Pose {i}: [{pose[0]:.1f}, {pose[1]:.1f}, {np.rad2deg(pose[2]):.0f}°]")
        print(f"      Added {len(scan)} points")
        print(f"      Total points in submap: {len(submap)}")
    
    print()
    print(f"3. Built submap from {submap.n_scans} scans")
    print(f"   Total points: {len(submap)}")
    print()
    
    # Demonstrate downsampling
    print("4. Downsampling submap...")
    original_count = len(submap)
    print(f"   Before: {original_count} points")
    
    # Downsample with 0.5m voxel size
    submap.downsample(voxel_size=0.5)
    downsampled_count = len(submap)
    
    print(f"   After:  {downsampled_count} points")
    print(f"   Reduction: {(1 - downsampled_count / original_count) * 100:.1f}%")
    print()
    
    # Demonstrate get_points with on-demand downsampling
    print("5. Getting points with different resolutions...")
    
    # Get all points
    all_points = submap.get_points()
    print(f"   get_points():              {len(all_points)} points")
    
    # Get with different voxel sizes (non-destructive)
    coarse = submap.get_points(voxel_size=1.0)
    fine = submap.get_points(voxel_size=0.1)
    print(f"   get_points(voxel_size=1.0): {len(coarse)} points")
    print(f"   get_points(voxel_size=0.1): {len(fine)} points")
    print()
    
    # Show coordinate frame verification
    print("6. Verifying coordinate transformation...")
    test_submap = Submap2D()
    
    # Add scan at identity pose
    pose1 = np.array([0.0, 0.0, 0.0])
    scan1 = np.array([[1.0, 0.0]])
    test_submap.add_scan(pose1, scan1)
    print(f"   Scan at identity: {scan1[0]} -> Map: {test_submap.points[0]}")
    
    # Add scan at translated pose
    pose2 = np.array([2.0, 3.0, 0.0])
    scan2 = np.array([[1.0, 0.0]])
    test_submap.add_scan(pose2, scan2)
    print(f"   Scan at [{pose2[0]}, {pose2[1]}]: {scan2[0]} -> Map: {test_submap.points[1]}")
    
    # Add scan at rotated pose (90 degrees)
    pose3 = np.array([0.0, 0.0, np.pi / 2])
    scan3 = np.array([[1.0, 0.0]])
    test_submap.add_scan(pose3, scan3)
    print(f"   Scan at 90° rotation: {scan3[0]} -> Map: {test_submap.points[2]}")
    print()
    
    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  - Submap accumulates scans in map frame")
    print("  - SE(2) transformations applied automatically")
    print("  - Downsampling reduces point density")
    print("  - get_points() can downsample on-demand (non-destructive)")
    print()
    print("Next steps: Integrate into SLAM front-end for scan-to-map odometry")
    print()


if __name__ == "__main__":
    main()
