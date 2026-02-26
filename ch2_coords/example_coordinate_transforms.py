"""Example: Coordinate transformations between LLH, ECEF, and ENU frames.

This example demonstrates the use of coordinate transformations for
indoor positioning applications:
1. Convert geodetic coordinates (LLH) to ECEF
2. Convert ECEF to local ENU frame
3. Convert rotation representations (Euler, quaternions, matrices)

Can run with:
    - Pre-generated dataset: python example_coordinate_transforms.py --data ch2_coords_san_francisco
    - Inline data (default): python example_coordinate_transforms.py

Reference: Chapter 2 - IPIN Fundamentals
    - Section 2.1: Coordinate Systems and Transformations
        - LLH representation: Eq. (2.8)
        - LLH→ECEF: Eq. (2.9)
        - ECEF→ENU: Eq. (2.10)
        - ECEF→LLH: Iterative method (see [2] in book references)
    - Section 2.2: Attitude Definition and Representation
        - Euler→Rotation matrix: Eq. (2.17)
        - Quaternion→Rotation matrix: Eq. (2.21)
        - Quaternion↔Euler: Eqs. (2.22)–(2.23)
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from core.coords import (
    ecef_to_enu,
    ecef_to_llh,
    enu_to_ecef,
    euler_to_quat,
    euler_to_rotation_matrix,
    llh_to_ecef,
    quat_to_euler,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
)


def load_dataset(data_dir: str) -> dict:
    """Load coordinate transforms dataset.
    
    Args:
        data_dir: Path to dataset directory (e.g., 'data/sim/ch2_coords_san_francisco')
    
    Returns:
        Dictionary with loaded data arrays and config
    """
    path = Path(data_dir)
    
    data = {
        'llh': np.loadtxt(path / 'llh_coordinates.txt'),
        'ecef': np.loadtxt(path / 'ecef_coordinates.txt'),
        'enu': np.loadtxt(path / 'enu_coordinates.txt'),
        'reference_llh': np.loadtxt(path / 'reference_llh.txt'),
        'euler_angles': np.loadtxt(path / 'euler_angles.txt'),
        'quaternions': np.loadtxt(path / 'quaternions.txt'),
    }
    
    with open(path / 'config.json') as f:
        data['config'] = json.load(f)
    
    return data


def run_with_dataset(data_dir: str) -> None:
    """Run coordinate transform examples using pre-generated dataset.
    
    Args:
        data_dir: Path to dataset directory
    """
    print("=" * 70)
    print("Chapter 2: Coordinate Transformation Examples")
    print(f"Using dataset: {data_dir}")
    print("=" * 70)
    
    # Load dataset
    data = load_dataset(data_dir)
    config = data['config']
    
    print(f"\nDataset Info:")
    print(f"  Location: {config.get('location', 'Unknown')}")
    print(f"  Points: {len(data['llh'])}")
    
    # Example 1: LLH to ECEF (verify dataset)
    print("\n1. LLH to ECEF Transformation (Dataset Verification)")
    print("-" * 70)
    
    llh_sample = data['llh'][0]
    ecef_dataset = data['ecef'][0]
    
    print(f"Dataset LLH: lat={np.rad2deg(llh_sample[0]):.6f}°, "
          f"lon={np.rad2deg(llh_sample[1]):.6f}°, h={llh_sample[2]:.2f}m")
    print(f"Dataset ECEF: [{ecef_dataset[0]:,.2f}, {ecef_dataset[1]:,.2f}, {ecef_dataset[2]:,.2f}] m")
    
    # Verify our transform matches
    ecef_computed = llh_to_ecef(llh_sample[0], llh_sample[1], llh_sample[2])
    diff = np.linalg.norm(ecef_computed - ecef_dataset)
    print(f"Computed ECEF: [{ecef_computed[0]:,.2f}, {ecef_computed[1]:,.2f}, {ecef_computed[2]:,.2f}] m")
    print(f"Difference: {diff:.6e} m (should be ~0)")
    
    # Example 2: Round-trip LLH -> ECEF -> LLH
    print("\n2. Round-Trip Accuracy Test")
    print("-" * 70)
    
    errors_lat = []
    errors_lon = []
    errors_h = []
    
    for i in range(min(10, len(data['llh']))):
        llh_orig = data['llh'][i]
        ecef = llh_to_ecef(llh_orig[0], llh_orig[1], llh_orig[2])
        llh_recovered = ecef_to_llh(ecef[0], ecef[1], ecef[2])
        
        errors_lat.append(np.abs(llh_recovered[0] - llh_orig[0]))
        errors_lon.append(np.abs(llh_recovered[1] - llh_orig[1]))
        errors_h.append(np.abs(llh_recovered[2] - llh_orig[2]))
    
    print(f"Round-trip errors (10 samples):")
    print(f"  Latitude:  {np.max(errors_lat):.2e} rad = {np.rad2deg(np.max(errors_lat)) * 3600:.2e} arcsec")
    print(f"  Longitude: {np.max(errors_lon):.2e} rad = {np.rad2deg(np.max(errors_lon)) * 3600:.2e} arcsec")
    print(f"  Height:    {np.max(errors_h):.2e} m")
    
    # Example 3: ENU Frame
    print("\n3. Local ENU Frame")
    print("-" * 70)
    
    ref_llh = data['reference_llh']
    if ref_llh.ndim == 1:
        lat_ref, lon_ref, h_ref = ref_llh[0], ref_llh[1], ref_llh[2]
    else:
        lat_ref, lon_ref, h_ref = ref_llh[0, 0], ref_llh[0, 1], ref_llh[0, 2]
    
    print(f"Reference point: lat={np.rad2deg(lat_ref):.6f}°, lon={np.rad2deg(lon_ref):.6f}°")
    
    # Show first few ENU coordinates
    print(f"\nSample ENU coordinates (from dataset):")
    for i in range(min(5, len(data['enu']))):
        enu = data['enu'][i]
        print(f"  Point {i}: E={enu[0]:.2f}m, N={enu[1]:.2f}m, U={enu[2]:.2f}m")
    
    # Example 4: Rotation Representations
    print("\n4. Rotation Representations")
    print("-" * 70)
    
    euler_sample = data['euler_angles'][0]
    quat_sample = data['quaternions'][0]
    
    print(f"Dataset Euler: roll={np.rad2deg(euler_sample[0]):.2f}°, "
          f"pitch={np.rad2deg(euler_sample[1]):.2f}°, yaw={np.rad2deg(euler_sample[2]):.2f}°")
    print(f"Dataset Quaternion: [{quat_sample[0]:.4f}, {quat_sample[1]:.4f}, "
          f"{quat_sample[2]:.4f}, {quat_sample[3]:.4f}]")
    
    # Convert and verify
    quat_computed = euler_to_quat(euler_sample[0], euler_sample[1], euler_sample[2])
    R_from_euler = euler_to_rotation_matrix(euler_sample[0], euler_sample[1], euler_sample[2])
    R_from_quat = quat_to_rotation_matrix(quat_sample)
    
    print(f"\nComputed Quaternion: [{quat_computed[0]:.4f}, {quat_computed[1]:.4f}, "
          f"{quat_computed[2]:.4f}, {quat_computed[3]:.4f}]")
    print(f"Quaternion norm: {np.linalg.norm(quat_computed):.6f} (should be 1.0)")
    print(f"Rotation matrix determinant: {np.linalg.det(R_from_euler):.6f} (should be 1.0)")
    
    # Example 5: Apply rotation to vector
    print("\n5. Applying Rotations")
    print("-" * 70)
    
    v_body = np.array([1.0, 0.0, 0.0])  # Forward in body frame
    v_nav = R_from_quat @ v_body
    
    print(f"Vector in body frame: {v_body}")
    print(f"Vector in navigation frame: [{v_nav[0]:.4f}, {v_nav[1]:.4f}, {v_nav[2]:.4f}]")
    
    print("\n" + "=" * 70)
    print("Dataset verification complete!")
    print("=" * 70)
    print("\nKey Learning Points:")
    print("  - LLH<->ECEF transforms have sub-nanometer accuracy")
    print("  - ENU provides intuitive local coordinates for indoor positioning")
    print("  - Quaternions avoid gimbal lock (use for computation)")
    print("  - Euler angles are human-readable (use for display)")


def run_with_inline_data() -> None:
    """Run coordinate transform examples with inline generated data (original behavior)."""
    print("=" * 70)
    print("Chapter 2: Coordinate Transformation Examples")
    print("(Using inline generated data)")
    print("=" * 70)

    # Example 1: LLH to ECEF transformation
    print("\n1. LLH to ECEF Transformation")
    print("-" * 70)

    # Define a location: San Francisco (37.7749°N, 122.4194°W)
    lat_sf = np.deg2rad(37.7749)
    lon_sf = np.deg2rad(-122.4194)
    height_sf = 0.0  # Sea level

    print(f"Location: San Francisco")
    print(f"  Latitude:  {np.rad2deg(lat_sf):.4f}°")
    print(f"  Longitude: {np.rad2deg(lon_sf):.4f}°")
    print(f"  Height:    {height_sf:.1f} m")

    # Convert to ECEF
    xyz_sf = llh_to_ecef(lat_sf, lon_sf, height_sf)
    print(f"\nECEF Coordinates:")
    print(f"  X: {xyz_sf[0]:,.2f} m")
    print(f"  Y: {xyz_sf[1]:,.2f} m")
    print(f"  Z: {xyz_sf[2]:,.2f} m")

    # Example 2: ECEF to LLH (round-trip)
    print("\n2. ECEF to LLH (Round-trip)")
    print("-" * 70)

    llh_result = ecef_to_llh(*xyz_sf)
    print(f"Recovered LLH:")
    print(f"  Latitude:  {np.rad2deg(llh_result[0]):.4f}°")
    print(f"  Longitude: {np.rad2deg(llh_result[1]):.4f}°")
    print(f"  Height:    {llh_result[2]:.2f} m")

    # Example 3: Local ENU frame
    print("\n3. Local ENU Frame Transformation")
    print("-" * 70)

    # Define reference point (e.g., a building entrance)
    lat_ref = np.deg2rad(37.7749)
    lon_ref = np.deg2rad(-122.4194)
    height_ref = 0.0

    # Define target points relative to reference
    targets = [
        ("100m East", np.deg2rad(37.7749), np.deg2rad(-122.4194) + 100 / 78800, 0.0),
        ("100m North", np.deg2rad(37.7749) + 100 / 111000, np.deg2rad(-122.4194), 0.0),
        (
            "50m Up",
            np.deg2rad(37.7749),
            np.deg2rad(-122.4194),
            50.0,
        ),
    ]

    for name, lat_tgt, lon_tgt, height_tgt in targets:
        # Convert target to ECEF
        xyz_tgt = llh_to_ecef(lat_tgt, lon_tgt, height_tgt)

        # Convert to ENU relative to reference
        enu = ecef_to_enu(*xyz_tgt, lat_ref, lon_ref, height_ref)

        print(f"\nTarget: {name}")
        print(f"  ENU: [{enu[0]:.2f}, {enu[1]:.2f}, {enu[2]:.2f}] m")

    # Example 4: Rotation representations
    print("\n4. Rotation Representations")
    print("-" * 70)

    # Define Euler angles (roll, pitch, yaw)
    roll = np.deg2rad(10.0)  # 10° roll
    pitch = np.deg2rad(20.0)  # 20° pitch
    yaw = np.deg2rad(30.0)  # 30° yaw

    print(f"Euler Angles:")
    print(f"  Roll:  {np.rad2deg(roll):.1f}°")
    print(f"  Pitch: {np.rad2deg(pitch):.1f}°")
    print(f"  Yaw:   {np.rad2deg(yaw):.1f}°")

    # Convert to rotation matrix
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    print(f"\nRotation Matrix:")
    print(f"{R}")
    print(f"  Determinant: {np.linalg.det(R):.6f} (should be 1.0)")

    # Convert to quaternion
    q = euler_to_quat(roll, pitch, yaw)
    print(f"\nQuaternion [qw, qx, qy, qz]:")
    print(f"  {q}")
    print(f"  Norm: {np.linalg.norm(q):.6f} (should be 1.0)")

    # Example 5: Rotation application
    print("\n5. Applying Rotation to Vector")
    print("-" * 70)

    # Vector in body frame (e.g., sensor pointing forward)
    v_body = np.array([1.0, 0.0, 0.0])
    print(f"Vector in body frame: {v_body}")

    # Rotate to navigation frame
    R_nav_body = quat_to_rotation_matrix(q)
    v_nav = R_nav_body @ v_body
    print(f"Vector in navigation frame: {v_nav}")

    # Example 6: Quaternion -> Euler (direct, Eqs. 2.22-2.23)
    print("\n6. Quaternion -> Euler (Eqs. 2.22-2.23)")
    print("-" * 70)

    euler_from_quat = quat_to_euler(q)
    print(f"Quaternion: {q}")
    print(f"Euler from quat_to_euler: "
          f"[{np.rad2deg(euler_from_quat[0]):.1f}°, "
          f"{np.rad2deg(euler_from_quat[1]):.1f}°, "
          f"{np.rad2deg(euler_from_quat[2]):.1f}°]")
    print(f"Original Euler:           "
          f"[{np.rad2deg(roll):.1f}°, "
          f"{np.rad2deg(pitch):.1f}°, "
          f"{np.rad2deg(yaw):.1f}°]")

    # Round-trip check: Euler -> Quat -> Euler
    q_rt = euler_to_quat(roll, pitch, yaw)
    euler_rt = quat_to_euler(q_rt)
    rt_error = np.max(np.abs(np.array([roll, pitch, yaw]) - euler_rt))
    print(f"\nRound-trip Euler->Quat->Euler error: {rt_error:.2e} rad "
          f"({'PASS' if rt_error < 1e-9 else 'FAIL'})")

    # Example 7: Round-trip rotation conversions (matrix path)
    print("\n7. Round-trip Rotation Conversions (Matrix Path)")
    print("-" * 70)

    R_from_euler = euler_to_rotation_matrix(roll, pitch, yaw)
    euler_recovered = rotation_matrix_to_euler(R_from_euler)

    print(f"Original Euler: [{np.rad2deg(roll):.1f}°, "
          f"{np.rad2deg(pitch):.1f}°, {np.rad2deg(yaw):.1f}°]")
    print(f"Recovered Euler: [{np.rad2deg(euler_recovered[0]):.1f}°, "
          f"{np.rad2deg(euler_recovered[1]):.1f}°, "
          f"{np.rad2deg(euler_recovered[2]):.1f}°]")

    # Example 8: Coordinate frame conversions
    print("\n8. Practical Indoor Positioning Scenario")
    print("-" * 70)

    # Scenario: Indoor positioning system with reference at building entrance
    print("Building entrance (reference): 37.7749°N, 122.4194°W")

    # User locations in ENU (relative to entrance)
    user_positions = [
        ("Lobby", np.array([0.0, 0.0, 0.0])),
        ("Room 101", np.array([15.0, 10.0, 0.0])),
        ("Room 201", np.array([15.0, 10.0, 3.5])),
        ("Parking", np.array([-5.0, -20.0, -2.5])),
    ]

    for name, enu_pos in user_positions:
        # Convert ENU to ECEF
        xyz = enu_to_ecef(*enu_pos, lat_ref, lon_ref, height_ref)

        # Convert ECEF to LLH
        llh = ecef_to_llh(*xyz)

        print(f"\n{name}:")
        print(f"  ENU:  [{enu_pos[0]:.1f}, {enu_pos[1]:.1f}, {enu_pos[2]:.1f}] m")
        print(f"  LLH:  [{np.rad2deg(llh[0]):.6f}°, "
              f"{np.rad2deg(llh[1]):.6f}°, {llh[2]:.2f} m]")

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
    print("\nTip: Run with --data ch2_coords_san_francisco to use pre-generated dataset")


def main() -> None:
    """Run coordinate transformation examples."""
    parser = argparse.ArgumentParser(
        description="Chapter 2: Coordinate Transformation Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with inline generated data (default)
  python example_coordinate_transforms.py
  
  # Run with pre-generated dataset
  python example_coordinate_transforms.py --data ch2_coords_san_francisco
  
  # Specify full path to dataset
  python example_coordinate_transforms.py --data data/sim/ch2_coords_san_francisco
        """
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset name or path (e.g., 'ch2_coords_san_francisco' or full path)"
    )
    
    args = parser.parse_args()
    
    if args.data:
        # Resolve dataset path
        data_path = Path(args.data)
        if not data_path.exists():
            # Try prepending data/sim/
            data_path = Path("data/sim") / args.data
        if not data_path.exists():
            print(f"Error: Dataset not found at '{args.data}' or 'data/sim/{args.data}'")
            print("Available datasets:")
            sim_dir = Path("data/sim")
            if sim_dir.exists():
                for d in sorted(sim_dir.iterdir()):
                    if d.is_dir() and d.name.startswith("ch2"):
                        print(f"  - {d.name}")
            return
        
        run_with_dataset(str(data_path))
    else:
        run_with_inline_data()


if __name__ == "__main__":
    main()
