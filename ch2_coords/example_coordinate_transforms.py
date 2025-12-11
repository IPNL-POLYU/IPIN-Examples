"""Example: Coordinate transformations between LLH, ECEF, and ENU frames.

This example demonstrates the use of coordinate transformations for
indoor positioning applications:
1. Convert geodetic coordinates (LLH) to ECEF
2. Convert ECEF to local ENU frame
3. Convert rotation representations (Euler, quaternions, matrices)

Reference: Chapter 2 - Coordinate Systems
"""

import numpy as np

from core.coords import (
    ecef_to_enu,
    ecef_to_llh,
    enu_to_ecef,
    euler_to_quat,
    euler_to_rotation_matrix,
    llh_to_ecef,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
)


def main() -> None:
    """Run coordinate transformation examples."""
    print("=" * 70)
    print("Chapter 2: Coordinate Transformation Examples")
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

    # Example 6: Round-trip rotation conversions
    print("\n6. Round-trip Rotation Conversions")
    print("-" * 70)

    # Euler -> matrix -> Euler
    R_from_euler = euler_to_rotation_matrix(roll, pitch, yaw)
    euler_recovered = rotation_matrix_to_euler(R_from_euler)

    print(f"Original Euler: [{np.rad2deg(roll):.1f}°, "
          f"{np.rad2deg(pitch):.1f}°, {np.rad2deg(yaw):.1f}°]")
    print(f"Recovered Euler: [{np.rad2deg(euler_recovered[0]):.1f}°, "
          f"{np.rad2deg(euler_recovered[1]):.1f}°, "
          f"{np.rad2deg(euler_recovered[2]):.1f}°]")

    # Example 7: Coordinate frame conversions
    print("\n7. Practical Indoor Positioning Scenario")
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


if __name__ == "__main__":
    main()

