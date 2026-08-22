"""
Generate Ch2 Coordinate Transforms Dataset.

This script generates practical coordinate transformation scenarios for indoor
positioning applications. Demonstrates LLH→ECEF→ENU conversions and rotation
representations commonly used in navigation systems.

Key Learning Objectives:
    - Understand coordinate frame transformations
    - Learn when to use LLH vs. ECEF vs. ENU
    - Study rotation representations (Euler, Quaternion, Matrix)
    - Explore numerical precision in transformations

Implements Equations:
    - Eq. (2.1): LLH → ECEF transformation
    - Eq. (2.2): ECEF → LLH transformation (iterative)
    - Eq. (2.3): ECEF → ENU transformation
    - Eqs. (2.5-2.10): Rotation representations

Author: Li-Ta Hsu
Date: December 2024
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.coords import (
    llh_to_ecef,
    ecef_to_llh,
    ecef_to_enu,
    enu_to_llh_offset,
    euler_to_quat,
    euler_to_rotation_matrix,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
)


def generate_building_trajectory_llh(
    lat_center: float,
    lon_center: float,
    height_ground: float,
    building_size_m: float = 50.0,
    n_points: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate trajectory in LLH around a building.

    Args:
        lat_center: Building center latitude (radians).
        lon_center: Building center longitude (radians).
        height_ground: Ground height above ellipsoid (m).
        building_size_m: Building footprint size (m).
        n_points: Number of trajectory points.
        seed: Random seed.

    Returns:
        Tuple of (latitudes, longitudes, heights) in radians and meters.
    """
    rng = np.random.default_rng(seed)

    # Sample the footprint in metres, then convert to radians.
    #
    # This used to compute `lat_offset_deg = building_size_m / 111000.0` and add
    # it straight to `lat_center`, which is in radians -- the variable name says
    # `_deg` and no deg2rad ever ran. Every offset came out 180/pi = 57.3x too
    # large, so a declared 50 m building was sampled across 2.7 km, and the
    # dataset README wrote the symptom up as "ENU coordinates in km instead of
    # m" under a troubleshooting entry blaming the reader's reference point.
    #
    # `enu_to_llh_offset` takes metres and returns radians, so there is no
    # per-degree quantity left for anyone to add to a radian coordinate.
    # Drawn north-then-east to match the old lat-then-lon draw order, so the
    # only thing this change moves is the scale.
    half = building_size_m / 2.0
    north_m = rng.uniform(-half, half, n_points)
    east_m = rng.uniform(-half, half, n_points)
    offsets = np.array([
        enu_to_llh_offset(e, n, lat_center) for e, n in zip(east_m, north_m)
    ])

    # Generate random positions within building footprint
    lats = lat_center + offsets[:, 0]
    lons = lon_center + offsets[:, 1]

    # Heights: random floors (0-5) × 3m per floor
    heights = height_ground + rng.integers(0, 6, n_points) * 3.0

    return lats, lons, heights


def generate_rotation_sequence(
    n_points: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate random rotation sequence (Euler angles).

    Args:
        n_points: Number of rotation samples.
        seed: Random seed.

    Returns:
        Array of Euler angles [N×3] (roll, pitch, yaw) in radians.
    """
    rng = np.random.default_rng(seed)

    # Random rotations (typical for handheld device)
    # Roll/Pitch: ±30°, Yaw: full 360°
    roll = rng.uniform(-np.pi/6, np.pi/6, n_points)
    pitch = rng.uniform(-np.pi/6, np.pi/6, n_points)
    yaw = rng.uniform(0, 2*np.pi, n_points)

    return np.column_stack([roll, pitch, yaw])


def save_dataset(
    output_dir: Path,
    lat_center: float,
    lon_center: float,
    height_ground: float,
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    ecef: np.ndarray,
    enu: np.ndarray,
    euler: np.ndarray,
    quaternions: np.ndarray,
    rotation_matrices: np.ndarray,
    config: Dict,
) -> None:
    """Save coordinate transforms dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LLH coordinates
    llh = np.column_stack([lats, lons, heights])
    np.savetxt(
        output_dir / "llh_coordinates.txt",
        llh,
        fmt="%.10f %.10f %.3f",
        header="latitude (rad), longitude (rad), height (m)",
    )

    # Save ECEF coordinates
    np.savetxt(
        output_dir / "ecef_coordinates.txt",
        ecef,
        fmt="%.3f",
        header="X (m), Y (m), Z (m) in ECEF frame",
    )

    # Save ENU coordinates
    np.savetxt(
        output_dir / "enu_coordinates.txt",
        enu,
        fmt="%.3f",
        header="East (m), North (m), Up (m) relative to reference",
    )

    # Save reference point. The height must be the one the ENU transform was
    # built with (height_ground), not heights[0] -- that is just the first
    # sampled point, and it drew floor 5, so the file advertised a reference
    # 15 m above the origin the East/North/Up axes were actually computed
    # about. A reader loading reference_llh.txt and calling ecef_to_enu got Up
    # wrong by exactly that 15 m on every point, while East and North matched.
    ref_llh = np.array([lat_center, lon_center, height_ground])
    np.savetxt(
        output_dir / "reference_llh.txt",
        ref_llh.reshape(1, -1),
        fmt="%.10f %.10f %.3f",
        header="Reference point: latitude (rad), longitude (rad), height (m)",
    )

    # Save Euler angles
    np.savetxt(
        output_dir / "euler_angles.txt",
        euler,
        fmt="%.6f",
        header="roll (rad), pitch (rad), yaw (rad)",
    )

    # Save quaternions
    np.savetxt(
        output_dir / "quaternions.txt",
        quaternions,
        fmt="%.6f",
        header="qw, qx, qy, qz (unit quaternion)",
    )

    # Save rotation matrices (flattened)
    matrices_flat = rotation_matrices.reshape(len(rotation_matrices), -1)
    np.savetxt(
        output_dir / "rotation_matrices.txt",
        matrices_flat,
        fmt="%.6f",
        header="3x3 rotation matrices (flattened row-major)",
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved dataset to: {output_dir}")
    print("    Files: 8 files (LLH, ECEF, ENU, reference, Euler, quat, matrix, config)")
    print(f"    Points: {len(lats)}")


def generate_dataset(
    output_dir: str,
    preset: str = None,
    latitude_deg: float = 37.7749,
    longitude_deg: float = -122.4194,
    building_size: float = 50.0,
    n_points: int = 20,
    seed: int = 42,
) -> None:
    """
    Generate coordinate transforms dataset.

    Args:
        output_dir: Output directory path.
        preset: Preset configuration name.
        latitude_deg: Building center latitude (degrees).
        longitude_deg: Building center longitude (degrees).
        building_size: Building footprint size (m).
        n_points: Number of sample points.
        seed: Random seed.
    """
    # Apply preset if specified
    if preset == "san_francisco":
        latitude_deg = 37.7749
        longitude_deg = -122.4194
        output_dir = output_dir or "data/sim/ch2_coords_san_francisco"
    elif preset == "tokyo":
        latitude_deg = 35.6762
        longitude_deg = 139.6503
        output_dir = output_dir or "data/sim/ch2_coords_tokyo"
    elif preset == "london":
        latitude_deg = 51.5074
        longitude_deg = -0.1278
        output_dir = output_dir or "data/sim/ch2_coords_london"

    # No preset and no --output: the module's own dataset.
    output_dir = output_dir or "data/sim/ch2_coords_san_francisco"

    print("\n" + "=" * 70)
    print(f"Generating Ch2 Coordinate Transforms Dataset: {Path(output_dir).name}")
    print("=" * 70)

    # Convert to radians
    lat_center = np.deg2rad(latitude_deg)
    lon_center = np.deg2rad(longitude_deg)
    height_ground = 0.0  # At sea level for simplicity

    # Generate trajectory in LLH
    print("\nStep 1: Generating building trajectory (LLH)...")
    lats, lons, heights = generate_building_trajectory_llh(
        lat_center, lon_center, height_ground, building_size, n_points, seed
    )
    print(f"  Center: {latitude_deg:.4f}°N, {longitude_deg:.4f}°E")
    print(f"  Building size: {building_size}m × {building_size}m")
    print(f"  Points: {n_points}")
    print(f"  Height range: {heights.min():.1f}m to {heights.max():.1f}m")

    # Convert to ECEF
    print("\nStep 2: Converting LLH -> ECEF...")
    ecef = np.array([llh_to_ecef(lat, lon, h) for lat, lon, h in zip(lats, lons, heights)])
    print(f"  ECEF X range: {ecef[:, 0].min()/1e3:.1f}km to {ecef[:, 0].max()/1e3:.1f}km")
    print(f"  ECEF Y range: {ecef[:, 1].min()/1e3:.1f}km to {ecef[:, 1].max()/1e3:.1f}km")
    print(f"  ECEF Z range: {ecef[:, 2].min()/1e3:.1f}km to {ecef[:, 2].max()/1e3:.1f}km")

    # Convert to ENU
    print("\nStep 3: Converting ECEF -> ENU (local frame)...")
    enu = np.array([ecef_to_enu(pt[0], pt[1], pt[2],
                                  lat_center, lon_center, height_ground)
                    for pt in ecef])
    print(f"  ENU East range: {enu[:, 0].min():.1f}m to {enu[:, 0].max():.1f}m")
    print(f"  ENU North range: {enu[:, 1].min():.1f}m to {enu[:, 1].max():.1f}m")
    print(f"  ENU Up range: {enu[:, 2].min():.1f}m to {enu[:, 2].max():.1f}m")

    # Verify round-trip accuracy
    print("\nStep 4: Verifying round-trip accuracy...")
    llh_recovered = np.array([ecef_to_llh(pt[0], pt[1], pt[2]) for pt in ecef])
    lat_errors = np.abs(lats - llh_recovered[:, 0])
    lon_errors = np.abs(lons - llh_recovered[:, 1])
    height_errors = np.abs(heights - llh_recovered[:, 2])
    print(f"  Latitude error: max {np.rad2deg(lat_errors.max())*3600:.3e} arcsec")
    print(f"  Longitude error: max {np.rad2deg(lon_errors.max())*3600:.3e} arcsec")
    print(f"  Height error: max {height_errors.max():.3e} m")

    # Generate rotations
    print("\nStep 5: Generating rotation representations...")
    euler = generate_rotation_sequence(n_points, seed)
    quaternions = np.array([euler_to_quat(e[0], e[1], e[2]) for e in euler])
    rotation_matrices = np.array([euler_to_rotation_matrix(e[0], e[1], e[2]) for e in euler])
    print(f"  Euler angles: roll ±{np.rad2deg(np.abs(euler[:, 0]).max()):.1f}°, "
          f"pitch ±{np.rad2deg(np.abs(euler[:, 1]).max()):.1f}°, "
          f"yaw 0-{np.rad2deg(euler[:, 2].max()):.1f}°")

    # Verify rotation round-trip
    print("\nStep 6: Verifying rotation round-trips...")
    euler_from_quat = np.array([rotation_matrix_to_euler(quat_to_rotation_matrix(q))
                                  for q in quaternions])
    # Wrap the difference to [-pi, pi] before taking its size. Yaw is sampled on
    # [0, 2pi) but recovered on (-pi, pi], so an exact round-trip of 4.4307 rad
    # comes back as -1.8525 rad and a raw subtraction calls that 2pi of error.
    # This reported "rotation_roundtrip_deg": 360.0 -- which the dataset README
    # then explained as gimbal lock, and offered quaternions as the fix. It was
    # neither: wrapped, the error is 0.0, and the round-trip was always exact.
    euler_errors = np.abs((euler - euler_from_quat + np.pi) % (2 * np.pi) - np.pi)
    print(f"  Euler <-> Quaternion error: max {np.rad2deg(euler_errors.max()):.3e} deg")

    # Save dataset
    config = {
        "dataset": "ch2_coordinate_transforms",
        "preset": preset,
        "reference_point": {
            "latitude_deg": latitude_deg,
            "longitude_deg": longitude_deg,
            "height_m": height_ground,
            "location": preset if preset else "custom",
        },
        "building": {
            "size_m": building_size,
            "num_points": n_points,
        },
        "accuracy": {
            "llh_roundtrip_lat_arcsec": float(np.rad2deg(lat_errors.max()) * 3600),
            "llh_roundtrip_lon_arcsec": float(np.rad2deg(lon_errors.max()) * 3600),
            "llh_roundtrip_height_m": float(height_errors.max()),
            "rotation_roundtrip_deg": float(np.rad2deg(euler_errors.max())),
        },
        "equations": [
            "2.1 (LLH->ECEF)",
            "2.2 (ECEF->LLH)",
            "2.3 (ECEF->ENU)",
            "2.5-2.10 (Rotations)"
        ],
        "seed": seed,
    }

    save_dataset(
        Path(output_dir),
        lat_center,
        lon_center,
        height_ground,
        lats,
        lons,
        heights,
        ecef,
        enu,
        euler,
        quaternions,
        rotation_matrices,
        config,
    )

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Ch2 Coordinate Transforms Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  san_francisco   San Francisco (37.77°N, 122.42°W)
  tokyo           Tokyo (35.68°N, 139.65°E)
  london          London (51.51°N, 0.13°W)

Examples:
  # Generate all presets for different locations
  python scripts/generate_ch2_coordinate_transforms_dataset.py --preset san_francisco
  python scripts/generate_ch2_coordinate_transforms_dataset.py --preset tokyo
  python scripts/generate_ch2_coordinate_transforms_dataset.py --preset london

  # Custom location
  python scripts/generate_ch2_coordinate_transforms_dataset.py \\
      --output data/sim/my_coords \\
      --latitude 40.7128 \\
      --longitude -74.0060

Learning Focus:
  - LLH (geodetic) -> ECEF (Cartesian) -> ENU (local)
  - Round-trip accuracy (numerical precision matters!)
  - Rotation representations: Euler <-> Quaternion <-> Matrix
  - Practical indoor positioning coordinate frames

Book Reference: Chapter 2, Sections 2.1-2.3
        """,
    )

    # Preset or custom
    parser.add_argument(
        "--preset",
        type=str,
        choices=["san_francisco", "tokyo", "london"],
        help="Use preset location (overrides lat/lon)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/sim/ch2_coords_san_francisco)",
    )

    # Location parameters
    loc_group = parser.add_argument_group("Location Parameters")
    loc_group.add_argument(
        "--latitude", type=float, default=37.7749, help="Center latitude in degrees (default: 37.7749)"
    )
    loc_group.add_argument(
        "--longitude", type=float, default=-122.4194, help="Center longitude in degrees (default: -122.4194)"
    )

    # Building parameters
    building_group = parser.add_argument_group("Building Parameters")
    building_group.add_argument(
        "--building-size", type=float, default=50.0, help="Building footprint size in meters (default: 50.0)"
    )
    building_group.add_argument(
        "--n-points", type=int, default=20, help="Number of sample points (default: 20)"
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        preset=args.preset,
        latitude_deg=args.latitude,
        longitude_deg=args.longitude,
        building_size=args.building_size,
        n_points=args.n_points,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

