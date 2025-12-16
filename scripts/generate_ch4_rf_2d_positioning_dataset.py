"""
Generate Ch4 RF 2D Positioning Dataset.

This script generates synthetic RF positioning datasets demonstrating TOA, TDOA,
and AOA techniques with various beacon geometries. Shows the critical impact of
geometry on DOP (Dilution of Precision) and positioning accuracy.

Key Learning Objectives:
    - Understand geometric DOP and its impact on accuracy
    - Compare TOA vs. TDOA vs. AOA positioning
    - Learn the effect of beacon placement
    - Study NLOS (Non-Line-of-Sight) impact
    - Explore measurement noise effects

Implements Equations:
    - Eq. (4.1-4.3): TOA range measurements
    - Eq. (4.27-4.33): TDOA range differences
    - Eq. (4.63-4.66): AOA angle measurements
    - Section 4.5: DOP calculations

Author: Navigation Engineer
Date: December 2024
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rf import (
    toa_range,
    tdoa_range_difference,
    aoa_azimuth,
    compute_geometry_matrix,
    compute_dop,
    TOAPositioner,
    TDOAPositioner,
    AOAPositioner,
)


def generate_trajectory(
    trajectory_type: str = "grid",
    area_size: float = 20.0,
    num_points: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate 2D trajectory for positioning evaluation.

    Args:
        trajectory_type: Type of trajectory ('grid', 'random', 'circle', 'corridor').
        area_size: Size of area in meters.
        num_points: Number of evaluation points.
        seed: Random seed.

    Returns:
        Array of 2D positions [N, 2] in meters.
    """
    rng = np.random.default_rng(seed)

    if trajectory_type == "grid":
        # Uniform grid
        grid_size = int(np.sqrt(num_points))
        x = np.linspace(2, area_size - 2, grid_size)
        y = np.linspace(2, area_size - 2, grid_size)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])[:num_points]

    elif trajectory_type == "random":
        # Random walk
        positions = rng.uniform(2, area_size - 2, (num_points, 2))

    elif trajectory_type == "circle":
        # Circular path
        center = np.array([area_size / 2, area_size / 2])
        radius = area_size / 3
        theta = np.linspace(0, 2 * np.pi, num_points)
        positions = center + radius * np.column_stack([np.cos(theta), np.sin(theta)])

    elif trajectory_type == "corridor":
        # Corridor walk (back and forth)
        y_center = area_size / 2
        x = np.linspace(2, area_size - 2, num_points)
        y = np.ones(num_points) * y_center
        positions = np.column_stack([x, y])

    else:
        raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

    return positions


def create_beacon_geometry(
    geometry_type: str = "square",
    area_size: float = 20.0,
) -> np.ndarray:
    """
    Create beacon geometry.

    Args:
        geometry_type: Type of geometry ('square', 'optimal', 'linear', 'lshape', 'poor').
        area_size: Size of area in meters.

    Returns:
        Beacon positions [N_beacons, 2] in meters.
    """
    if geometry_type == "square":
        # Beacons at corners (good GDOP in center)
        beacons = np.array([
            [0, 0],
            [area_size, 0],
            [area_size, area_size],
            [0, area_size]
        ], dtype=float)

    elif geometry_type == "optimal":
        # Beacons optimally placed (tetrahedral-like in 2D)
        center = area_size / 2
        radius = area_size / 2
        angles = np.linspace(0, 2 * np.pi, 5)[:-1]  # 4 beacons evenly spaced
        beacons = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])

    elif geometry_type == "linear":
        # Linear array (poor GDOP perpendicular to line)
        beacons = np.array([
            [area_size * 0.2, area_size / 2],
            [area_size * 0.4, area_size / 2],
            [area_size * 0.6, area_size / 2],
            [area_size * 0.8, area_size / 2]
        ], dtype=float)

    elif geometry_type == "lshape":
        # L-shaped array (poor GDOP in some regions)
        beacons = np.array([
            [0, 0],
            [area_size / 2, 0],
            [area_size, 0],
            [0, area_size / 2]
        ], dtype=float)

    elif geometry_type == "poor":
        # Clustered beacons (very poor GDOP)
        center = np.array([area_size * 0.3, area_size * 0.3])
        offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        beacons = center + offsets

    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type}")

    return beacons


def generate_measurements(
    beacons: np.ndarray,
    positions: np.ndarray,
    toa_noise: float = 0.1,
    tdoa_noise: float = 0.1,
    aoa_noise_deg: float = 2.0,
    nlos_beacons: Optional[List[int]] = None,
    nlos_bias: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate noisy RF measurements.

    Args:
        beacons: Beacon positions [N_beacons, 2] m.
        positions: Agent positions [N_positions, 2] m.
        toa_noise: TOA range noise std dev (m).
        tdoa_noise: TDOA range difference noise std dev (m).
        aoa_noise_deg: AOA angle noise std dev (degrees).
        nlos_beacons: List of beacon indices with NLOS bias.
        nlos_bias: NLOS positive bias (m).
        seed: Random seed.

    Returns:
        Tuple of (toa_ranges, tdoa_diffs, aoa_angles):
            - toa_ranges: [N_positions, N_beacons] in meters
            - tdoa_diffs: [N_positions, N_beacons-1] in meters
            - aoa_angles: [N_positions, N_beacons] in radians
    """
    rng = np.random.default_rng(seed)
    N_pos = len(positions)
    N_beacons = len(beacons)

    # Initialize arrays
    toa_ranges = np.zeros((N_pos, N_beacons))
    tdoa_diffs = np.zeros((N_pos, N_beacons - 1))
    aoa_angles = np.zeros((N_pos, N_beacons))

    # Generate measurements for each position
    for i, pos in enumerate(positions):
        # TOA ranges
        for j, beacon in enumerate(beacons):
            true_range = toa_range(beacon, pos)
            noise = rng.normal(0, toa_noise)
            
            # Add NLOS bias if applicable
            bias = nlos_bias if (nlos_beacons and j in nlos_beacons) else 0.0
            
            toa_ranges[i, j] = true_range + noise + bias

        # TDOA range differences (relative to first beacon)
        for j in range(1, N_beacons):
            true_diff = tdoa_range_difference(beacons[0], beacons[j], pos)
            noise = rng.normal(0, tdoa_noise)
            tdoa_diffs[i, j - 1] = true_diff + noise

        # AOA angles
        for j, beacon in enumerate(beacons):
            true_angle = aoa_azimuth(beacon, pos)
            noise_rad = np.deg2rad(rng.normal(0, aoa_noise_deg))
            aoa_angles[i, j] = true_angle + noise_rad

    return toa_ranges, tdoa_diffs, aoa_angles


def compute_dop_metrics(
    beacons: np.ndarray,
    positions: np.ndarray,
    measurement_type: str = "toa",
) -> np.ndarray:
    """
    Compute DOP metrics for all positions.

    Args:
        beacons: Beacon positions [N_beacons, 2].
        positions: Agent positions [N_positions, 2].
        measurement_type: Type ('toa', 'tdoa', 'aoa').

    Returns:
        GDOP values [N_positions].
    """
    gdop_values = np.zeros(len(positions))

    for i, pos in enumerate(positions):
        H = compute_geometry_matrix(beacons, pos, measurement_type)
        dop_dict = compute_dop(H)
        gdop_values[i] = dop_dict["GDOP"]

    return gdop_values


def run_positioning(
    beacons: np.ndarray,
    toa_ranges: np.ndarray,
    tdoa_diffs: np.ndarray,
    aoa_angles: np.ndarray,
    true_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run TOA, TDOA, and AOA positioning.

    Args:
        beacons: Beacon positions [N_beacons, 2].
        toa_ranges: TOA measurements [N_positions, N_beacons].
        tdoa_diffs: TDOA measurements [N_positions, N_beacons-1].
        aoa_angles: AOA measurements [N_positions, N_beacons].
        true_positions: True positions [N_positions, 2].

    Returns:
        Tuple of (toa_pos, tdoa_pos, aoa_pos) estimated positions.
    """
    N_pos = len(true_positions)

    toa_pos = np.zeros((N_pos, 2))
    tdoa_pos = np.zeros((N_pos, 2))
    aoa_pos = np.zeros((N_pos, 2))

    # TOA positioning
    toa_solver = TOAPositioner(beacons, method="iwls")
    for i in range(N_pos):
        try:
            pos_est, _ = toa_solver.solve(toa_ranges[i], initial_guess=true_positions[i] + 1.0)
            toa_pos[i] = pos_est
        except:
            toa_pos[i] = true_positions[i]  # Fallback

    # TDOA positioning
    tdoa_solver = TDOAPositioner(beacons, reference_idx=0)
    for i in range(N_pos):
        try:
            pos_est, _ = tdoa_solver.solve(tdoa_diffs[i], initial_guess=true_positions[i] + 1.0)
            tdoa_pos[i] = pos_est
        except:
            tdoa_pos[i] = true_positions[i]  # Fallback

    # AOA positioning
    aoa_solver = AOAPositioner(beacons)
    for i in range(N_pos):
        try:
            pos_est, _ = aoa_solver.solve(aoa_angles[i], initial_guess=true_positions[i] + 1.0)
            aoa_pos[i] = pos_est
        except:
            aoa_pos[i] = true_positions[i]  # Fallback

    return toa_pos, tdoa_pos, aoa_pos


def save_dataset(
    output_dir: Path,
    beacons: np.ndarray,
    positions: np.ndarray,
    toa_ranges: np.ndarray,
    tdoa_diffs: np.ndarray,
    aoa_angles: np.ndarray,
    gdop_toa: np.ndarray,
    gdop_tdoa: np.ndarray,
    gdop_aoa: np.ndarray,
    config: Dict,
) -> None:
    """Save dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save beacon positions
    np.savetxt(
        output_dir / "beacons.txt",
        beacons,
        fmt="%.6f",
        header="x (m), y (m)",
    )

    # Save ground truth positions
    np.savetxt(
        output_dir / "ground_truth_positions.txt",
        positions,
        fmt="%.6f",
        header="x (m), y (m)",
    )

    # Save measurements
    np.savetxt(
        output_dir / "toa_ranges.txt",
        toa_ranges,
        fmt="%.6f",
        header=f"ranges to {len(beacons)} beacons (m)",
    )
    np.savetxt(
        output_dir / "tdoa_diffs.txt",
        tdoa_diffs,
        fmt="%.6f",
        header=f"TDOA range differences relative to beacon 0 (m)",
    )
    np.savetxt(
        output_dir / "aoa_angles.txt",
        aoa_angles,
        fmt="%.6f",
        header=f"AOA angles from {len(beacons)} beacons (rad)",
    )

    # Save DOP metrics
    np.savetxt(
        output_dir / "gdop_toa.txt",
        gdop_toa,
        fmt="%.6f",
        header="GDOP for TOA",
    )
    np.savetxt(
        output_dir / "gdop_tdoa.txt",
        gdop_tdoa,
        fmt="%.6f",
        header="GDOP for TDOA",
    )
    np.savetxt(
        output_dir / "gdop_aoa.txt",
        gdop_aoa,
        fmt="%.6f",
        header="GDOP for AOA",
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved dataset to: {output_dir}")
    print(f"    Files: 8 files (beacons, positions, 3 measurements, 3 GDOP, config)")
    print(f"    Positions: {len(positions)}")
    print(f"    Beacons: {len(beacons)}")


def generate_dataset(
    output_dir: str,
    preset: Optional[str] = None,
    geometry: str = "square",
    trajectory: str = "grid",
    area_size: float = 20.0,
    num_points: int = 100,
    toa_noise: float = 0.1,
    tdoa_noise: float = 0.1,
    aoa_noise_deg: float = 2.0,
    add_nlos: bool = False,
    nlos_bias: float = 0.5,
    seed: int = 42,
) -> None:
    """
    Generate RF 2D positioning dataset.

    Args:
        output_dir: Output directory path.
        preset: Preset configuration name.
        geometry: Beacon geometry type.
        trajectory: Trajectory type.
        area_size: Area size (m).
        num_points: Number of evaluation points.
        toa_noise: TOA noise (m).
        tdoa_noise: TDOA noise (m).
        aoa_noise_deg: AOA noise (degrees).
        add_nlos: Add NLOS bias.
        nlos_bias: NLOS bias magnitude (m).
        seed: Random seed.
    """
    # Apply preset if specified
    if preset == "baseline":
        geometry = "square"
        trajectory = "grid"
        toa_noise = 0.1
        tdoa_noise = 0.1
        aoa_noise_deg = 2.0
        add_nlos = False
        output_dir = "data/sim/ch4_rf_2d_square"
    elif preset == "optimal":
        geometry = "optimal"
        trajectory = "grid"
        toa_noise = 0.1
        tdoa_noise = 0.1
        aoa_noise_deg = 2.0
        add_nlos = False
        output_dir = "data/sim/ch4_rf_2d_optimal"
    elif preset == "poor_geometry":
        geometry = "linear"
        trajectory = "grid"
        toa_noise = 0.1
        tdoa_noise = 0.1
        aoa_noise_deg = 2.0
        add_nlos = False
        output_dir = "data/sim/ch4_rf_2d_linear"
    elif preset == "nlos":
        geometry = "square"
        trajectory = "grid"
        toa_noise = 0.1
        tdoa_noise = 0.1
        aoa_noise_deg = 2.0
        add_nlos = True
        nlos_bias = 0.8
        output_dir = "data/sim/ch4_rf_2d_nlos"

    print("\n" + "=" * 70)
    print(f"Generating Ch4 RF 2D Positioning Dataset: {Path(output_dir).name}")
    print("=" * 70)

    # Create beacon geometry
    print("\nStep 1: Creating beacon geometry...")
    beacons = create_beacon_geometry(geometry, area_size)
    print(f"  Geometry: {geometry}")
    print(f"  Beacons: {len(beacons)}")
    print(f"  Area: {area_size}m x {area_size}m")

    # Generate trajectory
    print("\nStep 2: Generating trajectory...")
    positions = generate_trajectory(trajectory, area_size, num_points, seed)
    print(f"  Trajectory: {trajectory}")
    print(f"  Points: {len(positions)}")

    # Determine NLOS beacons
    nlos_beacons = [1, 2] if add_nlos else None

    # Generate measurements
    print("\nStep 3: Generating RF measurements...")
    print(f"  TOA noise: {toa_noise:.3f} m")
    print(f"  TDOA noise: {tdoa_noise:.3f} m")
    print(f"  AOA noise: {aoa_noise_deg:.1f} deg")
    print(f"  NLOS: {'YES' if add_nlos else 'NO'}")
    if add_nlos:
        print(f"  NLOS bias: {nlos_bias:.2f} m on beacons {nlos_beacons}")

    start = time.time()
    toa_ranges, tdoa_diffs, aoa_angles = generate_measurements(
        beacons, positions, toa_noise, tdoa_noise, aoa_noise_deg,
        nlos_beacons, nlos_bias, seed
    )
    elapsed = time.time() - start
    print(f"  Generation time: {elapsed:.3f} s")

    # Compute DOP metrics
    print("\nStep 4: Computing DOP metrics...")
    start = time.time()
    gdop_toa = compute_dop_metrics(beacons, positions, "toa")
    gdop_tdoa = compute_dop_metrics(beacons, positions, "tdoa")
    gdop_aoa = compute_dop_metrics(beacons, positions, "aoa")
    elapsed = time.time() - start

    print(f"  Computation time: {elapsed:.3f} s")
    print(f"  TOA GDOP: mean={gdop_toa.mean():.2f}, min={gdop_toa.min():.2f}, max={gdop_toa.max():.2f}")
    print(f"  TDOA GDOP: mean={gdop_tdoa.mean():.2f}, min={gdop_tdoa.min():.2f}, max={gdop_tdoa.max():.2f}")
    print(f"  AOA GDOP: mean={gdop_aoa.mean():.2f}, min={gdop_aoa.min():.2f}, max={gdop_aoa.max():.2f}")

    # Run positioning
    print("\nStep 5: Running positioning algorithms...")
    start = time.time()
    toa_pos, tdoa_pos, aoa_pos = run_positioning(
        beacons, toa_ranges, tdoa_diffs, aoa_angles, positions
    )
    elapsed = time.time() - start

    # Compute errors
    toa_errors = np.linalg.norm(toa_pos - positions, axis=1)
    tdoa_errors = np.linalg.norm(tdoa_pos - positions, axis=1)
    aoa_errors = np.linalg.norm(aoa_pos - positions, axis=1)

    print(f"  Positioning time: {elapsed:.3f} s")
    print(f"\nPositioning Errors:")
    print(f"  TOA:  mean={toa_errors.mean():.3f}m, std={toa_errors.std():.3f}m, max={toa_errors.max():.3f}m")
    print(f"  TDOA: mean={tdoa_errors.mean():.3f}m, std={tdoa_errors.std():.3f}m, max={tdoa_errors.max():.3f}m")
    print(f"  AOA:  mean={aoa_errors.mean():.3f}m, std={aoa_errors.std():.3f}m, max={aoa_errors.max():.3f}m")

    # Save dataset
    config = {
        "dataset": "ch4_rf_2d_positioning",
        "preset": preset,
        "geometry": {
            "type": geometry,
            "num_beacons": len(beacons),
            "area_size_m": area_size,
        },
        "trajectory": {
            "type": trajectory,
            "num_points": len(positions),
        },
        "measurements": {
            "toa_noise_std_m": toa_noise,
            "tdoa_noise_std_m": tdoa_noise,
            "aoa_noise_std_deg": aoa_noise_deg,
        },
        "nlos": {
            "enabled": add_nlos,
            "beacon_indices": nlos_beacons if add_nlos else [],
            "bias_m": nlos_bias if add_nlos else 0.0,
        },
        "dop": {
            "toa": {
                "mean": float(gdop_toa.mean()),
                "min": float(gdop_toa.min()),
                "max": float(gdop_toa.max()),
            },
            "tdoa": {
                "mean": float(gdop_tdoa.mean()),
                "min": float(gdop_tdoa.min()),
                "max": float(gdop_tdoa.max()),
            },
            "aoa": {
                "mean": float(gdop_aoa.mean()),
                "min": float(gdop_aoa.min()),
                "max": float(gdop_aoa.max()),
            },
        },
        "performance": {
            "toa_error_mean_m": float(toa_errors.mean()),
            "tdoa_error_mean_m": float(tdoa_errors.mean()),
            "aoa_error_mean_m": float(aoa_errors.mean()),
        },
        "equations": ["4.1-4.3", "4.27-4.33", "4.63-4.66", "4.5 (DOP)"],
        "seed": seed,
    }

    save_dataset(
        Path(output_dir),
        beacons,
        positions,
        toa_ranges,
        tdoa_diffs,
        aoa_angles,
        gdop_toa,
        gdop_tdoa,
        gdop_aoa,
        config,
    )

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Ch4 RF 2D Positioning Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  baseline        Square geometry, clean measurements (GDOP ~2-3)
  optimal         Optimal beacon placement (GDOP ~1.5-2)
  poor_geometry   Linear array (GDOP >10 in some regions)
  nlos            Square + NLOS bias on 2 beacons

Examples:
  # Generate baseline dataset
  python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset baseline

  # Generate with custom geometry
  python scripts/generate_ch4_rf_2d_positioning_dataset.py \\
      --output data/sim/my_rf \\
      --geometry optimal \\
      --toa-noise 0.2

  # Generate all presets
  python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset baseline
  python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset optimal
  python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset poor_geometry
  python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset nlos

Learning Focus:
  - Geometry is CRITICAL for positioning accuracy (DOP varies 10×!)
  - TOA, TDOA, AOA have different strengths/weaknesses
  - NLOS bias degrades all techniques (but differently)
  - Optimal beacon placement minimizes GDOP

Book Reference: Chapter 4, Sections 4.1-4.5
        """,
    )

    # Preset or custom
    parser.add_argument(
        "--preset",
        type=str,
        choices=["baseline", "optimal", "poor_geometry", "nlos"],
        help="Use preset configuration (overrides other parameters)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sim/ch4_rf_2d_square",
        help="Output directory (default: data/sim/ch4_rf_2d_square)",
    )

    # Geometry parameters
    geom_group = parser.add_argument_group("Geometry Parameters")
    geom_group.add_argument(
        "--geometry",
        type=str,
        choices=["square", "optimal", "linear", "lshape", "poor"],
        default="square",
        help="Beacon geometry type (default: square)",
    )
    geom_group.add_argument(
        "--area-size", type=float, default=20.0, help="Area size in meters (default: 20.0)"
    )

    # Trajectory parameters
    traj_group = parser.add_argument_group("Trajectory Parameters")
    traj_group.add_argument(
        "--trajectory",
        type=str,
        choices=["grid", "random", "circle", "corridor"],
        default="grid",
        help="Trajectory type (default: grid)",
    )
    traj_group.add_argument(
        "--num-points", type=int, default=100, help="Number of evaluation points (default: 100)"
    )

    # Measurement noise parameters
    noise_group = parser.add_argument_group("Measurement Noise Parameters")
    noise_group.add_argument(
        "--toa-noise", type=float, default=0.1, help="TOA noise std dev in meters (default: 0.1)"
    )
    noise_group.add_argument(
        "--tdoa-noise", type=float, default=0.1, help="TDOA noise std dev in meters (default: 0.1)"
    )
    noise_group.add_argument(
        "--aoa-noise", type=float, default=2.0, help="AOA noise std dev in degrees (default: 2.0)"
    )

    # NLOS parameters
    nlos_group = parser.add_argument_group("NLOS Parameters")
    nlos_group.add_argument(
        "--add-nlos", action="store_true", help="Add NLOS bias to beacons 1 and 2"
    )
    nlos_group.add_argument(
        "--nlos-bias", type=float, default=0.5, help="NLOS bias in meters (default: 0.5)"
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        preset=args.preset,
        geometry=args.geometry,
        trajectory=args.trajectory,
        area_size=args.area_size,
        num_points=args.num_points,
        toa_noise=args.toa_noise,
        tdoa_noise=args.tdoa_noise,
        aoa_noise_deg=args.aoa_noise,
        add_nlos=args.add_nlos,
        nlos_bias=args.nlos_bias,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

