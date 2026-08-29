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

Author: Li-Ta Hsu
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
    build_tdoa_covariance,
    solve_batch,
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
        beacons = np.array(
            [[0, 0], [area_size, 0], [area_size, area_size], [0, area_size]],
            dtype=float,
        )

    elif geometry_type == "optimal":
        # Beacons optimally placed (tetrahedral-like in 2D)
        center = area_size / 2
        radius = area_size / 2
        angles = np.linspace(0, 2 * np.pi, 5)[:-1]  # 4 beacons evenly spaced
        beacons = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])

    elif geometry_type == "linear":
        # Linear array (poor GDOP perpendicular to line)
        beacons = np.array(
            [
                [area_size * 0.2, area_size / 2],
                [area_size * 0.4, area_size / 2],
                [area_size * 0.6, area_size / 2],
                [area_size * 0.8, area_size / 2],
            ],
            dtype=float,
        )

    elif geometry_type == "lshape":
        # L-shaped array (poor GDOP in some regions)
        beacons = np.array(
            [[0, 0], [area_size / 2, 0], [area_size, 0], [0, area_size / 2]],
            dtype=float,
        )

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
    nlos_bias_deg: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate noisy RF measurements.

    Args:
        beacons: Beacon positions [N_beacons, 2] m.
        positions: Agent positions [N_positions, 2] m.
        toa_noise: TOA range noise std dev (m).
        tdoa_noise: TDOA *arrival-time* noise std dev (m), per beacon -- not
            per difference. Each difference carries two of these errors and
            so has std `sqrt(2) * tdoa_noise`, and any two of them correlate
            at rho = 0.5 through the shared reference beacon (Eq. 4.42).
        aoa_noise_deg: AOA angle noise std dev (degrees).
        nlos_beacons: List of beacon indices with NLOS bias.
        nlos_bias: NLOS positive *range* bias (m).
        nlos_bias_deg: NLOS *bearing* bias (degrees) on the same beacons. See
            the block comment in the AOA loop below for where its size comes
            from and why the model is phenomenological.
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

    # A TOA range and a TDOA difference are two views of one arrival-time
    # measurement, so there is one error process per beacon and `--tdoa-noise`
    # can only rescale it. Every preset sets the two equal, where the scale is
    # exactly 1 and `tdoa_diffs.txt` is `toa_ranges.txt` differenced against
    # its own first column.
    if toa_noise > 0:
        tdoa_error_scale = tdoa_noise / toa_noise
    elif tdoa_noise > 0:
        raise ValueError(
            "tdoa_noise > 0 needs toa_noise > 0: the range differences are "
            "built from the same per-beacon arrival-time errors the TOA "
            "ranges carry, so a noiseless TOA chain cannot produce noisy "
            "differences. Raise --toa-noise, or pass --tdoa-noise 0."
        )
    else:
        tdoa_error_scale = 0.0

    # Initialize arrays
    toa_ranges = np.zeros((N_pos, N_beacons))
    tdoa_diffs = np.zeros((N_pos, N_beacons - 1))
    aoa_angles = np.zeros((N_pos, N_beacons))

    # Generate measurements for each position
    for i, pos in enumerate(positions):
        # One arrival-time error and one NLOS bias per beacon, kept rather
        # than folded straight into `toa_ranges[i]` because the TDOA block
        # below is formed from the same numbers.
        range_errors = np.zeros(N_beacons)
        biases = np.zeros(N_beacons)
        # The same NLOS event seen by a bearing rather than by a clock. Set
        # here, beside the range bias, so that "which beacons are blocked" is
        # written down once and the two corruptions cannot name different
        # beacons.
        angle_biases_deg = np.zeros(N_beacons)

        # TOA ranges
        for j, beacon in enumerate(beacons):
            true_range = toa_range(beacon, pos)
            range_errors[j] = rng.normal(0, toa_noise)

            # Add NLOS bias if applicable
            blocked = bool(nlos_beacons) and j in nlos_beacons
            biases[j] = nlos_bias if blocked else 0.0
            angle_biases_deg[j] = nlos_bias_deg if blocked else 0.0

            toa_ranges[i, j] = true_range + range_errors[j] + biases[j]

        # TDOA range differences, d_j - d_ref, with beacon 0 as reference.
        # The argument order matters and used to be the other way round:
        # `tdoa_range_difference(anchor_i, anchor_j, pos)` returns d_i - d_j,
        # so passing (beacons[0], beacons[j]) stored -(d_j - d_0) while
        # `TDOAPositioner(reference_idx=0)` and Eqs. (4.34) onward both predict
        # d_j - d_0. Solving a negated range difference asks for the branch of
        # the hyperbola on the far side of the array: it cost a factor of 158
        # on the square geometry (13.753 m against the 0.087 m its GDOP
        # predicts) until it was corrected.
        #
        # The *error* on each difference is `e_j - e_0`, taken from the same
        # per-beacon draws the TOA ranges above carry. Every difference is
        # formed against the same reference beacon, so `e_0` is common mode:
        # it does not average away, and the differences are correlated with
        # one another (Eq. 4.42, and `build_tdoa_covariance`):
        #
        #     Var(z_j) = 2 sigma^2   Cov(z_j, z_k) = sigma^2   rho = 0.5
        #
        # This block used to draw an independent `rng.normal(0, tdoa_noise)`
        # per difference, which deletes the common term and hands TDOA
        # information it cannot physically have. It made TDOA's mean GDOP on
        # the square array read 0.8730 against a true 1.0665, and its median
        # error 0.0746 m against 0.0923 m -- beating TOA's 0.0881 m, which no
        # geometry can do. Differencing is a projection: it throws the
        # receiver clock away and cannot add information. TDOA and
        # TOA-with-an-estimated-clock are worth the same, and TDOA's real
        # trade is that it needs no synchronised clock at all.
        for j in range(1, N_beacons):
            true_diff = tdoa_range_difference(beacons[j], beacons[0], pos)
            tdoa_diffs[i, j - 1] = (
                true_diff
                + tdoa_error_scale * (range_errors[j] - range_errors[0])
                + (biases[j] - biases[0])
            )

        # Reproducibility, not physics. The model above consumes no random
        # numbers of its own, where the independent-noise model it replaces
        # consumed one per difference. Drawing and discarding them leaves the
        # AOA block below at the stream position it has always been at, so
        # `toa_ranges.txt` and `aoa_angles.txt` stay byte-identical across
        # this correction -- which is the evidence that it is confined to
        # TDOA. Delete this line and only the AOA files move.
        rng.normal(0, tdoa_noise, size=N_beacons - 1)

        # AOA angles.
        #
        # NLOS reaches the bearing too, and until recently it did not: the
        # bias was applied only as an additive *range* error, so
        # `aoa_angles.txt` was byte-identical between `ch4_rf_2d_square` and
        # `ch4_rf_2d_nlos` while `toa_ranges.txt` and `tdoa_diffs.txt` both
        # differed. Experiment 3 therefore taught that AOA is immune to NLOS,
        # which is false and is the opposite of the point the dataset exists
        # to make. A blocked direct path does not lengthen a bearing; the
        # signal arrives from the direction of the reflection, so the azimuth
        # is wrong by an *angle*.
        #
        # (The three `gdop_*.txt` files stay identical across that pair, and
        # that is correct rather than a second instance of this bug: DOP is a
        # function of the beacon geometry and the query point, both of which
        # this dataset shares with its baseline byte for byte. "DOP cannot see
        # a bias" is the lesson the NLOS README is built on.)
        #
        # **This is a phenomenological model, not a derived one**, and the
        # reason is worth stating rather than hiding. A real image-source
        # model determines the range excess and the bearing offset *together*
        # from the reflector's position -- but this dataset has no walls, only
        # beacons and an area size, and the range bias it already ships is a
        # *constant* 0.8 m, which no fixed reflector produces (a fixed wall
        # gives an excess that varies with where the agent stands). So the
        # angular term mirrors the range term exactly: one constant per
        # blocked beacon, which keeps the pair usable as the controlled
        # experiment its README describes.
        #
        # **The magnitude is derived even though the model is not.** Take the
        # textbook single-bounce geometry: a wall parallel to the direct path
        # at perpendicular offset w, so the image beacon sits 2w to the side.
        # At true range d,
        #
        #     excess path   dd    = sqrt(d^2 + 4w^2) - d
        #     bearing shift dpsi  = atan(2w / d)
        #
        # Invert the first for w at the declared dd = 0.8 m and read the
        # second. The agent-to-blocked-beacon range on this grid has median
        # 15.54 m, giving w = 2.53 m and **dpsi = 18.0 deg**, which is what
        # the `nlos` preset sets. Across the full range spread (2.83 to
        # 25.46 m) the relation gives 38.8 down to 14.2 deg, so 18 deg is the
        # median-range representative of a band, not a sharp value.
        #
        # The number is large, and that is the physics rather than a mistake:
        # **path length is second-order in the reflector offset while bearing
        # is first-order.** A bounce that barely lengthens the path can throw
        # the bearing badly, which is exactly why NLOS is harder for AOA than
        # for TOA. A second, independent calibration agrees: the 0.8 m bias
        # degrades the TOA median by 7.0x (0.088 -> 0.614 m), and matching
        # that factor on AOA (0.397 -> 2.77 m) needs 15.1 deg. Two arguments,
        # one physical and one pedagogical, landing within 20% of each other.
        #
        # Measured effect on `ch4_rf_2d_nlos`: AOA median error 0.3971 m ->
        # 3.2530 m, with 10 of 100 fixes no longer converging. Both are
        # reported rather than hidden -- `solve_batch` counts the failures and
        # `config.json` records them, because an error averaged over only the
        # solves that survived is the defect CLAUDE.md names.
        #
        # Consumes no random numbers, which is what keeps the three
        # non-NLOS datasets byte-identical: `nlos_bias_deg` is only ever
        # nonzero on beacons the caller named, and adding a constant after
        # the draw cannot move the stream.
        for j, beacon in enumerate(beacons):
            true_angle = aoa_azimuth(beacon, pos)
            noise_rad = np.deg2rad(rng.normal(0, aoa_noise_deg))
            aoa_angles[i, j] = true_angle + noise_rad + np.deg2rad(angle_biases_deg[j])

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
    # TOA and AOA measurements are independent, so W = I. TDOA differences are
    # not: they share a reference beacon, so W = C^-1 with C = I + 1 1^T --
    # Eq. (4.42), built by `build_tdoa_covariance`, which is where the
    # correlation was already written down. Unit sigmas, because DOP factors
    # the noise magnitude out by definition (Eq. 4.107).
    #
    # `compute_dop(H)` with no weights was used for all three here, and for
    # TDOA that is the same defect as drawing its noise independently, one
    # level up: it reports the DOP of a measurement set that does not exist.
    # On the square array it read 0.8730 against 1.0665.
    weights = None
    if measurement_type.lower() == "tdoa":
        weights = np.linalg.inv(build_tdoa_covariance(np.ones(len(beacons))))

    gdop_values = np.zeros(len(positions))

    for i, pos in enumerate(positions):
        H = compute_geometry_matrix(beacons, pos, measurement_type)
        dop_dict = compute_dop(H, weights=weights)
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
    # Every solver here used to be seeded with `true_positions[i] + 1.0` -- the
    # answer plus a metre. No user has that, so none of the reported errors
    # were reproducible. The beacon centroid is what a real system starts from.
    #
    # On the square geometry it changes nothing: TOA 0.0881 m, TDOA 0.0746 m
    # and AOA 0.3971 m from the centroid, identical to four decimals from
    # `truth + 1 m`. That is a statement about this geometry and not a general
    # one -- on the collinear `poor_geometry` beacons the seed is the whole
    # story. The centroid sits on the line of symmetry, where every TOA and
    # TDOA fix stalls; seeded at [10, 3] TOA solves all 100 and TDOA 83.
    # See data/sim/ch4_rf_2d_linear/README.md, which measures all three.
    initial_guess = beacons.mean(axis=0)

    # A solve can fail three ways, and only counting one of them is how AOA
    # came to look like the best-behaved method here:
    #   - it raises, or the solver reports converged=False;
    #   - it "converges" to somewhere absurd (1e15 m) -- caught downstream by
    #     the magnitude check;
    #   - it never leaves the initial guess. On the collinear `poor_geometry`
    #     beacons every method stalls at the seed, which then scores as the
    #     distance from the centroid to the truth -- an identical 6.77 m median
    #     for TOA, TDOA and AOA alike, which is the tell. AOA reports
    #     converged=True while doing this, with a residual of 2e10.
    # This policy lives in `core.rf.solve_batch`, extracted when Chapter 4's
    # geometry comparison turned out to be counting only the converged solves
    # and so reporting a different subset of methods per geometry. One
    # definition, used here and by `ch4_rf_point_positioning/example_comparison`,
    # so the example's failure counts and this file's `failed_count` cannot
    # drift apart. Verified bit-identical to the loop it replaced across all
    # four ch4 datasets.
    def solve_all(solver, measurements):
        outcome = solve_batch(
            solver,
            measurements,
            initial_guess,
            true_positions,
            divergence_m=np.inf,  # the magnitude check is applied downstream
        )
        return outcome.estimates, outcome.solved

    # `iterative_ls` is the book default and Eq. (4.20): W = I, which is the
    # maximum-likelihood weighting when every range carries the same noise --
    # and these datasets add `rng.normal(0, toa_noise)` with one fixed std, so
    # it does. This used to ask for `iwls`, a deprecated alias resolving to
    # `range_weighted` (W_ii = 1/d_i^2), whose stated assumption is sigma_i
    # proportional to d_i. That is a real weighting, but not one this data
    # justifies, and it left the generator and
    # `ch4_rf_point_positioning/example_comparison` reporting different TOA
    # medians for the same measurements -- 0.095 m against 0.088 m.
    toa_pos, toa_ok = solve_all(
        TOAPositioner(beacons, method="iterative_ls"), toa_ranges
    )
    tdoa_pos, tdoa_ok = solve_all(TDOAPositioner(beacons, reference_idx=0), tdoa_diffs)
    aoa_pos, aoa_ok = solve_all(AOAPositioner(beacons), aoa_angles)

    return (toa_pos, toa_ok), (tdoa_pos, tdoa_ok), (aoa_pos, aoa_ok)


def save_dataset(
    output_dir: Path,
    beacons: np.ndarray,
    positions: np.ndarray,
    toa_ranges: np.ndarray,
    tdoa_diffs: np.ndarray,
    aoa_angles: np.ndarray,
    gdop_toa: np.ndarray,
    gdop_tdoa: np.ndarray,
    aoa_sensitivity: np.ndarray,
    config: Dict,
) -> None:
    """Save dataset to disk.

    `aoa_sensitivity` is in metres per radian, not a dimensionless DOP; it is
    written to `gdop_aoa.txt` under the name that file has always had.
    """
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
        header="TDOA range differences relative to beacon 0 (m)",
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
    # The values are unchanged; only the header is. It used to read "GDOP for
    # AOA", and this quantity is not a GDOP -- see `aoa_sensitivity` in
    # `generate_dataset` and the AOA branch of `core.rf.compute_geometry_matrix`.
    # The filename stays `gdop_aoa.txt` because it is cited by the dataset
    # READMEs and the file-structure guard; the units now travel with the
    # numbers for anyone who opens it.
    np.savetxt(
        output_dir / "gdop_aoa.txt",
        aoa_sensitivity,
        fmt="%.6f",
        header="AOA position sensitivity, sqrt(trace (H^T H)^-1), metres per radian",
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved dataset to: {output_dir}")
    print("    Files: 8 files (beacons, positions, 3 measurements, 3 GDOP, config)")
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
    nlos_bias_deg: float = 0.0,
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
        tdoa_noise: TDOA arrival-time noise (m), per beacon. See
            `generate_measurements`.
        aoa_noise_deg: AOA noise (degrees).
        add_nlos: Add NLOS bias.
        nlos_bias: NLOS range bias magnitude (m).
        nlos_bias_deg: NLOS bearing bias magnitude (degrees), on the same
            beacons. Zero leaves `aoa_angles.txt` untouched.
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
        output_dir = output_dir or "data/sim/ch4_rf_2d_square"
    elif preset == "optimal":
        geometry = "optimal"
        trajectory = "grid"
        toa_noise = 0.1
        tdoa_noise = 0.1
        aoa_noise_deg = 2.0
        add_nlos = False
        output_dir = output_dir or "data/sim/ch4_rf_2d_optimal"
    elif preset == "poor_geometry":
        geometry = "linear"
        trajectory = "grid"
        toa_noise = 0.1
        tdoa_noise = 0.1
        aoa_noise_deg = 2.0
        add_nlos = False
        output_dir = output_dir or "data/sim/ch4_rf_2d_linear"
    elif preset == "nlos":
        geometry = "square"
        trajectory = "grid"
        toa_noise = 0.1
        tdoa_noise = 0.1
        aoa_noise_deg = 2.0
        add_nlos = True
        nlos_bias = 0.8
        # 18.0 deg is the bearing offset that accompanies 0.8 m of excess path
        # in a single-bounce image-source geometry at this grid's median
        # agent-to-beacon range of 15.54 m. Derived in the AOA block of
        # `generate_measurements`, which also records the two independent
        # calibrations that agree on it.
        nlos_bias_deg = 18.0
        output_dir = output_dir or "data/sim/ch4_rf_2d_nlos"

    # No preset and no --output: the module's own dataset.
    output_dir = output_dir or "data/sim/ch4_rf_2d_square"

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
        print(f"  NLOS bearing bias: {nlos_bias_deg:.1f} deg on the same beacons")

    start = time.time()
    # Keyword arguments from `nlos_beacons` on, because `nlos_bias_deg` was
    # inserted ahead of `seed`: positionally, the seed would have slid into
    # the bearing bias and silently regenerated everything from a new stream.
    toa_ranges, tdoa_diffs, aoa_angles = generate_measurements(
        beacons,
        positions,
        toa_noise,
        tdoa_noise,
        aoa_noise_deg,
        nlos_beacons=nlos_beacons,
        nlos_bias=nlos_bias,
        nlos_bias_deg=nlos_bias_deg,
        seed=seed,
    )
    elapsed = time.time() - start
    print(f"  Generation time: {elapsed:.3f} s")

    # Compute DOP metrics
    print("\nStep 4: Computing DOP metrics...")
    start = time.time()
    gdop_toa = compute_dop_metrics(beacons, positions, "toa")
    gdop_tdoa = compute_dop_metrics(beacons, positions, "tdoa")
    # Not a GDOP. The AOA geometry rows are [-dy/d^2, dx/d^2], units 1/m, so
    # Q = (H^T H)^-1 is in m^2 and sqrt(trace Q) is **metres per radian** -- a
    # sensitivity, not a dilution factor. The number is correct and its name
    # was not: `config.json` used to list it under `dop.aoa` beside a
    # dimensionless `dop.toa`, inviting "this array is 15x worse for AOA",
    # which compares m/rad against a pure ratio and means nothing. It is
    # `aoa_sensitivity_m_per_rad` now, and the units are in the key.
    aoa_sensitivity = compute_dop_metrics(beacons, positions, "aoa")

    # The dimensionless companion, reported in `config.json` only -- the
    # shipped `gdop_aoa.txt` keeps carrying the sensitivity, so no committed
    # measurement byte moves for the sake of a naming fix.
    #
    # Each AOA row is (1/d) times a unit vector perpendicular to the line of
    # sight, so for beacons all at range d the sensitivity is exactly d times
    # a pure geometry factor. Dividing by the mean range at each position
    # recovers that factor, and it *is* comparable to the TOA and TDOA GDOPs
    # -- on the square array it lands near 1.07 against TOA's 1.02, which says
    # the corner layout is about as good for bearings as for ranges once the
    # lever arm is accounted for. The reference distance has to be stated for
    # the number to mean anything; here it is the per-position mean beacon
    # range.
    mean_range = np.linalg.norm(
        positions[:, None, :] - beacons[None, :, :], axis=2
    ).mean(axis=1)
    aoa_dop_dimensionless = aoa_sensitivity / mean_range
    elapsed = time.time() - start

    print(f"  Computation time: {elapsed:.3f} s")
    print(
        f"  TOA GDOP: mean={gdop_toa.mean():.2f}, min={gdop_toa.min():.2f}, max={gdop_toa.max():.2f}"
    )
    print(
        f"  TDOA GDOP: mean={gdop_tdoa.mean():.2f}, min={gdop_tdoa.min():.2f}, max={gdop_tdoa.max():.2f}"
    )
    print(
        f"  AOA sensitivity: mean={aoa_sensitivity.mean():.2f}, "
        f"min={aoa_sensitivity.min():.2f}, max={aoa_sensitivity.max():.2f} m/rad"
    )
    print(
        f"  AOA GDOP (sensitivity / mean range): "
        f"mean={aoa_dop_dimensionless.mean():.2f}, "
        f"min={aoa_dop_dimensionless.min():.2f}, "
        f"max={aoa_dop_dimensionless.max():.2f}"
    )

    # Run positioning
    print("\nStep 5: Running positioning algorithms...")
    start = time.time()
    (toa_pos, toa_ok), (tdoa_pos, tdoa_ok), (aoa_pos, aoa_ok) = run_positioning(
        beacons, toa_ranges, tdoa_diffs, aoa_angles, positions
    )
    elapsed = time.time() - start

    # Compute errors
    toa_errors = np.linalg.norm(toa_pos - positions, axis=1)
    tdoa_errors = np.linalg.norm(tdoa_pos - positions, axis=1)
    aoa_errors = np.linalg.norm(aoa_pos - positions, axis=1)

    # Report the median and the diverged count, not the mean. An iterative
    # solver that misses can land 1e15 m away while reporting convergence, and
    # one such solve makes the mean a property of the worst outlier rather than
    # of the method. AOA does exactly this on 30 of 100 positions here. See
    # .cursor/rules/030-figures-and-claims.mdc, "One draw is not a measurement".
    DIVERGENCE_M = 100.0

    def summarise(errors, ok, label):
        finite = np.isfinite(errors)
        good = finite & ok & (errors < DIVERGENCE_M)
        n_fail = int(np.sum(~good))
        median = float(np.median(errors[finite])) if finite.any() else float("nan")
        mean_ok = float(errors[good].mean()) if good.any() else float("nan")
        max_ok = float(errors[good].max()) if good.any() else float("nan")
        print(
            f"  {label:5s} median={median:8.3f}m, mean(solved)={mean_ok:8.3f}m, "
            f"max(solved)={max_ok:8.3f}m, failed={n_fail}/{len(errors)}"
        )
        return {
            "median_m": median,
            "mean_solved_m": mean_ok,
            "max_solved_m": max_ok,
            "failed_count": n_fail,
            "n_positions": int(len(errors)),
        }

    print(f"  Positioning time: {elapsed:.3f} s")
    print(
        f"\nPositioning Errors (failed = did not converge, stalled at the "
        f"initial guess, or landed >{DIVERGENCE_M:.0f} m away):"
    )
    toa_stats = summarise(toa_errors, toa_ok, "TOA:")
    tdoa_stats = summarise(tdoa_errors, tdoa_ok, "TDOA:")
    aoa_stats = summarise(aoa_errors, aoa_ok, "AOA:")

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
            # The bearing half of the same NLOS event. Declared separately
            # from `bias_m` because it is a different physical quantity with
            # different units, and because
            # `tests/ch4_rf_point_positioning/test_shipped_measurements_match_the_solver_convention.py`
            # reads both keys to predict the shipped files.
            "bias_deg": nlos_bias_deg if add_nlos else 0.0,
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
            # NOT `aoa`, and not dimensionless. The AOA geometry rows carry
            # units of 1/m, so this is metres of position error per radian of
            # bearing error. Listing it as `aoa` beside a dimensionless `toa`
            # invited "15x worse for AOA", which compares two different
            # quantities. 15.04 m/rad x 2 deg = 0.525 m is the statement it
            # actually supports.
            "aoa_sensitivity_m_per_rad": {
                "mean": float(aoa_sensitivity.mean()),
                "min": float(aoa_sensitivity.min()),
                "max": float(aoa_sensitivity.max()),
            },
            # The dimensionless form, comparable to `toa` and `tdoa` above.
            # Sensitivity divided by the mean beacon range at each position;
            # the reference distance is named in the key's own description
            # because a ratio with an unstated denominator is not a fix.
            "aoa_dimensionless_ref_mean_range": {
                "mean": float(aoa_dop_dimensionless.mean()),
                "min": float(aoa_dop_dimensionless.min()),
                "max": float(aoa_dop_dimensionless.max()),
            },
        },
        "performance": {
            # Median and diverged count, not a mean: a single 1e15 m solve made
            # the old aoa_error_mean_m a property of the worst outlier.
            "divergence_threshold_m": DIVERGENCE_M,
            "toa": toa_stats,
            "tdoa": tdoa_stats,
            "aoa": aoa_stats,
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
        aoa_sensitivity,
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
        default=None,
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
        "--area-size",
        type=float,
        default=20.0,
        help="Area size in meters (default: 20.0)",
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
        "--num-points",
        type=int,
        default=100,
        help="Number of evaluation points (default: 100)",
    )

    # Measurement noise parameters
    noise_group = parser.add_argument_group("Measurement Noise Parameters")
    noise_group.add_argument(
        "--toa-noise",
        type=float,
        default=0.1,
        help="TOA noise std dev in meters (default: 0.1)",
    )
    noise_group.add_argument(
        "--tdoa-noise",
        type=float,
        default=0.1,
        help=(
            "TDOA arrival-time noise std dev in meters, per beacon "
            "(default: 0.1). Each range difference carries two of these, so "
            "its own std is sqrt(2) times this."
        ),
    )
    noise_group.add_argument(
        "--aoa-noise",
        type=float,
        default=2.0,
        help="AOA noise std dev in degrees (default: 2.0)",
    )

    # NLOS parameters
    nlos_group = parser.add_argument_group("NLOS Parameters")
    nlos_group.add_argument(
        "--add-nlos", action="store_true", help="Add NLOS bias to beacons 1 and 2"
    )
    nlos_group.add_argument(
        "--nlos-bias",
        type=float,
        default=0.5,
        help="NLOS range bias in meters (default: 0.5)",
    )
    nlos_group.add_argument(
        "--nlos-bias-deg",
        type=float,
        default=0.0,
        help=(
            "NLOS bearing bias in degrees on the same beacons (default: 0.0, "
            "which leaves AOA untouched; the 'nlos' preset uses 18.0)"
        ),
    )

    # Other
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

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
        nlos_bias_deg=args.nlos_bias_deg,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
