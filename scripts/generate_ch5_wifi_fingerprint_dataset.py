"""
Generate synthetic Wi-Fi fingerprint database for Chapter 5 examples.

Creates a realistic indoor RSS fingerprint database with:
    - 3 floors (0, 1, 2)
    - 100 reference points per floor (10×10 grid, 5m spacing)
    - 8 access points (APs) positioned strategically
    - Log-distance path loss, spatially correlated shadowing, fast fading
    - Multi-floor attenuation

The RSS model has three terms and the split between the last two is the
point of it::

    rss(p, ap) = pathloss(d(p, ap)) - floor_attenuation + S_ap(p) + fast

``S_ap(p)`` is a property of the *location*: the same wall attenuates the same
AP from the same spot on every visit, so it is drawn once per (floor, AP) as a
spatially correlated field (:class:`core.fingerprinting.ShadowingField`) and
evaluated wherever it is needed. ``fast`` is the only term that varies between
repeat visits to one reference point.

This used to be one term. The whole 4 dB was redrawn for every (RP, AP, sample),
which made the radio map a table of random numbers rather than a smooth function
of position -- and smoothness is the only property fingerprinting exploits.
Measured on the baseline grid before the split: nearest neighbour scored 6.93 m
against noiseless queries where the 5 m grid's own quantisation floor is 2.04 m
-- ``sqrt(2 s^2 / 12)``, the rms distance from a uniform position to the nearest
node. It scores 3.01 m now.

Saves to: data/sim/ch5_wifi_fingerprint_grid/

Author: Li-Ta Hsu
Date: December 2024
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fingerprinting import (
    DEFAULT_DECORRELATION_M,
    FingerprintDatabase,
    ShadowingField,
    save_fingerprint_database,
)

#: Standard deviation (dB) of the fast-fading / temporal term.
#:
#: This is the part of RSS variability that is *not* a property of the location:
#: small-scale multipath, receiver noise, and people moving between the AP and
#: the device. Reported values for a stationary receiver sit around 1-3 dB;
#: 1.5 dB is inside that range and well clear of both ends.
#:
#: It is deliberately much smaller than the 4 dB shadowing term, because that is
#: the physical claim being made: two surveys of one building disagree at a
#: point by *this*, not by the whole of the variability. Setting them equal
#: would be the old model wearing two names.
DEFAULT_FAST_FADING_STD = 1.5


def log_distance_path_loss(
    d: np.ndarray,
    P0: float = -30.0,
    d0: float = 1.0,
    n: float = 2.5,
) -> np.ndarray:
    """
    Compute mean RSS using the log-distance path-loss model.

    Model: P(d) = P0 - 10*n*log10(d/d0)

    This is the deterministic part only. It used to draw a shadow-fading term
    of its own, which made it impossible to ask for the mean RSS at a point --
    and made every caller's shadowing independent of every other caller's, which
    is precisely the defect this generator was rebuilt to remove. Shadowing now
    comes from :class:`core.fingerprinting.ShadowingField`, which is a function
    of position, and fast fading is added per sample by the caller.

    Args:
        d: Distance(s) from AP to reference point (meters). Scalar or array.
        P0: Reference power at distance d0 (dBm).
        d0: Reference distance (meters).
        n: Path-loss exponent (2.0 = free space, 2-4 = indoor).

    Returns:
        Mean RSS in dBm, same shape as ``d``.
    """
    # Avoid the singularity at the AP itself.
    d = np.maximum(np.asarray(d, dtype=float), 0.1)

    return P0 - 10 * n * np.log10(d / d0)


def generate_wifi_fingerprint_database(
    area_size: tuple = (50.0, 50.0),
    grid_spacing: float = 5.0,
    n_floors: int = 3,
    floor_height: float = 3.0,
    n_aps: int = 8,
    n_samples_per_rp: int = 1,
    seed: int = 42,
    shadow_fading_std: float = 4.0,
    fast_fading_std: float = DEFAULT_FAST_FADING_STD,
    decorrelation_m: float = DEFAULT_DECORRELATION_M,
) -> FingerprintDatabase:
    """
    Generate synthetic Wi-Fi fingerprint database.

    Args:
        area_size: (width, height) in meters.
        grid_spacing: Distance between reference points (meters).
        n_floors: Number of floors.
        floor_height: Height of each floor (meters).
        n_aps: Number of access points.
        n_samples_per_rp: Number of RSS samples to collect at each RP.
                          If 1 (default), creates single-sample DB (M, N).
                          If > 1, creates multi-sample DB (M, S, N) for
                          proper μ and σ estimation per Eq. 5.6.
        seed: Random seed for reproducibility. Seeds both the shadowing field
              and the fast-fading draws.
        shadow_fading_std: Marginal std (dB) of the spatially correlated
              shadowing field. A property of the location, not of the sample.
        fast_fading_std: Std (dB) of the per-sample term. The only thing that
              differs between repeat visits to one reference point.
        decorrelation_m: Correlation length (m) of the shadowing field.

    Returns:
        FingerprintDatabase with multi-floor RSS fingerprints.
    """
    # The fast-fading draws take an explicit Generator rather than the global
    # stream, so a caller can hold two independent databases at once without
    # them sharing a sequence. The shadowing field is seeded separately and
    # per (floor, AP), which is what keeps AP 3 on floor 1 the same field
    # whether the survey is the 5 m grid, the 2 m one or the 10 m one.
    rng = np.random.default_rng(seed)
    shadow_field = ShadowingField.build(
        n_aps=n_aps,
        n_floors=n_floors,
        sigma_dB=shadow_fading_std,
        seed=seed,
        decorrelation_m=decorrelation_m,
    )

    width, height = area_size

    # Generate reference point grid per floor
    x_coords = np.arange(0, width + grid_spacing / 2, grid_spacing)
    y_coords = np.arange(0, height + grid_spacing / 2, grid_spacing)

    print(f"\n{'='*60}")
    print("Generating Wi-Fi Fingerprint Database")
    print(f"{'='*60}")
    print(f"Area size: {width}m × {height}m")
    print(f"Grid spacing: {grid_spacing}m")
    print(
        f"Grid dimensions: {len(x_coords)} × {len(y_coords)} = {len(x_coords) * len(y_coords)} RPs per floor"
    )
    print(f"Floors: {n_floors}")
    print(f"Total reference points: {len(x_coords) * len(y_coords) * n_floors}")
    print(f"Access points: {n_aps}")
    print(
        f"Samples per RP: {n_samples_per_rp} {'(multi-sample DB)' if n_samples_per_rp > 1 else '(single-sample DB)'}"
    )

    # Generate AP positions (strategic placement on walls/ceiling)
    # APs at corners, mid-walls, and center ceiling of first floor
    ap_positions = np.array(
        [
            [0, 0, 2.5],  # Corner 1 (wall)
            [width, 0, 2.5],  # Corner 2 (wall)
            [width, height, 2.5],  # Corner 3 (wall)
            [0, height, 2.5],  # Corner 4 (wall)
            [width / 2, 0, 2.5],  # Mid-wall 1
            [width / 2, height, 2.5],  # Mid-wall 2
            [0, height / 2, 2.5],  # Mid-wall 3
            [width, height / 2, 2.5],  # Mid-wall 4
        ]
    )[:n_aps]

    ap_ids = [f"AP{i+1}" for i in range(n_aps)]

    print("\nAP Positions:")
    for i, pos in enumerate(ap_positions):
        print(f"  {ap_ids[i]}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})m")

    # Generate reference points and RSS measurements
    locations_list = []
    features_list = []
    floor_ids_list = []

    print("\nGenerating fingerprints...")

    for floor_id in range(n_floors):
        floor_z = floor_id * floor_height + 1.5  # Height of device (1.5m from floor)

        print(f"  Floor {floor_id}: z = {floor_z}m", end=" ")

        # One floor's reference points, in the x-then-y order the loops used.
        rp_xy = np.array([[x, y] for x in x_coords for y in y_coords])
        rp_3d = np.column_stack([rp_xy, np.full(len(rp_xy), floor_z)])

        # (n_rp, n_aps) distances, then the deterministic part of the model.
        distances = np.linalg.norm(rp_3d[:, None, :] - ap_positions[None, :, :], axis=2)
        mean_rss = log_distance_path_loss(distances, P0=-30.0, n=2.5)

        # Floor attenuation: 15 dB per floor between the device and the AP.
        ap_floors = (ap_positions[:, 2] / floor_height).astype(int)
        mean_rss -= np.abs(floor_id - ap_floors)[None, :] * 15.0

        # Shadowing is a property of the location, so it is added once here and
        # is identical in every sample taken at this reference point. This is
        # the whole change: the same wall attenuates the same AP from the same
        # spot on every visit.
        mean_rss = mean_rss + shadow_field(rp_xy, floor_id)

        # Fast fading is the only per-sample term.
        samples = mean_rss[:, None, :] + rng.normal(
            0.0, fast_fading_std, size=(len(rp_xy), n_samples_per_rp, n_aps)
        )

        for i in range(len(rp_xy)):
            locations_list.append(rp_xy[i])
            if n_samples_per_rp == 1:
                # Single sample: store as (N,)
                features_list.append(samples[i, 0])
            else:
                # Multiple samples: store as (S, N)
                features_list.append(samples[i])
            floor_ids_list.append(floor_id)

        print(f"OK ({len([f for f in floor_ids_list if f == floor_id])} RPs)")

    # Convert to arrays
    locations = np.array(locations_list)
    features = np.array(features_list)
    floor_ids = np.array(floor_ids_list, dtype=int)

    print(f"\n{'='*60}")
    print("Database Summary:")
    print(f"  Total reference points: {len(locations)}")
    print(f"  Location dimension: {locations.shape[1]}D")
    if n_samples_per_rp == 1:
        print(f"  Features shape: {features.shape} (M, N)")
        print(f"  Features per RP: {features.shape[1]} (APs)")
    else:
        print(f"  Features shape: {features.shape} (M, S, N)")
        print(f"  Samples per RP: {features.shape[1]}")
        print(f"  Features per sample: {features.shape[2]} (APs)")
    print(f"  Floors: {sorted(np.unique(floor_ids).tolist())}")
    print(f"  RSS range: [{features.min():.1f}, {features.max():.1f}] dBm")
    print(f"  RSS mean: {features.mean():.1f} dBm")
    print(f"  RSS std: {features.std():.1f} dBm")

    # Create database
    db = FingerprintDatabase(
        locations=locations,
        features=features,
        floor_ids=floor_ids,
        meta={
            "ap_ids": ap_ids,
            "ap_positions": ap_positions.tolist(),
            "area_size": list(area_size),
            "grid_spacing": grid_spacing,
            "floor_height": floor_height,
            "n_floors": n_floors,
            "n_samples_per_rp": n_samples_per_rp,
            "path_loss_model": {
                "type": "log_distance",
                "P0_dBm": -30.0,
                "path_loss_exponent": 2.5,
                # The std of the *spatial* shadowing field: what a location
                # contributes, and what two surveys of this building agree on.
                "shadow_fading_std_dBm": shadow_fading_std,
                # The std of the *per-sample* term: the only thing that differs
                # between repeat visits to one reference point.
                "fast_fading_std_dBm": fast_fading_std,
                "floor_attenuation_dB": 15.0,
            },
            # Everything needed to rebuild the shadowing field. A query
            # generator reads this and evaluates the same field at the query's
            # own position, which is what makes a query consistent with the map
            # where it stands. Without it the shadowing is unreconstructible and
            # a query can only redraw it, which is the defect this replaced.
            "shadow_field": shadow_field.to_meta(),
            "description": f"Synthetic Wi-Fi RSS fingerprint database ({'multi-sample' if n_samples_per_rp > 1 else 'single-sample'})",
            "generation_date": "2024-12-24",
            # The seed, and the parameters it has to be paired with. Without
            # these the shipped arrays cannot be regenerated exactly -- the ch5
            # datasets were the only three of twenty that could not say how they
            # were made. They always were reproducible; nothing recorded it.
            "seed": seed,
            "n_aps": n_aps,
            "generator": "scripts/generate_ch5_wifi_fingerprint_dataset.py",
        },
    )

    return db


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Ch5 Wi-Fi Fingerprint Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  baseline      Standard 5m grid, 8 APs, 3 floors, 1 sample/RP
  dense         Dense 2m grid, 8 APs, 3 floors, 1 sample/RP
  sparse        Sparse 10m grid, 8 APs, 3 floors, 1 sample/RP
  few_aps       Standard grid, only 4 APs, 1 sample/RP
  multisamples  Standard grid, 8 APs, 10 samples/RP (for μ/σ estimation)

Examples:
  # Generate baseline dataset
  python scripts/generate_wifi_fingerprint_dataset.py --preset baseline

  # Generate with custom parameters
  python scripts/generate_wifi_fingerprint_dataset.py \\
      --output data/sim/my_wifi_fp \\
      --grid-spacing 3.0 \\
      --n-aps 12

  # Generate all presets
  python scripts/generate_wifi_fingerprint_dataset.py --preset baseline
  python scripts/generate_wifi_fingerprint_dataset.py --preset dense
  python scripts/generate_wifi_fingerprint_dataset.py --preset sparse
  python scripts/generate_wifi_fingerprint_dataset.py --preset few_aps

Learning Focus:
  - Grid spacing affects positioning accuracy (2m vs 10m → 5× difference!)
  - Number of APs impacts RSS dimensionality
  - Dense databases → better accuracy but higher storage/computation

Book Reference: Chapter 5, Sections 5.1-5.3
        """,
    )

    # Preset or custom
    parser.add_argument(
        "--preset",
        type=str,
        choices=["baseline", "dense", "sparse", "few_aps", "multisamples"],
        help="Use preset configuration (overrides other parameters)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Defaults to the preset's own directory, or "
        "data/sim/ch5_wifi_fingerprint_grid without a preset. Given "
        "explicitly it always wins -- a preset does not override it.",
    )

    # Area parameters
    area_group = parser.add_argument_group("Area Parameters")
    area_group.add_argument(
        "--area-width",
        type=float,
        default=50.0,
        help="Area width in meters (default: 50.0)",
    )
    area_group.add_argument(
        "--area-height",
        type=float,
        default=50.0,
        help="Area height in meters (default: 50.0)",
    )
    area_group.add_argument(
        "--grid-spacing",
        type=float,
        default=5.0,
        help="Grid spacing in meters (default: 5.0)",
    )

    # Building parameters
    building_group = parser.add_argument_group("Building Parameters")
    building_group.add_argument(
        "--n-floors", type=int, default=3, help="Number of floors (default: 3)"
    )
    building_group.add_argument(
        "--floor-height",
        type=float,
        default=3.0,
        help="Floor height in meters (default: 3.0)",
    )

    # AP parameters
    ap_group = parser.add_argument_group("Access Point Parameters")
    ap_group.add_argument(
        "--n-aps", type=int, default=8, help="Number of access points (default: 8)"
    )

    # Survey parameters
    survey_group = parser.add_argument_group("Survey Parameters")
    survey_group.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of RSS samples per RP (default: 1). "
        "Use >1 for multi-sample DB to estimate μ and σ per Eq. 5.6",
    )

    # Propagation parameters
    prop_group = parser.add_argument_group("Propagation Parameters")
    prop_group.add_argument(
        "--shadow-fading-std",
        type=float,
        default=4.0,
        help="Std (dB) of the spatially correlated shadowing field, which is a "
        "property of the location (default: 4.0)",
    )
    prop_group.add_argument(
        "--fast-fading-std",
        type=float,
        default=DEFAULT_FAST_FADING_STD,
        help=f"Std (dB) of the per-sample term -- the only thing that varies "
        f"between repeat visits to one RP (default: {DEFAULT_FAST_FADING_STD})",
    )
    prop_group.add_argument(
        "--decorrelation",
        type=float,
        default=DEFAULT_DECORRELATION_M,
        help=f"Correlation length (m) of the shadowing field. Must exceed the "
        f"grid spacing for a survey to represent the field at all "
        f"(default: {DEFAULT_DECORRELATION_M})",
    )

    # Other
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Apply preset if specified
    if args.preset == "baseline":
        area_size = (50.0, 50.0)
        grid_spacing = 5.0
        n_floors = 3
        n_aps = 8
        n_samples = 1
        output_dir = args.output or "data/sim/ch5_wifi_fingerprint_grid"
    elif args.preset == "dense":
        area_size = (50.0, 50.0)
        grid_spacing = 2.0
        n_floors = 3
        n_aps = 8
        n_samples = 1
        output_dir = args.output or "data/sim/ch5_wifi_fingerprint_dense"
    elif args.preset == "sparse":
        area_size = (50.0, 50.0)
        grid_spacing = 10.0
        n_floors = 3
        n_aps = 8
        n_samples = 1
        output_dir = args.output or "data/sim/ch5_wifi_fingerprint_sparse"
    elif args.preset == "few_aps":
        area_size = (50.0, 50.0)
        grid_spacing = 5.0
        n_floors = 3
        n_aps = 4
        n_samples = 1
        output_dir = args.output or "data/sim/ch5_wifi_fingerprint_few_aps"
    elif args.preset == "multisamples":
        area_size = (50.0, 50.0)
        grid_spacing = 5.0
        n_floors = 3
        n_aps = 8
        n_samples = 10  # 10 samples per RP for proper μ/σ estimation
        output_dir = args.output or "data/sim/ch5_wifi_fingerprint_multisamples"
    else:
        area_size = (args.area_width, args.area_height)
        grid_spacing = args.grid_spacing
        n_floors = args.n_floors
        n_aps = args.n_aps
        n_samples = args.n_samples
        output_dir = args.output or "data/sim/ch5_wifi_fingerprint_grid"

    # Generate database
    db = generate_wifi_fingerprint_database(
        area_size=area_size,
        grid_spacing=grid_spacing,
        n_floors=n_floors,
        floor_height=args.floor_height if not args.preset else 3.0,
        n_aps=n_aps,
        n_samples_per_rp=n_samples,
        seed=args.seed,
        shadow_fading_std=args.shadow_fading_std,
        fast_fading_std=args.fast_fading_std,
        decorrelation_m=args.decorrelation,
    )

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Saving database...")
    save_fingerprint_database(db, output_path)
    print(f"OK Saved to: {output_path}")

    # Validate
    from core.fingerprinting import load_fingerprint_database, validate_database

    print(f"\n{'='*60}")
    print("Validating database...")
    db_loaded = load_fingerprint_database(output_path)
    stats = validate_database(db_loaded)

    print("\nValidation Results:")
    print("  OK Database loaded successfully")
    print("  OK All validation checks passed")
    if "floor_coverage" in stats:
        print(f"  Floor coverage: {stats['floor_coverage']}")
    if "feature_variance_min" in stats and "feature_variance_max" in stats:
        print(
            f"  Feature variance: min={stats['feature_variance_min']:.2f}, max={stats['feature_variance_max']:.2f}"
        )

    # Per-floor statistics
    print("\nPer-Floor Statistics:")
    for floor_id in sorted(np.unique(db.floor_ids)):
        mask = db.floor_ids == floor_id
        n_rps = np.sum(mask)
        rss_mean = db.features[mask].mean()
        rss_std = db.features[mask].std()
        print(
            f"  Floor {floor_id}: {n_rps} RPs, RSS mean={rss_mean:.1f} dBm, std={rss_std:.1f} dBm"
        )

    print(f"\n{'='*60}")
    print("SUCCESS: Dataset generation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
