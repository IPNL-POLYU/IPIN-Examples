"""
Generate synthetic Wi-Fi fingerprint database for Chapter 5 examples.

Creates a realistic indoor RSS fingerprint database with:
    - 3 floors (0, 1, 2)
    - 100 reference points per floor (10×10 grid, 5m spacing)
    - 8 access points (APs) positioned strategically
    - Log-distance path-loss model with shadow fading
    - Multi-floor attenuation

Saves to: data/sim/ch5_wifi_fingerprint_grid/

Author: Li-Ta Hsu
Date: December 2024
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fingerprinting import save_fingerprint_database, FingerprintDatabase


def log_distance_path_loss(
    d: float,
    P0: float = -30.0,
    d0: float = 1.0,
    n: float = 2.5,
    sigma: float = 4.0,
) -> float:
    """
    Compute RSS using log-distance path-loss model.
    
    Model: P(d) = P0 - 10*n*log10(d/d0) + X_sigma
    
    Args:
        d: Distance from AP to reference point (meters).
        P0: Reference power at distance d0 (dBm).
        d0: Reference distance (meters).
        n: Path-loss exponent (2.0 = free space, 2-4 = indoor).
        sigma: Shadow fading standard deviation (dBm).
    
    Returns:
        RSS in dBm.
    """
    if d < 0.1:
        d = 0.1  # Avoid singularity
    
    # Path loss
    path_loss = -10 * n * np.log10(d / d0)
    
    # Shadow fading (log-normal)
    shadow = np.random.randn() * sigma
    
    # RSS
    rss = P0 + path_loss + shadow
    
    return rss


def generate_wifi_fingerprint_database(
    area_size: tuple = (50.0, 50.0),
    grid_spacing: float = 5.0,
    n_floors: int = 3,
    floor_height: float = 3.0,
    n_aps: int = 8,
    n_samples_per_rp: int = 1,
    seed: int = 42,
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
        seed: Random seed for reproducibility.
    
    Returns:
        FingerprintDatabase with multi-floor RSS fingerprints.
    """
    np.random.seed(seed)
    
    width, height = area_size
    
    # Generate reference point grid per floor
    x_coords = np.arange(0, width + grid_spacing / 2, grid_spacing)
    y_coords = np.arange(0, height + grid_spacing / 2, grid_spacing)
    
    print(f"\n{'='*60}")
    print(f"Generating Wi-Fi Fingerprint Database")
    print(f"{'='*60}")
    print(f"Area size: {width}m × {height}m")
    print(f"Grid spacing: {grid_spacing}m")
    print(f"Grid dimensions: {len(x_coords)} × {len(y_coords)} = {len(x_coords) * len(y_coords)} RPs per floor")
    print(f"Floors: {n_floors}")
    print(f"Total reference points: {len(x_coords) * len(y_coords) * n_floors}")
    print(f"Access points: {n_aps}")
    print(f"Samples per RP: {n_samples_per_rp} {'(multi-sample DB)' if n_samples_per_rp > 1 else '(single-sample DB)'}")
    
    # Generate AP positions (strategic placement on walls/ceiling)
    # APs at corners, mid-walls, and center ceiling of first floor
    ap_positions = np.array([
        [0, 0, 2.5],           # Corner 1 (wall)
        [width, 0, 2.5],       # Corner 2 (wall)
        [width, height, 2.5],  # Corner 3 (wall)
        [0, height, 2.5],      # Corner 4 (wall)
        [width/2, 0, 2.5],     # Mid-wall 1
        [width/2, height, 2.5],# Mid-wall 2
        [0, height/2, 2.5],    # Mid-wall 3
        [width, height/2, 2.5],# Mid-wall 4
    ])[:n_aps]
    
    ap_ids = [f"AP{i+1}" for i in range(n_aps)]
    
    print(f"\nAP Positions:")
    for i, pos in enumerate(ap_positions):
        print(f"  {ap_ids[i]}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})m")
    
    # Generate reference points and RSS measurements
    locations_list = []
    features_list = []
    floor_ids_list = []
    
    print(f"\nGenerating fingerprints...")
    
    for floor_id in range(n_floors):
        floor_z = floor_id * floor_height + 1.5  # Height of device (1.5m from floor)
        
        print(f"  Floor {floor_id}: z = {floor_z}m", end=" ")
        
        for x in x_coords:
            for y in y_coords:
                # Reference point location (2D)
                rp_location = np.array([x, y])
                
                # Collect multiple samples at this RP if requested
                rp_samples = []  # Will be list of S samples, each of shape (N,)
                
                for sample_idx in range(n_samples_per_rp):
                    # RSS measurements from all APs for this sample
                    rss_vector = []
                    
                    for ap_pos in ap_positions:
                        # 3D distance from RP to AP
                        rp_3d = np.array([x, y, floor_z])
                        distance_3d = np.linalg.norm(rp_3d - ap_pos)
                        
                        # Floor attenuation factor (if AP on different floor)
                        ap_floor = int(ap_pos[2] / floor_height)
                        floor_diff = abs(floor_id - ap_floor)
                        floor_attenuation = floor_diff * 15.0  # 15 dB per floor
                        
                        # Compute RSS with path-loss model
                        # Each sample gets independent shadow fading
                        rss = log_distance_path_loss(
                            distance_3d,
                            P0=-30.0,
                            n=2.5,  # Indoor path-loss exponent
                            sigma=4.0,  # Shadow fading (varies per sample)
                        )
                        
                        # Apply floor attenuation
                        rss -= floor_attenuation
                        
                        rss_vector.append(rss)
                    
                    rp_samples.append(np.array(rss_vector))
                
                # Store
                locations_list.append(rp_location)
                if n_samples_per_rp == 1:
                    # Single sample: store as (N,)
                    features_list.append(rp_samples[0])
                else:
                    # Multiple samples: store as (S, N)
                    features_list.append(np.array(rp_samples))
                floor_ids_list.append(floor_id)
        
        print(f"OK ({len([f for f in floor_ids_list if f == floor_id])} RPs)")
    
    # Convert to arrays
    locations = np.array(locations_list)
    features = np.array(features_list)
    floor_ids = np.array(floor_ids_list, dtype=int)
    
    print(f"\n{'='*60}")
    print(f"Database Summary:")
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
                "shadow_fading_std_dBm": 4.0,
                "floor_attenuation_dB": 15.0,
            },
            "description": f"Synthetic Wi-Fi RSS fingerprint database ({'multi-sample' if n_samples_per_rp > 1 else 'single-sample'})",
            "generation_date": "2024-12-24",
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
        default="data/sim/ch5_wifi_fingerprint_grid",
        help="Output directory (default: data/sim/ch5_wifi_fingerprint_grid)",
    )
    
    # Area parameters
    area_group = parser.add_argument_group("Area Parameters")
    area_group.add_argument(
        "--area-width", type=float, default=50.0, help="Area width in meters (default: 50.0)"
    )
    area_group.add_argument(
        "--area-height", type=float, default=50.0, help="Area height in meters (default: 50.0)"
    )
    area_group.add_argument(
        "--grid-spacing", type=float, default=5.0, help="Grid spacing in meters (default: 5.0)"
    )
    
    # Building parameters
    building_group = parser.add_argument_group("Building Parameters")
    building_group.add_argument(
        "--n-floors", type=int, default=3, help="Number of floors (default: 3)"
    )
    building_group.add_argument(
        "--floor-height", type=float, default=3.0, help="Floor height in meters (default: 3.0)"
    )
    
    # AP parameters
    ap_group = parser.add_argument_group("Access Point Parameters")
    ap_group.add_argument(
        "--n-aps", type=int, default=8, help="Number of access points (default: 8)"
    )
    
    # Survey parameters
    survey_group = parser.add_argument_group("Survey Parameters")
    survey_group.add_argument(
        "--n-samples", type=int, default=1, 
        help="Number of RSS samples per RP (default: 1). "
             "Use >1 for multi-sample DB to estimate μ and σ per Eq. 5.6"
    )
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Apply preset if specified
    if args.preset == "baseline":
        area_size = (50.0, 50.0)
        grid_spacing = 5.0
        n_floors = 3
        n_aps = 8
        n_samples = 1
        output_dir = "data/sim/ch5_wifi_fingerprint_grid"
    elif args.preset == "dense":
        area_size = (50.0, 50.0)
        grid_spacing = 2.0
        n_floors = 3
        n_aps = 8
        n_samples = 1
        output_dir = "data/sim/ch5_wifi_fingerprint_dense"
    elif args.preset == "sparse":
        area_size = (50.0, 50.0)
        grid_spacing = 10.0
        n_floors = 3
        n_aps = 8
        n_samples = 1
        output_dir = "data/sim/ch5_wifi_fingerprint_sparse"
    elif args.preset == "few_aps":
        area_size = (50.0, 50.0)
        grid_spacing = 5.0
        n_floors = 3
        n_aps = 4
        n_samples = 1
        output_dir = "data/sim/ch5_wifi_fingerprint_few_aps"
    elif args.preset == "multisamples":
        area_size = (50.0, 50.0)
        grid_spacing = 5.0
        n_floors = 3
        n_aps = 8
        n_samples = 10  # 10 samples per RP for proper μ/σ estimation
        output_dir = "data/sim/ch5_wifi_fingerprint_multisamples"
    else:
        area_size = (args.area_width, args.area_height)
        grid_spacing = args.grid_spacing
        n_floors = args.n_floors
        n_aps = args.n_aps
        n_samples = args.n_samples
        output_dir = args.output
    
    # Generate database
    db = generate_wifi_fingerprint_database(
        area_size=area_size,
        grid_spacing=grid_spacing,
        n_floors=n_floors,
        floor_height=args.floor_height if not args.preset else 3.0,
        n_aps=n_aps,
        n_samples_per_rp=n_samples,
        seed=args.seed,
    )
    
    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Saving database...")
    save_fingerprint_database(db, output_path)
    print(f"OK Saved to: {output_path}")
    
    # Validate
    from core.fingerprinting import load_fingerprint_database, validate_database
    
    print(f"\n{'='*60}")
    print(f"Validating database...")
    db_loaded = load_fingerprint_database(output_path)
    stats = validate_database(db_loaded)
    
    print(f"\nValidation Results:")
    print(f"  OK Database loaded successfully")
    print(f"  OK All validation checks passed")
    if 'floor_coverage' in stats:
        print(f"  Floor coverage: {stats['floor_coverage']}")
    if 'feature_variance_min' in stats and 'feature_variance_max' in stats:
        print(f"  Feature variance: min={stats['feature_variance_min']:.2f}, max={stats['feature_variance_max']:.2f}")
    
    # Per-floor statistics
    print(f"\nPer-Floor Statistics:")
    for floor_id in sorted(np.unique(db.floor_ids)):
        mask = db.floor_ids == floor_id
        n_rps = np.sum(mask)
        rss_mean = db.features[mask].mean()
        rss_std = db.features[mask].std()
        print(f"  Floor {floor_id}: {n_rps} RPs, RSS mean={rss_mean:.1f} dBm, std={rss_std:.1f} dBm")
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: Dataset generation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

