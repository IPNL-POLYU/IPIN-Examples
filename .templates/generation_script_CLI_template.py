"""Generate [Dataset Name] for Chapter [X] examples.

Creates a [description] dataset with:
    - [Feature 1]
    - [Feature 2]
    - [Feature 3]

Saves to: data/sim/[dataset_name]/

Author: Navigation Engineer
Date: [Date]
References: Chapter [X] - [Title]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    'baseline': {
        'description': 'Standard configuration with nominal parameters',
        '[param1]': 0.1,
        '[param2]': 1.0,
        # Add all relevant parameters
    },
    '[preset2]': {
        'description': '[Description of this preset]',
        '[param1]': 0.5,
        '[param2]': 2.0,
    },
    '[preset3]': {
        'description': '[Description of this preset]',
        '[param1]': 0.01,
        '[param2]': 0.5,
    }
}


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_trajectory(
    param1: float,
    param2: float,
    dt: float,
    duration: float,
) -> Tuple[np.ndarray, ...]:
    """Generate [trajectory/data type].
    
    Args:
        param1: [Description and units].
        param2: [Description and units].
        dt: Time step (seconds).
        duration: Total duration (seconds).
    
    Returns:
        Tuple of (t, [field1], [field2], ...):
            t: timestamps (N,)
            [field1]: [description] (N, D)
            [field2]: [description] (N,)
    
    References:
        Based on [book section/equation].
    """
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # TODO: Implement trajectory/data generation logic
    
    return t, field1, field2


def generate_sensor_measurements(
    truth_data: Dict[str, np.ndarray],
    sensor_rate: float,
    noise_std: float,
    bias: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sensor measurements from ground truth.
    
    Args:
        truth_data: Ground truth data dictionary.
        sensor_rate: Sensor sampling rate (Hz).
        noise_std: Measurement noise standard deviation.
        bias: Constant bias (optional).
    
    Returns:
        Tuple of (t_sensor, measurements):
            t_sensor: sensor timestamps (M,)
            measurements: sensor readings (M, D)
    
    References:
        Implements measurement model from Eq. ([X.Y]).
    """
    # TODO: Implement sensor measurement generation
    
    return t_sensor, measurements


def generate_dataset(
    output_dir: str = "data/sim/[dataset_name]",
    seed: int = 42,
    # Trajectory parameters
    duration: float = 60.0,
    dt: float = 0.01,
    # Add all configurable parameters here
    param1: float = 1.0,
    param2: float = 0.1,
) -> None:
    """Generate and save [dataset name].
    
    Args:
        output_dir: Output directory path.
        seed: Random seed for reproducibility.
        duration: Dataset duration (seconds).
        dt: Time step for ground truth (seconds).
        param1: [Description].
        param2: [Description].
    """
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Generating [Dataset Name]")
    print(f"{'='*70}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate ground truth
    print(f"\n1. Generating ground truth...")
    print(f"   Duration: {duration} s")
    print(f"   Time step: {dt} s")
    
    t, field1, field2 = generate_trajectory(
        param1=param1,
        param2=param2,
        dt=dt,
        duration=duration
    )
    
    print(f"   Generated {len(t)} samples")
    
    # Save ground truth
    np.savez(
        output_path / "truth.npz",
        t=t,
        field1=field1,
        field2=field2
    )
    print(f"   Saved: truth.npz")
    
    # 2. Generate sensor measurements
    print(f"\n2. Generating sensor measurements...")
    # TODO: Add sensor generation logic
    
    # 3. Save configuration
    print(f"\n3. Saving configuration...")
    
    config = {
        "dataset_info": {
            "description": "[Dataset description]",
            "seed": seed,
            "duration_sec": duration,
            "num_samples": int(len(t))
        },
        "trajectory": {
            "type": "[trajectory type]",
            "param1": param1,
            "param2": param2
        },
        "sensor": {
            "rate_hz": 1.0 / dt,  # adjust as needed
            # Add sensor parameters
        },
        "coordinate_frame": {
            "description": "[Frame description, e.g., ENU]",
            "origin": "[Origin description]",
            "units": "meters"
        }
    }
    
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"   Saved: config.json")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Dataset generation complete!")
    print(f"{'='*70}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"\nFiles created:")
    print(f"  - truth.npz        : Ground truth data")
    print(f"  - [sensor].npz     : Sensor measurements")
    print(f"  - config.json      : Dataset configuration")
    print(f"\nDataset statistics:")
    print(f"  Duration        : {duration:.1f} s")
    print(f"  Samples         : {len(t)}")
    print(f"\n")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate [Dataset Name] for Chapter [X] examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default parameters
  python %(prog)s

  # Use a preset configuration
  python %(prog)s --preset baseline

  # Custom parameters
  python %(prog)s --param1 0.5 --param2 2.0 --duration 120

  # Generate multiple variants for experiments
  python %(prog)s --preset baseline --output data/sim/experiment_baseline
  python %(prog)s --preset [preset2] --output data/sim/experiment_[preset2]

Available presets: """ + ", ".join(PRESETS.keys())
    )
    
    # Preset configuration
    parser.add_argument(
        '--preset',
        type=str,
        choices=PRESETS.keys(),
        help='Use preset configuration (overrides individual parameters)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='data/sim/[dataset_name]',
        help='Output directory (default: data/sim/[dataset_name])'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Trajectory parameters
    traj_group = parser.add_argument_group('Trajectory Parameters')
    traj_group.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Trajectory duration in seconds (default: 60.0)'
    )
    traj_group.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Time step for ground truth in seconds (default: 0.01)'
    )
    traj_group.add_argument(
        '--param1',
        type=float,
        default=1.0,
        help='[Parameter 1 description] (default: 1.0)'
    )
    traj_group.add_argument(
        '--param2',
        type=float,
        default=0.1,
        help='[Parameter 2 description] (default: 0.1)'
    )
    
    # Sensor parameters
    sensor_group = parser.add_argument_group('Sensor Parameters')
    sensor_group.add_argument(
        '--sensor-rate',
        type=float,
        default=100.0,
        help='Sensor sampling rate in Hz (default: 100.0)'
    )
    sensor_group.add_argument(
        '--sensor-noise',
        type=float,
        default=0.1,
        help='Sensor noise std (default: 0.1)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # If preset is specified, override with preset values
    if args.preset:
        preset_config = PRESETS[args.preset]
        print(f"\nUsing preset: '{args.preset}'")
        print(f"Description: {preset_config['description']}")
        
        # Override parameters with preset values
        for key, value in preset_config.items():
            if key != 'description' and hasattr(args, key.replace('_', '-')):
                setattr(args, key.replace('_', '-'), value)
    
    # Validate parameters
    if args.duration <= 0:
        parser.error("Duration must be positive")
    if args.dt <= 0 or args.dt > args.duration:
        parser.error("Time step must be positive and less than duration")
    
    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        seed=args.seed,
        duration=args.duration,
        dt=args.dt,
        param1=args.param1,
        param2=args.param2,
        # Add all parameters
    )


if __name__ == "__main__":
    main()


