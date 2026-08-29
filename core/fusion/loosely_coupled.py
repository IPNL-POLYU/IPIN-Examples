"""Loosely coupled IMU + UWB fusion (Section 8.1.1).

Author: Li-Ta Hsu

The counterpart to :mod:`core.fusion.tightly_coupled`, separated from the demo
that plots it for the same reason: ``example_comparison`` and
``example_anchor_outage`` both need it, and an example that other examples
import is not an example.

References: Chapter 8, Section 8.1.1 (Loosely Coupled)
"""

from typing import Dict, List

import numpy as np

from core.fusion.adaptive import create_adaptive_manager_for_lc
from core.fusion.gating import chi_square_gate, mahalanobis_distance_squared
from core.fusion.lc_models import (
    create_lc_fusion_ekf,
    create_lc_position_measurement_model,
    solve_uwb_position_wls,
)
from core.fusion.tuning import innovation, innovation_covariance
from core.fusion.types import StampedMeasurement

__all__ = ["run_lc_fusion"]

# Used only if a caller hands run_lc_fusion a dataset dict whose config is
# missing uwb.range_noise_std_m -- every shipped dataset has it (see
# data/sim/*/config.json), so this is a defensive fallback, not the normal
# path. Matches solve_uwb_position_wls's own default. run_lc_fusion used to
# hardcode 0.1 m here unconditionally, diverging from both this fallback and
# the dataset's real 0.05 m, and from what run_tc_fusion reads for the same
# sensor -- see CLAUDE.md's Chapter 8 entries for the measurement.
_FALLBACK_RANGE_NOISE_STD_M = 0.05


def run_lc_fusion(
    dataset: Dict,
    use_gating: bool = True,
    gate_confidence: float = 0.95,
    verbose: bool = True,
) -> Dict:
    """Run loosely coupled IMU + UWB fusion.

    Args:
        dataset: Dataset dictionary from load_fusion_dataset
        use_gating: Whether to apply chi-square gating
        gate_confidence: Gating confidence level (default 0.95 for 95% confidence)
        verbose: Print progress

    Returns:
        Results dictionary with:
            - 't': timestamps (N,)
            - 'x_est': estimated states (N, 5)
            - 'P_trace': trace of covariance (N,)
            - 'innovations': list of innovations
            - 'nis': list of NIS values
            - 'gated': list of booleans
            - 'n_uwb_accepted': number of UWB position fixes accepted
            - 'n_uwb_rejected': number of UWB position fixes rejected
            - 'n_uwb_failed': number of UWB position solves that failed
    """
    if verbose:
        print("=" * 70)
        print("Loosely Coupled IMU + UWB EKF Fusion")
        print("=" * 70)

    # Extract data
    truth = dataset["truth"]
    imu = dataset["imu"]
    uwb = dataset["uwb"]
    anchors = dataset["uwb_anchors"]

    # Range noise std: read from the dataset config, exactly as run_tc_fusion
    # does (core/fusion/tightly_coupled.py:99). LC and TC observe the same
    # UWB sensor, so telling their WLS/EKF math two different noise levels
    # for it is not a modelling choice, it is a bug -- see CLAUDE.md.
    uwb_config = dataset.get("config", {}).get("uwb", {})
    if "range_noise_std_m" in uwb_config:
        range_noise_std = uwb_config["range_noise_std_m"]
    else:
        range_noise_std = _FALLBACK_RANGE_NOISE_STD_M
        if verbose:
            print(
                "  WARNING: dataset config missing uwb.range_noise_std_m; "
                f"falling back to {_FALLBACK_RANGE_NOISE_STD_M} m"
            )

    # Initialize EKF at true starting position
    x0 = np.array(
        [
            truth["p_xy"][0, 0],  # px
            truth["p_xy"][0, 1],  # py
            truth["v_xy"][0, 0],  # vx
            truth["v_xy"][0, 1],  # vy
            truth["yaw"][0],  # yaw
        ]
    )

    # Increase initial uncertainty to be more conservative (per book guidance on P0)
    # This prevents overconfidence in early stages before sufficient observations
    P0 = np.diag([1.0, 1.0, 1.0, 1.0, 0.5]) ** 2  # Larger initial uncertainty

    ekf = create_lc_fusion_ekf(initial_state=x0, initial_cov=P0)

    if verbose:
        print("\nInitialization:")
        print(f"  State: {x0}")
        print(f"  Gating: {'Enabled' if use_gating else 'Disabled'}")
        if use_gating:
            print(
                f"  Confidence: {gate_confidence} ({gate_confidence*100:.0f}% confidence)"
            )

    # Create position measurement model
    h, H_func, R_func = create_lc_position_measurement_model()

    # Prepare timestamped measurements
    measurements: List[StampedMeasurement] = []

    # Add IMU measurements
    for i in range(len(imu["t"])):
        measurements.append(
            StampedMeasurement(
                t=imu["t"][i],
                sensor="imu",
                z=np.hstack([imu["accel_xy"][i], imu["gyro_z"][i]]),  # [ax, ay, gz]
                R=np.eye(3),  # Not used
                meta={},
            )
        )

    # Add UWB measurements (aggregate by timestamp)
    for i in range(len(uwb["t"])):
        measurements.append(
            StampedMeasurement(
                t=uwb["t"][i],
                sensor="uwb",
                z=uwb["ranges"][i, :],  # All ranges at this timestamp
                R=np.eye(anchors.shape[0]),  # Not used (WLS computes own cov)
                meta={"epoch_idx": i},
            )
        )

    # Sort by timestamp
    measurements.sort(key=lambda m: m.t)

    if verbose:
        print("\nMeasurements:")
        print(f"  IMU samples: {len(imu['t'])}")
        print(f"  UWB epochs: {len([m for m in measurements if m.sensor == 'uwb'])}")
        print(f"  Total: {len(measurements)}")

    # Create adaptive gating manager (if gating enabled)
    adaptive_mgr = None
    if use_gating:
        adaptive_mgr = create_adaptive_manager_for_lc(
            consecutive_reject_limit=3,  # Lower limit for faster adaptation
            nis_window_size=20,
            nis_scale_threshold=2.0,  # More tolerant threshold (allow 2x NIS before scaling)
            P_inflation_factor=2.0,  # Larger inflation for faster recovery
            R_scale_factor=1.5,  # Larger R scaling steps
        )

    # Run fusion
    history = {
        "t": [],
        "x_est": [],
        "P_trace": [],
        "innovations": [],
        "nis": [],
        "gated": [],
        "uwb_positions": [],  # Store solved UWB positions for analysis
        "R_scales": [],
    }

    n_uwb_accepted = 0
    n_uwb_rejected = 0
    n_uwb_failed = 0
    t_prev = measurements[0].t

    for idx, meas in enumerate(measurements):
        dt = meas.t - t_prev

        if meas.sensor == "imu":
            # Propagate with IMU
            u = meas.z  # [ax, ay, gyro_z]
            ekf.predict(u=u, dt=dt)

        elif meas.sensor == "uwb":
            # Solve for UWB position fix
            ranges = meas.z  # All ranges at this epoch

            # Use current EKF position as initial guess
            initial_guess = ekf.state[:2]

            pos_uwb, cov_uwb, converged = solve_uwb_position_wls(
                ranges=ranges,
                anchor_positions=anchors,
                initial_guess=initial_guess,
                range_noise_std=range_noise_std,
                # No cov_floor_std: the dataset is line-of-sight with no
                # unmodeled error to floor for, so the honest WLS covariance
                # (H^T W H)^-1 is what the EKF and the chi-square gate should
                # see. See solve_uwb_position_wls's docstring.
            )

            if pos_uwb is None or not converged:
                # WLS solver failed (too few valid ranges or diverged)
                n_uwb_failed += 1
                continue

            # Store solved position
            history["uwb_positions"].append(pos_uwb)

            # Compute innovation (position residual)
            z_pred = h(ekf.state)  # Predicted position [px, py]
            y = innovation(pos_uwb, z_pred)

            # Compute innovation covariance
            # Use WLS covariance + state covariance
            H = H_func(ekf.state)
            R_base = cov_uwb  # Use WLS-computed covariance

            # Apply adaptive R scaling if using adaptive gating
            if adaptive_mgr is not None:
                R_scale = adaptive_mgr.get_R_scale()
                R = R_scale * R_base
            else:
                R = R_base
                R_scale = 1.0

            S = innovation_covariance(H, ekf.covariance, R)

            # Compute NIS for monitoring
            nis_value = mahalanobis_distance_squared(y, S)

            # Gating with adaptive management
            accept = True
            if use_gating:
                # First check with chi-square gate
                gate_accept = chi_square_gate(y, S, confidence=gate_confidence)

                # Update adaptive manager (may override decision or request action)
                accept, action = adaptive_mgr.update(nis_value, gate_accept)

                # Handle adaptive actions
                if action == "inflate_P":
                    # Apply covariance inflation to prevent filter starvation
                    ekf.covariance = adaptive_mgr.inflate_covariance(ekf.covariance)
                # 'scale_R' action is handled automatically via get_R_scale()

            if accept:
                # Perform EKF update with position fix
                K = ekf.covariance @ H.T @ np.linalg.inv(S)
                ekf.state = ekf.state + (K @ y).flatten()
                ekf.covariance = (np.eye(5) - K @ H) @ ekf.covariance
                n_uwb_accepted += 1
            else:
                n_uwb_rejected += 1

            # Log
            history["innovations"].append(np.linalg.norm(y))  # 2D innovation norm
            history["nis"].append(nis_value)
            history["gated"].append(accept)
            history["R_scales"].append(R_scale)

        # Record state
        history["t"].append(meas.t)
        history["x_est"].append(ekf.state.copy())
        history["P_trace"].append(np.trace(ekf.covariance))

        t_prev = meas.t

    # Convert to arrays
    history["t"] = np.array(history["t"])
    history["x_est"] = np.array(history["x_est"])
    history["P_trace"] = np.array(history["P_trace"])
    if history["uwb_positions"]:
        history["uwb_positions"] = np.array(history["uwb_positions"])
    history["n_uwb_accepted"] = n_uwb_accepted
    history["n_uwb_rejected"] = n_uwb_rejected
    history["n_uwb_failed"] = n_uwb_failed

    if verbose:
        print("\nFusion complete:")
        print(f"  UWB position fixes solved: {n_uwb_accepted + n_uwb_rejected}")
        print(f"  UWB fixes accepted: {n_uwb_accepted}")
        print(f"  UWB fixes rejected: {n_uwb_rejected}")
        print(f"  UWB solver failures: {n_uwb_failed}")
        if n_uwb_accepted + n_uwb_rejected > 0:
            print(
                f"  Acceptance rate: {100*n_uwb_accepted/(n_uwb_accepted+n_uwb_rejected):.1f}%"
            )

        # Print adaptive gating stats if enabled
        if adaptive_mgr is not None:
            stats = adaptive_mgr.get_stats()
            print("\nAdaptive Gating Stats:")
            print(
                f"  Mean NIS: {stats['mean_nis']:.2f} (expected: {stats['expected_nis']:.0f})"
            )
            print(f"  Final R scale: {stats['current_R_scale']:.2f}x")
            print(f"  Covariance inflations: {stats['total_adaptations']}")

    return history
