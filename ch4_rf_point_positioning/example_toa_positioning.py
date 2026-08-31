"""
TOA and RSS Positioning Example.

This script demonstrates Time of Arrival (TOA) and RSS-based positioning
using iterative least squares. Every solve here is `method="iterative_ls"`,
W = I and Eq. (4.20); Example 6 is the one that supplies a covariance and
so is genuinely weighted (Eq. 4.23).

One-way TOA carries an assumption that is easy to read past, so it is stated
wherever it is used rather than only where it is relaxed: Examples 1, 2 and 4
take the beacon and agent clocks to be already synchronised, which is what
makes a measured time of flight a range at all. At c, 1 ns of unmodelled
offset is 0.30 m of range error. Example 3 keeps the offset as an unknown
(Eqs. 4.24-4.26) and Example 7 prices the assumption against the two-way
protocol that removes it.

Implements:
    - Eq. (4.1)-(4.3): TOA range measurements
    - Eq. (4.6)-(4.9): Two-way TOA / RTT measurement model
    - Eq. (4.11)-(4.13): RSS path-loss model
    - Eq. (4.14)-(4.23): Nonlinear TOA iterative LS, and WLS in Example 6
    - Eq. (4.24)-(4.26): Joint position + clock bias estimation
    - Example 7: one-way (synchronised and not) against two-way, one budget

Author: Li-Ta Hsu
Date: December 2025
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import save_figure, show_figures_if_requested
from core.rf import (
    SPEED_OF_LIGHT,
    TOAPositioner,
    compute_dop,
    position_error_from_dop,
    range_to_rtt,
    rss_pathloss,
    rss_to_distance,
    rtt_to_range,
    simulate_rtt_measurement,
    solve_batch,
    toa_range,
    toa_solve_with_clock_bias,
)

# Shared by the global seeding and by the Monte Carlo in Example 2.
SEED = 42


def example_toa_perfect():
    """Example 1: TOA positioning with perfect measurements.

    "Perfect" here is two assumptions, not one: the ranges carry no
    measurement noise, *and* the beacon and agent clocks are already
    synchronised, so a one-way time of flight converts straight to a range
    (Eq. 4.1 with c*Delta_t = 0). The second is the one nothing on the
    screen would otherwise mention, and it is the expensive one -- Example 7
    solves this same protocol with the clocks left alone.
    """
    print("=" * 70)
    print("Example 1: TOA Positioning with Perfect Measurements")
    print("=" * 70)

    # Square anchor layout
    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos = np.array([5.0, 5.0])

    print(f"\nAnchor positions:\n{anchors}")
    print(f"True position: {true_pos}")

    print(
        "\nAssumption: one-way TOA as solved here needs the beacon and agent"
        "\n  clocks already synchronised -- that is what turns a measured time"
        f"\n  of flight into a range. At c, 1 ns of unmodelled offset is"
        f" {SPEED_OF_LIGHT * 1e-9:.3f} m"
        "\n  of range error. Example 3 estimates the offset; Example 7 prices"
        "\n  the assumption against two-way TOA, which does not need it."
    )

    # Compute true ranges
    ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])
    print(f"\nTrue ranges: {ranges}")

    # Solve using iterative LS (book default: Eq. 4.20)
    positioner = TOAPositioner(anchors, method="iterative_ls")
    estimated_pos, info = positioner.solve(ranges, initial_guess=np.array([6.0, 6.0]))

    # Results
    error = np.linalg.norm(estimated_pos - true_pos)
    print(f"\nEstimated position: {estimated_pos}")
    print(f"Position error: {error:.6f} m")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Residual: {info['residual']:.2e}")

    return anchors, true_pos, estimated_pos, info


def example_toa_with_noise():
    """Example 2: TOA positioning with measurement noise.

    Still one-way, and still synchronised: the 0.1 m here is *measurement*
    noise about a range the clocks already agree on. 0.1 m is 0.33 ns, so
    this example's whole error budget is smaller than the clock offset a
    consumer oscillator reaches in a fraction of a second. That is the
    comparison Example 7 makes, and it is why the numbers below should not be
    read as what one-way TOA achieves in the field.
    """
    print("\n" + "=" * 70)
    print("Example 2: TOA Positioning with Measurement Noise")
    print("=" * 70)

    np.random.seed(SEED)

    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos = np.array([3.0, 7.0])

    print(f"\nTrue position: {true_pos}")

    # Add Gaussian noise to ranges
    true_ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])
    noise_std = 0.1  # 10 cm standard deviation
    ranges_noisy = true_ranges + np.random.randn(4) * noise_std

    print(
        f"Range noise std: {noise_std} m"
        f"  ({noise_std * 1e9 / SPEED_OF_LIGHT:.2f} ns; clocks still assumed"
        " synchronised)"
    )
    print(f"True ranges:  {true_ranges}")
    print(f"Noisy ranges: {ranges_noisy}")

    # Solve using iterative LS (book default: Eq. 4.20)
    positioner = TOAPositioner(anchors, method="iterative_ls")
    estimated_pos, info = positioner.solve(
        ranges_noisy, initial_guess=np.array([5.0, 5.0])
    )

    # Results
    error = np.linalg.norm(estimated_pos - true_pos)
    print(f"\nEstimated position: {estimated_pos}")
    print(f"Position error: {error:.3f} m   <- ONE noise draw, not an accuracy")
    print(f"Iterations: {info['iterations']}")

    # A single draw says almost nothing: repeating this solve puts the error
    # anywhere between about 0.03 and 0.15 m. Dividing one draw by the noise
    # and calling it an "error/noise ratio", which this example used to do,
    # reads as though positioning were twice as good as its ranging.
    #
    # The quantity that does characterise the geometry is Eq. (4.107),
    # sigma_position = HDOP * sigma_range, and it is worth showing that the
    # solver attains it rather than asserting it.
    geometry = (true_pos - anchors) / np.linalg.norm(
        true_pos - anchors, axis=1, keepdims=True
    )
    hdop = compute_dop(geometry)["HDOP"]
    predicted = position_error_from_dop(hdop, noise_std)

    trials = 2000
    rng = np.random.default_rng(SEED)
    errors = np.empty(trials)
    for k in range(trials):
        noisy = true_ranges + rng.standard_normal(len(anchors)) * noise_std
        estimate, _ = TOAPositioner(anchors, method="iterative_ls").solve(
            noisy, initial_guess=np.array([5.0, 5.0])
        )
        errors[k] = np.linalg.norm(estimate - true_pos)
    measured = float(np.sqrt(np.mean(errors**2)))

    print(f"\nOver {trials} noise draws, against Eq. (4.107):")
    print(f"  HDOP for this geometry : {hdop:.3f}")
    print(f"  predicted HDOP x sigma : {predicted:.4f} m")
    print(
        f"  measured RMS error     : {measured:.4f} m  "
        f"({measured / predicted:.2f}x predicted)"
    )
    print(
        f"  a single draw lands anywhere in "
        f"[{np.percentile(errors, 10):.3f}, "
        f"{np.percentile(errors, 90):.3f}] m (10th-90th percentile)"
    )

    return anchors, true_pos, estimated_pos


def example_toa_with_clock_bias():
    """
    Example 3: TOA positioning with unknown clock bias.

    Demonstrates the unit convention for clock bias:
    - Measurement model (`toa_range`): clock_bias_s in SECONDS
    - Positioning solver: clock_bias_m in METERS (book Eq. 4.24)
    - Conversion: bias_m = c * bias_s, bias_s = bias_m / c

    The book uses meters because the Jacobian ∂h/∂(c*Δt) = 1 (Eq. 4.26).
    """
    print("\n" + "=" * 70)
    print("Example 3: Joint Position and Clock Bias Estimation")
    print("=" * 70)

    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos = np.array([5.0, 5.0])

    # Define clock bias in SECONDS (timing domain)
    # Then convert to METERS for the solver
    true_clock_bias_s = 10e-9  # 10 nanoseconds
    true_clock_bias_m = true_clock_bias_s * SPEED_OF_LIGHT  # ~3.0 meters

    print(f"\nTrue position: {true_pos}")
    print(
        f"True clock bias: {true_clock_bias_s*1e9:.2f} ns = {true_clock_bias_m:.3f} m"
    )
    print(f"  (1 ns = {SPEED_OF_LIGHT*1e-9:.3f} m, 1 m = {1e9/SPEED_OF_LIGHT:.3f} ns)")

    # Compute ranges WITH clock bias using measurement model
    # toa_range() takes clock_bias_s in SECONDS
    ranges_biased = np.array(
        [
            toa_range(anchor, true_pos, clock_bias_s=true_clock_bias_s)
            for anchor in anchors
        ]
    )

    # Also compute true geometric ranges (no bias)
    true_ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])

    print(f"\nTrue geometric ranges: {true_ranges}")
    print(f"Measured pseudoranges: {ranges_biased}")
    print(f"Difference (bias_m):   {ranges_biased - true_ranges}")

    # Solve with clock bias estimation
    # The solver estimates bias in METERS (book convention)
    initial_guess = np.array([6.0, 6.0, 0.0])  # [x, y, bias_m]
    pos, bias_m, info = toa_solve_with_clock_bias(anchors, ranges_biased, initial_guess)

    # Convert estimated bias from meters to seconds for interpretation
    bias_s = bias_m / SPEED_OF_LIGHT

    # Results
    pos_error = np.linalg.norm(pos - true_pos)
    bias_error_m = abs(bias_m - true_clock_bias_m)
    bias_error_ns = abs(bias_s - true_clock_bias_s) * 1e9

    print("\n--- Results ---")
    print(f"Estimated position: {pos}")
    print(f"Position error: {pos_error:.6f} m")
    print("\nEstimated clock bias:")
    print(f"  In meters:  {bias_m:.6f} m (error: {bias_error_m:.6f} m)")
    print(f"  In seconds: {bias_s*1e9:.3f} ns (error: {bias_error_ns:.3f} ns)")
    print(f"Iterations: {info['iterations']}, Converged: {info['converged']}")

    return anchors, true_pos, pos


def example_rss_positioning():
    """Example 4: RSS-based ranging and positioning.

    Uses the log-distance path-loss model (Eq. 4.11):
        P_R(d) = p_ref - 10 * eta * log10(d / d_ref)

    where ``p_ref`` is the received power at the reference distance
    ``d_ref = 1 m``.  A typical Wi-Fi beacon yields about -40 dBm at 1 m.
    """
    print("\n" + "=" * 70)
    print("Example 4: RSS-Based Ranging")
    print("=" * 70)

    # Reference received power at d_ref = 1 m (typical Wi-Fi beacon)
    p_ref_dbm = -40.0  # dBm at 1 m
    path_loss_exp = 2.5  # Indoor environment

    distances = np.array([1.0, 5.0, 10.0, 20.0])

    print(f"\nReference power (d=1m): {p_ref_dbm} dBm")
    print(f"Path loss exponent: {path_loss_exp}")
    print("\nDistance -> RSS -> Estimated Distance:")

    for dist in distances:
        rss = rss_pathloss(p_ref_dbm, dist, path_loss_exp)
        dist_est = rss_to_distance(rss, p_ref_dbm, path_loss_exp)
        print(f"  {dist:5.1f} m -> {rss:7.2f} dBm -> {dist_est:5.1f} m")

    # RSS-based positioning example
    print("\nRSS-Based Positioning:")
    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos = np.array([5.0, 5.0])

    rss_measurements = []
    for anchor in anchors:
        dist = np.linalg.norm(anchor - true_pos)
        rss = rss_pathloss(p_ref_dbm, dist, path_loss_exp)
        rss_measurements.append(rss)

    rss_measurements = np.array(rss_measurements)
    print(f"RSS measurements: {rss_measurements}")

    ranges_from_rss = np.array(
        [rss_to_distance(rss, p_ref_dbm, path_loss_exp) for rss in rss_measurements]
    )
    print(f"Estimated ranges: {ranges_from_rss}")

    positioner = TOAPositioner(anchors, method="iterative_ls")
    estimated_pos, info = positioner.solve(
        ranges_from_rss, initial_guess=np.array([6.0, 6.0])
    )

    error = np.linalg.norm(estimated_pos - true_pos)
    print(f"Estimated position: {estimated_pos}")
    print(f"Position error: {error:.3f} m")


def plot_toa_positioning(anchors, true_pos, estimated_pos, history=None):
    """Visualize TOA positioning results."""
    plt.figure(figsize=(8, 8))

    # Plot anchors
    plt.scatter(
        anchors[:, 0],
        anchors[:, 1],
        s=200,
        c="red",
        marker="^",
        label="Anchors",
        zorder=5,
    )

    # Label anchors
    for i, anchor in enumerate(anchors):
        plt.text(
            anchor[0],
            anchor[1] + 0.5,
            f"A{i+1}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Plot true position
    plt.scatter(
        true_pos[0],
        true_pos[1],
        s=150,
        c="green",
        marker="o",
        label="True Position",
        zorder=4,
    )

    # Plot estimated position
    plt.scatter(
        estimated_pos[0],
        estimated_pos[1],
        s=150,
        c="blue",
        marker="x",
        label="Estimated Position",
        linewidths=3,
        zorder=4,
    )

    # Plot iteration history if available
    if history is not None and len(history) > 1:
        plt.plot(
            history[:, 0],
            history[:, 1],
            "b--",
            alpha=0.5,
            label="Convergence Path",
            zorder=3,
        )

    # Plot range circles
    for anchor in anchors:
        dist = np.linalg.norm(anchor - true_pos)
        circle = plt.Circle(
            anchor, dist, fill=False, color="red", alpha=0.2, linestyle="--"
        )
        plt.gca().add_patch(circle)

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("East (m)", fontsize=12)
    plt.ylabel("North (m)", fontsize=12)
    plt.title("TOA Positioning", fontsize=14, fontweight="bold")
    plt.legend(loc="best")

    plt.tight_layout()
    return plt.gcf()


def example_rtt_measurement():
    """Example 5: Two-way TOA / RTT measurement model (Eqs. 4.6-4.9).

    This walks the mechanism -- processing time, drift, and what each costs
    if it is not corrected. Its timing budget is deliberately loose (5 ns of
    processing-time uncertainty, 2 ns of drift) so the effects are visible,
    which means the metre-scale numbers below are **not** comparable with
    Examples 1-2: those solve one-way ranges at 0.1 m of measurement noise
    and assume the clocks are already synchronised, an assumption worth
    metres in its own right. Example 7 makes the comparison properly, with
    one timing budget shared by both protocols.
    """
    print("\n" + "=" * 70)
    print("Example 5: Two-Way TOA / RTT Measurement Model (Eqs. 4.6-4.9)")
    print("=" * 70)

    print("\n--- Basic RTT to Range Conversion (Eq. 4.7) ---")

    # Example: Wi-Fi FTM measurement
    print("\nWi-Fi FTM Example:")
    print(f"  Speed of light: {SPEED_OF_LIGHT:.0f} m/s")
    print(f"  1 nanosecond timing -> {SPEED_OF_LIGHT * 1e-9 / 2:.3f} m range error")

    # RTT for 15m distance
    distance = 15.0
    rtt_ideal = range_to_rtt(distance)
    print(f"\n  True distance: {distance:.1f} m")
    print(f"  Ideal RTT: {rtt_ideal * 1e9:.2f} ns")

    # Convert back to range
    range_est = rtt_to_range(rtt_ideal)
    print(f"  Range from RTT: {range_est:.6f} m")

    print("\n--- RTT with Processing Time (Eq. 4.7) ---")

    # Typical Wi-Fi FTM processing time: 10-100 ns
    processing_time = 50e-9  # 50 ns
    print(f"\n  Beacon processing time: {processing_time * 1e9:.0f} ns")

    # RTT includes processing time
    rtt_with_proc = range_to_rtt(distance, processing_time=processing_time)
    print(f"  RTT with processing: {rtt_with_proc * 1e9:.2f} ns")

    # Without correction: overestimate distance
    range_wrong = rtt_to_range(rtt_with_proc)
    print(
        f"\n  Range without correction: {range_wrong:.2f} m (ERROR: +{range_wrong - distance:.2f} m)"
    )

    # With correction: correct distance
    range_correct = rtt_to_range(rtt_with_proc, processing_time=processing_time)
    print(f"  Range with correction: {range_correct:.6f} m")

    print("\n--- RTT with Clock Drift (Eq. 4.8) ---")

    # TCXO clock drift example: ~1-2 ppm
    # For 100ns RTT, 1 ppm drift -> 0.1 ns error
    clock_drift = 5e-9  # 5 ns drift
    print(f"\n  Agent clock drift: {clock_drift * 1e9:.0f} ns")

    rtt_with_drift = rtt_with_proc + clock_drift
    print(f"  RTT with drift: {rtt_with_drift * 1e9:.2f} ns")

    # Correct for both processing and drift
    range_corrected = rtt_to_range(
        rtt_with_drift, processing_time=processing_time, clock_drift=clock_drift
    )
    print(f"  Range with full correction: {range_corrected:.6f} m")

    print("\n--- Simulated RTT Measurement with Noise (Eq. 4.9) ---")

    np.random.seed(SEED)

    anchor = np.array([0.0, 0.0, 0.0])
    agent = np.array([15.0, 0.0, 0.0])

    # Single measurement
    rtt, info = simulate_rtt_measurement(
        anchor,
        agent,
        processing_time=50e-9,
        processing_time_std=5e-9,  # 5 ns std
        clock_drift_std=2e-9,  # 2 ns std
    )

    print(f"\n  True range: {info['true_range']:.2f} m")
    print(f"  Processing time (actual): {info['processing_time_actual'] * 1e9:.2f} ns")
    print(f"  Clock drift (actual): {info['clock_drift_actual'] * 1e9:.2f} ns")
    print(f"  Measured RTT: {rtt * 1e9:.2f} ns")
    print(f"  Estimated range: {info['range_estimate']:.3f} m")
    print(f"  Range error: {info['range_estimate'] - 15.0:.3f} m")

    # Monte Carlo simulation
    print("\n  Monte Carlo (100 trials):")
    errors = []
    for _ in range(100):
        _, info = simulate_rtt_measurement(
            anchor,
            agent,
            processing_time=50e-9,
            processing_time_std=5e-9,
            clock_drift_std=2e-9,
        )
        errors.append(info["range_estimate"] - 15.0)

    errors = np.array(errors)
    # What should this be? Only the *uncorrected* parts of the RTT survive:
    # sigma_RTT = sqrt(sigma_proc^2 + sigma_drift^2), halved by Eq. (4.7).
    predicted_rtt_sigma = SPEED_OF_LIGHT * np.hypot(5e-9, 2e-9) / 2.0
    print(f"    Mean error: {np.mean(errors):.4f} m")
    print(f"    Std dev: {np.std(errors):.4f} m")
    print(f"    RMSE: {np.sqrt(np.mean(errors**2)):.4f} m")
    print(
        f"    predicted c*sqrt(5ns^2 + 2ns^2)/2: {predicted_rtt_sigma:.4f} m"
        "   (100 draws, so +/- ~7%)"
    )

    print("\n--- RTT-Based Positioning Example ---")

    # Multiple anchors
    anchors = np.array(
        [
            [0, 0, 0],
            [20, 0, 0],
            [20, 20, 0],
            [0, 20, 0],
        ],
        dtype=float,
    )
    true_pos = np.array([8.0, 12.0, 0.0])

    print(f"\n  True position: {true_pos[:2]}")

    # Simulate RTT measurements from each anchor
    ranges_from_rtt = []
    for i, anchor in enumerate(anchors):
        rtt, info = simulate_rtt_measurement(
            anchor,
            true_pos,
            processing_time=50e-9,
            processing_time_std=3e-9,
        )
        ranges_from_rtt.append(info["range_estimate"])
        print(
            f"  Anchor {i+1}: RTT={rtt*1e9:.1f}ns -> Range={info['range_estimate']:.3f}m "
            f"(true: {info['true_range']:.2f}m)"
        )

    ranges_from_rtt = np.array(ranges_from_rtt)

    # Position using TOA solver
    positioner = TOAPositioner(anchors[:, :2], method="iterative_ls")
    est_pos, info = positioner.solve(
        ranges_from_rtt, initial_guess=np.array([10.0, 10.0])
    )

    error = np.linalg.norm(est_pos - true_pos[:2])
    # This run leaves clock_drift_std at 0, so only the 3 ns of processing-time
    # uncertainty reaches the range: sigma_range = c * 3 ns / 2.
    single_draw_sigma = SPEED_OF_LIGHT * 3e-9 / 2.0
    print(f"\n  Estimated position: {est_pos}")
    print(f"  Position error: {error:.3f} m   <- ONE draw, not an accuracy")
    print(
        f"  (range sigma here is c*3ns/2 = {single_draw_sigma:.3f} m, and this"
        " array's HDOP is ~1,"
        f"\n   so ~{single_draw_sigma:.2f} m is the scale to expect."
        " Example 7 reports a distribution.)"
    )

    return


#: Iteration budget for Example 6. The library default is 10, and the LS arm
#: needs more than that here: see the docstring below for what happened when it
#: did not get them.
WLS_DEMO_MAX_ITERS = 50


def example_wls_vs_ls():
    """Example 6: Weighted Least Squares vs ordinary LS (Eq. 4.23).

    Uses an asymmetric anchor layout with heterogeneous per-anchor
    measurement noise to demonstrate the benefit of WLS weighting.

    **Every draw is reported, and the iteration budget is what made that
    possible.** This used to keep only the draws whose ``info["converged"]``
    was True, which at the library's default ``max_iters=10`` was 120 of 200
    for LS against 200 of 200 for WLS -- so the two columns were computed over
    different samples, and the LS column over a sample selected by how quickly
    LS converged. The 80 discarded draws were the *worse* half: RMSE 0.416 m
    against 0.181 m for the survivors, and 0.298 m over all 200. The
    "improvement" therefore read 23.1% where the honest figure is 53.4%.

    Nothing had actually failed. ``converged`` here is a **step-tolerance
    stop** -- ``norm(delta) < tol`` with ``tol = 1e-6`` m -- and the residual
    test above it cannot fire at all on noisy data, since four ranges over two
    unknowns leave a residual the solve can never drive to zero. Raising the
    budget to 50 gives 200 of 200 for both arms and moves the LS RMSE by 1e-4 m:
    the extra iterations were polishing a converged answer, not rescuing a
    failed one. A flag whose real meaning is "finished within the budget" is
    not a sample selector, and reporting over its survivors is the survivor
    statistic ``.cursor/rules/030-figures-and-claims.mdc`` warns about.

    The number is checked against theory rather than admired: for a linear
    model with covariance Sigma, WLS attains ``(H' Sigma^-1 H)^-1`` and LS
    attains ``(H'H)^-1 H' Sigma H (H'H)^-1``, which for this geometry predict
    0.143 m and 0.287 m -- a 50.1% improvement, against the 53.4% measured.
    """
    print("\n" + "=" * 70)
    print("Example 6: WLS vs LS with Asymmetric Geometry")
    print("=" * 70)

    np.random.seed(SEED)

    # Deliberately asymmetric layout: three close anchors on the left,
    # one distant anchor on the right.
    anchors = np.array(
        [[0, 0], [0, 8], [2, 4], [15, 5]],
        dtype=float,
    )
    true_pos = np.array([6.0, 4.0])

    # Per-anchor noise std (far anchor has much higher noise)
    sigma_per_anchor = np.array([0.1, 0.1, 0.1, 0.8])

    print(f"\nAnchors:\n{anchors}")
    print(f"True position: {true_pos}")
    print(f"Per-anchor noise std (m): {sigma_per_anchor}")

    n_trials = 200
    errors_ls = []
    errors_wls = []
    stopped_ls = 0
    stopped_wls = 0
    cov = np.diag(sigma_per_anchor**2)

    for _ in range(n_trials):
        true_ranges = np.array([toa_range(a, true_pos) for a in anchors])
        noisy_ranges = true_ranges + np.random.randn(len(anchors)) * sigma_per_anchor

        init = np.array([5.0, 5.0])

        pos_ls, info_ls = TOAPositioner(anchors, method="iterative_ls").solve(
            noisy_ranges,
            initial_guess=init,
            max_iters=WLS_DEMO_MAX_ITERS,
        )
        pos_wls, info_wls = TOAPositioner(anchors, method="iterative_wls").solve(
            noisy_ranges,
            initial_guess=init,
            covariance=cov,
            max_iters=WLS_DEMO_MAX_ITERS,
        )

        # Every draw is scored. The flag is recorded beside it rather than used
        # to select the sample -- it says "the step fell below 1e-6 m within
        # the budget", which is not the same question as "did this solve work".
        errors_ls.append(np.linalg.norm(pos_ls - true_pos))
        errors_wls.append(np.linalg.norm(pos_wls - true_pos))
        stopped_ls += bool(info_ls["converged"])
        stopped_wls += bool(info_wls["converged"])

    errors_ls = np.array(errors_ls)
    errors_wls = np.array(errors_wls)

    rmse_ls = np.sqrt(np.mean(errors_ls**2))
    rmse_wls = np.sqrt(np.mean(errors_wls**2))

    # What should these be? Eq. (4.23) is BLUE for this covariance, and both
    # estimators are linear in the measurements once the geometry is fixed.
    ranges_true = np.array([toa_range(a, true_pos) for a in anchors])
    jac = -(anchors - true_pos) / ranges_true[:, None]
    pinv = np.linalg.inv(jac.T @ jac) @ jac.T
    rmse_ls_theory = np.sqrt(np.trace(pinv @ cov @ pinv.T))
    rmse_wls_theory = np.sqrt(np.trace(np.linalg.inv(jac.T @ np.linalg.inv(cov) @ jac)))

    print(f"\nMonte-Carlo results over all {n_trials} trials (no draw discarded):")
    print(
        f"  LS  RMSE: {rmse_ls:.3f} m   (median {np.median(errors_ls):.3f} m, "
        f"step tolerance reached {stopped_ls}/{n_trials})"
    )
    print(
        f"  WLS RMSE: {rmse_wls:.3f} m   (median {np.median(errors_wls):.3f} m, "
        f"step tolerance reached {stopped_wls}/{n_trials})"
    )
    print(f"  WLS improvement: {(1 - rmse_wls / rmse_ls) * 100:.1f}%")
    print(
        f"  Predicted by Eq. (4.23): LS {rmse_ls_theory:.3f} m, "
        f"WLS {rmse_wls_theory:.3f} m, "
        f"improvement {(1 - rmse_wls_theory / rmse_ls_theory) * 100:.1f}%"
    )
    print("\n  -> WLS down-weights the noisy far anchor, improving accuracy.")
    print("     Reporting only the draws that reported convergence made this")
    print("     23.1%: at the default max_iters=10 the LS arm reached the step")
    print("     tolerance on 120 of 200 draws, and the 80 it dropped were the")
    print("     worse half (RMSE 0.416 m against 0.181 m). Nothing had failed --")
    print(f"     with max_iters={WLS_DEMO_MAX_ITERS} both arms finish on every")
    print("     draw and the LS RMSE moves by 1e-4 m.")


#: Example 7's timing budget. One piece of hardware, three protocols: the
#: agent's clock stamps an event with this much jitter whatever the protocol
#: asks it to do, so this is the single number every case is derived from.
TIMESTAMP_JITTER_S = 1.0e-9

#: The receiver clock offset Case B carries. An uncorrected consumer
#: oscillator at a few ppm reaches tens of nanoseconds in well under a second,
#: which is why one-way systems need a synchronisation protocol and not merely
#: a good crystal.
CLOCK_OFFSET_S = 20.0e-9

#: What is left of the responder's turnaround time after calibration. This is
#: the term two-way TOA pays *instead of* synchronisation, and the comparison
#: is only honest with it present -- see `two_way_range_sigma`.
TURNAROUND_RESIDUAL_S = 1.0e-9

#: Nominal responder turnaround, known to the agent and removed by
#: `rtt_to_range` (Eq. 4.7). Only the residual above survives the correction.
TURNAROUND_NOMINAL_S = 50.0e-9

#: Draws per case in Example 7. Large enough that the RMSE column is a
#: measurement rather than a draw: the standard error on an RMSE over n draws
#: is about 1/sqrt(2n), so 2000 gives ~1.6%.
TWO_WAY_TRIALS = 2000

#: Iteration budget for Example 7, for the reason recorded on
#: WLS_DEMO_MAX_ITERS: the library default of 10 makes `converged` a report on
#: how quickly a solve finished rather than on whether it worked.
TWO_WAY_MAX_ITERS = 50


def one_way_range_sigma(jitter_s: float) -> float:
    """Ranging noise of a one-way TOA measurement, from per-timestamp jitter.

    A downlink TOA is one *receive* timestamp read against a transmit epoch
    the synchronised network schedules, so the whole jitter lands on the
    range:

        sigma_range = c * sigma_t.

    The scheduled epoch is treated as noiseless, which is the assumption that
    flatters one-way most. Giving the transmit event its own sigma_t would
    make this c * sqrt(2) * sigma_t and double the gap to two-way below, so
    the bookkeeping here is the conservative one.
    """
    return SPEED_OF_LIGHT * jitter_s


def two_way_range_sigma(jitter_s: float, turnaround_residual_s: float) -> float:
    """Ranging noise of a two-way (RTT) measurement, from the same jitter.

    The agent stamps departure and arrival on its **own** clock, so two
    independent jitters enter the round-trip interval, and the responder's
    calibrated turnaround time contributes whatever survives calibration:

        sigma_RTT   = sqrt(2 * sigma_t^2 + sigma_proc^2)     (Eq. 4.9)
        sigma_range = c * sigma_RTT / 2                      (Eq. 4.7)

    That division is worth pausing on, because it is why the measured table
    does not simply order itself the way "two-way costs an extra hop" would
    suggest. With sigma_proc = 0 the two-way figure is c * sigma_t / sqrt(2),
    a factor sqrt(2) **better** than one-way: a round trip puts twice the
    distance into the interval being timed while the jitter grows only as
    sqrt(2). The turnaround residual buys it back, and the two protocols are
    exactly equal at sigma_proc = sqrt(2) * sigma_t.

    So the honest statement of two-way's advantage is not that it ranges
    better. It is that it needs no synchronised clock, and the sqrt(2) is a
    property of the arithmetic that has to be reported rather than hidden.
    """
    sigma_rtt = float(np.hypot(np.sqrt(2.0) * jitter_s, turnaround_residual_s))
    return SPEED_OF_LIGHT * sigma_rtt / 2.0


class _ClockBiasSolver:
    """Adapter letting :func:`core.rf.solve_batch` drive Eqs. (4.24)-(4.26).

    ``toa_solve_with_clock_bias`` is a function taking an ``(x, y, c*dt)``
    seed, not a positioner. Wrapping it keeps Case B(ii) on the same four
    failure tests as the other three arms -- raised, refused, stalled,
    diverged -- rather than on whichever subset a hand-rolled loop remembers.
    ``example_comparison`` carries the same adapter for the same reason.
    """

    def __init__(self, anchors):
        self.anchors = np.asarray(anchors, dtype=float)

    def solve(self, ranges, initial_guess, **kwargs):
        state = np.concatenate([np.asarray(initial_guess, dtype=float), [0.0]])
        position, _bias, info = toa_solve_with_clock_bias(
            self.anchors, ranges, state, **kwargs
        )
        return position, info


def _simulate_rtt_ranges(anchors, true_pos, jitter_s, turnaround_residual_s, rng):
    """One row of RTT-derived ranges, one per anchor (Eqs. 4.7-4.9).

    The agent's two timestamps enter through ``clock_drift_std`` and the
    responder's calibration residual through ``processing_time_std``, which is
    what ``simulate_rtt_measurement`` already models: both are added to the
    RTT and only the *nominal* turnaround is corrected out again.
    """
    ranges = np.empty(len(anchors))
    for i, anchor in enumerate(anchors):
        _, info = simulate_rtt_measurement(
            anchor,
            true_pos,
            processing_time=TURNAROUND_NOMINAL_S,
            processing_time_std=turnaround_residual_s,
            clock_drift_std=np.sqrt(2.0) * jitter_s,
            rng=rng,
        )
        ranges[i] = info["range_estimate"]
    return ranges


def example_one_way_vs_two_way():
    """Example 7: one-way TOA (synchronised or not) against two-way TOA.

    Chapter 4 introduces two-way TOA (Eqs. 4.6-4.9) because it removes the
    synchronisation requirement one-way TOA carries silently. Read in file
    order, though, this file used to argue the opposite: Examples 1 and 2
    solve one-way ranges at 0.1 m of measurement noise and never mention the
    clock, while Example 5 measures RTT with an honest nanosecond timing
    budget and prints an error seven times larger. That is not a comparison
    between protocols, it is a comparison between one protocol's assumptions
    and another's physics.

    So this example puts all three on one anchor array, one agent position and
    one number: ``sigma_t``, the agent clock's per-timestamp jitter.

        A     one-way, clocks synchronised. The textbook assumption.
        B(i)  one-way, receiver clock offset by b, solved for position only.
        B(ii) the same measurements with the clock as a third unknown
              (Eqs. 4.24-4.26) -- correct, at the cost of one degree of
              freedom.
        C     two-way RTT, same hardware, no synchronisation anywhere.

    **The measured ordering is C < A ~ B(ii) << B(i), which is not quite the
    ordering the design predicted** -- C was expected to sit at or just behind
    A. It comes out 13% ahead instead, and the reason is arithmetic rather
    than a thumb on the scale: see :func:`two_way_range_sigma` for the sqrt(2)
    a round trip buys and the ``sigma_proc = sqrt(2) * sigma_t`` at which it
    is exactly repaid. The crossover is printed rather than argued, and the
    parameters were not moved to recover the expected order.

    What the table does support, and what the chapter needs it to say, is the
    B-versus-C half. Twenty nanoseconds of unmodelled offset -- less than a
    second of free-running drift on a few-ppm oscillator -- puts a position-
    only one-way fix 1.8 m out, on an array where the same hardware two-way
    lands at 0.22 m. The offset can be estimated instead, and B(ii) recovers
    almost all of it, which is the honest statement of the trade: one-way
    needs *either* a synchronisation protocol *or* a spare degree of freedom;
    two-way needs neither.

    Two-way does not thereby beat TDOA, and no run here claims it does.
    Chapter 4's own DOP result is that TDOA and TOA-with-an-estimated-clock
    carry the same information -- see the chapter README, where the two
    position DOPs agree to machine precision at all 100 grid points. The
    axis two-way TOA wins on is infrastructure, not accuracy.
    """
    print("\n" + "=" * 70)
    print("Example 7: One-Way vs Two-Way TOA Under One Timing Budget")
    print("=" * 70)

    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    true_pos = np.array([8.0, 12.0])
    seed_pos = np.array([10.0, 10.0])

    offset_m = CLOCK_OFFSET_S * SPEED_OF_LIGHT
    sigma_one_way = one_way_range_sigma(TIMESTAMP_JITTER_S)
    sigma_two_way = two_way_range_sigma(TIMESTAMP_JITTER_S, TURNAROUND_RESIDUAL_S)
    sigma_rtt_s = 2.0 * sigma_two_way / SPEED_OF_LIGHT

    print(
        f"\nAnchors: 20 m square, agent at {true_pos}, all solves seeded at {seed_pos}"
    )
    print("\nOne timing budget, three protocols:")
    print(f"  per-timestamp jitter sigma_t     : {TIMESTAMP_JITTER_S * 1e9:.3f} ns")
    print(
        f"  receiver clock offset b (case B) : {CLOCK_OFFSET_S * 1e9:.1f} ns"
        f" = {offset_m:.3f} m"
    )
    print(f"  turnaround residual (case C)     : {TURNAROUND_RESIDUAL_S * 1e9:.3f} ns")

    print("\nRange noise, derived from sigma_t rather than chosen per case:")
    print(
        f"  one-way : sigma_range = c*sigma_t                  = {sigma_one_way:.4f} m"
    )
    print(
        "  two-way : sigma_RTT = sqrt(2*st^2 + sp^2)          = "
        f"{sigma_rtt_s * 1e9:.4f} ns"
    )
    print(
        "            sigma_range = c*sigma_RTT/2              = "
        f"{sigma_two_way:.4f} m   ({sigma_two_way / sigma_one_way:.3f}x one-way)"
    )
    print(
        "  the round trip times twice the distance while the jitter grows only"
        "\n  as sqrt(2), so two-way ranges sqrt(2) better before the turnaround"
        "\n  residual is charged; the two are equal at sigma_proc = "
        f"{np.sqrt(2.0) * TIMESTAMP_JITTER_S * 1e9:.3f} ns."
    )

    # Geometry. Rows of H are the unit vectors from each anchor toward the
    # agent, which is the same convention Example 2 hands to compute_dop.
    offsets = true_pos - anchors
    geometry = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    hdop = compute_dop(geometry)["HDOP"]

    # With the clock as a third unknown the design matrix gains a column of
    # ones (Eq. 4.26), and the position DOP is the leading 2x2 block of the
    # inverse -- the extra unknown can only make it larger.
    augmented = np.column_stack([geometry, np.ones(len(geometry))])
    cofactor = np.linalg.inv(augmented.T @ augmented)
    hdop_with_clock = float(np.sqrt(np.trace(cofactor[:2, :2])))

    print("\nPosition DOP, and what the third unknown costs:")
    print(f"  position only        (A, B(i), C) : {hdop:.4f}")
    print(
        f"  position + clock     (B(ii))      : {hdop_with_clock:.4f}"
        f"   (+{(hdop_with_clock / hdop - 1) * 100:.2f}%)"
    )

    # --- Zero-noise sanity -------------------------------------------------
    # Every case must be exact when nothing is wrong with the clock and
    # nothing is wrong with the timing. A nonzero number on this line is a
    # model that cannot represent its own data, never noise.
    clean_ranges = np.array([toa_range(a, true_pos) for a in anchors])
    exact_a, _ = TOAPositioner(anchors, method="iterative_ls").solve(
        clean_ranges, initial_guess=seed_pos, max_iters=TWO_WAY_MAX_ITERS
    )
    exact_b, exact_bias, _ = toa_solve_with_clock_bias(
        anchors,
        clean_ranges,
        np.array([*seed_pos, 0.0]),
        max_iters=TWO_WAY_MAX_ITERS,
    )
    # Both stds are zero, so `simulate_rtt_measurement` draws nothing; the
    # Generator is passed anyway rather than None, so this line can never
    # reach into the global stream if someone later gives it a nonzero std.
    exact_c_ranges = _simulate_rtt_ranges(
        anchors, true_pos, 0.0, 0.0, np.random.default_rng(SEED)
    )
    exact_c, _ = TOAPositioner(anchors, method="iterative_ls").solve(
        exact_c_ranges, initial_guess=seed_pos, max_iters=TWO_WAY_MAX_ITERS
    )
    # At b = 0 case B(i) *is* case A -- same measurements, same solver -- so
    # it shares the number rather than being solved twice under a second name.
    print("\nZero-noise sanity (sigma_t = 0, b = 0, turnaround residual = 0):")
    print(
        f"  A and B(i) {np.linalg.norm(exact_a - true_pos):.6f} m"
        f"   B(ii) {np.linalg.norm(exact_b - true_pos):.6f} m"
        f" (bias {exact_bias:.6f} m)"
        f"   C {np.linalg.norm(exact_c - true_pos):.6f} m"
    )

    # --- What the offset does to a position-only fix -----------------------
    # Linearising about the truth, a common range bias b displaces the LS fix
    # by (H'H)^-1 H' (b * 1) = b * (H'H)^-1 sum(u_i). At the centre of a
    # symmetric array sum(u_i) is zero and the bias is invisible, which is why
    # the agent here is off-centre.
    gain = np.linalg.inv(geometry.T @ geometry) @ geometry.sum(axis=0)
    predicted_bias = offset_m * float(np.linalg.norm(gain))
    biased_ranges = clean_ranges + offset_m
    biased_fix, _ = TOAPositioner(anchors, method="iterative_ls").solve(
        biased_ranges, initial_guess=seed_pos, max_iters=TWO_WAY_MAX_ITERS
    )
    measured_bias = float(np.linalg.norm(biased_fix - true_pos))

    # The prediction is first order in b, and b is 6 m against ranges of
    # 11-17 m. Solving a deliberately tiny offset shows the prediction is
    # right where it applies and the gap above is nonlinearity.
    small_m = 0.01
    small_fix, _ = TOAPositioner(anchors, method="iterative_ls").solve(
        clean_ranges + small_m,
        initial_guess=seed_pos,
        max_iters=200,
        tol=1e-12,
    )
    small_ratio = float(np.linalg.norm(small_fix - true_pos)) / (
        small_m * float(np.linalg.norm(gain))
    )

    print(f"\nCase B(i): what {CLOCK_OFFSET_S * 1e9:.0f} ns of unmodelled offset does:")
    print(f"  linearised prediction b*||(H'H)^-1 sum(u_i)|| : {predicted_bias:.4f} m")
    print(f"  measured, noiseless                           : {measured_bias:.4f} m")
    print(
        f"  the gap is nonlinearity, not a wrong prediction: at b = {small_m} m the"
        f"\n  measured displacement is {small_ratio:.3f}x predicted, and"
        f" {offset_m:.2f} m of bias"
        "\n  on 11-17 m ranges is not a small perturbation."
    )

    # --- Monte Carlo -------------------------------------------------------
    rng = np.random.default_rng(SEED)
    truth = np.tile(true_pos, (TWO_WAY_TRIALS, 1))

    draws = rng.standard_normal((TWO_WAY_TRIALS, len(anchors)))
    measurements_a = clean_ranges + draws * sigma_one_way
    measurements_b = measurements_a + offset_m
    measurements_c = np.array(
        [
            _simulate_rtt_ranges(
                anchors, true_pos, TIMESTAMP_JITTER_S, TURNAROUND_RESIDUAL_S, rng
            )
            for _ in range(TWO_WAY_TRIALS)
        ]
    )

    ls_solver = TOAPositioner(anchors, method="iterative_ls")
    cases = [
        (
            "A     one-way, clocks synchronised",
            measurements_a,
            ls_solver,
            hdop * sigma_one_way,
        ),
        ("B(i)  one-way, offset, position only", measurements_b, ls_solver, np.nan),
        (
            "B(ii) one-way, offset, clock solved",
            measurements_b,
            _ClockBiasSolver(anchors),
            hdop_with_clock * sigma_one_way,
        ),
        (
            "C     two-way RTT, no sync needed",
            measurements_c,
            ls_solver,
            hdop * sigma_two_way,
        ),
    ]

    print(
        f"\nMonte Carlo, {TWO_WAY_TRIALS} draws per case."
        " Every draw is scored: `converged`"
        "\nis a step-tolerance stop, not a success flag, so failures are counted"
        "\nbeside the medians rather than removed from them."
    )
    print(
        f"  {'case':38s} {'median':>8s} {'RMSE':>8s} {'failed':>7s} {'predicted':>10s}"
    )
    for label, measurements, solver, predicted in cases:
        outcome = solve_batch(
            solver,
            measurements,
            seed_pos,
            truth,
            max_iters=TWO_WAY_MAX_ITERS,
        )
        finite = np.isfinite(outcome.errors)
        rmse = float(np.sqrt(np.mean(outcome.errors[finite] ** 2)))
        if np.isnan(predicted):
            # B(i) is bias plus scatter, not DOP x sigma: the systematic part
            # measured above and the same scatter as A, added in quadrature.
            predicted = float(np.hypot(measured_bias, hdop * sigma_one_way))
        print(
            f"  {label:38s} {outcome.median_m:8.4f} {rmse:8.4f}"
            f" {outcome.n_failed:7d} {predicted:10.4f}"
        )

    print(
        "\n  The predicted column is DOP x sigma_range for A, B(ii) and C, and"
        "\n  those three land on it. B(i) has no such prediction -- its number is"
        "\n  the noiseless bias and A's scatter in quadrature, and the measured"
        "\n  RMSE runs a few per cent above it because the response to the offset"
        "\n  is not linear, which is the same nonlinearity the block above"
        "\n  measures. A bias-dominated error is not a DOP result."
    )

    print("\n  -> Two-way TOA removes the synchronisation one-way TOA assumes.")
    print("     Against unsynchronised one-way it is not close: 0.2 m against")
    print("     1.8 m, on the same hardware and the same array. Against")
    print("     one-way that estimates its clock it is comparable, and it gets")
    print("     there without spending a degree of freedom on the offset.")
    print("     It does NOT follow that two-way beats TDOA: the chapter's DOP")
    print("     result is that TDOA and TOA-with-an-estimated-clock carry the")
    print("     same information. What differs between the methods is which")
    print("     clock somebody has to build, not how much of the geometry each")
    print("     one extracts.")


def main():
    """Run all TOA positioning examples."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    print("\n" + "=" * 70)
    print("Chapter 4: TOA and RSS Positioning Examples")
    print("=" * 70)

    # Example 1: Perfect measurements
    anchors1, true_pos1, est_pos1, info1 = example_toa_perfect()

    # Example 2: With noise
    anchors2, true_pos2, est_pos2 = example_toa_with_noise()

    # Example 3: With clock bias
    anchors3, true_pos3, est_pos3 = example_toa_with_clock_bias()

    # Example 4: RSS-based
    example_rss_positioning()

    # Example 5: RTT model
    example_rtt_measurement()

    # Example 6: WLS vs LS
    example_wls_vs_ls()

    # Example 7: one-way (synchronised or not) against two-way, one budget
    example_one_way_vs_two_way()

    # Visualization
    print("\n" + "=" * 70)
    print("Generating visualization...")
    print("=" * 70)

    fig = plot_toa_positioning(anchors1, true_pos1, est_pos1, info1["history"])
    paths = save_figure(fig, Path(__file__).parent / "figs", "toa_positioning_example")
    print(f"\nFigure saved: {paths[0]}")

    show_figures_if_requested()

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
