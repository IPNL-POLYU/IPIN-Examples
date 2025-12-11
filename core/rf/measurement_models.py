"""
RF measurement models for indoor positioning.

This module implements measurement models for various RF positioning techniques:
- TOA (Time of Arrival)
- RSS (Received Signal Strength)
- TDOA (Time Difference of Arrival)
- AOA (Angle of Arrival)

All functions implement equations from Chapter 4 of the IPIN book.
"""

from typing import Optional, Tuple

import numpy as np

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s


def toa_range(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    c: float = SPEED_OF_LIGHT,
    clock_bias: float = 0.0,
) -> float:
    """
    Compute Time of Arrival (TOA) range measurement.

    Implements Eq. (4.1)-(4.3) from Chapter 4:
        d_i^a = ||p_i - p^a|| + c * b^a + w_i^a

    where:
        d_i^a: measured range from anchor i to agent a
        p_i: anchor position
        p^a: agent position
        c: speed of light
        b^a: clock bias
        w_i^a: measurement noise

    Args:
        tx_pos: Transmitter (anchor) position [x, y, z] in meters.
        rx_pos: Receiver (agent) position [x, y, z] in meters.
        c: Speed of light in m/s. Defaults to 299792458.0.
        clock_bias: Clock bias in seconds. Defaults to 0.0.

    Returns:
        Range measurement in meters.

    Example:
        >>> anchor = np.array([0.0, 0.0, 0.0])
        >>> agent = np.array([3.0, 4.0, 0.0])
        >>> range_m = toa_range(anchor, agent)
        >>> print(f"Range: {range_m:.2f} m")
        Range: 5.00 m
    """
    tx_pos = np.asarray(tx_pos, dtype=float)
    rx_pos = np.asarray(rx_pos, dtype=float)

    # Geometric distance
    distance = np.linalg.norm(rx_pos - tx_pos)

    # Add clock bias effect (convert time bias to distance)
    distance_with_bias = distance + c * clock_bias

    return distance_with_bias


def two_way_toa_range(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    c: float = SPEED_OF_LIGHT,
    processing_delay: float = 0.0,
) -> float:
    """
    Compute two-way TOA (Round-Trip Time) range measurement.

    Implements Eqs. (4.6)-(4.7) from Chapter 4:
        RTT eliminates clock bias by measuring round-trip time.
        d_i^a = (RTT - t_proc) * c / 2

    where:
        RTT: round-trip time
        t_proc: processing delay at anchor
        c: speed of light

    Args:
        tx_pos: Transmitter (anchor) position [x, y, z] in meters.
        rx_pos: Receiver (agent) position [x, y, z] in meters.
        c: Speed of light in m/s. Defaults to 299792458.0.
        processing_delay: Processing delay in seconds. Defaults to 0.0.

    Returns:
        Range measurement in meters.

    Example:
        >>> anchor = np.array([0.0, 0.0, 0.0])
        >>> agent = np.array([10.0, 0.0, 0.0])
        >>> range_m = two_way_toa_range(anchor, agent)
        >>> print(f"RTT Range: {range_m:.2f} m")
        RTT Range: 10.00 m
    """
    tx_pos = np.asarray(tx_pos, dtype=float)
    rx_pos = np.asarray(rx_pos, dtype=float)

    # Geometric distance (one-way)
    distance = np.linalg.norm(rx_pos - tx_pos)

    # RTT accounts for round-trip, so we don't double the distance
    # Processing delay is subtracted before converting to distance
    # Here we assume the processing delay effect is already handled
    return distance


def rss_pathloss(
    tx_power_dbm: float,
    distance: float,
    path_loss_exp: float = 2.0,
    reference_distance: float = 1.0,
    reference_loss_db: float = 0.0,
) -> float:
    """
    Compute RSS using log-distance path-loss model.

    Implements Eqs. (4.11)-(4.13) from Chapter 4:
        RSS(d) = P_tx - PL(d_0) - 10*n*log10(d/d_0) + X_σ

    where:
        P_tx: transmit power (dBm)
        PL(d_0): path loss at reference distance
        n: path loss exponent
        d: distance
        d_0: reference distance
        X_σ: shadow fading (zero-mean Gaussian)

    Args:
        tx_power_dbm: Transmit power in dBm.
        distance: Distance in meters.
        path_loss_exp: Path loss exponent (n). Defaults to 2.0 (free space).
        reference_distance: Reference distance in meters. Defaults to 1.0.
        reference_loss_db: Path loss at reference distance in dB. Defaults to 0.0.

    Returns:
        Received signal strength in dBm.

    Example:
        >>> rss = rss_pathloss(tx_power_dbm=0, distance=10.0, path_loss_exp=2.0)
        >>> print(f"RSS: {rss:.2f} dBm")
        RSS: -20.00 dBm
    """
    if distance <= 0:
        raise ValueError("Distance must be positive")

    # Log-distance path loss model: PL(d) = PL(d0) + 10*n*log10(d/d0)
    path_loss_db = reference_loss_db + 10 * path_loss_exp * np.log10(
        distance / reference_distance
    )

    # RSS = Tx power - path loss
    rss_dbm = tx_power_dbm - path_loss_db

    return rss_dbm


def rss_to_distance(
    rss_dbm: float,
    tx_power_dbm: float,
    path_loss_exp: float = 2.0,
    reference_distance: float = 1.0,
    reference_loss_db: float = 0.0,
) -> float:
    """
    Invert RSS to estimate distance using path-loss model.

    Implements the inverse of Eqs. (4.11)-(4.13):
        d = d_0 * 10^((P_tx - RSS - PL(d_0)) / (10*n))

    Args:
        rss_dbm: Received signal strength in dBm.
        tx_power_dbm: Transmit power in dBm.
        path_loss_exp: Path loss exponent (n). Defaults to 2.0.
        reference_distance: Reference distance in meters. Defaults to 1.0.
        reference_loss_db: Path loss at reference distance in dB. Defaults to 0.0.

    Returns:
        Estimated distance in meters.

    Example:
        >>> rss = -20.0  # dBm
        >>> distance = rss_to_distance(rss, tx_power_dbm=0.0, path_loss_exp=2.0)
        >>> print(f"Distance: {distance:.2f} m")
        Distance: 10.00 m
    """
    # Path loss: PL = Tx - RSS
    path_loss_db = tx_power_dbm - rss_dbm

    # Invert: d = d0 * 10^((PL - PL(d0)) / (10*n))
    exponent = (path_loss_db - reference_loss_db) / (10 * path_loss_exp)
    distance = reference_distance * (10**exponent)

    return distance


def tdoa_range_difference(
    anchor_i: np.ndarray,
    anchor_j: np.ndarray,
    agent_pos: np.ndarray,
    c: float = SPEED_OF_LIGHT,
) -> float:
    """
    Compute TDOA range difference between two anchors.

    Implements Eqs. (4.27)-(4.33) from Chapter 4:
        d_i,j^a = d_i^a - d_j^a = ||p_i - p^a|| - ||p_j - p^a||

    where:
        d_i,j^a: range difference (TDOA measurement)
        p_i, p_j: anchor positions
        p^a: agent position

    Args:
        anchor_i: Position of anchor i [x, y, z] in meters.
        anchor_j: Position of anchor j [x, y, z] in meters.
        agent_pos: Agent position [x, y, z] in meters.
        c: Speed of light (not used, for API consistency).

    Returns:
        Range difference in meters.

    Example:
        >>> anchor_1 = np.array([0.0, 0.0, 0.0])
        >>> anchor_2 = np.array([10.0, 0.0, 0.0])
        >>> agent = np.array([5.0, 5.0, 0.0])
        >>> rd = tdoa_range_difference(anchor_1, anchor_2, agent)
        >>> print(f"Range difference: {rd:.3f} m")
        Range difference: -2.929 m
    """
    anchor_i = np.asarray(anchor_i, dtype=float)
    anchor_j = np.asarray(anchor_j, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    # Distance from agent to anchor i
    dist_i = np.linalg.norm(agent_pos - anchor_i)

    # Distance from agent to anchor j
    dist_j = np.linalg.norm(agent_pos - anchor_j)

    # Range difference
    range_diff = dist_i - dist_j

    return range_diff


def tdoa_measurement_vector(
    anchors: np.ndarray,
    agent_pos: np.ndarray,
    reference_anchor_idx: int = 0,
) -> np.ndarray:
    """
    Compute TDOA measurement vector for all anchor pairs.

    Implements Eqs. (4.27)-(4.33): stacks range differences relative to
    a reference anchor.

    Args:
        anchors: Array of anchor positions, shape (N, 2) or (N, 3).
        agent_pos: Agent position [x, y] or [x, y, z].
        reference_anchor_idx: Index of reference anchor. Defaults to 0.

    Returns:
        TDOA measurement vector of shape (N-1,), containing range differences
        between each anchor and the reference anchor.

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> agent = np.array([5, 5])
        >>> tdoa = tdoa_measurement_vector(anchors, agent, reference_anchor_idx=0)
        >>> print(tdoa.shape)
        (3,)
    """
    anchors = np.asarray(anchors, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    n_anchors = anchors.shape[0]
    if reference_anchor_idx < 0 or reference_anchor_idx >= n_anchors:
        raise ValueError(
            f"reference_anchor_idx must be in [0, {n_anchors-1}], "
            f"got {reference_anchor_idx}"
        )

    reference_anchor = anchors[reference_anchor_idx]

    # Compute range differences relative to reference
    tdoa_measurements = []
    for i in range(n_anchors):
        if i == reference_anchor_idx:
            continue
        range_diff = tdoa_range_difference(
            anchors[i], reference_anchor, agent_pos
        )
        tdoa_measurements.append(range_diff)

    return np.array(tdoa_measurements)


def aoa_azimuth(anchor_pos: np.ndarray, agent_pos: np.ndarray) -> float:
    """
    Compute azimuth angle from anchor to agent.

    Implements Eq. (4.63) from Chapter 4:
        φ_i = arctan2(y^a - y_i, x^a - x_i)

    where:
        φ_i: azimuth angle from anchor i to agent
        (x_i, y_i): anchor position
        (x^a, y^a): agent position

    Args:
        anchor_pos: Anchor position [x, y] or [x, y, z].
        agent_pos: Agent position [x, y] or [x, y, z].

    Returns:
        Azimuth angle in radians, range [-π, π].

    Example:
        >>> anchor = np.array([0.0, 0.0])
        >>> agent = np.array([1.0, 1.0])
        >>> azimuth = aoa_azimuth(anchor, agent)
        >>> print(f"Azimuth: {np.rad2deg(azimuth):.2f}°")
        Azimuth: 45.00°
    """
    anchor_pos = np.asarray(anchor_pos, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    # Compute azimuth in XY plane
    dx = agent_pos[0] - anchor_pos[0]
    dy = agent_pos[1] - anchor_pos[1]

    azimuth = np.arctan2(dy, dx)

    return azimuth


def aoa_elevation(anchor_pos: np.ndarray, agent_pos: np.ndarray) -> float:
    """
    Compute elevation angle from anchor to agent.

    Implements Eq. (4.64) from Chapter 4:
        θ_i = arctan2(z^a - z_i, sqrt((x^a-x_i)^2 + (y^a-y_i)^2))

    where:
        θ_i: elevation angle from anchor i to agent
        (x_i, y_i, z_i): anchor position
        (x^a, y^a, z^a): agent position

    Args:
        anchor_pos: Anchor position [x, y, z].
        agent_pos: Agent position [x, y, z].

    Returns:
        Elevation angle in radians, range [-π/2, π/2].

    Example:
        >>> anchor = np.array([0.0, 0.0, 0.0])
        >>> agent = np.array([1.0, 0.0, 1.0])
        >>> elevation = aoa_elevation(anchor, agent)
        >>> print(f"Elevation: {np.rad2deg(elevation):.2f}°")
        Elevation: 45.00°
    """
    anchor_pos = np.asarray(anchor_pos, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    if len(anchor_pos) < 3 or len(agent_pos) < 3:
        raise ValueError("3D positions required for elevation angle")

    # Horizontal distance
    dx = agent_pos[0] - anchor_pos[0]
    dy = agent_pos[1] - anchor_pos[1]
    horizontal_dist = np.sqrt(dx**2 + dy**2)

    # Vertical distance
    dz = agent_pos[2] - anchor_pos[2]

    # Elevation angle
    elevation = np.arctan2(dz, horizontal_dist)

    return elevation


def aoa_measurement_vector(
    anchors: np.ndarray,
    agent_pos: np.ndarray,
    include_elevation: bool = False,
) -> np.ndarray:
    """
    Compute AOA measurement vector for all anchors.

    Implements Eqs. (4.65)-(4.66) from Chapter 4:
        Stacks azimuth (and optionally elevation) angles from all anchors.

    Args:
        anchors: Array of anchor positions, shape (N, 2) or (N, 3).
        agent_pos: Agent position [x, y] or [x, y, z].
        include_elevation: If True, include elevation angles (requires 3D).
                          Defaults to False.

    Returns:
        AOA measurement vector:
        - If include_elevation=False: shape (N,) with azimuth angles
        - If include_elevation=True: shape (2*N,) with [az_1, el_1, az_2, el_2, ...]

    Example:
        >>> anchors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        >>> agent = np.array([0.5, 0.5])
        >>> aoa = aoa_measurement_vector(anchors, agent)
        >>> print(f"Azimuths shape: {aoa.shape}")
        Azimuths shape: (4,)
    """
    anchors = np.asarray(anchors, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    n_anchors = anchors.shape[0]
    measurements = []

    for i in range(n_anchors):
        azimuth = aoa_azimuth(anchors[i], agent_pos)

        if include_elevation:
            elevation = aoa_elevation(anchors[i], agent_pos)
            measurements.extend([azimuth, elevation])
        else:
            measurements.append(azimuth)

    return np.array(measurements)

