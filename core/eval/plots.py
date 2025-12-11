"""
Visualization Utilities for Indoor Positioning.

This module provides plotting functions for trajectories, errors,
and positioning geometry.

All functions return matplotlib Figure objects for flexible display/saving.

Author: Navigation Engineering Team
Date: December 2025
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_2d(
    truth_xy: np.ndarray,
    est_xy_dict: Dict[str, np.ndarray],
    anchors_xy: Optional[np.ndarray] = None,
    title: str = "2D Trajectory",
) -> plt.Figure:
    """
    Plot 2D trajectory with true and estimated paths.

    Args:
        truth_xy: True trajectory, shape (N, 2)
        est_xy_dict: Dictionary of estimated trajectories {name: array}
        anchors_xy: Anchor positions, shape (M, 2) (optional)
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot true trajectory
    ax.plot(
        truth_xy[:, 0],
        truth_xy[:, 1],
        "k-",
        linewidth=2,
        label="Ground Truth",
        zorder=10,
    )
    ax.plot(
        truth_xy[0, 0],
        truth_xy[0, 1],
        "go",
        markersize=10,
        label="Start",
        zorder=11,
    )
    ax.plot(
        truth_xy[-1, 0],
        truth_xy[-1, 1],
        "ro",
        markersize=10,
        label="End",
        zorder=11,
    )

    # Plot estimated trajectories
    colors = ["blue", "red", "green", "orange", "purple"]
    linestyles = ["-", "--", "-.", ":", "-"]

    for i, (name, est_xy) in enumerate(est_xy_dict.items()):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        ax.plot(
            est_xy[:, 0],
            est_xy[:, 1],
            linestyle=linestyle,
            color=color,
            linewidth=1.5,
            label=name,
            alpha=0.7,
        )

    # Plot anchors if provided
    if anchors_xy is not None:
        ax.plot(
            anchors_xy[:, 0],
            anchors_xy[:, 1],
            "s",
            color="blue",
            markersize=8,
            label="Anchors",
            zorder=5,
        )

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    plt.tight_layout()
    return fig


def plot_position_error_time(
    errors_dict: Dict[str, np.ndarray],
    dt: float = 1.0,
    axes: str = "enu",
    title: str = "Position Error vs Time",
) -> plt.Figure:
    """
    Plot position error components over time.

    Args:
        errors_dict: Dictionary of error arrays {name: errors}
                     Each array has shape (N, 2) or (N, 3)
        dt: Time step in seconds
        axes: Axis labels ("enu" or "xyz")
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    # Determine number of dimensions
    sample_errors = next(iter(errors_dict.values()))
    n_dims = sample_errors.shape[1]

    if axes.lower() == "enu":
        axis_labels = ["East", "North", "Up"][:n_dims]
    else:
        axis_labels = ["X", "Y", "Z"][:n_dims]

    fig, axes_arr = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims))
    if n_dims == 1:
        axes_arr = [axes_arr]

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, axis_label in enumerate(axis_labels):
        ax = axes_arr[i]

        for j, (name, errors) in enumerate(errors_dict.items()):
            time = np.arange(len(errors)) * dt
            color = colors[j % len(colors)]
            ax.plot(time, errors[:, i], label=name, color=color, linewidth=1.5)

        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel(f"{axis_label} Error (m)", fontsize=11)
        ax.set_title(
            f"{axis_label}-axis Error", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    return fig


def plot_error_hist(
    errors_dict: Dict[str, np.ndarray],
    bins: int = 30,
    title: str = "Error Distribution",
) -> plt.Figure:
    """
    Plot histogram of position error magnitudes.

    Args:
        errors_dict: Dictionary of error arrays {name: errors}
        bins: Number of histogram bins
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, (name, errors) in enumerate(errors_dict.items()):
        # Compute error magnitudes
        if errors.ndim > 1:
            error_magnitudes = np.linalg.norm(errors, axis=1)
        else:
            error_magnitudes = np.abs(errors)

        color = colors[i % len(colors)]
        ax.hist(
            error_magnitudes,
            bins=bins,
            alpha=0.6,
            label=name,
            color=color,
            edgecolor="black",
        )

    ax.set_xlabel("Position Error (m)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_error_cdf(
    errors_dict: Dict[str, np.ndarray], title: str = "Error CDF"
) -> plt.Figure:
    """
    Plot Cumulative Distribution Function (CDF) of position errors.

    Args:
        errors_dict: Dictionary of error arrays {name: errors}
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "red", "green", "orange", "purple"]
    linestyles = ["-", "--", "-.", ":", "-"]

    for i, (name, errors) in enumerate(errors_dict.items()):
        # Compute error magnitudes
        if errors.ndim > 1:
            error_magnitudes = np.linalg.norm(errors, axis=1)
        else:
            error_magnitudes = np.abs(errors)

        # Compute CDF
        sorted_errors = np.sort(error_magnitudes)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        ax.plot(
            sorted_errors,
            cdf,
            label=name,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )

    ax.set_xlabel("Position Error (m)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


def plot_rf_geometry(
    anchors_xy: np.ndarray,
    traj_xy: Optional[np.ndarray] = None,
    title: str = "RF Geometry",
) -> plt.Figure:
    """
    Plot RF anchor geometry and optional trajectory.

    Args:
        anchors_xy: Anchor positions, shape (M, 2)
        traj_xy: Trajectory positions, shape (N, 2) (optional)
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot anchors
    ax.plot(
        anchors_xy[:, 0],
        anchors_xy[:, 1],
        "s",
        color="blue",
        markersize=12,
        label="Anchors",
    )

    # Label anchors
    for i, anchor in enumerate(anchors_xy):
        ax.text(
            anchor[0],
            anchor[1] + 0.5,
            f"A{i}",
            fontsize=10,
            ha="center",
            color="blue",
        )

    # Plot trajectory if provided
    if traj_xy is not None:
        ax.plot(
            traj_xy[:, 0],
            traj_xy[:, 1],
            "r-",
            linewidth=2,
            label="Trajectory",
            alpha=0.7,
        )
        ax.plot(
            traj_xy[0, 0],
            traj_xy[0, 1],
            "go",
            markersize=8,
            label="Start",
        )
        ax.plot(
            traj_xy[-1, 0],
            traj_xy[-1, 1],
            "ro",
            markersize=8,
            label="End",
        )

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    plt.tight_layout()
    return fig


def plot_dop_map(
    dop_grid: np.ndarray,
    extent: Tuple[float, float, float, float],
    anchors_xy: Optional[np.ndarray] = None,
    dop_type: str = "GDOP",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot DOP heatmap over a 2D area.

    Args:
        dop_grid: DOP values, shape (ny, nx)
        extent: (xmin, xmax, ymin, ymax) for grid
        anchors_xy: Anchor positions, shape (M, 2) (optional)
        dop_type: Type of DOP (GDOP, HDOP, VDOP, PDOP)
        title: Plot title (default: "{dop_type} Map")

    Returns:
        fig: Matplotlib figure
    """
    if title is None:
        title = f"{dop_type} Map"

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot DOP heatmap
    im = ax.imshow(
        dop_grid,
        extent=extent,
        origin="lower",
        cmap="jet",
        aspect="auto",
        interpolation="bilinear",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(dop_type, fontsize=12)

    # Plot anchors if provided
    if anchors_xy is not None:
        ax.plot(
            anchors_xy[:, 0],
            anchors_xy[:, 1],
            "ws",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=2,
            label="Anchors",
        )

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if anchors_xy is not None:
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

    plt.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    out_dir: Union[str, Path],
    name: str,
    formats: Tuple[str, ...] = ("svg", "pdf", "png"),
) -> List[Path]:
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure to save
        out_dir: Output directory
        name: Base filename (without extension)
        formats: Tuple of format extensions

    Returns:
        paths: List of saved file paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for fmt in formats:
        filepath = out_dir / f"{name}.{fmt}"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        paths.append(filepath)

    return paths

