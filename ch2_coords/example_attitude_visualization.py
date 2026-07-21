"""Attitude and Frame Visualization for Chapter 2.

Chapter 2 defines the frames and the attitude convention that every later
chapter inherits, and its convention is *not* the aerospace default: the book
rotates roll about **Y** (2.15) and pitch about **X** (2.16), composing them as
C = Rx(pitch) Ry(roll) Rz(yaw) (2.17). Read with the usual roll-about-X habit,
every downstream rotation is silently wrong -- which is exactly what had
happened in this repository's own code before the Chapter 2 audit.

This example draws that convention rather than asserting it, producing four
figures:

1. ``ch2_euler_convention``   -- the three elemental rotations of (2.14)-(2.16)
                                 and their composition (2.17), each axis of
                                 rotation marked so the Y/X pairing is visible.
2. ``ch2_passive_vs_active``  -- the transpose trap: Chapter 2's passive C
                                 (2.21) beside the active body-to-map rotation
                                 used in Chapter 6 (6.13).
3. ``ch2_gimbal_lock``        -- the singularity, which in this convention
                                 occurs at **roll** = +/-90 degrees, not pitch.
4. ``ch2_frame_chain``        -- ENU -> NED -> body, the local-frame chain of
                                 (2.5)-(2.7). ECEF is omitted deliberately: it
                                 is an earth-sized global frame and shares no
                                 useful scale with a unit triad.

Run:
    python ch2_coords/example_attitude_visualization.py
    python ch2_coords/example_attitude_visualization.py --no-show

Author: Li-Ta Hsu
References: Chapter 2, Sections 2.1 (frames) and 2.2 (attitude),
            Eqs. (2.5)-(2.7), (2.14)-(2.17), (2.21)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.coords import (
    enu_to_ned,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
)
from core.eval import plot_frame_3d, save_figure, set_axes_equal_3d

FIGS_DIR = Path(__file__).parent / "figs"

# Axis of rotation for each book Euler angle. This mapping is the whole point
# of figure 1: roll turns about Y and pitch about X, per Eqs. (2.15)-(2.16).
ROLL_AXIS = "y"
PITCH_AXIS = "x"
YAW_AXIS = "z"


# One viewing angle for every panel: at the default azimuth the Y axis points
# up-and-right and visually collides with Z, which makes "roll about Y" look
# like the Y axis moved when it has not.
VIEW_ELEV = 20.0
VIEW_AZIM = 35.0


def _style_3d(ax, elev: float = VIEW_ELEV, azim: float = VIEW_AZIM) -> None:
    """Apply a view angle and strip numeric ticks.

    Args:
        ax: Matplotlib 3-D axes.
        elev: Elevation angle in degrees.
        azim: Azimuth in degrees. Override when a rotation would otherwise be
            foreshortened: a turn about an axis pointing away from the camera
            is nearly invisible.
    """
    set_axes_equal_3d(ax)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def _draw_reference(ax, axis_names=("X", "Y", "Z")) -> None:
    """Draw the faint unrotated frame, without labels, behind the subject."""
    plot_frame_3d(ax, None, alpha=0.22, linewidth=1.4, linestyle="--",
                  axis_names=axis_names, show_labels=False)


def _draw_rotation_axis(ax, axis: str, radius: float = 1.45) -> None:
    """Mark the axis a rotation happens about, as a dashed grey line."""
    direction = {"x": np.array([1.0, 0.0, 0.0]),
                 "y": np.array([0.0, 1.0, 0.0]),
                 "z": np.array([0.0, 0.0, 1.0])}[axis]
    span = np.outer(np.array([-radius, radius]), direction)
    ax.plot(span[:, 0], span[:, 1], span[:, 2],
            color="0.25", linestyle=":", linewidth=2.0, zorder=0)


def plot_euler_convention(angle_deg: float = 35.0) -> plt.Figure:
    """Figure 1: the elemental rotations of (2.14)-(2.16) and (2.17).

    Each panel applies a single non-zero angle so the axis it turns about is
    unambiguous, then the fourth panel composes all three.

    Args:
        angle_deg: Magnitude used for each elemental rotation, in degrees.

    Returns:
        The matplotlib figure.
    """
    angle = np.deg2rad(angle_deg)
    panels = [
        (f"Yaw  psi = {angle_deg:g} deg, about Z\nEq. (2.14)",
         euler_to_rotation_matrix(0.0, 0.0, angle), YAW_AXIS),
        (f"Roll  phi = {angle_deg:g} deg, about Y\nEq. (2.15)",
         euler_to_rotation_matrix(angle, 0.0, 0.0), ROLL_AXIS),
        (f"Pitch  theta = {angle_deg:g} deg, about X\nEq. (2.16)",
         euler_to_rotation_matrix(0.0, angle, 0.0), PITCH_AXIS),
        ("Composed C = Rx(theta) Ry(phi) Rz(psi)\nEq. (2.17)",
         euler_to_rotation_matrix(angle, angle, angle), None),
    ]

    fig = plt.figure(figsize=(14, 4.2))
    for index, (title, C, axis) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        _draw_reference(ax)
        plot_frame_3d(ax, C, label="'", linewidth=2.6)
        if axis is not None:
            _draw_rotation_axis(ax, axis)
        _style_3d(ax)
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        "Chapter 2 Euler convention: roll about Y, pitch about X "
        "(dashed = reference frame, dotted = axis of rotation)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.02, right=0.98, wspace=0.05)
    return fig


def plot_passive_vs_active(roll_deg: float = 0.0,
                           pitch_deg: float = 0.0,
                           yaw_deg: float = 50.0) -> plt.Figure:
    """Figure 2: the passive/active transpose trap.

    Chapter 2's C is passive: ``x_new = C x_old`` rotates the *coordinates*.
    Chapter 6's strapdown mechanization (6.13) uses the active body-to-map
    rotation, which is C transposed. The two look identical for zero rotation
    and diverge by twice the angle otherwise -- a bug that hides until the
    vehicle turns.

    Args:
        roll_deg: Roll angle in degrees.
        pitch_deg: Pitch angle in degrees.
        yaw_deg: Yaw angle in degrees.

    Returns:
        The matplotlib figure.
    """
    C = euler_to_rotation_matrix(
        np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    )

    fig = plt.figure(figsize=(11, 4.6))
    for index, (title, matrix) in enumerate(
        [
            ("Passive: C, Eq. (2.21)\n'x_new = C x_old' -- rotates coordinates",
             C),
            ("Active: C^T, Ch. 6 Eq. (6.13)\nbody-to-map -- rotates the vector",
             C.T),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        _draw_reference(ax)
        plot_frame_3d(ax, matrix, label="'", linewidth=2.6)
        _draw_rotation_axis(ax, YAW_AXIS)
        _style_3d(ax)
        ax.set_title(title, fontsize=9)

    separation = np.rad2deg(
        np.arccos(np.clip((np.trace(C @ C) - 1.0) / 2.0, -1.0, 1.0))
    )
    fig.suptitle(
        f"Passive and active differ by a transpose: for yaw = {yaw_deg:g} deg "
        f"the two frames sit {separation:.0f} deg apart. Never mix them.",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.02, right=0.98, wspace=0.05)
    return fig


def plot_gimbal_lock() -> plt.Figure:
    """Figure 3: the singularity at roll = +/-90 degrees.

    At roll = +/-90 the yaw and pitch axes become parallel, so only their sum
    (or difference) is recoverable. The top row sweeps roll toward the
    singularity; the bottom panel shows two different (yaw, pitch) pairs that
    produce the *same* attitude once roll = 90.

    Returns:
        The matplotlib figure.
    """
    fig = plt.figure(figsize=(13, 7.5))

    # Facing down the Y axis: roll turns about Y, so from here it reads as a
    # plain in-plane rotation of the x'/z' pair rather than a foreshortened one.
    gimbal_view = {"elev": 8.0, "azim": 88.0}

    approach = [0.0, 45.0, 80.0, 90.0]
    for index, roll_deg in enumerate(approach, start=1):
        ax = fig.add_subplot(2, 4, index, projection="3d")
        C = euler_to_rotation_matrix(np.deg2rad(roll_deg), 0.0, np.deg2rad(30.0))
        _draw_reference(ax)
        plot_frame_3d(ax, C, label="'", linewidth=2.6)
        _draw_rotation_axis(ax, ROLL_AXIS)
        _style_3d(ax, **gimbal_view)
        ax.set_title(f"roll = {roll_deg:g} deg", fontsize=9)

    # Two distinct (yaw, pitch) pairs collapsing to one attitude at roll = 90.
    locked = [(30.0, 0.0), (0.0, -30.0)]
    for offset, (yaw_deg, pitch_deg) in enumerate(locked):
        ax = fig.add_subplot(2, 4, 5 + offset, projection="3d")
        C = euler_to_rotation_matrix(
            np.pi / 2.0, np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
        )
        _draw_reference(ax)
        plot_frame_3d(ax, C, label="'", linewidth=2.6)
        _draw_rotation_axis(ax, ROLL_AXIS)
        _style_3d(ax, **gimbal_view)
        ax.set_title(
            f"roll = 90, yaw = {yaw_deg:g}, pitch = {pitch_deg:g}\n"
            "(identical to its neighbour -- that is the lock)", fontsize=9
        )

    # Show numerically that the recovery collapses to a single angle.
    C_a = euler_to_rotation_matrix(np.pi / 2.0, 0.0, np.deg2rad(30.0))
    C_b = euler_to_rotation_matrix(np.pi / 2.0, np.deg2rad(-30.0), 0.0)
    recovered_a = np.rad2deg(rotation_matrix_to_euler(C_a))
    recovered_b = np.rad2deg(rotation_matrix_to_euler(C_b))

    ax_text = fig.add_subplot(2, 2, 4)
    ax_text.axis("off")
    ax_text.text(
        0.0,
        0.5,
        "Gimbal lock in the Chapter 2 convention\n\n"
        "The singularity is at ROLL = +/-90 deg (not pitch), because roll is\n"
        "the middle rotation about Y in C = Rx(theta) Ry(phi) Rz(psi).\n\n"
        f"  (roll, pitch, yaw) = (90, 0, 30)  ->  recovered {np.round(recovered_a, 1)}\n"
        f"  (roll, pitch, yaw) = (90, -30, 0) ->  recovered {np.round(recovered_b, 1)}\n\n"
        f"Both matrices agree to {np.abs(C_a - C_b).max():.1e}: the two inputs are\n"
        "the same attitude. Only pitch -/+ yaw survives, so the recovery pins\n"
        "yaw = 0 and folds the remainder into pitch.",
        fontsize=9.5,
        family="monospace",
        va="center",
    )

    fig.suptitle(
        "Gimbal lock occurs at roll = +/-90 deg in this convention", fontsize=12
    )
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.02, right=0.98, wspace=0.05)
    return fig


def plot_frame_chain() -> plt.Figure:
    """Figure 4: the ECEF -> ENU -> NED -> body chain of (2.5)-(2.7).

    ENU and NED differ by the swap-and-flip of (2.5); the body frame then sits
    at the vehicle attitude via (2.6)/(2.7).

    Returns:
        The matplotlib figure.
    """
    # ENU->NED as an explicit matrix, obtained by mapping the basis vectors
    # through the library function rather than hard-coding it here.
    C_enu_to_ned = np.column_stack(
        [enu_to_ned(basis) for basis in np.eye(3)]
    )

    attitude = euler_to_rotation_matrix(
        np.deg2rad(15.0), np.deg2rad(10.0), np.deg2rad(40.0)
    )

    panels = [
        ("ENU (local tangent)\nEast, North, Up", np.eye(3), ("E", "N", "U")),
        ("NED, Eq. (2.5)\nNorth, East, Down", C_enu_to_ned, ("N", "E", "D")),
        ("Body, Eqs. (2.6)/(2.7)\nroll 15, pitch 10, yaw 40",
         attitude, ("x", "y", "z")),
    ]

    fig = plt.figure(figsize=(12, 4.4))
    for index, (title, C, names) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        _draw_reference(ax, axis_names=("E", "N", "U"))
        plot_frame_3d(ax, C, linewidth=2.6, axis_names=names)
        _style_3d(ax)
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        "Frame chain: dashed ENU reference behind each frame. "
        "NED is a handedness-preserving swap-and-flip of ENU, not a rotation.",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.02, right=0.98, wspace=0.05)
    return fig


def main() -> None:
    """Generate and save all Chapter 2 attitude figures."""
    parser = argparse.ArgumentParser(
        description="Chapter 2 attitude and frame visualizations"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save figures without displaying"
    )
    parser.add_argument(
        "--out-dir", default=str(FIGS_DIR), help="Output directory for figures"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Chapter 2: Attitude and Frame Visualization")
    print("=" * 70)
    print("Convention: roll about Y (2.15), pitch about X (2.16), yaw about Z")
    print("            C = Rx(pitch) Ry(roll) Rz(yaw), Eq. (2.17), passive")
    print()

    figures = [
        ("ch2_euler_convention", plot_euler_convention()),
        ("ch2_passive_vs_active", plot_passive_vs_active()),
        ("ch2_gimbal_lock", plot_gimbal_lock()),
        ("ch2_frame_chain", plot_frame_chain()),
    ]

    for name, fig in figures:
        paths = save_figure(fig, args.out_dir, name)
        print(f"  saved {name}: {', '.join(p.suffix.lstrip('.') for p in paths)}")

    print()
    print(f"Figures written to {args.out_dir}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
