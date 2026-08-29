"""Regression tests for named, tuple-compatible Ch6 generated trajectories."""

from ch6_dead_reckoning.example_comparison import generate_mixed_trajectory
from ch6_dead_reckoning.example_environment import generate_building_walk
from ch6_dead_reckoning.example_imu_strapdown import generate_figure8_trajectory
from ch6_dead_reckoning.example_pdr import generate_corridor_walk
from ch6_dead_reckoning.example_wheel_odometry import generate_vehicle_trajectory
from ch6_dead_reckoning.example_zupt import generate_walking_trajectory


def test_imu_strapdown_trajectory_has_named_fields_and_unpacking() -> None:
    trajectory = generate_figure8_trajectory(duration=0.1, dt=0.01)

    t, pos_true, *_ = trajectory

    assert trajectory.timestamps_s is t
    assert trajectory.true_positions_map_m is pos_true
    assert trajectory.t is t
    assert trajectory.pos_true is pos_true
    assert trajectory._fields == (
        "timestamps_s",
        "true_positions_map_m",
        "true_velocities_map_mps",
        "true_attitudes_quat_wxyz",
        "specific_force_body_mps2",
        "angular_rates_body_rad_s",
    )


def test_zupt_trajectory_has_named_fields_and_unpacking() -> None:
    trajectory = generate_walking_trajectory(duration=1.0, dt=0.01)

    t, pos_true, *_ = trajectory

    assert trajectory.timestamps_s is t
    assert trajectory.true_positions_map_m is pos_true
    assert trajectory.t is t
    assert trajectory.pos_true is pos_true
    assert trajectory._fields == (
        "timestamps_s",
        "true_positions_map_m",
        "true_velocities_map_mps",
        "true_attitudes_quat_wxyz",
        "specific_force_body_mps2",
        "angular_rates_body_rad_s",
        "stance_mask_stationary",
    )


def test_environment_walk_has_named_fields_and_unpacking() -> None:
    walk = generate_building_walk(duration=1.0, dt=0.1)

    t, pos_true, *_ = walk

    assert walk.timestamps_s is t
    assert walk.true_positions_map_m is pos_true
    assert walk.t is t
    assert walk.pos_true is pos_true
    assert walk._fields == (
        "timestamps_s",
        "true_positions_map_m",
        "true_attitudes_rpy_rad",
        "true_magnetic_field_body",
        "true_pressure_pa",
        "true_floor_ids",
    )


def test_wheel_trajectory_has_named_fields_and_unpacking() -> None:
    trajectory = generate_vehicle_trajectory(duration=1.0, dt=0.01)

    t, pos_true, *_ = trajectory

    assert trajectory.timestamps_s is t
    assert trajectory.true_positions_map_m is pos_true
    assert trajectory.t is t
    assert trajectory.pos_true is pos_true
    assert trajectory._fields == (
        "timestamps_s",
        "true_positions_map_m",
        "true_velocities_map_mps",
        "true_attitudes_quat_wxyz",
        "true_wheel_speed_mps",
        "true_angular_rates_body_rad_s",
    )


def test_pdr_corridor_walk_has_named_fields_and_unpacking() -> None:
    walk = generate_corridor_walk(duration=1.0, dt=0.01)

    t, pos_2d, *_ = walk

    assert walk.timestamps_s is t
    assert walk.true_positions_map_m is pos_2d
    assert walk.t is t
    assert walk.pos_2d is pos_2d
    assert walk._fields == (
        "timestamps_s",
        "true_positions_map_m",
        "true_headings_rad",
        "specific_force_body_mps2",
        "angular_rates_body_rad_s",
        "true_magnetic_field_body",
        "expected_step_count",
    )


def test_comparison_trajectory_has_named_fields_and_unpacking() -> None:
    trajectory = generate_mixed_trajectory(duration=5.0, dt=0.05)

    t, pos_true, *_ = trajectory

    assert trajectory.timestamps_s is t
    assert trajectory.true_positions_map_m is pos_true
    assert trajectory.t is t
    assert trajectory.pos_true is pos_true
    assert trajectory._fields == (
        "timestamps_s",
        "true_positions_map_m",
        "true_velocities_map_mps",
        "specific_force_body_mps2",
        "angular_rates_body_rad_s",
        "true_headings_rad",
        "true_magnetic_field_body",
        "stance_mask_stationary",
        "true_wheel_speed_mps",
    )
