"""What Chapter 6's PDR example claims about where its error comes from.

The example prints an error budget rather than a single accuracy figure, and
auditing it showed the budget had two of its three causes wrong, and both were
in ``generate_corridor_walk`` rather than in the PDR algorithm: the corners
turned 90 degrees inside one 0.01 s sample, and the gait oscillation kept
running through 36 s of standing still. Both are fixed -- corners rounded to
CORNER_RADIUS_M, gait gated on motion -- and the final error went from 80.7 m
to 1.4 m over a 117 m lap.

These tests now guard the fixed state: that the truth stays achievable, that
the gyro remains consistent with the heading it is supposed to encode, and
that what is left is the step-length model rather than an artefact.

The earlier version of this file pinned the *broken* state on the KNOWN_FROZEN
pattern, and it did its job: making the fix turned four of these tests red, so
the change could not land silently and the example's prose had to move with it.

The same shape as KNOWN_FROZEN and FRONTEND_IS_KNOWN_NO_OP: assert the
limitation *persists*, so its disappearance is an event rather than a silence.

Author: Li-Ta Hsu
References: Chapter 6, Section 6.3, Eqs. (6.46)-(6.50)
"""

import numpy as np
import pytest

from ch6_dead_reckoning.example_pdr import (
    add_sensor_noise,
    compute_step_length,
    generate_corridor_walk,
    run_pdr_gyro_heading,
    run_pdr_mag_heading,
)
from core.eval import path_length
from core.sensors import FrameConvention, IMUNoiseParams, units

DURATION_S = 120.0
DT_S = 0.01
HEIGHT_M = 1.75
STEP_FREQ_HZ = 2.0

#: The gait the generator actually walks, from generate_corridor_walk.
SIMULATED_STEP_LENGTH_M = 0.5


@pytest.fixture(scope="module")
def pdr_run():
    """One inline PDR run, wired exactly as run_with_inline_data wires it."""
    frame = FrameConvention.create_enu()
    imu_params = IMUNoiseParams(
        gyro_bias_rad_s=units.deg_per_hour_to_rad_per_sec(50.0),
        gyro_arw_rad_sqrt_s=units.deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.5),
        gyro_rrw_rad_s_sqrt_s=0.0,
        accel_bias_mps2=units.mg_to_mps2(10.0),
        accel_vrw_mps_sqrt_s=units.mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.01),
        grade='consumer (high gyro drift)',
    )

    t, pos_true, heading_true, accel, gyro, mag, expected_steps = (
        generate_corridor_walk(DURATION_S, DT_S, step_freq=STEP_FREQ_HZ, frame=frame)
    )
    accel_m, gyro_m, mag_m = add_sensor_noise(accel, gyro, mag, DT_S, imu_params)

    pos_gyro, heading_gyro, steps_gyro = run_pdr_gyro_heading(
        t, accel_m, gyro_m, HEIGHT_M
    )
    pos_mag, heading_mag, steps_mag = run_pdr_mag_heading(
        t, accel_m, gyro_m, mag_m, HEIGHT_M
    )

    return {
        "t": t,
        "pos_true": pos_true,
        "heading_true": heading_true,
        "gyro_meas": gyro_m,
        "gyro_true": gyro,
        "expected_steps": expected_steps,
        "pos_gyro": pos_gyro,
        "heading_gyro": heading_gyro,
        "steps_gyro": steps_gyro,
        "pos_mag": pos_mag,
        "steps_mag": steps_mag,
        "imu_params": imu_params,
    }


class TestTheErrorBudgetAddsUp:
    """Each numbered cause in the printed budget, checked separately."""

    def test_the_walk_and_gait_are_consistent(self):
        """239 found against 171 taken -- and that is most of the error.

        The budget used to say the opposite, having assumed a 0.5 m gait and
        divided the distance by it to get a "true" 240. Measured, the walk
        runs at 1.400 m/s for 85.7 s and then stands still for 34.3 s while
        generate_corridor_walk keeps emitting the same gait oscillation, so
        the detector faithfully counts about 2 Hz x 120 s of steps for a
        2 Hz x 85.7 s walk.
        """
        frame = FrameConvention.create_enu()
        t, pos, _, _, _, _, _ = generate_corridor_walk(
            DURATION_S, DT_S, step_freq=STEP_FREQ_HZ, frame=frame
        )
        speed = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) / DT_S
        moving = speed > 0.05
        walking_time = float(np.sum(moving)) * DT_S

        assert walking_time == pytest.approx(83.3, abs=2.0), (
            f"the lap now takes {walking_time:.1f} s. Rounding changed its "
            f"length; if that moved again, the printed budget moves too."
        )
        assert float(np.mean(speed[moving])) / STEP_FREQ_HZ == pytest.approx(
            0.700, abs=0.02
        ), "the simulated gait is no longer 0.700 m per step"

    def test_the_distance_is_overstated_by_about_half(self, pdr_run):
        """PDR believes 178.5 m against a true 120.0 m.

        The first line of the budget, and the one that is actually explained:
        the step-length model is uncalibrated for this walker.
        """
        walked = path_length(pdr_run["pos_true"][:, :2])
        believed = path_length(pdr_run["pos_gyro"][:, :2])

        assert 115.0 < walked < 125.0, f"truth walked {walked:.1f} m"
        assert 1.02 < believed / walked < 1.12, (
            f"PDR traced {believed:.1f} m against {walked:.1f} m walked, a "
            f"ratio of {believed / walked:.2f}. The example says +7%, all of it the step-length model."
        )

    def test_the_step_length_model_disagrees_with_the_simulated_gait(self):
        """Eq. (6.49) returns 0.747 m/step where the gait uses 0.500.

        This is the cause the example does name, so it is worth pinning at the
        source rather than only through its effect on distance.
        """
        modelled = compute_step_length(
            height=HEIGHT_M, f_step=STEP_FREQ_HZ, model="book"
        )

        assert modelled == pytest.approx(0.747, abs=0.02), (
            f"Eq. (6.49) returns {modelled:.3f} m/step for a {HEIGHT_M} m "
            f"walker at {STEP_FREQ_HZ} Hz."
        )
        assert modelled / SIMULATED_STEP_LENGTH_M == pytest.approx(1.49, abs=0.05)


class TestTheHeadingErrorComesFromTheTrajectory:
    """Not bias drift, and not the estimator."""

    def test_the_gyro_heading_ends_far_from_truth(self, pdr_run):
        """163 degrees, from a shared 0.0 degree start."""
        final_error_deg = abs(
            np.degrees(
                np.arctan2(
                    np.sin(pdr_run["heading_gyro"][-1] - pdr_run["heading_true"][-1]),
                    np.cos(pdr_run["heading_gyro"][-1] - pdr_run["heading_true"][-1]),
                )
            )
        )

        assert final_error_deg < 5.0, (
            f"the gyro heading ends {final_error_deg:.1f} deg from truth. It "
            f"should now be about 1.2 deg -- the realised bias and nothing "
            f"else. A large value means the trajectory has become "
            f"unrepresentable by the gyro again."
        )

    def test_the_realised_bias_does_not_explain_it(self, pdr_run):
        """1.2 degrees over 120 s, against a 163 degree error.

        The measurement that rules out drift. Note it must difference the
        measured gyro against the *true* one: the walker turns, so integrating
        the measured rate alone gives the total rotation and not the error.
        Doing that by mistake returns 163.25 deg, which agrees with the
        heading error to three figures and looks like a confirmation.
        """
        gyro_error = (
            np.asarray(pdr_run["gyro_meas"])[:, 2]
            - np.asarray(pdr_run["gyro_true"])[:, 2]
        )
        bias_heading_deg = abs(np.degrees(np.sum(gyro_error) * DT_S))

        assert bias_heading_deg < 5.0, (
            f"the realised gyro error integrates to {bias_heading_deg:.2f} deg "
            f"over {DURATION_S:.0f} s. The example argues the 163 deg heading "
            f"error cannot be bias drift because this figure is small; if it "
            f"has grown, that argument no longer holds."
        )

    def test_the_simulated_corner_is_what_the_gyro_cannot_carry(self, pdr_run):
        """The cause, and it is in the trajectory rather than the estimator.

        ``generate_corridor_walk`` turns 90 degrees between two samples, so
        the true heading rate reaches 9000 deg/s. The IMU forward model cannot
        represent a step that large, so the *true* gyro integrates to about
        162 deg over a walk whose heading really comes round to 360. An
        estimator integrating that gyro loses the missing ~198 deg, and that
        is the whole heading error.

        Chapter 8 had the identical defect at the identical 9000 deg/s, which
        is what ``tests/test_simulated_truth_is_physical.py`` was written for
        -- but that test globs ``data/sim/*/truth.npz``, and this generator is
        inline, so it was never covered.

        Asserted as *present*, on the KNOWN_FROZEN pattern: rounding these
        corners is the fix, and it will move every number this example prints,
        so it should be a deliberate change rather than a silent one.
        """
        heading = np.unwrap(np.asarray(pdr_run["heading_true"]))
        turn_rate_deg_s = np.degrees(np.gradient(heading, DT_S))
        integrated_true_gyro_deg = np.degrees(
            np.sum(np.asarray(pdr_run["gyro_true"])[:, 2]) * DT_S
        )
        heading_travelled_deg = abs(np.degrees(heading[-1] - heading[0]))

        assert np.abs(turn_rate_deg_s).max() < 180.0, (
            f"the corridor turns at {np.abs(turn_rate_deg_s).max():.0f} deg/s. "
            f"A pedestrian manages well under 180; above that the gyro forward "
            f"model cannot encode the rotation and the estimator will be "
            f"blamed for losing it."
        )
        assert heading_travelled_deg == pytest.approx(360.0, abs=1.0)
        assert integrated_true_gyro_deg == pytest.approx(
            heading_travelled_deg, abs=1.0
        ), (
            f"the true gyro integrates to {integrated_true_gyro_deg:.1f} deg "
            f"against {heading_travelled_deg:.1f} deg of actual rotation. They "
            f"must agree: the gap used to be 198 deg and it was reported as "
            f"the estimator's heading error."
        )

    def test_the_example_names_the_measured_cause(self):
        """The prose must keep matching the measurements above.

        This used to assert the example still said the cause was "not
        established". It is established now -- instantaneous corners -- so the
        check moves with it. The half a numeric test cannot cover: the tests
        above would still pass if the explanation were deleted or replaced by
        a confident wrong one.
        """
        from pathlib import Path

        import ch6_dead_reckoning.example_pdr as example

        source = Path(example.__file__).read_text(encoding="utf-8")

        assert "9000" in source and "rounded" in source, (
            "example_pdr.py no longer explains the heading error by the "
            "instantaneous corners, or the distance error by the phantom "
            "steps. Both were measured; if the generator was fixed, update "
            "these tests and the prose together."
        )
