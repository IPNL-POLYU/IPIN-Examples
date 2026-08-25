"""Allan variance runs on synthetic noise, so the right answer is known.

That is the whole reason to demonstrate it on simulated data, and the example
never did it: it printed what the analysis recovered and never put the injected
value beside it. Three unit errors survived that gap, each plausible alone:

  - ARW printed without the sqrt(3600) that takes rad/sqrt(s) to deg/sqrt(hr),
    so a 0.5 deg/sqrt(hr) gyro was reported as 0.0090 -- which the reference
    table printed twelve lines lower calls better than navigation grade
  - RRW multiplied by 3600 where the same conversion needs 60, sixty times too
    large in the other direction
  - the consumer accelerometer's bias instability divided by 3600 for no
    reason, making it 2.8e-8 m/s^2: 360x *better* than the tactical entry three
    lines below it in the same table

The conversion the first two got wrong is written out in
``characterize_imu_noise``'s own docstring, and the third contradicted its own
neighbour. Both are the repo's recurring shape -- the correct version was
already in the file.

Author: Li-Ta Hsu
"""

import numpy as np
import pytest

from ch6_dead_reckoning.example_allan_variance import (
    DEFAULT_SEED,
    IMU_SPECS,
    generate_imu_stationary_data,
    injected_si,
)
from core.sensors import characterize_imu_noise

DURATION_S = 3600.0
FS_HZ = 100.0
GRADE = "consumer"


@pytest.fixture(scope="module")
def recovered():
    """Characterise one seeded record; shared, because generating it is slow."""
    _, gyro, accel = generate_imu_stationary_data(
        duration=DURATION_S, fs=FS_HZ, imu_grade=GRADE
    )
    # The RRW and accel-BI regions are absent from a 1-hour record, and
    # characterize_imu_noise says so through warnings. That is asserted below
    # rather than ignored; here it would only be noise.
    with pytest.warns(UserWarning):
        return characterize_imu_noise(gyro, accel, fs=FS_HZ)


def _long_tau_slope(result):
    """Slope of log10(adev) against log10(tau) over the last decade."""
    taus = np.asarray(result["taus"])
    adev = np.asarray(result["adev"])
    tail = taus >= taus[-1] / 10.0
    return float(np.polyfit(np.log10(taus[tail]), np.log10(adev[tail]), 1)[0])


class TestShortTauParametersComeBack:
    """ARW and VRW are read where the record actually has data."""

    def test_angle_random_walk_within_20_percent(self, recovered):
        """The one the missing sqrt(3600) hid."""
        ratio = recovered["gyro"]["angle_random_walk"] / injected_si(GRADE)["gyro_arw"]

        assert 0.8 < ratio < 1.2, (
            f"gyro ARW recovered {ratio:.2f}x the injected value. White noise "
            f"is read at short tau, where an hour of data at {FS_HZ:.0f} Hz is "
            f"plentiful, so this should be close."
        )

    def test_velocity_random_walk_within_20_percent(self, recovered):
        """Same argument, accelerometer side."""
        ratio = (
            recovered["accel"]["velocity_random_walk"] / injected_si(GRADE)["accel_vrw"]
        )

        assert (
            0.8 < ratio < 1.2
        ), f"accel VRW recovered {ratio:.2f}x the injected value."

    def test_gyro_bias_instability_within_a_factor_of_two(self, recovered):
        """Looser on purpose: the shoulder is broad and the minimum is noisy."""
        ratio = (
            recovered["gyro"]["bias_instability"]
            / injected_si(GRADE)["gyro_bias_instability"]
        )

        assert (
            0.5 < ratio < 2.0
        ), f"gyro bias instability recovered {ratio:.2f}x the injected value."


class TestTheReportedNumbersAgreeWithTheReferenceTable:
    """The example prints a grade table; its own results must fit in it."""

    def test_arw_lands_in_the_consumer_band(self, recovered):
        """The tell that the ARW conversion was wrong.

        The table says consumer is 0.1-1.0 deg/sqrt(hr) and navigation is
        below 0.01. The example reported 0.0090 for a gyro it had just
        described as consumer, contradicting a table twelve lines away.
        """
        arw = np.rad2deg(recovered["gyro"]["angle_random_walk"]) * 60

        assert 0.1 <= arw <= 1.0, (
            f"ARW reports {arw:.4f} deg/sqrt(hr), outside the 0.1-1.0 band the "
            f"example prints for consumer grade. A value below 0.01 would be "
            f"navigation grade, i.e. the conversion is out by sqrt(3600)."
        )

    def test_bias_instability_lands_in_the_consumer_band(self, recovered):
        """Same check for the column beside it."""
        bi = np.rad2deg(recovered["gyro"]["bias_instability"]) * 3600

        assert 10 <= bi <= 100, (
            f"bias instability reports {bi:.2f} deg/hr, outside the 10-100 "
            f"band the example prints for consumer grade."
        )


class TestTheSpecTableIsInternallyConsistent:
    """A worse grade must be worse on every axis."""

    @pytest.mark.parametrize(
        "key",
        [
            "gyro_arw",
            "gyro_bias_instability",
            "gyro_rrw",
            "accel_vrw",
            "accel_bias_instability",
        ],
    )
    def test_consumer_is_noisier_than_tactical(self, key):
        """The accel bias instability failed this by 360x.

        Cheap, and it is the check that would have caught a stray "/ 3600.0"
        without anyone having to know what a plausible accelerometer bias
        instability is.
        """
        assert IMU_SPECS["consumer"][key] > IMU_SPECS["tactical"][key], (
            f"consumer {key} is {IMU_SPECS['consumer'][key]:.3e}, better than "
            f"tactical's {IMU_SPECS['tactical'][key]:.3e}. A consumer-grade "
            f"part cannot be quieter than a tactical one; suspect a unit "
            f"conversion."
        )


class TestTheUnidentifiableParametersAreStillUnidentifiable:
    """Two parameters this record cannot measure, asserted as such.

    Not a complaint about the estimator. Rate random walk shows at a tau this
    record does not reach, and the accelerometer's white noise does not fall to
    its bias-instability floor until tau = 22681 s -- which, since tau_max is
    duration/10, needs a 63-hour record.

    These assert the limitation *persists*, on the same reasoning as
    KNOWN_FROZEN and FRONTEND_IS_KNOWN_NO_OP: if a change makes these regions
    appear, the example's text about them is now wrong and should be rewritten
    deliberately rather than silently left behind.
    """

    def test_the_gyro_curve_has_not_reached_the_rate_random_walk_region(
        self, recovered
    ):
        slope = _long_tau_slope(recovered["gyro"])

        assert slope < 0.3, (
            f"the gyro long-tau slope is {slope:+.2f}, close enough to +0.5 "
            f"that the rate random walk may now be readable. If that is "
            f"intended, the example's 'NOT REACHED' text needs updating."
        )

    def test_the_accel_curve_is_still_falling_as_white_noise(self, recovered):
        slope = _long_tau_slope(recovered["accel"])

        assert slope < -0.3, (
            f"the accel long-tau slope is {slope:+.2f} rather than about "
            f"-0.5, so the curve may now be flattening onto its bias "
            f"instability. Update the example's text if so."
        )


def test_the_record_is_reproducible():
    """Same seed, same record -- the reason DEFAULT_SEED exists.

    The generator used a bare np.random.default_rng(). Bias instability came
    out 11.15 deg/hr on one run and 7.85 on the next, so the committed figure
    could not be regenerated and a diff against it meant nothing.
    """
    short = dict(duration=60.0, fs=FS_HZ, imu_grade=GRADE)

    _, gyro_a, accel_a = generate_imu_stationary_data(**short)
    _, gyro_b, accel_b = generate_imu_stationary_data(**short)

    assert np.array_equal(gyro_a, gyro_b), "same seed gave a different gyro record"
    assert np.array_equal(accel_a, accel_b), "same seed gave a different accel record"

    _, gyro_c, _ = generate_imu_stationary_data(**short, seed=DEFAULT_SEED + 1)
    assert not np.array_equal(gyro_a, gyro_c), (
        "a different seed gave an identical record, so the seed is not "
        "actually reaching the draws"
    )
