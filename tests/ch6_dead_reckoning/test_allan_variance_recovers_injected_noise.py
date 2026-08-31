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
    # The RRW region is absent from a 1-hour record, and the gyro's ARW read
    # sits on its bias-instability shoulder; characterize_imu_noise says so
    # through warnings. Asserted below rather than ignored; here it is noise.
    with pytest.warns(UserWarning):
        return characterize_imu_noise(gyro, accel, fs=FS_HZ)


def _long_tau_slope(result):
    """Slope of log10(adev) against log10(tau) over the last decade."""
    taus = np.asarray(result["taus"])
    adev = np.asarray(result["adev"])
    tail = taus >= taus[-1] / 10.0
    return float(np.polyfit(np.log10(taus[tail]), np.log10(adev[tail]), 1)[0])


class TestShortTauParametersComeBack:
    """ARW and VRW are read at short tau -- but only where the slope is -1/2.

    "Short tau" is not the same as "readable". `identify_random_walk` reads at
    tau = 1 s by convention, and that lands in the white-noise region only
    while the bias-instability floor is still below the curve there. The
    crossover is at tau = (RW / 0.664 B)^2, and for the shared consumer spec
    that is 0.8 s for the gyro and 6.3 s for the accelerometer -- so the two
    sensors are in different situations at the same tau, from the same table.
    """

    def test_angle_random_walk_is_readable_but_biased_by_the_early_shoulder(
        self, recovered
    ):
        """The gyro's white-noise region ends before tau = 1 s.

        With ARW 0.1 deg/sqrt(hr) and B 10 deg/hr, sigma_ARW(tau) drops below
        0.664 B at tau = 0.8 s, so the conventional read at 1 s is already on
        the shoulder and comes back high -- measured, 2.7x. That is a property
        of this spec, not an estimator error, and lengthening the record does
        not help: a longer run moves the far end of the curve, not the near
        one. Bounded rather than pinned so that a genuine regression (an order
        of magnitude, a unit slip) still fails.

        This used to assert 0.8-1.2x, and passed because the example injected
        an ARW five times larger than `IMUNoiseParams.consumer_grade()` -- a
        second "consumer" spec whose shoulder sat at tau = 20 s.
        """
        ratio = recovered["gyro"]["angle_random_walk"] / injected_si(GRADE)["gyro_arw"]

        assert 1.5 < ratio < 4.0, (
            f"gyro ARW recovered {ratio:.2f}x the injected value. Expected "
            f"about 2.7x: the read at tau = 1 s sits past the shoulder. A "
            f"ratio near 1 would mean the shoulder has moved (check the "
            f"IMUNoiseParams consumer spec); a ratio near 60 or 1/60 is a "
            f"sqrt(3600) unit slip."
        )

    def test_velocity_random_walk_within_50_percent(self, recovered):
        """Same argument, accelerometer side, where tau = 1 s is still clean.

        Its shoulder is at 6.3 s, so the read at 1 s is inside the -1/2 region
        and the recovery is good: measured 1.3x, the residual being that the
        floor is already within a factor of two of the curve by then.
        """
        ratio = (
            recovered["accel"]["velocity_random_walk"] / injected_si(GRADE)["accel_vrw"]
        )

        assert (
            0.8 < ratio < 1.5
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
    """What this record can and cannot see, asserted rather than assumed.

    Rate random walk shows at a tau a one-hour record does not reach, and that
    is still true.

    The accelerometer's bias instability used to be in the same category --
    "does not fall to its floor until tau = 22681 s, which needs a 63-hour
    record". It is readable now, and the reason is worth recording because the
    old sentence was arithmetically correct and diagnosed the wrong thing: the
    injected VRW was 60x too large (0.01 read as m/s/sqrt(s) where
    `IMUNoiseParams` means m/s/sqrt(hr)), so the white noise started 60x higher
    and had 3600x further to fall. With the two specs unified the floor arrives
    at tau = 6.3 s. Nothing about the estimator or the record length changed.

    The remaining assertion holds the limitation that is real, on the same
    reasoning as KNOWN_FROZEN and FRONTEND_IS_KNOWN_NO_OP: if a change makes
    the region appear, the example's text about it is now wrong and should be
    rewritten deliberately rather than silently left behind. This test did
    exactly that job for the accelerometer -- it went red on the fix, and its
    own message said "Update the example's text if so."
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

    def test_the_accel_curve_now_reaches_its_bias_instability_floor(self, recovered):
        """The half of this class that flipped, and it flipped the right way.

        A flat long-tau slope means the bias-instability plateau is present,
        which is what makes the accelerometer's BI recoverable at all: it comes
        back within 20% now, against 5.9x before the VRW unit slip was fixed.
        """
        slope = _long_tau_slope(recovered["accel"])

        assert abs(slope) < 0.2, (
            f"the accel long-tau slope is {slope:+.2f} rather than about 0, so "
            f"the curve is no longer flattening onto its bias instability. If "
            f"the injected VRW grew again, the floor has moved out of reach."
        )

    def test_the_accel_bias_instability_comes_back(self, recovered):
        """The consequence, stated as a number rather than as a slope."""
        ratio = (
            recovered["accel"]["bias_instability"]
            / injected_si(GRADE)["accel_bias_instability"]
        )

        assert 0.8 < ratio < 1.5, (
            f"accel bias instability recovered {ratio:.2f}x the injected "
            f"value; it was 5.9x when the curve never reached the floor."
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
