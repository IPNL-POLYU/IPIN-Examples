"""Chapter 6's ZUPT detector used to fire on almost every sample.

`example_zupt` ran `detect_zupt_windowed` at ``gamma=1000.0``, commented "much
higher threshold for noisy consumer IMU".  Measured on the example's own data:
it reported zero velocity on **98.2% of swing samples** and scored **26.7%**
accuracy -- exactly the 26.67% stance base rate, which is what a constant
"always stationary" predictor scores.  The transcript in the chapter README
even printed ``ZUPT detections: 97.0% of samples`` next to a stance ratio of
26.7%, and nobody read the two lines together.

Same signature as the ch5 floor classifier landing on its base rate and the
ch7 SLAM front-end reporting "-0.00% improvement": a stage that has stopped
doing anything rarely says so, it just reports the arithmetic of doing
nothing.

**And it looked like a success.** The always-fires configuration reported a
91.7% RMSE reduction, because clamping velocity to zero on a 61.6 m walk keeps
the estimate near the origin, which beats an IMU-only track that has exploded
to 237 m.  That number is not achievable by any detector: an ORACLE detector
handed the true stance mask scores 35.08 m RMSE, and 9.22 m is better than the
oracle.  A result that beats the physical ceiling is measuring something else.

So this asserts what the floor-detector guard asserts -- that the detector
beats a constant predictor -- rather than an accuracy or an error threshold,
because a threshold anywhere near the base rate would have passed the bug.

Author: Li-Ta Hsu
"""

import numpy as np
import pytest

from ch6_dead_reckoning.example_zupt import (
    ZUPT_GAMMA,
    add_imu_noise,
    generate_walking_trajectory,
)
from core.sensors import (
    FrameConvention,
    IMUNoiseParams,
    detect_zupt_windowed,
    random_walk_to_rate_sample_std,
)

DT = 0.01
DURATION_S = 60.0
WINDOW = 10

#: The detector has to beat "always stationary" by this much, in percentage
#: points. Measured: the shipped threshold scores 98.4% against a 26.7% base
#: rate, a margin of 71.7 points, and the broken one scored 0.04 points.
#: 20 points sits an order of magnitude below the working margin and two
#: orders above the broken one.
MARGIN_PERCENTAGE_POINTS = 20.0


@pytest.fixture(scope="module")
def walk():
    """The example's own trajectory and noise realisation."""
    np.random.seed(42)
    frame = FrameConvention.create_enu()
    imu_params = IMUNoiseParams.consumer_grade()
    _, _, _, _, accel_body, gyro_body, stance = generate_walking_trajectory(
        DURATION_S, DT, 2.0, 0.7, frame
    )
    accel_meas, gyro_meas = add_imu_noise(accel_body, gyro_body, DT, imu_params)
    return accel_meas, gyro_meas, stance, imu_params


def detector_scores(accel_meas, gyro_meas, stance, imu_params, gamma):
    """(true positive rate, false positive rate, accuracy, base rate), all %."""
    sigma_a = random_walk_to_rate_sample_std(imu_params.accel_vrw_mps_sqrt_s, DT)
    sigma_g = random_walk_to_rate_sample_std(imu_params.gyro_arw_rad_sqrt_s, DT)
    n_samples = len(stance)
    predicted = np.array(
        [
            detect_zupt_windowed(
                accel_meas[k - WINDOW : k],
                gyro_meas[k - WINDOW : k],
                gamma=gamma,
                g=9.81,
                sigma_a=sigma_a,
                sigma_g=sigma_g,
            )
            for k in range(WINDOW, n_samples)
        ]
    )
    truth = stance[WINDOW:]
    return (
        predicted[truth].mean() * 100.0,
        predicted[~truth].mean() * 100.0,
        (predicted == truth).mean() * 100.0,
        truth.mean() * 100.0,
    )


def test_the_detector_beats_a_constant_predictor(walk):
    """The assertion itself: accuracy must clear the base rate by a margin."""
    tpr, fpr, accuracy, base_rate = detector_scores(*walk, gamma=ZUPT_GAMMA)

    assert accuracy > base_rate + MARGIN_PERCENTAGE_POINTS, (
        f"ZUPT detector at gamma={ZUPT_GAMMA} scores {accuracy:.2f}% against a "
        f"{base_rate:.2f}% stance base rate (TPR {tpr:.2f}%, FPR {fpr:.2f}%). "
        "An accuracy at the base rate means the detector has one output."
    )


def test_the_detector_output_actually_varies(walk):
    """One line that would have caught it, and is cheaper than any of this."""
    accel_meas, gyro_meas, stance, imu_params = walk
    tpr, fpr, _, _ = detector_scores(*walk, gamma=ZUPT_GAMMA)

    assert tpr > 50.0, f"detector never fires during stance (TPR {tpr:.2f}%)"
    assert fpr < 10.0, f"detector fires during swing (FPR {fpr:.2f}%)"


def test_the_old_threshold_would_fail_this(walk):
    """Guard the guard: gamma=1000 must be red, or the margin above is decorative.

    Run it and the numbers are TPR 95.00%, FPR 98.18%, accuracy 26.71% against
    a 26.67% base rate -- 0.04 percentage points of margin.
    """
    tpr, fpr, accuracy, base_rate = detector_scores(*walk, gamma=1000.0)

    assert fpr > 90.0, "gamma=1000 no longer fires on swing; re-derive this test"
    assert accuracy < base_rate + MARGIN_PERCENTAGE_POINTS, (
        f"gamma=1000 now scores {accuracy:.2f}% against base rate {base_rate:.2f}%, "
        "so the margin no longer separates the broken configuration from the "
        "working one"
    )


def test_the_threshold_is_not_perched_on_a_cliff(walk):
    """A tuned constant is only honest if the answer is flat around it.

    Measured: accuracy is 98.40% at gamma 16-20, falls to 95.04% at 13.5 and
    to 79.07% at 21.5. +-15% around 17.0 stays inside the plateau, so the
    value was chosen from a region rather than fitted to an edge.
    """
    for factor in (0.85, 1.0, 1.15):
        _, _, accuracy, base_rate = detector_scores(*walk, gamma=ZUPT_GAMMA * factor)
        assert accuracy > base_rate + MARGIN_PERCENTAGE_POINTS, (
            f"gamma={ZUPT_GAMMA * factor:.2f} ({factor:.2f}x) scores "
            f"{accuracy:.2f}% -- the threshold sits on a cliff"
        )
