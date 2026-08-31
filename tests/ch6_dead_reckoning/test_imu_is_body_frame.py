"""A shipped accelerometer must be specific force in the body frame.

Five datasets ship `imu.npz/accel_xy` beside a `truth.npz` the reader is meant
to reproduce by integrating it: Chapter 6's `ch6_strapdown_basic` and
`ch6_foot_zupt_walk`, and Chapter 8's three fusion datasets. Their READMEs
state the contract: `f~ = f + b_a + n_a` is a *measured specific force*, and
Eq. (6.19) integrates it as `v += C(theta) f~ dt`. That `C(theta)` rotates body
to map, so a map-frame vector handed to it is rotated twice. Chapter 8's
filters do the same thing in `core/fusion/tc_models.py`, whose process model
rotates u = [ax, ay, .] by the state yaw before adding it to the map-frame
velocity.

**Both chapters shipped the same defect, and Chapter 8's outlived the fix to
Chapter 6's because this file hand-listed the two datasets it knew about.**
Each generator differentiated the truth velocity and wrote the result out
unrotated, taking `yaw` as an argument and using it only for the gyro. So the
list is now discovered by scanning `data/sim` for an accelerometer: the next
dataset to ship one is covered without anyone remembering to add it.

The tell in both cases was that the residual against *map-frame* acceleration
was 0.1002 m/s^2 -- indistinguishable from the declared 0.1 noise -- while the
residual against body frame carried a systematic component. A frame error hides
as clean noise in whichever frame you compare against, so compare against both
and let the systematic mean pick the winner.

`ch6_foot_zupt_walk` was unaffected in value because its walk is a straight
line with yaw identically zero, which makes body and map the same frame. It is
tested here anyway: the next preset that turns would have inherited the bug.

**The two trajectory assertions catch different shapes, which is why there are
two.** The extent check is what caught Chapter 6: on a circle the double
rotation drew a 16.9 m radius against a true 10.0 m. It is *blind* to Chapter
8's instance -- the broken rectangular walk integrates to an extent ratio of
1.00, because a path that wanders can still have a normal spread about its own
centroid. What catches that one is the deviation from the true path, sample by
sample: a median of 9.84 m against a true extent of 9.66 m, where the corrected
data gives 1.38 m. A statistic that reduces a trajectory to its size cannot see
an error that preserves the size.

Author: Li-Ta Hsu
References: Chapter 6, Eqs. (6.5), (6.9), (6.19); Chapter 8, Section 8.1.2
"""

import json
import unittest
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "sim"

#: Every dataset that ships an accelerometer, found rather than remembered.
DATASETS = tuple(sorted(d.name for d in DATA.iterdir() if (d / "imu.npz").is_file()))

#: What the scan found when this was written. A glob that silently matches
#: nothing turns every assertion below into a no-op, and a vacuous green is
#: the failure mode this whole file exists to prevent -- so the discovery is
#: itself checked. Raise this when a new IMU dataset lands.
EXPECTED_AT_LEAST = 5


def _load(name):
    p = DATA / name
    imu = np.load(p / "imu.npz")
    truth = np.load(p / "truth.npz")
    cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
    return imu, truth, cfg


def _accel_in_body_frame(truth):
    """The specific force the shipped accelerometer should be reporting."""
    t, vel, yaw = truth["t"], truth["v_xy"], truth["yaw"]
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    a_map = np.gradient(vel, axis=0) / dt[:, None]
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    return np.column_stack(
        [
            cos_y * a_map[:, 0] + sin_y * a_map[:, 1],
            -sin_y * a_map[:, 0] + cos_y * a_map[:, 1],
        ]
    )


def _dead_reckon(imu, truth):
    """Integrate the shipped IMU per Eq. (6.19) and return the track."""
    t, accel, gyro = imu["t"], imu["accel_xy"], imu["gyro_z"]
    pos, vel, yaw = truth["p_xy"], truth["v_xy"], truth["yaw"]
    dt = float(np.median(np.diff(t)))

    theta, v, x = yaw[0], vel[0].copy(), pos[0].copy()
    track = [x.copy()]
    for k in range(1, len(t)):
        theta += gyro[k - 1] * dt
        c, s = np.cos(theta), np.sin(theta)
        a_map = np.array(
            [
                c * accel[k - 1, 0] - s * accel[k - 1, 1],
                s * accel[k - 1, 0] + c * accel[k - 1, 1],
            ]
        )
        v = v + a_map * dt
        x = x + v * dt
        track.append(x.copy())
    return np.asarray(track)


class TestImuIsBodyFrame(unittest.TestCase):

    def test_the_scan_found_the_datasets(self):
        """A guard that discovers its own inputs must say what it found."""
        self.assertGreaterEqual(
            len(DATASETS),
            EXPECTED_AT_LEAST,
            f"only found {DATASETS}; a scan matching nothing makes every "
            f"other assertion in this file vacuous",
        )

    def test_accel_matches_the_truth_rotated_into_body(self):
        """Residual must be the declared noise, with no systematic part."""
        for name in DATASETS:
            imu, truth, cfg = _load(name)
            residual = imu["accel_xy"] - _accel_in_body_frame(truth)
            sigma = cfg["imu"]["accel_noise_std_m_s2"]

            with self.subTest(dataset=name):
                self.assertLess(
                    float(residual.std()),
                    1.5 * sigma,
                    "accelerometer does not match the body-frame truth",
                )
                # A frame error shows up here rather than in the spread.
                self.assertLess(
                    float(np.abs(residual.mean(axis=0)).max()),
                    0.25 * sigma,
                    "residual has a systematic component, which is what a "
                    "wrong frame looks like",
                )

    def test_integrating_the_imu_reproduces_the_trajectory(self):
        """Eq. (6.19) on the shipped bytes must trace the shipped truth.

        Drift is the lesson, so the bounds are loose -- but they have to be
        the right *shape*, and one bound cannot do it. Both were measured on
        the corrected data and on the map-frame data it replaces:

            statistic (normalised by extent)   corrected      map-frame ch8
            mean extent about the centroid     0.96 - 1.01    1.00  (blind)
            median deviation from the truth    0.01 - 0.14    1.02

        The extent check is what caught Chapter 6's circle; only the deviation
        check catches Chapter 8's rectangle.
        """
        for name in DATASETS:
            imu, truth, _ = _load(name)
            pos = truth["p_xy"]
            track = _dead_reckon(imu, truth)

            true_extent = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
            got_extent = float(
                np.linalg.norm(track - track.mean(axis=0), axis=1).mean()
            )
            median_deviation = float(np.median(np.linalg.norm(track - pos, axis=1)))

            with self.subTest(dataset=name):
                self.assertAlmostEqual(
                    got_extent,
                    true_extent,
                    delta=0.25 * true_extent,
                    msg=f"{name}: integrated path spans {got_extent:.2f} m about "
                    f"its centre against {true_extent:.2f} m for the truth",
                )
                # Measured 0.14 at worst on the corrected data against 1.02 on
                # the map-frame data, so this sits well clear of both.
                self.assertLess(
                    median_deviation,
                    0.35 * true_extent,
                    f"{name}: integrated path sits a median "
                    f"{median_deviation:.2f} m from the truth, on a trajectory "
                    f"only {true_extent:.2f} m across",
                )


if __name__ == "__main__":
    unittest.main()
