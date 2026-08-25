"""A shipped accelerometer must be specific force in the body frame.

`ch6_strapdown_basic` and `ch6_foot_zupt_walk` ship `imu.npz/accel_xy` beside a
`truth.npz` the reader is meant to reproduce by integrating it. Their READMEs
state the contract: `f~ = f + b_a + n_a` is a *measured specific force*, and
Eq. (6.19) integrates it as `v += C(theta) f~ dt`. That `C(theta)` rotates body
to map, so a map-frame vector handed to it is rotated twice.

`accel_xy` was map-frame. The generator differentiated the truth velocity and
wrote the result out unrotated, taking `yaw` as an argument and using it only
for the gyro. On the circular trajectory that meant the accelerometer carried
no centripetal term at all: integrating per Eq. (6.19) drew a 16.9 m radius
against a true 10.0 m and finished 69 m away after 60 s.

The tell was that the residual against *map-frame* acceleration was 0.1002
m/s^2 -- indistinguishable from the declared 0.1 noise -- while the residual
against body frame carried a systematic 0.1 m/s^2, exactly the centripetal
magnitude v^2/r. A frame error hides as a constant offset in the rotating
frame, so comparing against the wrong frame looks like clean noise.

ch6_foot_zupt_walk was unaffected in value because its walk is a straight line
with yaw identically zero, which makes body and map the same frame. It is
tested here anyway: the next preset that turns would have inherited the bug.

Author: Li-Ta Hsu
References: Chapter 6, Eqs. (6.5), (6.9), (6.19)
"""

import json
import unittest
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "sim"
DATASETS = ("ch6_strapdown_basic", "ch6_foot_zupt_walk")


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


class TestImuIsBodyFrame(unittest.TestCase):

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

        Drift is the lesson, so the bound is loose -- but it has to be the
        right *shape*. The mean radius check is what the map-frame data failed:
        16.9 m against a true 10.0 m, which no amount of drift explains.
        """
        for name in DATASETS:
            imu, truth, _ = _load(name)
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
            track = np.asarray(track)

            with self.subTest(dataset=name):
                true_extent = float(
                    np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean()
                )
                got_extent = float(
                    np.linalg.norm(track - track.mean(axis=0), axis=1).mean()
                )
                self.assertAlmostEqual(
                    got_extent,
                    true_extent,
                    delta=0.25 * true_extent,
                    msg=f"{name}: integrated path spans {got_extent:.2f} m about "
                    f"its centre against {true_extent:.2f} m for the truth",
                )


if __name__ == "__main__":
    unittest.main()
