"""A reported heading error must be the shorter arc, and nothing else.

`generate_ch6_env_sensors_dataset.py` reduced its heading error with::

    heading_error = np.abs(heading_est - heading_true)
    heading_error = np.minimum(heading_error, 2 * np.pi - heading_error)

which is the shorter arc only while ``|d| <= 2*pi``. It is not: the building
walk's first phase runs the true yaw up to 7.84 rad while ``mag_heading``
returns ``(-pi, pi]``, so ``|d|`` reached 11 rad and ``2*pi - |d|`` went
**negative**. 221 of 1800 samples were handed a negative "error", and the mean
they dragged down was written into `config.json` as ``mean_error_deg: 2.66``
against a true 3.51 -- understated by 24%.

Three things about the shape, all of which recur:

- **It was invisible on clean data.** With a noiseless magnetometer the
  difference is exactly 0 or exactly 2*pi, and ``2*pi - 2*pi`` is 0, so the
  broken form gives the right answer. Only the noisy path -- the one that
  ships -- produces ``2*pi + epsilon`` and therefore ``-epsilon``. Testing the
  formula on ideal input would have confirmed it.
- **The ratio survived.** The README's tilt-compensation experiment divided two
  of these means and printed "1.9x worse". It still prints 1.9x, because both
  sides were understated alike. A ratio can stay convincing while the
  magnitudes under it are wrong.
- **The library already had the answer.** `core.sensors.wrap_angle_diff` exists
  for exactly this, and `ch6_dead_reckoning/example_environment.py` already used
  it for the same computation. The generator and the dataset README each
  hand-rolled a copy instead, and both got it wrong -- duplicated policy only
  has to be forgotten once.

Author: Li-Ta Hsu
References: Chapter 6, Eqs. (6.51)-(6.53)
"""

import json
import unittest
from pathlib import Path

import numpy as np

from core.sensors import mag_heading, wrap_angle_diff

DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "sim"
    / "ch6_env_sensors_heading_altitude"
)


class TestHeadingErrorIsWrapped(unittest.TestCase):
    """The heading accuracy in config.json is the one the data supports."""

    @classmethod
    def setUpClass(cls):
        with open(DATA_DIR / "config.json") as handle:
            cls.config = json.load(handle)
        att = np.loadtxt(DATA_DIR / "ground_truth_attitude.txt")
        mag = np.loadtxt(DATA_DIR / "magnetometer.txt")
        cls.yaw_true = att[:, 2]
        cls.heading_est = np.array(
            [
                mag_heading(mag[k], att[k, 0], att[k, 1], declination=0.0)
                for k in range(len(att))
            ]
        )
        cls.error = np.abs(
            np.array(
                [wrap_angle_diff(e, y) for e, y in zip(cls.heading_est, cls.yaw_true)]
            )
        )

    def test_the_trajectory_still_exercises_the_wrap(self) -> None:
        """Guard the guard: the true yaw must leave (-pi, pi].

        If the trajectory were ever rewritten to keep yaw wrapped, every
        assertion below would hold for a broken implementation too, and this
        file would quietly stop testing anything.
        """
        self.assertGreater(
            self.yaw_true.max(),
            np.pi,
            "the building walk no longer drives yaw past pi, so the wrap this "
            "file exists to check is no longer exercised.",
        )

    def _naive_reduction(self) -> np.ndarray:
        """The old `min(|d|, 2pi - |d|)`, kept so its failure stays visible."""
        d = np.abs(self.heading_est - self.yaw_true)
        return np.minimum(d, 2 * np.pi - d)

    def test_the_naive_reduction_really_does_break_here(self) -> None:
        """The hazard is live on this data, not hypothetical.

        Asserting that ``wrap_angle_diff`` returns something non-negative and
        below 180 deg would test numpy, not the dataset -- both hold by
        construction. What is worth pinning is that the *alternative* fails on
        exactly this trajectory, which is what makes the assertion below a real
        constraint rather than a restatement of the helper's contract.
        """
        naive = self._naive_reduction()
        self.assertGreater(
            int((naive < 0).sum()),
            0,
            "the naive reduction no longer produces negative errors here, so "
            "this dataset no longer demonstrates the defect and the check "
            "below has stopped discriminating.",
        )

    def test_config_does_not_match_the_naive_reduction(self) -> None:
        """A regression to the old formula fails loudly rather than silently.

        The two reductions differ by 0.84 deg on this data, well outside the
        0.05 tolerance below, so they cannot both satisfy the config check.
        """
        naive_mean = float(np.rad2deg(self._naive_reduction()).mean())
        reported = self.config["performance"]["magnetometer_heading"]["mean_error_deg"]
        self.assertGreater(
            abs(reported - naive_mean),
            0.05,
            f"config.json's {reported:.4f} deg matches the naive reduction "
            f"({naive_mean:.4f}), which counts negative errors toward the mean.",
        )

    def test_config_matches_the_shipped_measurements(self) -> None:
        """config.json reports what the shipped sensor data actually gives.

        Recomputed from `magnetometer.txt` and `ground_truth_attitude.txt`, not
        from a remembered number -- so this fails if either the data or the
        reduction drifts from the claim.
        """
        reported = self.config["performance"]["magnetometer_heading"]
        for name, actual in (
            ("mean_error_deg", float(np.rad2deg(self.error).mean())),
            ("max_error_deg", float(np.rad2deg(self.error).max())),
        ):
            with self.subTest(metric=name):
                self.assertAlmostEqual(
                    reported[name],
                    actual,
                    delta=0.05,
                    msg=f"config.json says {name}={reported[name]:.4f}, the "
                    f"shipped data gives {actual:.4f}.",
                )


if __name__ == "__main__":
    unittest.main()
