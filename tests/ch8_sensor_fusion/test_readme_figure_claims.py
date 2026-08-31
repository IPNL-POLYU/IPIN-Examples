"""Pin reader-facing Chapter 8 figure descriptions to current outputs."""

import re
from pathlib import Path

README = Path("ch8_sensor_fusion/README.md")


def test_anchor_outage_ratio_is_directional_not_zero_x():
    """The peak ratio must name which side is larger, and by how much.

    This used to pin the literal string "TC peak is 13.5x LC". That number
    was a mirror-branch flip caused by the shipped accelerometer being in the
    map frame, and correcting it reversed the direction as well as the
    magnitude -- so pinning either would just have to be re-pinned again. The
    invariant worth holding is the *form*: a named direction and a real
    multiplier, never the "(0x)" this test was written to prevent. The exact
    line is checked against a live run by
    tests/docs/test_readme_example_output.py.
    """
    text = README.read_text(encoding="utf-8")

    assert "(0x)" not in text
    assert re.search(
        r"\((?:LC|TC) peak is \d+(?:\.\d+)?x (?:TC|LC)\)", text
    ), "the anchor-outage transcript no longer states a directional peak ratio"
    assert "outage + 3 s recovery" in text


def test_lc_tc_color_keys_and_mixed_unit_bars_are_explicit():
    text = README.read_text(encoding="utf-8")

    assert "ground truth (blue) vs EKF estimate (orange)" not in text
    assert "truth is black, TC EKF is blue" in text
    assert "black is ground truth, LC is" in text
    assert "mixed-unit summary" in text
    assert "Updates [×100]" in text


def test_imu_calibration_describes_four_axes():
    text = README.read_text(encoding="utf-8")

    assert "Three-panel IMU" not in text
    assert "Four-axis IMU" in text
    assert "Bottom-left" in text
    assert "Bottom-right" in text
