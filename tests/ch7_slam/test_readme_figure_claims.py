"""Pin reader-facing Chapter 7 figure descriptions to the current figures."""

from pathlib import Path

README = Path("ch7_slam/README.md")


def test_slam_with_maps_layout_matches_current_static_figure():
    text = README.read_text(encoding="utf-8")

    assert "four panels in a 1x3 grid" not in text
    assert "four axes arranged on a 2x3 layout" in text
    assert "Static loop-closure edges are intentionally not drawn" in text
    assert "orange dash-dot" in text


def test_slam_frontend_demo_has_reader_guidance():
    text = README.read_text(encoding="utf-8")

    assert "figs/slam_frontend_demo.svg" in text
    assert "two-panel teaching example" in text
    assert "step index" in text
    assert "it never sees it" in text
