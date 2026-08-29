"""Reader-facing guidance checks for the Ch2-Ch4 beginner path.

These tests intentionally check source text, not rendered pixels. They lock the
audit fixes that make the notebooks/READMEs usable for a first-time indoor
positioning reader while staying lightweight enough for docs CI.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _notebook(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _notebook_text(path: str) -> str:
    nb = _notebook(path)
    return "\n\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in nb["cells"]
    )


def test_ch2_notebook_has_beginner_controls_and_inline_visuals() -> None:
    text = _notebook_text("notebooks/ch2_coordinate_systems.ipynb")

    assert "## Beginner Experiment controls" in text
    assert 'EXPERIMENT_PRESET = "baseline"' in text
    assert '"baseline"' in text
    assert '"nearby_room"' in text
    assert '"gimbal_lock_demo"' in text
    assert "Try changing" in text
    assert "Observe" in text
    assert "Why" in text
    assert "Cannot conclude" in text

    assert 'SVG(filename="ch2_coords/figs/ch2_frame_chain.svg")' in text
    assert 'SVG(filename="ch2_coords/figs/ch2_gimbal_lock.svg")' in text
    assert "roll about +Y" in text
    assert "pitch about +X" in text


def test_ch3_notebook_has_beginner_controls_and_presets() -> None:
    text = _notebook_text("notebooks/ch3_state_estimation.ipynb")

    assert "## Beginner Experiment controls" in text
    assert 'EXPERIMENT_PRESET = "baseline"' in text
    assert '"baseline"' in text
    assert '"noisy_ranges"' in text
    assert '"smooth_filter"' in text
    assert "Try changing" in text
    assert "Observe" in text
    assert "Why" in text
    assert "Cannot conclude" in text
    assert "range_noise_std_m" in text
    assert "process_noise_std_mps2" in text
    assert "measurement_noise_std_m" in text


def test_ch2_ch3_notebooks_have_no_stored_outputs() -> None:
    for path in [
        "notebooks/ch2_coordinate_systems.ipynb",
        "notebooks/ch3_state_estimation.ipynb",
    ]:
        nb = _notebook(path)
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell.get("outputs", []) == [], f"{path} cell {index} stores outputs"
            assert (
                cell.get("execution_count") is None
            ), f"{path} cell {index} stores an execution_count"


def test_ch3_readme_explains_multi_panel_figures() -> None:
    readme = (ROOT / "ch3_estimators/README.md").read_text(encoding="utf-8")

    for figure in [
        "ch3_least_squares_examples.png",
        "ch3_kalman_1d_tracking.png",
        "ch3_ekf_range_bearing.png",
        "ch3_iekf_vs_ekf_comparison.png",
        "ch3_estimator_comparison.png",
    ]:
        start = readme.index(figure)
        window = readme[start : start + 1400]
        assert "Read" in window, f"{figure} lacks panel-reading guidance"
        assert "takeaway" in window.lower(), f"{figure} lacks a takeaway"


def test_ch4_readme_clarifies_aoa_direction_and_lists_new_figures() -> None:
    readme = (ROOT / "ch4_rf_point_positioning/README.md").read_text(encoding="utf-8")

    assert "model angle from the agent toward each anchor" in readme
    assert "reciprocal" in readme
    assert "dashed ray from A0" in readme
    assert "northeast" in readme

    for figure, command in [
        (
            "ch4_dop_geometry.png",
            "python -m ch4_rf_point_positioning.example_dop_geometry",
        ),
        (
            "ch4_initial_guess_basin.png",
            "python -m ch4_rf_point_positioning.example_initial_guess_basin",
        ),
    ]:
        assert figure in readme
        assert command in readme
