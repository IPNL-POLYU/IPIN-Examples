"""Notebook experiment-control cells should expose beginner presets."""

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _assignment_value(source: str, name: str):
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found")


def _code_sources(notebook_name: str) -> list[str]:
    notebook = json.loads(
        (REPO_ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
    )
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_ch5_notebook_has_three_experiment_presets() -> None:
    source = next(
        src for src in _code_sources("ch5_fingerprinting.ipynb") if "CH5_PRESETS" in src
    )

    presets = _assignment_value(source, "CH5_PRESETS")

    assert "baseline" in presets
    assert len(presets) >= 3


def test_ch6_notebook_has_three_experiment_presets() -> None:
    source = next(
        src for src in _code_sources("ch6_dead_reckoning.ipynb") if "CH6_PRESETS" in src
    )

    presets = _assignment_value(source, "CH6_PRESETS")

    assert "baseline" in presets
    assert len(presets) >= 3
