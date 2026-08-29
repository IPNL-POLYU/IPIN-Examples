"""Each chapter notebook exposes one small, repeatable learner experiment."""

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("ch*.ipynb"))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_has_beginner_experiment_controls(notebook_path: Path):
    """Learners get presets and an interpretation loop, not vague tuning advice."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    text_lower = text.casefold()

    required_guidance = (
        "experiment controls",
        "try changing",
        "observe",
        "why",
    )
    missing = [phrase for phrase in required_guidance if phrase not in text_lower]
    if "cannot conclude" not in text_lower and "do not conclude" not in text_lower:
        missing.append("cannot/do not conclude")
    assert not missing, f"{notebook_path.name} is missing {missing}"

    preset_dicts = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            target_names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
            if any(
                name.endswith("PRESETS") or name == "SCENARIOS" for name in target_names
            ):
                keys = {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                preset_dicts.append(keys)

    assert preset_dicts, f"{notebook_path.name} has no named preset dictionary"
    assert any("baseline" in keys and len(keys) >= 3 for keys in preset_dicts), (
        f"{notebook_path.name} must offer baseline plus at least two experiments; "
        f"found {preset_dicts}"
    )


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_playground_notebooks_keep_outputs_clear(notebook_path: Path):
    """Committed tutorials remain clean even after their controls are exercised."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)
