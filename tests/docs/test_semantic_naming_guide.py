"""The semantic API contract must stay discoverable and complete.

PEP 8 can enforce spelling and layout, but it cannot tell a new indoor-
positioning reader whether an array is truth or an estimate, metres or radians,
ENU or body frame. The repository therefore carries a domain-specific naming
guide. These checks keep it linked from the two places contributors actually
read and prevent a future rewrite from silently dropping one of its contracts.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs" / "api_naming_conventions.md"


def test_semantic_naming_guide_is_linked_from_reader_and_contributor_docs():
    """Readers and code contributors must both be able to discover the guide."""
    expected_path = "docs/api_naming_conventions.md"
    for relative_path in ("README.md", ".github/copilot-instructions.md"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert expected_path in text, f"{relative_path} does not link {expected_path}"


@pytest.mark.parametrize(
    "required_heading",
    (
        "## Public names and mathematical notation",
        "## Roles",
        "## Units",
        "## Coordinate frames and transforms",
        "## Domain vocabulary",
        "## Method verbs and side effects",
        "## Return values",
        "## Compatibility policy",
        "## Review checklist",
    ),
)
def test_semantic_naming_guide_keeps_each_required_contract(required_heading):
    """The guide must cover every semantic dimension named by issue #113."""
    text = GUIDE.read_text(encoding="utf-8")
    assert required_heading in text
