"""Semantic ambiguity in public ``core`` APIs may only shrink.

PEP 8 cannot tell whether ``R`` is a rotation or measurement covariance, or
whether the third item in a tuple is RMSE, a score, or convergence metadata.
Issue #113 introduces descriptive APIs without breaking every historical caller
at once. This ratchet records the remaining legacy boundary debt and rejects a
new ambiguous parameter or raw tuple/dict result.

The baseline is per violation kind rather than one total: improving ``ref_idx``
must not pay for adding a new public ``noise_std`` parameter.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1] / "core"

AMBIGUOUS_PUBLIC_PARAMETERS = {
    "F",
    "H",
    "P",
    "Q",
    "R",
    "S",
    "cov",
    "dt",
    "est",
    "noise_std",
    "pos",
    "ref_idx",
    "reference_anchor_idx",
    "reference_idx",
    "t",
    "x",
    "z",
}

# Counts after the compatibility-first migration in issue #113. Edit only
# downwards; delete a zero entry. The test below reports exact replacements.
BASELINE: dict[str, int] = {
    "parameter:F": 5,
    "parameter:H": 5,
    "parameter:P": 2,
    "parameter:Q": 4,
    "parameter:R": 7,
    "parameter:S": 5,
    "parameter:dt": 40,
    "parameter:est": 1,
    "parameter:noise_std": 3,
    "parameter:ref_idx": 4,
    "parameter:reference_anchor_idx": 1,
    "parameter:reference_idx": 2,
    "parameter:t": 2,
    "parameter:x": 28,
    "parameter:z": 27,
    "return:raw_dict": 25,
    "return:raw_tuple": 47,
}


def _public_functions(tree: ast.Module):
    """Yield module functions and direct class methods that form public APIs."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                yield node
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    not child.name.startswith("_") or child.name == "__init__"
                ):
                    yield child


def _semantic_debt_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in CORE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in _public_functions(tree):
            arguments = (
                function.args.posonlyargs
                + function.args.args
                + function.args.kwonlyargs
            )
            if function.args.vararg is not None:
                arguments.append(function.args.vararg)
            if function.args.kwarg is not None:
                arguments.append(function.args.kwarg)
            for argument in arguments:
                if argument.arg in AMBIGUOUS_PUBLIC_PARAMETERS:
                    counts[f"parameter:{argument.arg}"] += 1

            if function.returns is None:
                continue
            return_names = {
                node.id
                for node in ast.walk(function.returns)
                if isinstance(node, ast.Name)
            }
            is_property = any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in function.decorator_list
            )
            if return_names & {"tuple", "Tuple"}:
                counts["return:raw_tuple"] += 1
            # An explicitly named conversion is not a primary result contract.
            if (
                return_names & {"dict", "Dict"}
                and function.name != "as_dict"
                and not is_property
            ):
                counts["return:raw_dict"] += 1
    return counts


def test_no_semantic_api_violation_count_grows_or_appears():
    """No legacy ambiguity category may grow, and no new one may appear."""
    counts = _semantic_debt_counts()
    grown = {
        key: (count, BASELINE[key])
        for key, count in counts.items()
        if key in BASELINE and count > BASELINE[key]
    }
    appeared = {key: count for key, count in counts.items() if key not in BASELINE}

    assert not grown and not appeared, (
        "Semantic public-API debt grew:\n"
        + "".join(
            f"  {key}: {now}, was {then}\n"
            for key, (now, then) in sorted(grown.items())
        )
        + "".join(f"  {key}: {now}, new\n" for key, now in sorted(appeared.items()))
    )


def test_semantic_api_baseline_is_not_stale():
    """Record every reduction immediately so the ratchet keeps tightening."""
    counts = _semantic_debt_counts()
    stale = {
        key: (counts.get(key, 0), recorded)
        for key, recorded in BASELINE.items()
        if counts.get(key, 0) < recorded
    }
    assert not stale, "Lower these semantic API baselines:\n" + "".join(
        f"  {key}: {now}, recorded {then}\n"
        for key, (now, then) in sorted(stale.items())
    )
