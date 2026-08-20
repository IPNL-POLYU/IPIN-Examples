"""Repository-relative path resolution for the shipped datasets.

Author: Li-Ta Hsu

Every example cites its dataset the way the book does, as ``data/sim/<name>``.
That spelling only resolves when the working directory happens to be the
repository root. Running an example from inside its own chapter folder -- the
first thing a reader does after opening that folder -- used to fail with

    FileNotFoundError: Required file not found: data/sim/ch5_.../locations.npy

which names the dataset and says nothing about the working directory, so it
reads as "the data is missing" rather than "you are standing somewhere else".

:func:`resolve_data_path` keeps the cwd-relative spelling working (so a reader
can still point an example at their own copy) and falls back to the same path
under the repository root.
"""

from pathlib import Path
from typing import Union

# core/utils/paths.py -> core/utils -> core -> repository root
_REPO_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["resolve_data_path", "repo_root"]


def repo_root() -> Path:
    """Return the repository root, inferred from this file's location."""
    return _REPO_ROOT


def resolve_data_path(
    path: Union[str, Path],
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a dataset path against the cwd, then the repository root.

    Args:
        path: Dataset path as written in the examples, e.g.
            ``data/sim/ch5_wifi_fingerprint_grid``. An absolute path is
            returned unchanged.
        must_exist: Raise :class:`FileNotFoundError` naming both candidates
            when neither resolves. Defaults to ``False``, which returns the
            cwd-relative spelling so the caller's own error message keeps
            reporting the path the reader actually asked for.

    Returns:
        The first candidate that exists: ``path`` relative to the working
        directory, else ``path`` relative to the repository root. Falls back to
        the cwd-relative candidate when neither exists.

    Raises:
        FileNotFoundError: If ``must_exist`` and neither candidate exists.

    Examples:
        >>> from core.utils import resolve_data_path
        >>> p = resolve_data_path("data/sim/ch5_wifi_fingerprint_grid")
        >>> p.exists()
        True
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    if candidate.exists():
        return candidate

    from_root = _REPO_ROOT / candidate
    if from_root.exists():
        return from_root

    if must_exist:
        raise FileNotFoundError(
            f"Dataset not found as '{candidate}' (relative to the working "
            f"directory {Path.cwd()}) nor as '{from_root}' (relative to the "
            f"repository root)."
        )
    return candidate
