"""Shared pytest configuration for the whole test suite.

Author: Li-Ta Hsu
"""

import os
import tempfile

import pytest

from core.eval.plots import FIGS_DIR_ENV_VAR


@pytest.fixture(scope="session", autouse=True)
def divert_figure_output():
    """Keep a test run from writing figures into the working tree.

    Several tests run a chapter example end to end to prove its figures are
    still produced. Examples write to ``chX_*/figs`` next to their own source,
    so those runs rewrote committed figures and left the working tree dirty --
    `git status` showed a pile of modified binaries after every `pytest`, which
    buries real changes and invites committing them by accident.

    Pointing ``IPIN_FIGS_DIR`` at a temporary directory for the session fixes
    this everywhere at once. Tests build subprocess environments with
    ``os.environ.copy()``, so child processes inherit it too, and tests added
    later are covered without having to remember any of this.

    Yields:
        Path to the temporary figure root, as a string.
    """
    previous = os.environ.get(FIGS_DIR_ENV_VAR)

    with tempfile.TemporaryDirectory(prefix="ipin-figs-") as figs_root:
        os.environ[FIGS_DIR_ENV_VAR] = figs_root
        try:
            yield figs_root
        finally:
            if previous is None:
                os.environ.pop(FIGS_DIR_ENV_VAR, None)
            else:
                os.environ[FIGS_DIR_ENV_VAR] = previous
