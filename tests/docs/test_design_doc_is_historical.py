"""references/design_doc.md is a record of intent, not an API reference.

It was written before the implementation and never updated against it, and the
result is the worst state a reference can be in: **mostly right.** Measured, of
the 117 core APIs it specifies, 86 exist and 31 do not -- so a reader who trusts
it is correct three times out of four, which is exactly often enough to stop
checking. Of the 31, sixteen were renamed and fifteen were never built; both
facts are useful and the register below keeps them apart.

It also cannot resolve its own cross-references. The document makes 42 references
to its Sections 5 through 8 and contains no heading numbered above 4; those
sections were planned and never written. Seven files elsewhere in the repository
cited them, including `tools/validate_dataset_docs.py`, whose docstring said it
validated "the standards defined in Section 5.3 of the design document" -- a
standard that lives in `.templates/dataset_README_template.md` and never lived in
the design doc at all.

So the document is marked as historical rather than updated. Updating 4,980 lines
of specification to mirror the code would produce a second source of truth that
drifts again; the repository already has better ones, all of them checked:

    docs/equation_index.yml          equation-to-code, validated in CI
    chX_*/README.md                  per-chapter API tables and transcripts
    data/sim/*/README.md             per-dataset structure and loading
    .cursor/rules/                   the conventions, authoritative
    .templates/                      the dataset README standard

This file keeps the marking honest. It recomputes the drift rather than trusting
the header's numbers, pins the superseded names so the translation table cannot
rot, and fails if a new reference to a non-existent section appears.

**If a name here becomes real, delete its entry.** If you are tempted to add one,
you are adding aspiration to a historical document -- put it in the code instead.

Author: Li-Ta Hsu
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DESIGN_DOC = REPO_ROOT / "references/design_doc.md"

#: The header the document must carry, so its status cannot be quietly dropped.
HISTORICAL_MARKER = "> **Status: historical.**"

#: Core APIs the design doc specifies that do not exist, and what to use.
#:
#: A value is the real replacement, or None where the thing was never built.
#: Both cases are useful to a reader; conflating them is not.
SUPERSEDED = {
    # --- core/coords: the attitude conversions were all renamed -------------
    "rpy_to_rotmat": "core.coords.rotations.euler_to_rotation_matrix",
    "rotmat_to_rpy": "core.coords.rotations.rotation_matrix_to_euler",
    "rotmat_to_quat": "core.coords.rotations.rotation_matrix_to_quat",
    # --- core/estimators: the solvers are functions, not step helpers -------
    "gauss_newton_solve": "core.estimators.nonlinear_least_squares.gauss_newton",
    "gauss_newton_step": "core.estimators.nonlinear_least_squares.gauss_newton",
    "levenberg_marquardt_step": "core.estimators.nonlinear_least_squares.levenberg_marquardt",
    "solve_fgo": "core.estimators.factor_graph.FactorGraph.optimize",
    "gradient_descent_step": None,
    # Factor's interface was named differently in the end.
    "jacobian": "core.estimators.factor_graph.Factor.linearize",
    "evaluate": "core.estimators.factor_graph.FactorGraph.compute_error",
    # --- core/rf ------------------------------------------------------------
    "tdoa_range_diff": "core.rf.measurement_models.tdoa_range_difference",
    "aoa_bearing": "core.rf.measurement_models.aoa_azimuth",
    "simulate_rf_measurements": None,
    # --- core/sensors -------------------------------------------------------
    "ZaruMeasurementModel": "core.sensors.constraints.ZaruMeasurementModelPlaceholder",
    "InsWheelProcessModel": None,
    "WheelSpeedMeasurementModel": None,
    # --- core/sim: the trajectory generators live in scripts/ instead -------
    "generate_2d_trajectory": None,
    "generate_3d_trajectory": None,
    "add_imu_noise": None,
    # --- core/eval: five planned plots were never built ---------------------
    "plot_trajectory_3d": None,
    "plot_nees": None,
    "plot_nis": None,
    "plot_occupancy_grid": None,
    "plot_factor_graph_skeleton": None,
    # --- core/fingerprinting ------------------------------------------------
    "log_likelihoods": "core.fingerprinting.probabilistic.log_likelihood",
    "posterior": "core.fingerprinting.probabilistic.log_posterior",
    "gaussian_likelihood": None,
    "raytrace_likelihood": None,
    # --- outside core/ ------------------------------------------------------
    "run_multisensor_ekf": None,
    "generate_fusion_2d_imu_uwb_dataset": "scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py",
}

#: Names the extraction picks up that are not API claims.
#:
#: `main` is the entry point of an illustrative generation script inside a code
#: block, and `print` is a builtin the signature regex picks up from an example
#: line -- neither is a core API the document is specifying.
NOT_API_CLAIMS = {"main", "print"}

#: Modules the design doc names that do not exist, and where the code went.
SUPERSEDED_MODULES = {
    "core/estimators/kalman.py": "core/estimators/kalman_filter.py",
    "core/estimators/particle.py": "core/estimators/particle_filter.py",
    "core/estimators/fgo.py": "core/estimators/factor_graph.py",
    "core/estimators/optim.py": "core/estimators/nonlinear_least_squares.py",
    "core/sensors/ins_ekf_models.py": "core/sensors/ins_ekf.py",
    "core/fingerprinting/model_based.py": None,
    "core/fusion/calibration.py": None,
    "core/fusion/run.py": None,
    "core/fusion/time_alignment.py": None,
    "core/slam/loam.py": None,
}


def _core_names():
    """Every top-level function and class defined under core/."""
    names = set()
    for path in sorted(REPO_ROOT.glob("core/**/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
    return names


def _doc_api_names(text):
    """Names the design doc presents as API.

    Three styles, because the document uses three:

      - full signatures in fenced blocks, ``def name(...) -> T``
      - class declarations, ``class Name:``
      - bullet-list entries, ``name(args) -> T`` or ``name(args) - description``,
        which is how Sections 4.1, 4.3, 4.5 and 4.6 list most of their API

    The bullet form is worth catching: ``gaussian_likelihood(z, z_pred, sigma) -
    measurement likelihood reused by demos`` is as much a specification as a
    fenced signature, and it is equally absent from the code.
    """
    found = set(re.findall(r"^\s*def\s+(\w+)\s*\(", text, re.M))
    found |= set(re.findall(r"^\s*class\s+(\w+)", text, re.M))
    found |= set(re.findall(r"^\s*(\w+)\([^)]*\)\s*(?:->|[-–—]|$)", text, re.M))
    return {n for n in found if not n.startswith("_")} - NOT_API_CLAIMS


def test_the_document_says_it_is_historical():
    """The status marker is present, so the framing cannot be lost silently."""
    text = DESIGN_DOC.read_text(encoding="utf-8")

    assert HISTORICAL_MARKER in text, (
        f"{DESIGN_DOC.name} must carry {HISTORICAL_MARKER!r} near the top. It "
        f"specifies APIs that do not exist and cross-references sections it "
        f"does not contain; without the marker a reader has no way to know."
    )
    # Near the top, not buried.
    assert text.index(HISTORICAL_MARKER) < 400, (
        "The status marker must be in the opening lines, before the reader has "
        "started trusting the content."
    )


def test_superseded_api_register_is_exact():
    """The names that do not exist are exactly the ones registered."""
    text = DESIGN_DOC.read_text(encoding="utf-8")
    core = _core_names()
    absent = {n for n in _doc_api_names(text) if n not in core}

    unregistered = sorted(absent - set(SUPERSEDED))
    stale = sorted(set(SUPERSEDED) - absent)

    assert not unregistered, (
        f"design_doc.md specifies core APIs that do not exist and are not in "
        f"SUPERSEDED: {unregistered}\n\n"
        f"If a core symbol was just renamed, add the mapping here -- do not "
        f"edit the design document, which is a historical record."
    )
    assert not stale, (
        f"These SUPERSEDED entries now exist in core/, so the register is out "
        f"of date: {stale}\n\nDelete them."
    )


@pytest.mark.parametrize(
    "old,new",
    sorted((k, v) for k, v in SUPERSEDED.items() if v and not v.endswith(".py")),
)
def test_every_replacement_exists(old, new):
    """A translation table that points at nothing is worse than none."""
    leaf = new.split(".")[-1]
    parent = new.split(".")[-2]
    core = _core_names()

    assert leaf in core or parent in core, (
        f"SUPERSEDED maps {old} -> {new}, but neither {leaf!r} nor its parent "
        f"{parent!r} is defined under core/."
    )


def test_superseded_module_register_is_exact():
    """The module paths the document names either exist or are registered."""
    text = DESIGN_DOC.read_text(encoding="utf-8")
    named = set(re.findall(r"\b(core/[\w/]+\.py)\b", text))
    absent = {m for m in named if not (REPO_ROOT / m).exists()}

    assert absent == set(SUPERSEDED_MODULES), (
        f"module register out of date.\n"
        f"  unregistered: {sorted(absent - set(SUPERSEDED_MODULES))}\n"
        f"  now present:  {sorted(set(SUPERSEDED_MODULES) - absent)}"
    )

    for old, new in SUPERSEDED_MODULES.items():
        if new is not None:
            assert (
                REPO_ROOT / new
            ).exists(), f"SUPERSEDED_MODULES maps {old} -> {new}, which does not exist."


def test_nothing_cites_a_design_doc_section_that_does_not_exist():
    """The document has no heading above 4; nothing may point at one.

    Nine files did, `tools/validate_dataset_docs.py` among them. Its docstring
    claimed to validate "the standards defined in Section 5.3 of the design
    document" -- a standard that has always lived in
    .templates/dataset_README_template.md.
    """
    text = DESIGN_DOC.read_text(encoding="utf-8")
    headings = set(re.findall(r"^#+\s*(\d+)(?:\.\d+)*\.?\s", text, re.M))
    available = {int(h) for h in headings}

    offenders = []
    for path in sorted(REPO_ROOT.glob("**/*")):
        if path.suffix not in (".py", ".md", ".mdc") or not path.is_file():
            continue
        parts = set(path.parts)
        if parts & {".git", "__pycache__"} or "worktrees" in path.parts:
            continue
        # This file quotes the offending citations in order to explain them.
        if path in (DESIGN_DOC, Path(__file__).resolve()):
            continue
        body = path.read_text(encoding="utf-8", errors="replace")

        # Only citations that are *about* the design document. Several files
        # cite book sections by the same "Section 7.3" spelling -- docs/ch7_slam.md
        # is almost entirely book section numbers -- so the design-doc reference
        # and the section number have to appear together on one line.
        for line in body.splitlines():
            lowered = line.lower()
            if "design doc" not in lowered and "design_doc" not in line:
                continue
            for cited in re.findall(r"Section (\d+)(?:\.\d+)*", line):
                if int(cited) not in available:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}: Section {cited}"
                    )

    assert not offenders, (
        "These cite a design-doc section that does not exist:\n  "
        + "\n  ".join(sorted(set(offenders)))
        + f"\n\ndesign_doc.md has top-level sections {sorted(available)}. The "
        f"dataset README standard lives in .templates/dataset_README_template.md; "
        f"equation traceability lives in docs/equation_index.yml.\n\n"
        f"If the citation is a *historical* note recording what was believed at "
        f"the time -- .dev/ is full of those -- say so in the sentence rather "
        f"than exempting the file. An exemption here is a hole in the only thing "
        f"stopping these pointers coming back."
    )
