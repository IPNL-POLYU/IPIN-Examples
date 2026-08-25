"""Validate dataset documentation completeness.

This tool checks that each dataset in data/sim/ has complete documentation
against .templates/dataset_README_template.md, which is where the required
section list comes from.

This docstring used to cite a chapter of the design document as the source
of that standard. No such chapter was ever written -- that document stops
partway -- so the citation had always pointed at nothing.

Usage:
    python tools/validate_dataset_docs.py                       # every dataset
    python tools/validate_dataset_docs.py ch4_rf_2d_square      # just one
    python tools/validate_dataset_docs.py --quiet                # summary only
    python tools/validate_dataset_docs.py --strict               # warnings fail too

Exits 0 when every failing dataset is listed in KNOWN_INCOMPLETE below, and
non-zero on a gap that is not -- or on a registered dataset that has become
valid, so the register cannot go stale. `tests/test_repo_validators.py` runs it.

This docstring used to advertise a `--fix` flag that auto-created missing
READMEs. There is no such flag and no such code; the four arguments above are
all argparse defines. It also named a `fusion_2d_imu_uwb` dataset, which has
been `ch8_fusion_2d_imu_uwb` since the chapter prefixes went in.

Author: Li-Ta Hsu
Date: December 2025
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


# Unicode-safe symbols (fallback for Windows)
CHECK = "OK"  # Was: ✓
CROSS = "X"  # Was: ✗
WARN = "!"  # Was: ⚠


# Datasets whose documentation is genuinely incomplete, with what is missing.
#
# THIS REGISTER MUST ONLY SHRINK. It exists so this tool can run in CI against
# today's real state: before it, the tool reported 72 errors across 12 datasets,
# and a check that is always red is a check nobody reads. Most of those 72 were
# the tool's own fault and are fixed -- it demanded .npz of text datasets,
# demanded one specific set of table column names, and could not see the three
# ch5 datasets at all. What is left is real, and each entry says what it needs
# rather than merely naming the dataset.
#
# It started at nine and is empty. Six were simply a missing README -- the same
# six data/sim/README.md gave a [README](...) link to and did not have. The last
# three were READMEs that existed and lacked required sections.
#
# The third of those, ch4_rf_2d_linear, was recorded here as a decision to make:
# it is written to a different shape from the template, leading with the
# reflection ambiguity rather than the template's order, and that reads better.
# The framing was wrong. This tool checks that a section is *present*, never
# where it sits, so the narrative sections were never in conflict with the
# template -- the four required ones were simply missing, and adding them left
# the narrative intact. There was no template-versus-practice question.
KNOWN_INCOMPLETE: dict = {}

# Not a documentation gap, so not in the register: the three ch5 datasets carry
# metadata.json where every other dataset uses config.json. The naming is
# inconsistent and the file does the same job. It now also records the seed,
# so all twenty datasets regenerate exactly -- see
# tests/ch5_fingerprinting/test_dataset_reproduces_from_its_seed.py.

# Required sections in dataset README, from .templates/dataset_README_template.md
REQUIRED_SECTIONS = [
    "## Overview",
    "## Scenario Description",
    "## Files and Data Structure",
    "## Loading Example",
    "## Configuration Parameters",
    "## Parameter Effects and Learning Experiments",
    "## Visualization Example",
    "## Connection to Book Equations",
    "## Recommended Experiments",
]

# Optional but recommended sections
RECOMMENDED_SECTIONS = [
    "## Dataset Variants",
    "## Troubleshooting / Common Student Questions",
    "## Generation",
]


def check_dataset_files(dataset_path: Path) -> Tuple[List[str], List[str]]:
    """Check for required dataset files.

    Args:
        dataset_path: Path to dataset directory.

    Returns:
        Tuple of (found_files, missing_files)
    """
    # A dataset needs its generation parameters and at least one data file.
    #
    # .txt counts. This used to demand .npz or .npy, which failed nine of the
    # shipped datasets -- every text-based one -- while data/sim/README.md
    # documents .txt as a supported format and answers "why do some datasets
    # use .txt and others .npz?" with "both formats are supported". The design
    # doc says NPZ *preferred*, not required. So the check was reporting a
    # house-style preference as a missing file, and that is most of why this
    # tool sat red and unread.
    data_files = [
        path
        for suffix in ("*.npz", "*.npy", "*.txt")
        for path in dataset_path.glob(suffix)
    ]

    found = []
    missing = []

    # config.json is the documented name; the three ch5 fingerprint datasets
    # use metadata.json instead. It carries the scenario parameters -- AP
    # positions, grid spacing, path-loss model -- so it serves the purpose, and
    # it is accepted here rather than reported as a missing file. The naming
    # inconsistency is noted above.
    config = next(
        (
            name
            for name in ("config.json", "metadata.json")
            if (dataset_path / name).exists()
        ),
        None,
    )
    if config:
        found.append(config)
    else:
        missing.append("config.json")

    if data_files:
        found.append(f"data files ({len(data_files)} found)")
    else:
        missing.append("data files (.npz, .npy or .txt)")

    return found, missing


def check_readme_sections(readme_path: Path) -> Dict[str, bool]:
    """Check which required sections are present in README.

    Args:
        readme_path: Path to README.md file.

    Returns:
        Dictionary mapping section names to presence (True/False).
    """
    if not readme_path.exists():
        return {section: False for section in REQUIRED_SECTIONS + RECOMMENDED_SECTIONS}

    content = readme_path.read_text(encoding="utf-8")

    results = {}
    for section in REQUIRED_SECTIONS + RECOMMENDED_SECTIONS:
        results[section] = section in content

    return results


def check_readme_code_blocks(readme_path: Path) -> Tuple[int, List[str]]:
    """Check for code examples in README.

    Args:
        readme_path: Path to README.md file.

    Returns:
        Tuple of (num_code_blocks, languages_found)
    """
    if not readme_path.exists():
        return 0, []

    content = readme_path.read_text(encoding="utf-8")

    # Count code blocks
    code_blocks = content.count("```")
    num_blocks = code_blocks // 2  # Each block has opening and closing

    # Detect languages (python, bash, json)
    languages = set()
    if "```python" in content:
        languages.add("python")
    if "```bash" in content:
        languages.add("bash")
    if "```json" in content:
        languages.add("json")

    return num_blocks, sorted(languages)


def check_parameter_table(readme_path: Path) -> bool:
    """Check if README contains a parameter effects table.

    Args:
        readme_path: Path to README.md file.

    Returns:
        True if parameter table found.
    """
    if not readme_path.exists():
        return False

    content = readme_path.read_text(encoding="utf-8")

    # A markdown table inside the parameter-effects section.
    #
    # This used to require the literal column headers "| Parameter |" and
    # "| Effect", which is a check on one table's column naming rather than on
    # whether the section says anything. It failed seven READMEs that carry
    # several parameter-effects tables each, because theirs are headed with the
    # quantity actually being varied -- "| Geometry | Mean GDOP | ... |",
    # "| TOA Noise (m) | Position Error (m) | ... |". Renaming those columns to
    # "Parameter" would make them worse, so the check moved instead.
    section = "## Parameter Effects and Learning Experiments"
    if section not in content:
        return False

    body = content.split(section, 1)[1]
    lines = []
    for line in body.splitlines():
        if line.startswith("## "):
            break
        lines.append(line)

    # A table needs the |---|---| separator row under its header.
    separator = set("|-: \t")
    return any(
        line.lstrip().startswith("|") and set(line) <= separator and "-" in line
        for line in lines
    )


def validate_dataset(dataset_path: Path, verbose: bool = True) -> Tuple[bool, Dict]:
    """Validate a single dataset directory.

    Args:
        dataset_path: Path to dataset directory.
        verbose: Print detailed output.

    Returns:
        Tuple of (is_valid, results_dict)
    """
    results = {
        "path": dataset_path,
        "dataset": dataset_path.name,
        "has_readme": False,
        "has_config": False,
        "has_data_files": False,
        "required_sections": {},
        "recommended_sections": {},
        "num_code_blocks": 0,
        "code_languages": [],
        "has_parameter_table": False,
        "warnings": [],
        "errors": [],
    }

    if verbose:
        print(f"\n{Colors.BOLD}Checking: {dataset_path.name}{Colors.END}")
        print(f"Path: {dataset_path}")

    # Check files
    found_files, missing_files = check_dataset_files(dataset_path)
    results["has_config"] = any(
        name in found_files for name in ("config.json", "metadata.json")
    )
    results["has_data_files"] = any("data files" in f for f in found_files)

    if missing_files:
        for mf in missing_files:
            results["errors"].append(f"Missing required file: {mf}")
            if verbose:
                print(f"  {Colors.RED}[{CROSS}]{Colors.END} Missing: {mf}")
    else:
        if verbose:
            print(f"  {Colors.GREEN}[{CHECK}]{Colors.END} All required files present")

    # Check README
    readme_path = dataset_path / "README.md"
    results["has_readme"] = readme_path.exists()

    if not readme_path.exists():
        results["errors"].append("Missing README.md")
        if verbose:
            print(f"  {Colors.RED}[{CROSS}]{Colors.END} Missing README.md")
        return False, results
    else:
        if verbose:
            print(f"  {Colors.GREEN}[{CHECK}]{Colors.END} README.md exists")

    # Check sections
    sections = check_readme_sections(readme_path)
    for section in REQUIRED_SECTIONS:
        results["required_sections"][section] = sections[section]
        if not sections[section]:
            results["errors"].append(f"Missing required section: {section}")
            if verbose:
                print(f"  {Colors.RED}[{CROSS}]{Colors.END} Missing section: {section}")

    for section in RECOMMENDED_SECTIONS:
        results["recommended_sections"][section] = sections[section]
        if not sections[section]:
            results["warnings"].append(f"Missing recommended section: {section}")

    if verbose and not results["errors"]:
        print(f"  {Colors.GREEN}[{CHECK}]{Colors.END} All required sections present")

    # Check code examples
    num_blocks, languages = check_readme_code_blocks(readme_path)
    results["num_code_blocks"] = num_blocks
    results["code_languages"] = languages

    if num_blocks < 3:
        results["warnings"].append(f"Only {num_blocks} code blocks (recommend ≥3)")
        if verbose:
            print(
                f"  {Colors.YELLOW}[{WARN}]{Colors.END} Only {num_blocks} code blocks (recommend ≥3)"
            )
    elif verbose:
        print(f"  {Colors.GREEN}[{CHECK}]{Colors.END} {num_blocks} code blocks found")

    if "python" not in languages:
        results["warnings"].append("No Python loading examples found")
        if verbose:
            print(f"  {Colors.YELLOW}[{WARN}]{Colors.END} No Python loading examples")

    # Check parameter table
    results["has_parameter_table"] = check_parameter_table(readme_path)
    if not results["has_parameter_table"]:
        results["errors"].append("Missing parameter effects table")
        if verbose:
            print(
                f"  {Colors.RED}[{CROSS}]{Colors.END} Missing parameter effects table"
            )
    elif verbose:
        print(f"  {Colors.GREEN}[{CHECK}]{Colors.END} Parameter effects table present")

    is_valid = len(results["errors"]) == 0

    if verbose:
        if is_valid:
            print(f"  {Colors.GREEN}{Colors.BOLD}Status: VALID [{CHECK}]{Colors.END}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}Status: INVALID [{CROSS}]{Colors.END}")

        if results["warnings"]:
            print(f"  {Colors.YELLOW}Warnings: {len(results['warnings'])}{Colors.END}")

    return is_valid, results


def find_datasets(data_sim_path: Path) -> List[Path]:
    """Find all dataset directories in data/sim/.

    Args:
        data_sim_path: Path to data/sim/ directory.

    Returns:
        List of dataset directory paths.
    """
    datasets = []

    # Anything that carries a README or a data file of any kind.
    #
    # This used to require config.json or *.npz, which silently skipped the
    # three ch5 fingerprint datasets -- they carry metadata.json and *.npy --
    # so the tool reported "17 datasets checked" out of 20 and a reader had no
    # way to know which three were missing. ch5_wifi_fingerprint_grid has a
    # full README that nothing had ever validated.
    markers = ("config.json", "metadata.json")
    suffixes = ("*.npz", "*.npy", "*.txt")

    for item in data_sim_path.iterdir():
        if not item.is_dir() or item.name.startswith("."):
            continue
        looks_like_dataset = (
            (item / "README.md").exists()
            or any((item / marker).exists() for marker in markers)
            or any(item.glob(suffix) for suffix in suffixes)
        )
        if looks_like_dataset:
            datasets.append(item)

    return sorted(datasets)


def print_summary(results_list: List[Tuple[bool, Dict]]):
    """Print summary of validation results.

    Args:
        results_list: List of (is_valid, results_dict) tuples.
    """
    valid_count = sum(1 for is_valid, _ in results_list if is_valid)
    total_count = len(results_list)

    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}VALIDATION SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")

    print(f"Total datasets checked: {total_count}")
    print(f"Valid datasets: {Colors.GREEN}{valid_count}{Colors.END}")
    print(f"Invalid datasets: {Colors.RED}{total_count - valid_count}{Colors.END}")

    if valid_count == total_count:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}All datasets have complete documentation! [{CHECK}]{Colors.END}"
        )
    else:
        print(
            f"\n{Colors.RED}{Colors.BOLD}Some datasets need documentation fixes.{Colors.END}"
        )
        print("\nDatasets needing attention:")
        for is_valid, results in results_list:
            if not is_valid:
                print(f"  - {results['path'].name}: {len(results['errors'])} errors")

    # Print statistics
    total_errors = sum(len(r["errors"]) for _, r in results_list)
    total_warnings = sum(len(r["warnings"]) for _, r in results_list)

    print(f"\nTotal errors: {Colors.RED}{total_errors}{Colors.END}")
    print(f"Total warnings: {Colors.YELLOW}{total_warnings}{Colors.END}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate dataset documentation completeness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all datasets
  python tools/validate_dataset_docs.py
  
  # Check specific dataset
  python tools/validate_dataset_docs.py fusion_2d_imu_uwb
  
  # Quiet mode (only summary)
  python tools/validate_dataset_docs.py --quiet
  
  # Strict mode (warnings treated as errors)
  python tools/validate_dataset_docs.py --strict
        """,
    )

    parser.add_argument(
        "dataset", nargs="?", help="Specific dataset to check (default: check all)"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only print summary (no per-dataset details)",
    )

    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sim",
        help="Path to data/sim directory (default: data/sim)",
    )

    args = parser.parse_args()

    # Find data/sim directory
    data_sim_path = Path(args.data_dir)
    if not data_sim_path.exists():
        print(f"{Colors.RED}Error: Directory not found: {data_sim_path}{Colors.END}")
        return 1

    # Find datasets to check
    if args.dataset:
        dataset_path = data_sim_path / args.dataset
        if not dataset_path.exists():
            print(f"{Colors.RED}Error: Dataset not found: {dataset_path}{Colors.END}")
            return 1
        datasets_to_check = [dataset_path]
    else:
        datasets_to_check = find_datasets(data_sim_path)
        if not datasets_to_check:
            print(f"{Colors.YELLOW}No datasets found in {data_sim_path}{Colors.END}")
            return 0

    print(f"{Colors.BOLD}Dataset Documentation Validator{Colors.END}")
    print(f"Checking {len(datasets_to_check)} dataset(s)...")

    # Validate each dataset
    results_list = []
    for dataset_path in datasets_to_check:
        is_valid, results = validate_dataset(dataset_path, verbose=not args.quiet)

        # In strict mode, treat warnings as errors
        if args.strict and results["warnings"]:
            is_valid = False
            results["errors"].extend(results["warnings"])
            results["warnings"] = []

        results_list.append((is_valid, results))

    # Print summary
    print_summary(results_list)

    # --strict is a "show me everything" mode, so the register does not apply.
    #
    # It promotes the recommended-section warnings to errors, which fails eight
    # datasets that are complete by the required-section standard the register
    # tracks. Running them through the register logic reported those eight as
    # "New documentation gaps" and advised adding them to KNOWN_INCOMPLETE,
    # which would have been wrong: they are not incomplete, they simply lack
    # optional sections.
    if args.strict:
        failing = sorted(
            results["dataset"] for is_valid, results in results_list if not is_valid
        )
        if failing:
            print(
                f"\n{Colors.YELLOW}{Colors.BOLD}Strict mode: recommended "
                f"sections missing{Colors.END}"
            )
            for name in failing:
                print(f"  - {name}")
            print(
                "\nStrict mode treats the RECOMMENDED_SECTIONS as required. "
                "These are not KNOWN_INCOMPLETE candidates -- that register is "
                "for the required set. Run without --strict for the answer CI "
                "uses."
            )
            return 1
        print(f"\n{Colors.GREEN}{Colors.BOLD}[PASSED]{Colors.END} strict")
        return 0

    # Exit code: a registered failure is debt, not a regression.
    #
    # An unregistered failure fails the build. A *registered* dataset that has
    # become valid also fails it, so the register cannot quietly grow stale --
    # the same reasoning as the ratchets in tests/test_repo_conventions.py.
    unregistered = sorted(
        results["dataset"]
        for is_valid, results in results_list
        if not is_valid and results["dataset"] not in KNOWN_INCOMPLETE
    )
    fixed = sorted(
        results["dataset"]
        for is_valid, results in results_list
        if is_valid and results["dataset"] in KNOWN_INCOMPLETE
    )

    if unregistered:
        print(f"\n{Colors.RED}{Colors.BOLD}New documentation gaps:{Colors.END}")
        for name in unregistered:
            print(f"  - {name}")
        print(
            "\nFix the dataset's README, or add it to KNOWN_INCOMPLETE with a "
            "line saying what it still needs."
        )
    if fixed:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}Now valid, so drop from "
            f"KNOWN_INCOMPLETE:{Colors.END}"
        )
        for name in fixed:
            print(f"  - {name}")

    if not unregistered and not fixed:
        registered = sum(
            1
            for is_valid, results in results_list
            if not is_valid and results["dataset"] in KNOWN_INCOMPLETE
        )
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}[PASSED]{Colors.END} "
            f"no new gaps ({registered} known, listed in KNOWN_INCOMPLETE)"
        )

    return 1 if (unregistered or fixed) else 0


if __name__ == "__main__":
    sys.exit(main())
