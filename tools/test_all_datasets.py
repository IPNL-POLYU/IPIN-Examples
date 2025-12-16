"""
Comprehensive Dataset Testing Script.

Tests all code examples in all dataset READMEs across all chapters.
Validates that documentation is accurate and examples are runnable.

Usage:
    python tools/test_all_datasets.py              # Test all datasets
    python tools/test_all_datasets.py --chapter 8   # Test specific chapter
    python tools/test_all_datasets.py --verbose     # Detailed output

Author: Navigation Engineer
Date: December 2024
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# Dataset paths by chapter
DATASETS = {
    "ch8_fusion": [
        "data/sim/fusion_2d_imu_uwb",
        "data/sim/fusion_2d_imu_uwb_nlos",
        "data/sim/fusion_2d_imu_uwb_timeoffset",
    ],
    "ch6_dead_reckoning": [
        "data/sim/ch6_strapdown_basic",
        "data/sim/ch6_foot_zupt_walk",
        "data/sim/ch6_wheel_odom_square",
        "data/sim/ch6_pdr_corridor_walk",
        "data/sim/ch6_env_sensors_heading_altitude",
    ],
    "ch4_rf": [
        "data/sim/ch4_rf_2d_square",
    ],
    "ch5_fingerprint": [
        "data/sim/wifi_fingerprint_grid",
    ],
    "ch7_slam": [
        "data/sim/ch7_slam_2d_square",
    ],
    "ch3_estimators": [
        "data/sim/ch3_estimator_nonlinear",
    ],
    "ch2_coords": [
        "data/sim/ch2_coords_san_francisco",
    ],
}


def extract_code_blocks(readme_path: Path) -> List[Tuple[str, str, int]]:
    """
    Extract Python code blocks from README.

    Args:
        readme_path: Path to README file.

    Returns:
        List of (code, section_name, line_number) tuples.
    """
    if not readme_path.exists():
        return []

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all Python code blocks
    # Pattern: ```python\n<code>\n```
    pattern = r"```python\n(.*?)\n```"
    matches = re.finditer(pattern, content, re.DOTALL)

    code_blocks = []
    current_section = "Unknown"

    for match in matches:
        code = match.group(1)
        line_number = content[:match.start()].count("\n") + 1

        # Find section name (look backwards for ## heading)
        before = content[:match.start()]
        section_matches = re.findall(r"##\s+(.+)", before)
        if section_matches:
            current_section = section_matches[-1].strip()

        code_blocks.append((code, current_section, line_number))

    return code_blocks


def is_runnable_code(code: str) -> bool:
    """
    Determine if code block is meant to be runnable.

    Filters out:
    - Snippets with ellipsis (...)
    - Snippets with comments like "# ... more code ..."
    - Very short snippets (<3 lines, likely just imports)
    - Configuration examples (JSON, etc.)

    Args:
        code: Code block content.

    Returns:
        True if code should be runnable, False otherwise.
    """
    # Skip if contains ellipsis
    if "..." in code or "# ..." in code:
        return False

    # Skip if too short (just imports)
    lines = [l for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 3:
        return False

    # Skip if it's just a docstring or comment
    if code.strip().startswith('"""') or code.strip().startswith("'''"):
        return False

    # Skip if it looks like JSON/config
    if code.strip().startswith("{") or code.strip().startswith("["):
        return False

    # Skip if it's output (starts with numbers or arrows)
    if re.match(r"^\s*[\d\-]+", code):
        return False

    return True


def test_code_block(code: str, dataset_path: Path, verbose: bool = False) -> Dict:
    """
    Test a single code block.

    Args:
        code: Python code to test.
        dataset_path: Path to dataset (for imports).
        verbose: Print detailed output.

    Returns:
        Dict with test results.
    """
    # Create temporary script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        # Add necessary imports and setup
        test_code = f"""
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root
import os
os.chdir(project_root)

# Run the code block
{code}
"""
        f.write(test_code)
        temp_file = Path(f.name)

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(temp_file)],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
        )

        # Clean up
        temp_file.unlink()

        if result.returncode == 0:
            return {
                "status": "PASS",
                "output": result.stdout if verbose else "",
            }
        else:
            return {
                "status": "FAIL",
                "error": result.stderr,
                "returncode": result.returncode,
            }

    except subprocess.TimeoutExpired:
        temp_file.unlink()
        return {
            "status": "TIMEOUT",
            "error": "Code execution exceeded 30 second timeout",
        }
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        return {
            "status": "ERROR",
            "error": str(e),
        }


def test_dataset(dataset_path: Path, verbose: bool = False) -> Dict:
    """
    Test a single dataset.

    Args:
        dataset_path: Path to dataset directory.
        verbose: Print detailed output.

    Returns:
        Dict with test results.
    """
    results = {
        "dataset": str(dataset_path),
        "readme_exists": False,
        "files_exist": True,
        "config_valid": False,
        "code_blocks_total": 0,
        "code_blocks_runnable": 0,
        "code_blocks_passed": 0,
        "code_blocks_failed": 0,
        "issues": [],
    }

    # Check README exists
    readme_path = dataset_path / "README.md"
    if not readme_path.exists():
        results["issues"].append("README.md missing")
        return results

    results["readme_exists"] = True

    # Check config.json
    config_path = dataset_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                json.load(f)
            results["config_valid"] = True
        except json.JSONDecodeError as e:
            results["issues"].append(f"config.json invalid: {e}")

    # Extract and test code blocks
    code_blocks = extract_code_blocks(readme_path)
    results["code_blocks_total"] = len(code_blocks)

    if verbose:
        print(f"\n  Found {len(code_blocks)} code blocks")

    for i, (code, section, line_num) in enumerate(code_blocks, 1):
        if not is_runnable_code(code):
            if verbose:
                print(f"  [{i}/{len(code_blocks)}] Skipping snippet in '{section}' (not runnable)")
            continue

        results["code_blocks_runnable"] += 1

        if verbose:
            print(f"  [{i}/{len(code_blocks)}] Testing code in '{section}' (line {line_num})...")

        test_result = test_code_block(code, dataset_path, verbose)

        if test_result["status"] == "PASS":
            results["code_blocks_passed"] += 1
            if verbose:
                print(f"    PASS")
        else:
            results["code_blocks_failed"] += 1
            error_msg = f"Code block at line {line_num} in '{section}' failed: {test_result.get('error', 'Unknown error')}"
            results["issues"].append(error_msg)
            if verbose:
                print(f"    FAIL: {test_result.get('error', 'Unknown error')[:100]}")

    return results


def print_summary(all_results: Dict[str, List[Dict]]) -> None:
    """Print testing summary."""
    print("\n" + "=" * 70)
    print("TESTING SUMMARY")
    print("=" * 70)

    total_datasets = sum(len(results) for results in all_results.values())
    total_passed = 0
    total_failed = 0
    total_issues = 0

    for chapter, results_list in all_results.items():
        print(f"\n{chapter.upper().replace('_', ' ')}:")

        for result in results_list:
            status = "PASS" if not result["issues"] else "FAIL"
            symbol = "[PASS]" if status == "PASS" else "[FAIL]"

            dataset_name = Path(result["dataset"]).name
            print(f"  {symbol} {dataset_name}")

            if result["issues"]:
                total_failed += 1
                for issue in result["issues"][:3]:  # Show first 3 issues
                    print(f"      - {issue[:80]}")
                if len(result["issues"]) > 3:
                    print(f"      ... and {len(result['issues']) - 3} more issues")
                total_issues += len(result["issues"])
            else:
                total_passed += 1

            # Show code block stats
            if result["code_blocks_runnable"] > 0:
                passed = result["code_blocks_passed"]
                total = result["code_blocks_runnable"]
                print(f"      Code examples: {passed}/{total} passed")

    print("\n" + "=" * 70)
    print(f"OVERALL: {total_passed}/{total_datasets} datasets passed")
    print(f"Total issues found: {total_issues}")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test all dataset documentation and code examples"
    )
    parser.add_argument(
        "--chapter",
        type=str,
        choices=["2", "3", "4", "5", "6", "7", "8"],
        help="Test specific chapter only",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    # Filter datasets by chapter if specified
    if args.chapter:
        chapter_map = {
            "2": ["ch2_coords"],
            "3": ["ch3_estimators"],
            "4": ["ch4_rf"],
            "5": ["ch5_fingerprint"],
            "6": ["ch6_dead_reckoning"],
            "7": ["ch7_slam"],
            "8": ["ch8_fusion"],
        }
        datasets_to_test = {k: v for k, v in DATASETS.items() if k in chapter_map[args.chapter]}
    else:
        datasets_to_test = DATASETS

    # Run tests
    all_results = {}

    print("=" * 70)
    print("DATASET TESTING SUITE")
    print("=" * 70)
    print(f"Testing {sum(len(v) for v in datasets_to_test.values())} datasets...")

    for chapter, dataset_paths in datasets_to_test.items():
        print(f"\n{chapter.upper().replace('_', ' ')}:")
        results_list = []

        for dataset_path_str in dataset_paths:
            dataset_path = Path(dataset_path_str)
            print(f"\n  Testing {dataset_path.name}...")

            result = test_dataset(dataset_path, args.verbose)
            results_list.append(result)

        all_results[chapter] = results_list

    # Print summary
    print_summary(all_results)

    # Exit with error code if any tests failed
    total_failed = sum(
        1 for results in all_results.values()
        for result in results
        if result["issues"]
    )
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()

