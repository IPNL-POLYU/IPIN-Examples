"""Validate dataset documentation completeness.

This tool checks that each dataset in data/sim/ has complete documentation
following the standards defined in Section 5.3 of the design document.

Usage:
    python tools/validate_dataset_docs.py                    # Check all datasets
    python tools/validate_dataset_docs.py fusion_2d_imu_uwb  # Check specific dataset
    python tools/validate_dataset_docs.py --fix              # Auto-create missing READMEs

Author: Software Engineer
Date: December 2025
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


# Required sections in dataset README (from Section 5.3.2)
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
    required_files = ["config.json"]
    
    # Check for at least one .npz or .npy file
    data_files = list(dataset_path.glob("*.npz")) + list(dataset_path.glob("*.npy"))
    
    found = []
    missing = []
    
    if (dataset_path / "config.json").exists():
        found.append("config.json")
    else:
        missing.append("config.json")
    
    if data_files:
        found.append(f"data files ({len(data_files)} found)")
    else:
        missing.append("data files (.npz or .npy)")
    
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
    
    content = readme_path.read_text(encoding='utf-8')
    
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
    
    content = readme_path.read_text(encoding='utf-8')
    
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
    
    content = readme_path.read_text(encoding='utf-8')
    
    # Look for table with "Parameter" and "Effect" columns
    has_table = "| Parameter |" in content and "| Effect" in content
    
    return has_table


def validate_dataset(dataset_path: Path, verbose: bool = True) -> Tuple[bool, Dict]:
    """Validate a single dataset directory.
    
    Args:
        dataset_path: Path to dataset directory.
        verbose: Print detailed output.
    
    Returns:
        Tuple of (is_valid, results_dict)
    """
    results = {
        'path': dataset_path,
        'has_readme': False,
        'has_config': False,
        'has_data_files': False,
        'required_sections': {},
        'recommended_sections': {},
        'num_code_blocks': 0,
        'code_languages': [],
        'has_parameter_table': False,
        'warnings': [],
        'errors': [],
    }
    
    if verbose:
        print(f"\n{Colors.BOLD}Checking: {dataset_path.name}{Colors.END}")
        print(f"Path: {dataset_path}")
    
    # Check files
    found_files, missing_files = check_dataset_files(dataset_path)
    results['has_config'] = "config.json" in found_files
    results['has_data_files'] = any("data files" in f for f in found_files)
    
    if missing_files:
        for mf in missing_files:
            results['errors'].append(f"Missing required file: {mf}")
            if verbose:
                print(f"  {Colors.RED}✗{Colors.END} Missing: {mf}")
    else:
        if verbose:
            print(f"  {Colors.GREEN}✓{Colors.END} All required files present")
    
    # Check README
    readme_path = dataset_path / "README.md"
    results['has_readme'] = readme_path.exists()
    
    if not readme_path.exists():
        results['errors'].append("Missing README.md")
        if verbose:
            print(f"  {Colors.RED}✗{Colors.END} Missing README.md")
        return False, results
    else:
        if verbose:
            print(f"  {Colors.GREEN}✓{Colors.END} README.md exists")
    
    # Check sections
    sections = check_readme_sections(readme_path)
    for section in REQUIRED_SECTIONS:
        results['required_sections'][section] = sections[section]
        if not sections[section]:
            results['errors'].append(f"Missing required section: {section}")
            if verbose:
                print(f"  {Colors.RED}✗{Colors.END} Missing section: {section}")
    
    for section in RECOMMENDED_SECTIONS:
        results['recommended_sections'][section] = sections[section]
        if not sections[section]:
            results['warnings'].append(f"Missing recommended section: {section}")
    
    if verbose and not results['errors']:
        print(f"  {Colors.GREEN}✓{Colors.END} All required sections present")
    
    # Check code examples
    num_blocks, languages = check_readme_code_blocks(readme_path)
    results['num_code_blocks'] = num_blocks
    results['code_languages'] = languages
    
    if num_blocks < 3:
        results['warnings'].append(f"Only {num_blocks} code blocks (recommend ≥3)")
        if verbose:
            print(f"  {Colors.YELLOW}⚠{Colors.END} Only {num_blocks} code blocks (recommend ≥3)")
    elif verbose:
        print(f"  {Colors.GREEN}✓{Colors.END} {num_blocks} code blocks found")
    
    if 'python' not in languages:
        results['warnings'].append("No Python loading examples found")
        if verbose:
            print(f"  {Colors.YELLOW}⚠{Colors.END} No Python loading examples")
    
    # Check parameter table
    results['has_parameter_table'] = check_parameter_table(readme_path)
    if not results['has_parameter_table']:
        results['errors'].append("Missing parameter effects table")
        if verbose:
            print(f"  {Colors.RED}✗{Colors.END} Missing parameter effects table")
    elif verbose:
        print(f"  {Colors.GREEN}✓{Colors.END} Parameter effects table present")
    
    is_valid = len(results['errors']) == 0
    
    if verbose:
        if is_valid:
            print(f"  {Colors.GREEN}{Colors.BOLD}Status: VALID ✓{Colors.END}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}Status: INVALID ✗{Colors.END}")
        
        if results['warnings']:
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
    
    for item in data_sim_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it looks like a dataset (has config.json or data files)
            if (item / "config.json").exists() or list(item.glob("*.npz")):
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
        print(f"\n{Colors.GREEN}{Colors.BOLD}All datasets have complete documentation! ✓{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some datasets need documentation fixes.{Colors.END}")
        print(f"\nDatasets needing attention:")
        for is_valid, results in results_list:
            if not is_valid:
                print(f"  - {results['path'].name}: {len(results['errors'])} errors")
    
    # Print statistics
    total_errors = sum(len(r['errors']) for _, r in results_list)
    total_warnings = sum(len(r['warnings']) for _, r in results_list)
    
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
        """
    )
    
    parser.add_argument(
        'dataset',
        nargs='?',
        help='Specific dataset to check (default: check all)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only print summary (no per-dataset details)'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/sim',
        help='Path to data/sim directory (default: data/sim)'
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
        if args.strict and results['warnings']:
            is_valid = False
            results['errors'].extend(results['warnings'])
            results['warnings'] = []
        
        results_list.append((is_valid, results))
    
    # Print summary
    print_summary(results_list)
    
    # Exit code: 0 if all valid, 1 if any invalid
    all_valid = all(is_valid for is_valid, _ in results_list)
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())

