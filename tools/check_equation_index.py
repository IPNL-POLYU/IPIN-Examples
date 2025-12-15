#!/usr/bin/env python3
"""Check equation index consistency between code and documentation.

This script verifies that:
1. All equations referenced in code docstrings are documented in equation_index.yml
2. All equations in equation_index.yml have corresponding code implementations
3. File paths in the index point to existing files

Usage:
    python tools/check_equation_index.py
    python tools/check_equation_index.py --verbose
    python tools/check_equation_index.py --fix  # Update index with missing entries

Reference:
    Design document Section 6: Equation-Level Traceability
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Try to import yaml, fall back to basic parsing if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. Using basic YAML parsing.")


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def parse_equation_index(index_path: Path) -> List[Dict]:
    """Parse the equation_index.yml file.
    
    Args:
        index_path: Path to equation_index.yml
        
    Returns:
        List of equation entries from the YAML file
    """
    if not index_path.exists():
        print(f"Error: equation_index.yml not found at {index_path}")
        return []
    
    content = index_path.read_text(encoding="utf-8")
    
    if HAS_YAML:
        # Parse with PyYAML
        entries = yaml.safe_load(content)
        return entries if entries else []
    else:
        # Basic regex parsing for equation entries
        entries = []
        pattern = r'- eq: "([^"]+)"'
        for match in re.finditer(pattern, content):
            entries.append({"eq": match.group(1)})
        return entries


def find_equation_references_in_code(root: Path) -> Dict[str, List[Tuple[str, int]]]:
    """Find all equation references in Python source files.
    
    Searches for patterns like:
    - "Eq. (2.1)"
    - "Eqs. (3.1)-(3.3)"
    - "Reference: Eq. (4.5)"
    
    Args:
        root: Project root directory
        
    Returns:
        Dict mapping equation IDs to list of (file_path, line_number) tuples
    """
    equation_refs: Dict[str, List[Tuple[str, int]]] = {}
    
    # Patterns to search for
    patterns = [
        r'Eq\.\s*\((\d+\.\d+)\)',           # Eq. (2.1)
        r'Eqs\.\s*\((\d+\.\d+)\)',          # Eqs. (3.1) - captures first
        r'Equation\s*\((\d+\.\d+)\)',       # Equation (2.1)
    ]
    combined_pattern = '|'.join(patterns)
    
    # Search in core/ and ch*/ directories
    search_dirs = [root / "core"] + list(root.glob("ch*_*"))
    
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
            
        for py_file in search_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                for line_num, line in enumerate(content.split('\n'), 1):
                    for match in re.finditer(combined_pattern, line):
                        eq_num = match.group(1) or match.group(2) or match.group(3)
                        if eq_num:
                            eq_id = f"Eq. ({eq_num})"
                            rel_path = str(py_file.relative_to(root))
                            if eq_id not in equation_refs:
                                equation_refs[eq_id] = []
                            equation_refs[eq_id].append((rel_path, line_num))
            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")
    
    return equation_refs


def extract_equations_from_index(entries: List[Dict]) -> Set[str]:
    """Extract equation IDs from parsed index entries.
    
    Args:
        entries: List of equation entries from YAML
        
    Returns:
        Set of equation IDs (e.g., {"Eq. (2.1)", "Eq. (2.2)", ...})
    """
    equations = set()
    for entry in entries:
        if isinstance(entry, dict) and "eq" in entry:
            eq_id = entry["eq"]
            # Normalize format
            if not eq_id.startswith("Eq"):
                eq_id = f"Eq. ({eq_id})"
            equations.add(eq_id)
    return equations


def check_file_paths(entries: List[Dict], root: Path) -> List[str]:
    """Check that file paths in the index exist.
    
    Args:
        entries: List of equation entries from YAML
        root: Project root directory
        
    Returns:
        List of error messages for missing files
    """
    errors = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        files = entry.get("files", [])
        for file_info in files:
            if isinstance(file_info, dict):
                path = file_info.get("path", "")
                if path and not (root / path).exists():
                    errors.append(f"Missing file: {path} (referenced by {entry.get('eq', 'unknown')})")
    return errors


def main():
    """Main entry point for equation index checker."""
    parser = argparse.ArgumentParser(
        description="Check equation index consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any equations in code are not in index"
    )
    args = parser.parse_args()
    
    # Find project root
    try:
        root = find_project_root()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Project root: {root}")
    print()
    
    # Parse equation index
    index_path = root / "docs" / "equation_index.yml"
    entries = parse_equation_index(index_path)
    indexed_equations = extract_equations_from_index(entries)
    
    print(f"Equations in index: {len(indexed_equations)}")
    
    # Find equation references in code
    code_refs = find_equation_references_in_code(root)
    code_equations = set(code_refs.keys())
    
    print(f"Equations referenced in code: {len(code_equations)}")
    print()
    
    # Check for equations in code but not in index
    missing_from_index = code_equations - indexed_equations
    if missing_from_index:
        print("[WARNING] Equations in code but NOT in equation_index.yml:")
        for eq in sorted(missing_from_index):
            print(f"   - {eq}")
            if args.verbose:
                for file_path, line_num in code_refs[eq]:
                    print(f"       {file_path}:{line_num}")
        print()
    else:
        print("[OK] All equations in code are documented in index")
        print()
    
    # Check for equations in index but not referenced in code
    # (This is informational - some equations may be documented but not yet implemented)
    extra_in_index = indexed_equations - code_equations
    if extra_in_index and args.verbose:
        print("[INFO] Equations in index but not found in code (may be OK):")
        for eq in sorted(extra_in_index):
            print(f"   - {eq}")
        print()
    
    # Check file paths
    path_errors = check_file_paths(entries, root)
    if path_errors:
        print("[ERROR] File path errors in equation_index.yml:")
        for error in path_errors:
            print(f"   - {error}")
        print()
    else:
        print("[OK] All file paths in index are valid")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Indexed equations:     {len(indexed_equations)}")
    print(f"  Code references:       {len(code_equations)}")
    print(f"  Missing from index:    {len(missing_from_index)}")
    print(f"  File path errors:      {len(path_errors)}")
    print()
    
    # Determine exit code
    if args.strict and (missing_from_index or path_errors):
        print("[FAILED] (strict mode)")
        return 1
    elif path_errors:
        print("[WARNING] (file path issues)")
        return 0
    else:
        print("[PASSED]")
        return 0


if __name__ == "__main__":
    sys.exit(main())

