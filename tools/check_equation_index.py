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
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def check_objects(entries: List[Dict], root: Path) -> List[str]:
    """Check that each ``object`` named in the index is defined in its file.

    A path that exists is not enough: an index entry pointing at a function or
    class that was since renamed or removed still looks green while documenting
    something that is not there. Chapter 8 carried three such dangling names
    (``compute_innovation``, ``create_process_model``,
    ``create_position_measurement_model``) until this check was added.

    Args:
        entries: List of equation entries from YAML
        root: Project root directory

    Returns:
        List of error messages for objects that cannot be resolved.
    """
    errors = []
    cache: Dict[Path, Optional[ast.Module]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for file_info in entry.get("files", []):
            if not isinstance(file_info, dict):
                continue
            path = file_info.get("path", "")
            obj = file_info.get("object", "")
            if not path or not obj:
                continue
            file_path = root / path
            if not file_path.exists():
                continue  # already reported by check_file_paths

            if file_path not in cache:
                try:
                    cache[file_path] = ast.parse(
                        file_path.read_text(encoding="utf-8")
                    )
                except (OSError, SyntaxError):
                    cache[file_path] = None
            tree = cache[file_path]
            if tree is None:
                continue  # unparsable: not this check's business

            if not _resolve_object(tree, obj):
                errors.append(
                    f"Missing object: {obj} in {path} "
                    f"(referenced by {entry.get('eq', 'unknown')})"
                )
    return errors


def _resolve_object(tree: ast.Module, dotted_name: str) -> bool:
    """Resolve a possibly dotted ``Class.method`` name against a parsed module.

    Walks the AST one segment at a time, so ``Foo.bar`` requires ``bar`` to be
    defined inside ``Foo`` -- a module-level ``bar`` does not satisfy it.
    Module-level assignments (constants, aliases) count as definitions.
    """
    scope: List[ast.stmt] = list(tree.body)

    for segment in dotted_name.split("."):
        for node in scope:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and node.name == segment:
                scope = list(getattr(node, "body", []))
                break
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == segment for t in node.targets
            ):
                scope = []
                break
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == segment
            ):
                scope = []
                break
        else:
            return False

    return True


def resolve_test_node(node: str, root: Path) -> bool:
    """Check that a pytest-style test node id actually exists in the source.

    Accepts ``path/to/test.py``, ``...::ClassName`` or
    ``...::ClassName::test_method``. Verifies the file exists and that each
    class/function name after ``::`` is defined in that file.

    Args:
        node: Test node id from a ``verified_by`` field.
        root: Project root directory.

    Returns:
        True if the file exists and all referenced names are defined.
    """
    parts = node.split("::")
    file_path = root / parts[0]
    if not file_path.exists():
        return False
    if len(parts) == 1:
        return True
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    for name in parts[1:]:
        if not re.search(rf"(?:class|def)\s+{re.escape(name)}\b", content):
            return False
    return True


def check_verification(
    entries: List[Dict], root: Path
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Check that each *implemented* equation is backed by a conformance test.

    Entries with a non-empty ``files`` list are considered implemented and must
    carry a ``verified_by`` node that resolves to a real test. Definition-only
    entries (empty ``files``) are exempt.

    Args:
        entries: Parsed index entries.
        root: Project root directory.

    Returns:
        (verified_eq_ids, unverified) where ``unverified`` is a list of
        (eq_id, verified_by_value) for implemented equations lacking a
        resolvable ``verified_by``.
    """
    verified: List[str] = []
    unverified: List[Tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not entry.get("files"):
            continue  # definition-only entry; no verification required
        eq_id = str(entry.get("eq", "unknown"))
        vb = entry.get("verified_by", "")
        if vb and resolve_test_node(vb, root):
            verified.append(eq_id)
        else:
            unverified.append((eq_id, vb or "<missing>"))
    return verified, unverified


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

    # Check that named objects actually exist in those files
    object_errors = check_objects(entries, root)
    if object_errors:
        print("[ERROR] Object reference errors in equation_index.yml:")
        for error in object_errors:
            print(f"   - {error}")
        print()
    else:
        print("[OK] All indexed objects resolve in their files")
        print()

    # Check verification: implemented equations must be backed by a real test.
    verified, unverified = check_verification(entries, root)
    implemented = len(verified) + len(unverified)
    if unverified:
        print("[WARNING] Implemented equations without a resolvable verified_by:")
        for eq, vb in sorted(unverified):
            print(f"   - {eq}  (verified_by: {vb})")
        print()
    else:
        print("[OK] All implemented equations are backed by a conformance test")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Indexed equations:     {len(indexed_equations)}")
    print(f"  Code references:       {len(code_equations)}")
    print(f"  Missing from index:    {len(missing_from_index)}")
    print(f"  File path errors:      {len(path_errors)}")
    print(f"  Object ref errors:     {len(object_errors)}")
    print(f"  Verified equations:    {len(verified)}/{implemented} implemented")
    print()
    
    # Determine exit code
    if args.strict and (missing_from_index or path_errors or object_errors or unverified):
        print("[FAILED] (strict mode)")
        return 1
    elif path_errors or object_errors or unverified:
        print("[WARNING] (unresolved file paths, object references, or unverified equations)")
        return 0
    else:
        print("[PASSED]")
        return 0


if __name__ == "__main__":
    sys.exit(main())

