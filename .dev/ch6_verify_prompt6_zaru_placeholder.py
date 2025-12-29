#!/usr/bin/env python3
"""
Acceptance Verification for Prompt 6: ZARU Placeholder

This script verifies that:
1. No "stub" class claims to implement a numbered equation without actually doing so
2. ZARU is properly renamed to ZaruMeasurementModelPlaceholder
3. Documentation clearly states incomplete status
4. Equation references are softened or removed

Acceptance Criterion:
    No "stub" class claims to implement a numbered equation.

Author: Li-Ta Hsu
Date: December 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_class_renamed():
    """Check that ZaruMeasurementModel is renamed to ZaruMeasurementModelPlaceholder."""
    print("\n[1/6] Checking class is renamed to Placeholder...")
    
    constraints_file = Path(__file__).parent.parent / "core" / "sensors" / "constraints.py"
    content = constraints_file.read_text(encoding='utf-8')
    
    checks = []
    
    # Should NOT have old name as a class definition
    if "class ZaruMeasurementModel:" in content or "class ZaruMeasurementModel(" in content:
        checks.append(("Old class name removed", False))
    else:
        checks.append(("Old class name removed", True))
    
    # Should HAVE new name
    if "class ZaruMeasurementModelPlaceholder:" in content:
        checks.append(("New placeholder class name exists", True))
    else:
        checks.append(("New placeholder class name exists", False))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def check_no_false_equation_claims():
    """Check that the placeholder does NOT claim to implement Eq. (6.60)."""
    print("\n[2/6] Checking no false equation claims...")
    
    constraints_file = Path(__file__).parent.parent / "core" / "sensors" / "constraints.py"
    content = constraints_file.read_text(encoding='utf-8')
    
    # Extract the class definition
    class_start = content.find("class ZaruMeasurementModelPlaceholder:")
    if class_start == -1:
        print("  [FAIL] Class not found")
        return False
    
    # Find next class definition or end of file
    next_class = content.find("\nclass ", class_start + 1)
    if next_class == -1:
        class_section = content[class_start:]
    else:
        class_section = content[class_start:next_class]
    
    checks = []
    
    # Should NOT have strong implementation claims
    bad_phrases = [
        "Implements Eq. (6.60)",
        "implements Eq. (6.60)",
        "Implementation of Eq. (6.60)",
        "Based on Eq. (6.60)",
    ]
    
    has_bad_phrase = False
    for phrase in bad_phrases:
        if phrase in class_section:
            checks.append((f"No false claim: '{phrase}'", False))
            has_bad_phrase = True
    
    if not has_bad_phrase:
        checks.append(("No false 'implements Eq. (6.60)' claims", True))
    
    # SHOULD have disclaimer
    if "INCOMPLETE" in class_section or "PLACEHOLDER" in class_section:
        checks.append(("Has INCOMPLETE/PLACEHOLDER warning", True))
    else:
        checks.append(("Has INCOMPLETE/PLACEHOLDER warning", False))
    
    # Should explain why incomplete
    if "interface" in class_section.lower() or "omega_meas" in class_section:
        checks.append(("Explains interface limitation", True))
    else:
        checks.append(("Explains interface limitation", False))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def check_method_docstrings_honest():
    """Check that method docstrings are honest about limitations."""
    print("\n[3/6] Checking method docstrings are honest...")
    
    constraints_file = Path(__file__).parent.parent / "core" / "sensors" / "constraints.py"
    content = constraints_file.read_text(encoding='utf-8')
    
    # Extract class section
    class_start = content.find("class ZaruMeasurementModelPlaceholder:")
    next_class = content.find("\nclass ", class_start + 1)
    if next_class == -1:
        class_section = content[class_start:]
    else:
        class_section = content[class_start:next_class]
    
    checks = []
    
    # Check h() method
    if "def h(self, x:" in class_section:
        h_start = class_section.find("def h(self, x:")
        h_end = class_section.find("\n    def ", h_start + 1)
        h_section = class_section[h_start:h_end] if h_end != -1 else class_section[h_start:]
        
        if "INCOMPLETE" in h_section or "LIMITATION" in h_section:
            checks.append(("h() docstring warns about incompleteness", True))
        else:
            checks.append(("h() docstring warns about incompleteness", False))
    else:
        checks.append(("h() method exists", False))
    
    # Check H() method
    if "def H(self, x:" in class_section:
        H_start = class_section.find("def H(self, x:")
        H_end = class_section.find("\n    def ", H_start + 1)
        H_section = class_section[H_start:H_end] if H_end != -1 else class_section[H_start:]
        
        if "partially correct" in H_section.lower() or "correct" in H_section.lower():
            checks.append(("H() docstring acknowledges partial correctness", True))
        else:
            checks.append(("H() docstring acknowledges partial correctness", False))
    else:
        checks.append(("H() method exists", False))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def check_imports_updated():
    """Check that imports use the new placeholder name."""
    print("\n[4/6] Checking imports are updated...")
    
    init_file = Path(__file__).parent.parent / "core" / "sensors" / "__init__.py"
    content = init_file.read_text(encoding='utf-8')
    
    checks = []
    
    # Should import placeholder
    if "ZaruMeasurementModelPlaceholder" in content:
        checks.append(("__init__.py imports ZaruMeasurementModelPlaceholder", True))
    else:
        checks.append(("__init__.py imports ZaruMeasurementModelPlaceholder", False))
    
    # Should export placeholder
    if '"ZaruMeasurementModelPlaceholder"' in content:
        checks.append(("__init__.py exports ZaruMeasurementModelPlaceholder", True))
    else:
        checks.append(("__init__.py exports ZaruMeasurementModelPlaceholder", False))
    
    # Should NOT have old name in exports (unless as comment/deprecation)
    export_lines = [line for line in content.split('\n') if '"ZaruMeasurementModel"' in line and not line.strip().startswith('#')]
    if len(export_lines) == 0:
        checks.append(("Old name removed from exports", True))
    else:
        checks.append(("Old name removed from exports", False))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def check_tests_updated():
    """Check that tests use the new placeholder name."""
    print("\n[5/6] Checking tests are updated...")
    
    test_files = [
        Path(__file__).parent.parent / "tests" / "core" / "sensors" / "test_sensors_constraints.py",
        Path(__file__).parent.parent / "tests" / "core" / "test_ins_state_ordering.py",
    ]
    
    checks = []
    
    for test_file in test_files:
        if not test_file.exists():
            checks.append((f"{test_file.name} exists", False))
            continue
        
        content = test_file.read_text(encoding='utf-8')
        
        # Should import placeholder
        if "ZaruMeasurementModelPlaceholder" in content:
            checks.append((f"{test_file.name} imports ZaruMeasurementModelPlaceholder", True))
        else:
            checks.append((f"{test_file.name} imports ZaruMeasurementModelPlaceholder", False))
        
        # Should NOT import old name
        if "from" in content and "ZaruMeasurementModel" in content and "Placeholder" not in content:
            # Check if it's actually importing the old name
            import_lines = [line for line in content.split('\n') if 'import' in line and 'ZaruMeasurementModel' in line and 'Placeholder' not in line]
            if len(import_lines) > 0:
                checks.append((f"{test_file.name} does not import old name", False))
            else:
                checks.append((f"{test_file.name} does not import old name", True))
        else:
            checks.append((f"{test_file.name} does not import old name", True))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def check_documentation_updated():
    """Check that documentation reflects placeholder status."""
    print("\n[6/6] Checking documentation is updated...")
    
    readme_file = Path(__file__).parent.parent / "ch6_dead_reckoning" / "README.md"
    content = readme_file.read_text(encoding='utf-8')
    
    checks = []
    
    # Should reference placeholder
    if "ZaruMeasurementModelPlaceholder" in content:
        checks.append(("README uses ZaruMeasurementModelPlaceholder", True))
    else:
        checks.append(("README uses ZaruMeasurementModelPlaceholder", False))
    
    # Should NOT claim Eq. (6.60) is implemented
    if "Eq. (6.60)" in content:
        # Check if it's in a line with ZARU
        lines_with_eq = [line for line in content.split('\n') if 'Eq. (6.60)' in line and 'Zaru' in line]
        if len(lines_with_eq) > 0:
            checks.append(("README does not claim Eq. (6.60) is implemented", False))
        else:
            checks.append(("README does not claim Eq. (6.60) is implemented", True))
    else:
        checks.append(("README does not claim Eq. (6.60) is implemented", True))
    
    # Should have warning marker
    if "INCOMPLETE" in content or "⚠" in content or "WARNING" in content:
        zaru_lines = [line for line in content.split('\n') if 'Zaru' in line]
        has_warning = any("INCOMPLETE" in line or "⚠" in line or "WARNING" in line for line in zaru_lines)
        checks.append(("README has warning marker for ZARU", has_warning))
    else:
        checks.append(("README has warning marker for ZARU", False))
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name}")
            all_passed = False
    
    return all_passed


def main():
    """Run all acceptance checks."""
    print("="*70)
    print("Prompt 6 Acceptance Verification: ZARU Placeholder")
    print("="*70)
    
    checks = [
        ("Class renamed to Placeholder", check_class_renamed),
        ("No false equation claims", check_no_false_equation_claims),
        ("Method docstrings honest", check_method_docstrings_honest),
        ("Imports updated", check_imports_updated),
        ("Tests updated", check_tests_updated),
        ("Documentation updated", check_documentation_updated),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("ACCEPTANCE SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status:8} {name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL CHECKS PASSED -> Prompt 6 acceptance criteria met!")
        print("Acceptance Criterion: No stub class claims to implement a numbered equation")
        print("Status: SATISFIED - ZARU is now an honest placeholder")
        print("="*70)
        return 0
    else:
        print("SOME CHECKS FAILED -> Please review failures above")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

