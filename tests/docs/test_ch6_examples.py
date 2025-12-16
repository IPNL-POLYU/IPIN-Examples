"""
Test Ch6 dataset code examples to ensure they work correctly.

This script extracts and tests Python code blocks from Ch6 dataset READMEs.

Location: tests/docs/
Purpose: Documentation validation (not unit tests for core/ modules)

Note: Unit tests for core/sensors/ are in tests/core/sensors/
"""

import io
import re
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr


def extract_python_blocks(readme_path):
    """Extract Python code blocks from README."""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all Python code blocks
    pattern = r'```python\n(.*?)\n```'
    blocks = re.findall(pattern, content, re.DOTALL)
    return blocks


def test_code_block(code, dataset_name, block_num):
    """Test a single code block."""
    # Skip blocks that are just examples (contain plt.show() or are incomplete)
    if 'plt.show()' in code or '# ... rest is' in code or '...' in code:
        return True, "Skipped (plotting or incomplete example)"
    
    # Create a safe execution environment
    exec_globals = {
        '__name__': '__main__',
        '__file__': f'test_{dataset_name}.py'
    }
    
    try:
        # Redirect output
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, exec_globals)
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def main():
    """Main test execution."""
    print("\n" + "=" * 70)
    print("Testing Ch6 Dataset Code Examples")
    print("=" * 70)
    
    # Ch6 datasets to test
    datasets = [
        'ch6_strapdown_basic',
        'ch6_foot_zupt_walk',
        'ch6_wheel_odom_square',
        'ch6_pdr_corridor_walk',
        'ch6_env_sensors_heading_altitude',
    ]
    
    total_tests = 0
    total_passed = 0
    total_skipped = 0
    total_failed = 0
    
    for dataset in datasets:
        readme_path = Path(f"data/sim/{dataset}/README.md")
        
        if not readme_path.exists():
            print(f"\n[X] {dataset}: README not found")
            continue
        
        print(f"\n\nTesting: {dataset}")
        print("-" * 70)
        
        blocks = extract_python_blocks(readme_path)
        print(f"Found {len(blocks)} Python code blocks")
        
        for i, block in enumerate(blocks, 1):
            total_tests += 1
            success, message = test_code_block(block, dataset, i)
            
            if success:
                if "Skipped" in message:
                    status = "[SKIP]"
                    total_skipped += 1
                else:
                    status = "[OK]"
                    total_passed += 1
            else:
                status = "[FAIL]"
                total_failed += 1
            
            # Truncate message for display
            display_msg = message[:50] + "..." if len(message) > 50 else message
            print(f"  Block {i:2d}: {status} {display_msg}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests:   {total_tests}")
    print(f"Passed:        {total_passed} (executable)")
    print(f"Skipped:       {total_skipped} (plotting/incomplete)")
    print(f"Failed:        {total_failed}")
    print()
    
    if total_failed == 0:
        print("[OK] All testable code examples work correctly!")
        return 0
    else:
        print(f"[FAIL] {total_failed} code examples failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

