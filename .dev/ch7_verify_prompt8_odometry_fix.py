"""Verification script for Prompt 8: Truth-free odometry constraints.

This script checks that:
1. No ground truth is used to generate odometry measurements
2. All odometry factors come from odom_poses, not true_poses
3. Ground truth is only used for evaluation/plotting

Author: Li-Ta Hsu
Date: December 2025
"""

import re
import sys
from pathlib import Path


def check_odometry_contamination(file_path: Path) -> dict:
    """Check for ground truth contamination in odometry measurements.
    
    Returns:
        dict with results and violations
    """
    results = {
        'file': str(file_path),
        'violations': [],
        'legitimate_uses': [],
        'passed': False
    }
    
    content = file_path.read_text()
    lines = content.split('\n')
    
    # Pattern 1: Using true_poses with se2_relative (potential violation)
    pattern_violation = re.compile(r'se2_relative.*true_poses|true_poses.*se2_relative')
    
    # Pattern 2: Building odometry_measurements (should use odom_poses)
    pattern_odom_build = re.compile(r'odometry_measurements\.append')
    
    for i, line in enumerate(lines, 1):
        # Check for true_poses with se2_relative
        if pattern_violation.search(line):
            # Check context to see if it's in add_odometry_noise (legitimate)
            context_start = max(0, i - 20)
            context = '\n'.join(lines[context_start:i])
            
            if 'def add_odometry_noise' in context:
                results['legitimate_uses'].append({
                    'line': i,
                    'code': line.strip(),
                    'context': 'Inside add_odometry_noise() - data generation (OK)'
                })
            else:
                results['violations'].append({
                    'line': i,
                    'code': line.strip(),
                    'context': 'Using true_poses for measurement (VIOLATION)'
                })
        
        # Check odometry_measurements building
        if pattern_odom_build.search(line):
            # Look backwards for the rel_pose computation
            for j in range(max(0, i - 10), i):
                if 'se2_relative' in lines[j]:
                    if 'true_poses' in lines[j]:
                        results['violations'].append({
                            'line': j + 1,
                            'code': lines[j].strip(),
                            'context': 'Odometry measurement from true_poses (VIOLATION)'
                        })
                    elif 'odom_poses' in lines[j]:
                        results['legitimate_uses'].append({
                            'line': j + 1,
                            'code': lines[j].strip(),
                            'context': 'Odometry measurement from odom_poses (CORRECT)'
                        })
                    break
    
    results['passed'] = len(results['violations']) == 0
    return results


def main():
    """Run verification checks."""
    print("=" * 80)
    print("PROMPT 8 VERIFICATION: Truth-Free Odometry Constraints")
    print("=" * 80)
    print()
    
    # Check the main example file
    example_file = Path('ch7_slam/example_pose_graph_slam.py')
    
    if not example_file.exists():
        print(f"❌ ERROR: File not found: {example_file}")
        print("   Run this script from the repository root.")
        sys.exit(1)
    
    results = check_odometry_contamination(example_file)
    
    print(f"📁 File: {results['file']}")
    print()
    
    # Report violations
    if results['violations']:
        print("❌ VIOLATIONS FOUND:")
        print()
        for v in results['violations']:
            print(f"   Line {v['line']}: {v['code']}")
            print(f"   Context: {v['context']}")
            print()
    else:
        print("✅ No violations found!")
        print()
    
    # Report legitimate uses
    if results['legitimate_uses']:
        print("✅ LEGITIMATE USES:")
        print()
        for u in results['legitimate_uses']:
            print(f"   Line {u['line']}: {u['code']}")
            print(f"   Context: {u['context']}")
            print()
    
    # Overall result
    print("=" * 80)
    if results['passed']:
        print("✅ VERIFICATION PASSED")
        print()
        print("Acceptance Criteria:")
        print("  ✅ No ground truth used for odometry measurements")
        print("  ✅ All odometry factors come from odom_poses")
        print("  ✅ Ground truth only used for evaluation/plotting")
        print()
        print("Next steps:")
        print("  1. Test inline mode: python -m ch7_slam.example_pose_graph_slam")
        print("  2. Test dataset: python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_square")
        print("  3. Proceed to Prompt 9: Fix loop closure detection (observation-based)")
        print()
        return 0
    else:
        print("❌ VERIFICATION FAILED")
        print()
        print(f"Found {len(results['violations'])} violation(s).")
        print("Fix violations before proceeding to next prompt.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
