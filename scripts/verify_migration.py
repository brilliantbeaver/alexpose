#!/usr/bin/env python3
"""
Verification script for notebooks/utils to ambient/utils migration.

This script verifies that:
1. New imports work correctly
2. Old imports fail as expected
3. All functions are accessible
4. Package structure is correct
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_new_imports():
    """Test that new imports work correctly."""
    print("\n" + "="*60)
    print("Testing New Imports")
    print("="*60)
    
    try:
        # Test direct module imports
        from ambient.utils.eval_keypoints import (
            visualize_keypoints,
            extract_pose_from_sequence,
            pose_estimation_for_frames,
            pose_estimation_all_sequences,
            get_keypoints,
            ensure_model_downloaded,
            calculate_angles,
        )
        print("✅ Direct module imports work")
        
        # Test package-level imports
        from ambient.utils import (
            get_keypoints,
            visualize_frame,
            visualize_pose_with_skeleton,
            draw_bounding_box,
        )
        print("✅ Package-level imports work")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_old_imports_fail():
    """Test that old imports fail as expected."""
    print("\n" + "="*60)
    print("Testing Old Imports (Should Fail)")
    print("="*60)
    
    try:
        from notebooks.utils.eval_keypoints import get_keypoints
        print("❌ Old import should have failed but didn't!")
        return False
    except ModuleNotFoundError:
        print("✅ Old import fails as expected (ModuleNotFoundError)")
        return True
    except Exception as e:
        print(f"⚠️  Old import failed with unexpected error: {e}")
        return True


def test_function_accessibility():
    """Test that all functions are accessible and callable."""
    print("\n" + "="*60)
    print("Testing Function Accessibility")
    print("="*60)
    
    try:
        from ambient.utils import (
            get_keypoints,
            ensure_model_downloaded,
            calculate_angles,
            visualize_keypoints,
            extract_pose_from_sequence,
            pose_estimation_for_frames,
            pose_estimation_all_sequences,
            visualize_frame,
            visualize_pose_with_skeleton,
            draw_bounding_box,
        )
        
        # Check that functions are callable
        functions = [
            get_keypoints,
            ensure_model_downloaded,
            calculate_angles,
            visualize_keypoints,
            extract_pose_from_sequence,
            pose_estimation_for_frames,
            pose_estimation_all_sequences,
            visualize_frame,
            visualize_pose_with_skeleton,
            draw_bounding_box,
        ]
        
        for func in functions:
            if not callable(func):
                print(f"❌ {func.__name__} is not callable")
                return False
        
        print(f"✅ All {len(functions)} functions are accessible and callable")
        return True
        
    except Exception as e:
        print(f"❌ Function accessibility test failed: {e}")
        return False


def test_package_structure():
    """Test that package structure is correct."""
    print("\n" + "="*60)
    print("Testing Package Structure")
    print("="*60)
    
    # Check that new files exist
    new_files = [
        project_root / "ambient" / "utils" / "eval_keypoints.py",
        project_root / "ambient" / "utils" / "viz.py",
    ]
    
    for file_path in new_files:
        if file_path.exists():
            print(f"✅ {file_path.relative_to(project_root)} exists")
        else:
            print(f"❌ {file_path.relative_to(project_root)} missing")
            return False
    
    # Check that old files don't exist
    old_files = [
        project_root / "notebooks" / "utils" / "eval_keypoints.py",
        project_root / "notebooks" / "utils" / "viz.py",
        project_root / "notebooks" / "utils" / "__init__.py",
    ]
    
    for file_path in old_files:
        if not file_path.exists():
            print(f"✅ {file_path.relative_to(project_root)} removed")
        else:
            print(f"⚠️  {file_path.relative_to(project_root)} still exists")
    
    return True


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("MIGRATION VERIFICATION")
    print("notebooks/utils → ambient/utils")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("New Imports", test_new_imports()))
    results.append(("Old Imports Fail", test_old_imports_fail()))
    results.append(("Function Accessibility", test_function_accessibility()))
    results.append(("Package Structure", test_package_structure()))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(result for _, result in results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("Migration completed successfully!")
    else:
        print("❌ SOME VERIFICATION TESTS FAILED")
        print("Please review the output above.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
