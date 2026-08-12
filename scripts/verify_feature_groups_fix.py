#!/usr/bin/env python3
"""
Verification script for the _feature_groups_enabled initialization fix.

This script demonstrates that the AttributeError bug has been fixed and
that GaitFeatureVector instances can be created and used correctly.

Run this script to verify the fix:
    python3 scripts/verify_feature_groups_fix.py

Author: AlexPose Team
Date: 2026-01-27
"""

import sys
import numpy as np
from ambient.classification.features import GaitFeatureVector
from ambient.classification.knn_classifier import KNNGaitClassifier, KNNClassifierConfig


def test_direct_instantiation():
    """Test 1: Direct instantiation works."""
    print("Test 1: Direct instantiation")
    print("-" * 60)
    
    try:
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            condition_label="normal"
        )
        
        # Check _feature_groups_enabled exists
        assert hasattr(fv, "_feature_groups_enabled")
        assert isinstance(fv._feature_groups_enabled, dict)
        
        print("✅ Direct instantiation works")
        print(f"   _feature_groups_enabled: {len(fv._feature_groups_enabled)} groups")
        return True
    except AttributeError as e:
        print(f"❌ FAILED: {e}")
        return False


def test_to_array():
    """Test 2: to_array() works after direct instantiation."""
    print("\nTest 2: to_array() functionality")
    print("-" * 60)
    
    try:
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            condition_label="normal"
        )
        
        # Test default (all 82 features)
        arr = fv.to_array()
        assert arr.shape == (82,)
        print(f"✅ to_array() works - shape: {arr.shape}")
        
        # Test legacy mode (15 features)
        arr_legacy = fv.to_array(feature_groups=["core_angles"])
        assert arr_legacy.shape == (15,)
        print(f"✅ to_array(feature_groups=['core_angles']) works - shape: {arr_legacy.shape}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_classifier_training():
    """Test 3: Classifier training works (exact bug scenario)."""
    print("\nTest 3: Classifier training (exact bug scenario)")
    print("-" * 60)
    
    try:
        # Create features exactly as in the bug report
        features = []
        for i in range(10):
            features.append(GaitFeatureVector(
                left_hip_mean=45 + np.random.randn() * 3,
                left_knee_mean=60 + np.random.randn() * 5,
                left_ankle_mean=30 + np.random.randn() * 2,
                right_hip_mean=44 + np.random.randn() * 3,
                right_knee_mean=59 + np.random.randn() * 5,
                right_ankle_mean=31 + np.random.randn() * 2,
                condition_label="normal",
                sample_id=f"normal_{i}"
            ))
        
        for i in range(10):
            features.append(GaitFeatureVector(
                left_hip_mean=35 + np.random.randn() * 3,
                left_knee_mean=45 + np.random.randn() * 5,
                left_ankle_mean=25 + np.random.randn() * 2,
                right_hip_mean=40 + np.random.randn() * 3,
                right_knee_mean=50 + np.random.randn() * 5,
                right_ankle_mean=28 + np.random.randn() * 2,
                condition_label="stroke",
                sample_id=f"stroke_{i}"
            ))
        
        # Train classifier (this is where the bug occurred)
        config = KNNClassifierConfig(n_neighbors=3, cv_n_jobs=1)
        classifier = KNNGaitClassifier(config)
        
        metrics = classifier.train(
            features=features,
            validate=True,
            auto_remove_invalid=True
        )
        
        print(f"✅ Classifier training successful")
        print(f"   Train accuracy: {metrics['train_accuracy']:.3f}")
        print(f"   N samples: {metrics['n_samples']}")
        print(f"   N features: {metrics['n_features']}")
        print(f"   Classes: {metrics['classes']}")
        
        return True
    except AttributeError as e:
        if "_feature_groups_enabled" in str(e):
            print(f"❌ FAILED: The bug still exists!")
            print(f"   Error: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_instances():
    """Test 4: Multiple instances are independent."""
    print("\nTest 4: Multiple instances independence")
    print("-" * 60)
    
    try:
        fv1 = GaitFeatureVector(condition_label="normal")
        fv2 = GaitFeatureVector(condition_label="stroke")
        
        # Modify one instance
        fv1._feature_groups_enabled["core_angles"] = False
        
        # Verify the other is not affected
        assert fv2._feature_groups_enabled["core_angles"] is True
        
        print("✅ Multiple instances are independent")
        print("   Each instance has its own _feature_groups_enabled dict")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_batch_creation():
    """Test 5: Batch creation pattern (common in training)."""
    print("\nTest 5: Batch creation pattern")
    print("-" * 60)
    
    try:
        # Create batch of features
        features = [
            GaitFeatureVector(
                left_hip_mean=45 + np.random.randn(),
                condition_label="normal" if i < 10 else "stroke"
            )
            for i in range(20)
        ]
        
        # Convert to array (this is where the bug occurred)
        X = np.array([f.to_array() for f in features])
        
        assert X.shape == (20, 82)
        
        print(f"✅ Batch creation works")
        print(f"   Created {len(features)} feature vectors")
        print(f"   Array shape: {X.shape}")
        
        return True
    except AttributeError as e:
        if "_feature_groups_enabled" in str(e):
            print(f"❌ FAILED: The bug still exists in batch creation!")
            print(f"   Error: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("VERIFICATION: _feature_groups_enabled Initialization Fix")
    print("=" * 60)
    print()
    
    tests = [
        test_direct_instantiation,
        test_to_array,
        test_classifier_training,
        test_multiple_instances,
        test_batch_creation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe _feature_groups_enabled initialization bug has been fixed.")
        print("You can now:")
        print("  1. Restart your Jupyter kernel")
        print("  2. Re-run your classifier training code")
        print("  3. It should work without AttributeError")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("\nThe bug may still exist. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
