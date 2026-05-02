"""
Test suite for GaitFeatureVector initialization and _feature_groups_enabled attribute.

This test suite specifically addresses the AttributeError that occurred when
_feature_groups_enabled was not properly initialized in dataclass instances.

Root Cause:
-----------
The _feature_groups_enabled field was defined with default_factory=lambda: {...}
in the dataclass, but Python dataclasses don't properly handle mutable defaults
when instances are created directly (not through factory methods).

Solution:
---------
Changed _feature_groups_enabled to Optional[Dict[str, bool]] with default=None,
then initialize it properly in __post_init__ if it's None.

Author: AlexPose Team
Date: 2026-01-27
"""

import pytest
import numpy as np
from ambient.classification.features import GaitFeatureVector


class TestFeatureVectorInitialization:
    """Test proper initialization of GaitFeatureVector instances."""

    def test_direct_instantiation_with_values(self):
        """Test that direct instantiation properly initializes _feature_groups_enabled."""
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            left_ankle_mean=30.0,
            right_hip_mean=44.0,
            right_knee_mean=59.0,
            right_ankle_mean=31.0,
            condition_label="normal",
            sample_id="test_001",
        )

        # Verify _feature_groups_enabled exists and is properly initialized
        assert hasattr(fv, "_feature_groups_enabled")
        assert isinstance(fv._feature_groups_enabled, dict)
        assert len(fv._feature_groups_enabled) == 13
        assert fv._feature_groups_enabled["core_angles"] is True

    def test_empty_instantiation(self):
        """Test that empty instantiation properly initializes _feature_groups_enabled."""
        fv = GaitFeatureVector()

        # Verify _feature_groups_enabled exists and is properly initialized
        assert hasattr(fv, "_feature_groups_enabled")
        assert isinstance(fv._feature_groups_enabled, dict)
        assert len(fv._feature_groups_enabled) == 13

    def test_to_array_after_direct_instantiation(self):
        """Test that to_array() works after direct instantiation."""
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            condition_label="normal",
        )

        # This should not raise AttributeError
        arr = fv.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (82,)  # All 82 features

    def test_to_array_with_feature_groups(self):
        """Test that to_array(feature_groups) works after direct instantiation."""
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            condition_label="normal",
        )

        # Test with specific feature groups
        arr_core = fv.to_array(feature_groups=["core_angles"])
        assert arr_core.shape == (15,)

        arr_spatio = fv.to_array(
            feature_groups=["core_angles", "spatiotemporal"]
        )
        assert arr_spatio.shape == (19,)

    def test_multiple_instances_independent(self):
        """Test that multiple instances have independent _feature_groups_enabled."""
        fv1 = GaitFeatureVector(condition_label="normal")
        fv2 = GaitFeatureVector(condition_label="stroke")

        # Modify one instance
        fv1._feature_groups_enabled["core_angles"] = False

        # Verify the other instance is not affected
        assert fv2._feature_groups_enabled["core_angles"] is True

    def test_classifier_training_scenario(self):
        """Test the exact scenario that caused the original bug."""
        # Create features as they would be in a training scenario
        features = []
        for i in range(10):
            features.append(
                GaitFeatureVector(
                    left_hip_mean=45 + np.random.randn() * 3,
                    left_knee_mean=60 + np.random.randn() * 5,
                    left_ankle_mean=30 + np.random.randn() * 2,
                    right_hip_mean=44 + np.random.randn() * 3,
                    right_knee_mean=59 + np.random.randn() * 5,
                    right_ankle_mean=31 + np.random.randn() * 2,
                    condition_label="normal",
                    sample_id=f"normal_{i}",
                )
            )

        # This is what the classifier does - should not raise AttributeError
        X = np.array([f.to_array() for f in features])
        assert X.shape == (10, 82)

    def test_all_feature_groups_enabled_by_default(self):
        """Test that all feature groups are enabled by default."""
        fv = GaitFeatureVector()

        expected_groups = [
            "core_angles",
            "spatiotemporal",
            "temporal_phases",
            "symmetry_indices",
            "kinematic",
            "variability",
            "postural",
            "extended_angles",
            "temporal_extended",
            "stability",
            "stride_extended",
            "symmetry_extended",
            "kinematic_extended",
        ]

        for group in expected_groups:
            assert group in fv._feature_groups_enabled
            assert fv._feature_groups_enabled[group] is True

    def test_custom_feature_groups_enabled(self):
        """Test that custom _feature_groups_enabled can be provided."""
        custom_groups = {
            "core_angles": True,
            "spatiotemporal": False,
            "temporal_phases": False,
            "symmetry_indices": True,
            "kinematic": False,
            "variability": False,
            "postural": False,
            "extended_angles": False,
            "temporal_extended": False,
            "stability": False,
            "stride_extended": False,
            "symmetry_extended": False,
            "kinematic_extended": False,
        }

        fv = GaitFeatureVector(_feature_groups_enabled=custom_groups)

        assert fv._feature_groups_enabled == custom_groups
        assert fv._feature_groups_enabled["core_angles"] is True
        assert fv._feature_groups_enabled["spatiotemporal"] is False


class TestFeatureVectorRegressionPrevention:
    """Regression tests to prevent the AttributeError from recurring."""

    def test_no_attribute_error_on_to_array(self):
        """Ensure to_array() never raises AttributeError for _feature_groups_enabled."""
        fv = GaitFeatureVector()

        try:
            arr = fv.to_array()
            assert True  # Success
        except AttributeError as e:
            if "_feature_groups_enabled" in str(e):
                pytest.fail(
                    f"AttributeError for _feature_groups_enabled: {e}\n"
                    "This is the bug that was fixed. It should not occur."
                )
            else:
                raise

    def test_no_attribute_error_with_various_instantiation_patterns(self):
        """Test various instantiation patterns that might trigger the bug."""
        patterns = [
            # Empty
            GaitFeatureVector(),
            # With condition only
            GaitFeatureVector(condition_label="normal"),
            # With sample_id only
            GaitFeatureVector(sample_id="test_001"),
            # With some features
            GaitFeatureVector(left_hip_mean=45.0, right_hip_mean=44.0),
            # With all core features
            GaitFeatureVector(
                left_hip_mean=45.0,
                left_knee_mean=60.0,
                left_ankle_mean=30.0,
                right_hip_mean=44.0,
                right_knee_mean=59.0,
                right_ankle_mean=31.0,
            ),
        ]

        for i, fv in enumerate(patterns):
            try:
                arr = fv.to_array()
                assert isinstance(arr, np.ndarray)
            except AttributeError as e:
                if "_feature_groups_enabled" in str(e):
                    pytest.fail(
                        f"Pattern {i} raised AttributeError: {e}\n"
                        f"Feature vector: {fv}"
                    )
                else:
                    raise

    def test_batch_creation_like_training(self):
        """Test batch creation pattern used in classifier training."""
        # This mimics the exact pattern from the error traceback
        features = []
        for i in range(20):
            features.append(
                GaitFeatureVector(
                    left_hip_mean=45 + np.random.randn(),
                    left_knee_mean=60 + np.random.randn(),
                    condition_label="normal" if i < 10 else "stroke",
                )
            )

        # This is the line that failed in the original bug
        try:
            X = np.array([f.to_array() for f in features])
            assert X.shape == (20, 82)
        except AttributeError as e:
            if "_feature_groups_enabled" in str(e):
                pytest.fail(
                    f"Batch creation raised AttributeError: {e}\n"
                    "This is the exact scenario from the bug report."
                )
            else:
                raise

    def test_none_feature_vectors_filtered(self):
        """Test that None feature vectors are filtered out during training."""
        from ambient.classification.knn_classifier import KNNGaitClassifier, KNNClassifierConfig
        
        # Create features with some None values
        features = []
        for i in range(10):
            features.append(
                GaitFeatureVector(
                    left_hip_mean=45 + np.random.randn(),
                    condition_label="normal",
                )
            )
        
        # Add None values (simulating failed extraction)
        features.append(None)
        features.append(None)
        
        for i in range(10):
            features.append(
                GaitFeatureVector(
                    left_hip_mean=35 + np.random.randn(),
                    condition_label="stroke",
                )
            )
        
        # Train classifier (should handle None values)
        config = KNNClassifierConfig(n_neighbors=3, cv_n_jobs=1)
        classifier = KNNGaitClassifier(config)
        
        metrics = classifier.train(
            features=features,
            validate=False,  # Skip CV for speed
            auto_remove_invalid=True
        )
        
        # Should have filtered out the 2 None values
        assert metrics["n_samples"] == 20
        assert classifier.is_trained


class TestPickleCompatibility:
    """Test pickle compatibility for old feature vectors."""

    def test_pickle_with_none_feature_groups_enabled(self):
        """Test that unpickling old feature vectors initializes _feature_groups_enabled."""
        import pickle
        
        # Create a feature vector
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            condition_label="normal",
            sample_id="test_001"
        )
        
        # Simulate old pickle by setting _feature_groups_enabled to None
        fv._feature_groups_enabled = None
        
        # Pickle it
        pickled = pickle.dumps(fv)
        
        # Unpickle it (should trigger __setstate__)
        fv_loaded = pickle.loads(pickled)
        
        # Verify _feature_groups_enabled was initialized
        assert hasattr(fv_loaded, "_feature_groups_enabled")
        assert fv_loaded._feature_groups_enabled is not None
        assert isinstance(fv_loaded._feature_groups_enabled, dict)
        assert len(fv_loaded._feature_groups_enabled) == 13
        
        # Verify to_array() works
        arr = fv_loaded.to_array()
        assert arr.shape == (82,)

    def test_pickle_without_feature_groups_enabled_attribute(self):
        """Test that unpickling very old feature vectors adds _feature_groups_enabled."""
        import pickle
        
        # Create a feature vector
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            condition_label="normal"
        )
        
        # Simulate very old pickle by removing the attribute entirely
        delattr(fv, "_feature_groups_enabled")
        
        # Pickle it
        pickled = pickle.dumps(fv)
        
        # Unpickle it (should trigger __setstate__)
        fv_loaded = pickle.loads(pickled)
        
        # Verify _feature_groups_enabled was added
        assert hasattr(fv_loaded, "_feature_groups_enabled")
        assert fv_loaded._feature_groups_enabled is not None
        assert len(fv_loaded._feature_groups_enabled) == 13
        
        # Verify to_array() works
        arr = fv_loaded.to_array()
        assert arr.shape == (82,)

    def test_pickle_roundtrip_with_new_code(self):
        """Test that pickling with new code works correctly."""
        import pickle
        
        # Create a feature vector with new code
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            condition_label="normal"
        )
        
        # Verify it has _feature_groups_enabled
        assert fv._feature_groups_enabled is not None
        
        # Pickle and unpickle
        pickled = pickle.dumps(fv)
        fv_loaded = pickle.loads(pickled)
        
        # Verify everything still works
        assert fv_loaded._feature_groups_enabled is not None
        assert len(fv_loaded._feature_groups_enabled) == 13
        arr = fv_loaded.to_array()
        assert arr.shape == (82,)

    def test_batch_pickle_with_mixed_versions(self):
        """Test loading a batch of features with mixed pickle versions."""
        import pickle
        
        features = []
        
        # Add new-style feature (with _feature_groups_enabled)
        fv1 = GaitFeatureVector(left_hip_mean=45.0, condition_label="normal")
        features.append(fv1)
        
        # Add old-style feature (with None)
        fv2 = GaitFeatureVector(left_hip_mean=35.0, condition_label="stroke")
        fv2._feature_groups_enabled = None
        features.append(fv2)
        
        # Add very old-style feature (without attribute)
        fv3 = GaitFeatureVector(left_hip_mean=40.0, condition_label="parkinsons")
        delattr(fv3, "_feature_groups_enabled")
        features.append(fv3)
        
        # Pickle the batch
        pickled = pickle.dumps(features)
        
        # Unpickle
        features_loaded = pickle.loads(pickled)
        
        # Verify all features work
        assert len(features_loaded) == 3
        for fv in features_loaded:
            assert hasattr(fv, "_feature_groups_enabled")
            assert fv._feature_groups_enabled is not None
            arr = fv.to_array()
            assert arr.shape == (82,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
