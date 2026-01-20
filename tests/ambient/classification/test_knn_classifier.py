"""
Unit tests for KNN Gait Classifier.

This module provides comprehensive tests for the KNN classifier including:
- Feature vector creation
- Classifier training
- Prediction and evaluation
- Model persistence
- Edge cases and error handling

Author: AlexPose Team
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from ambient.classification.knn_classifier import (
    KNNGaitClassifier,
    KNNClassifierConfig,
    GaitFeatureVector
)


@pytest.fixture
def sample_joint_angles():
    """Create mock joint angle sequence for testing."""
    mock_sequence = Mock()
    mock_sequence.get_statistics = Mock(side_effect=lambda joint: {
        "left_hip": {"mean": 175.0, "std": 5.0, "min": 165.0, "max": 185.0, "range": 20.0},
        "left_knee": {"mean": 170.0, "std": 8.0, "min": 155.0, "max": 185.0, "range": 30.0},
        "left_ankle": {"mean": 120.0, "std": 10.0, "min": 100.0, "max": 140.0, "range": 40.0},
        "right_hip": {"mean": 173.0, "std": 5.5, "min": 163.0, "max": 183.0, "range": 20.0},
        "right_knee": {"mean": 168.0, "std": 7.5, "min": 153.0, "max": 183.0, "range": 30.0},
        "right_ankle": {"mean": 118.0, "std": 9.5, "min": 98.0, "max": 138.0, "range": 40.0},
    }[joint])
    return mock_sequence


@pytest.fixture
def sample_feature_vectors():
    """Create sample feature vectors for testing."""
    features = []
    
    # Normal gait samples
    for i in range(5):
        fv = GaitFeatureVector(
            left_hip_mean=175.0 + np.random.randn(),
            left_knee_mean=170.0 + np.random.randn(),
            left_ankle_mean=120.0 + np.random.randn(),
            right_hip_mean=174.0 + np.random.randn(),
            right_knee_mean=169.0 + np.random.randn(),
            right_ankle_mean=119.0 + np.random.randn(),
            hip_asymmetry=1.0 + abs(np.random.randn()),
            knee_asymmetry=1.0 + abs(np.random.randn()),
            ankle_asymmetry=1.0 + abs(np.random.randn()),
            left_hip_range=20.0,
            left_knee_range=30.0,
            right_hip_range=20.0,
            right_knee_range=30.0,
            sample_id=f"normal_{i}",
            condition_label="normal"
        )
        features.append(fv)

    
    # Stroke gait samples (higher asymmetry)
    for i in range(5):
        fv = GaitFeatureVector(
            left_hip_mean=175.0 + np.random.randn(),
            left_knee_mean=170.0 + np.random.randn(),
            left_ankle_mean=120.0 + np.random.randn(),
            right_hip_mean=165.0 + np.random.randn(),  # Lower
            right_knee_mean=160.0 + np.random.randn(),  # Lower
            right_ankle_mean=110.0 + np.random.randn(),  # Lower
            hip_asymmetry=10.0 + abs(np.random.randn()),  # Higher
            knee_asymmetry=10.0 + abs(np.random.randn()),  # Higher
            ankle_asymmetry=10.0 + abs(np.random.randn()),  # Higher
            left_hip_range=20.0,
            left_knee_range=30.0,
            right_hip_range=15.0,  # Reduced
            right_knee_range=25.0,  # Reduced
            sample_id=f"stroke_{i}",
            condition_label="stroke"
        )
        features.append(fv)
    
    return features


class TestGaitFeatureVector:
    """Tests for GaitFeatureVector class."""
    
    def test_feature_vector_creation(self):
        """Test creating a feature vector."""
        fv = GaitFeatureVector(
            left_hip_mean=175.0,
            left_knee_mean=170.0,
            condition_label="normal"
        )
        
        assert fv.left_hip_mean == 175.0
        assert fv.left_knee_mean == 170.0
        assert fv.condition_label == "normal"
    
    def test_to_array(self):
        """Test converting feature vector to array."""
        fv = GaitFeatureVector(
            left_hip_mean=175.0,
            left_knee_mean=170.0,
            left_ankle_mean=120.0,
            right_hip_mean=174.0,
            right_knee_mean=169.0,
            right_ankle_mean=119.0,
            hip_asymmetry=1.0,
            knee_asymmetry=1.0,
            ankle_asymmetry=1.0,
            left_hip_range=20.0,
            left_knee_range=30.0,
            right_hip_range=20.0,
            right_knee_range=30.0
        )
        
        arr = fv.to_array()
        
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 13
        assert arr[0] == 175.0  # left_hip_mean
        assert arr[6] == 1.0    # hip_asymmetry
    
    def test_from_joint_angles(self, sample_joint_angles):
        """Test creating feature vector from joint angles."""
        fv = GaitFeatureVector.from_joint_angles(
            sample_joint_angles,
            sample_id="test_sample",
            condition_label="normal"
        )
        
        assert fv.sample_id == "test_sample"
        assert fv.condition_label == "normal"
        assert fv.left_hip_mean == 175.0
        assert fv.right_hip_mean == 173.0
        assert fv.hip_asymmetry == 2.0  # |175 - 173|
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        names = GaitFeatureVector.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) == 13
        assert "left_hip_mean" in names
        assert "hip_asymmetry" in names


class TestKNNClassifierConfig:
    """Tests for KNNClassifierConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = KNNClassifierConfig()
        
        assert config.n_neighbors == 5
        assert config.weights == "distance"
        assert config.metric == "euclidean"
        assert config.normalize_features is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = KNNClassifierConfig(
            n_neighbors=7,
            weights="uniform",
            metric="manhattan"
        )
        
        assert config.n_neighbors == 7
        assert config.weights == "uniform"
        assert config.metric == "manhattan"


class TestKNNGaitClassifier:
    """Tests for KNNGaitClassifier class."""
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = KNNGaitClassifier()
        
        assert classifier.is_trained is False
        assert classifier.config.n_neighbors == 5
        assert len(classifier.feature_names) == 13
    
    def test_classifier_with_custom_config(self):
        """Test classifier with custom config."""
        config = KNNClassifierConfig(n_neighbors=7)
        classifier = KNNGaitClassifier(config=config)
        
        assert classifier.config.n_neighbors == 7
    
    def test_train_classifier(self, sample_feature_vectors):
        """Test training the classifier."""
        classifier = KNNGaitClassifier()
        metrics = classifier.train(sample_feature_vectors, validate=True)
        
        assert classifier.is_trained is True
        assert "train_accuracy" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == len(sample_feature_vectors)
        assert "classes" in metrics
        assert set(metrics["classes"]) == {"normal", "stroke"}
    
    def test_train_with_insufficient_data(self):
        """Test training with insufficient data."""
        classifier = KNNGaitClassifier()
        
        with pytest.raises(ValueError, match="No training features"):
            classifier.train([])
    
    def test_classify_before_training(self, sample_feature_vectors):
        """Test classification before training raises error."""
        classifier = KNNGaitClassifier()
        
        with pytest.raises(RuntimeError, match="must be trained"):
            classifier.classify_gait(sample_feature_vectors[0])
    
    def test_classify_gait(self, sample_feature_vectors):
        """Test gait classification."""
        classifier = KNNGaitClassifier()
        classifier.train(sample_feature_vectors)
        
        test_sample = sample_feature_vectors[0]
        result = classifier.classify_gait(test_sample)
        
        assert "predicted_condition" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "neighbors" in result
        assert 0 <= result["confidence"] <= 1
        assert result["predicted_condition"] in ["normal", "stroke"]
    
    def test_classify_with_dict_input(self, sample_feature_vectors):
        """Test classification with dictionary input."""
        classifier = KNNGaitClassifier()
        classifier.train(sample_feature_vectors)
        
        feature_dict = {
            "left_hip_mean": 175.0,
            "left_knee_mean": 170.0,
            "left_ankle_mean": 120.0,
            "right_hip_mean": 174.0,
            "right_knee_mean": 169.0,
            "right_ankle_mean": 119.0,
            "hip_asymmetry": 1.0,
            "knee_asymmetry": 1.0,
            "ankle_asymmetry": 1.0,
            "left_hip_range": 20.0,
            "left_knee_range": 30.0,
            "right_hip_range": 20.0,
            "right_knee_range": 30.0
        }
        
        result = classifier.classify_gait(feature_dict)
        
        assert "predicted_condition" in result
        assert result["predicted_condition"] in ["normal", "stroke"]
    
    def test_get_classification_confidence(self, sample_feature_vectors):
        """Test getting classification confidence."""
        classifier = KNNGaitClassifier()
        classifier.train(sample_feature_vectors)
        
        result = classifier.classify_gait(sample_feature_vectors[0])
        confidence = classifier.get_classification_confidence(result)
        
        assert 0 <= confidence <= 1
        assert confidence == result["confidence"]
    
    def test_explain_classification(self, sample_feature_vectors):
        """Test classification explanation."""
        classifier = KNNGaitClassifier()
        classifier.train(sample_feature_vectors)
        
        result = classifier.classify_gait(sample_feature_vectors[0])
        explanation = classifier.explain_classification(result)
        
        assert isinstance(explanation, str)
        assert "Predicted Condition" in explanation
        assert "Confidence" in explanation
        assert "Probability Distribution" in explanation
    
    def test_evaluate_classifier(self, sample_feature_vectors):
        """Test classifier evaluation."""
        # Split data
        train_features = sample_feature_vectors[:8]
        test_features = sample_feature_vectors[8:]
        
        classifier = KNNGaitClassifier()
        classifier.train(train_features)
        
        metrics = classifier.evaluate(test_features)
        
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_save_and_load_classifier(self, sample_feature_vectors):
        """Test saving and loading classifier."""
        classifier = KNNGaitClassifier()
        classifier.train(sample_feature_vectors)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_classifier.pkl"
            
            # Save
            classifier.save(model_path)
            assert model_path.exists()
            
            # Load
            loaded_classifier = KNNGaitClassifier.load(model_path)
            assert loaded_classifier.is_trained is True
            assert loaded_classifier.classes_ is not None
            
            # Test loaded classifier
            result = loaded_classifier.classify_gait(sample_feature_vectors[0])
            assert "predicted_condition" in result
    
    def test_save_untrained_classifier_raises_error(self):
        """Test saving untrained classifier raises error."""
        classifier = KNNGaitClassifier()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_classifier.pkl"
            
            with pytest.raises(RuntimeError, match="Cannot save untrained"):
                classifier.save(model_path)
    
    def test_tune_hyperparameters(self, sample_feature_vectors):
        """Test hyperparameter tuning."""
        classifier = KNNGaitClassifier()
        
        param_grid = {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance']
        }
        
        results = classifier.tune_hyperparameters(
            sample_feature_vectors,
            param_grid=param_grid
        )
        
        assert "best_params" in results
        assert "best_score" in results
        assert classifier.is_trained is True
        assert 0 <= results["best_score"] <= 1


@pytest.mark.integration
class TestKNNClassifierIntegration:
    """Integration tests for KNN classifier."""
    
    def test_end_to_end_workflow(self, sample_feature_vectors):
        """Test complete workflow from training to prediction."""
        # Train
        classifier = KNNGaitClassifier()
        train_metrics = classifier.train(sample_feature_vectors, validate=True)
        
        assert train_metrics["train_accuracy"] > 0.5
        
        # Predict
        test_sample = sample_feature_vectors[0]
        result = classifier.classify_gait(test_sample)
        
        assert result["confidence"] > 0.0
        
        # Explain
        explanation = classifier.explain_classification(result)
        assert len(explanation) > 0
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "classifier.pkl"
            classifier.save(model_path)
            
            loaded = KNNGaitClassifier.load(model_path)
            result2 = loaded.classify_gait(test_sample)
            
            # Results should be identical
            assert result["predicted_condition"] == result2["predicted_condition"]
            assert abs(result["confidence"] - result2["confidence"]) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
