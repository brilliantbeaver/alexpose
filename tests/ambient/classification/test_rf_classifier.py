"""
Unit tests for Random Forest Gait Classifier.

Tests cover:
- Configuration and initialization
- Training with various datasets
- Classification and prediction
- Feature importance analysis
- Model evaluation
- Hyperparameter tuning
- Model persistence (save/load)
- Error handling and edge cases
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List

from ambient.classification.rf_classifier import (
    RFGaitClassifier,
    RFClassifierConfig,
    FeatureImportance,
)
from ambient.classification.knn_classifier import GaitFeatureVector


@pytest.fixture
def default_config():
    """Default classifier configuration."""
    return RFClassifierConfig(
        n_estimators=10,  # Small for fast tests
        random_state=42,
        n_jobs=1,  # Single thread for reproducibility
    )


@pytest.fixture
def classifier(default_config):
    """Create a classifier instance."""
    return RFGaitClassifier(default_config)


@pytest.fixture
def sample_features() -> List[GaitFeatureVector]:
    """Generate sample training features."""
    np.random.seed(42)
    features = []

    # Normal gait samples
    for i in range(20):
        features.append(
            GaitFeatureVector(
                left_hip_mean=45 + np.random.randn() * 5,
                left_knee_mean=60 + np.random.randn() * 5,
                left_ankle_mean=20 + np.random.randn() * 3,
                right_hip_mean=45 + np.random.randn() * 5,
                right_knee_mean=60 + np.random.randn() * 5,
                right_ankle_mean=20 + np.random.randn() * 3,
                hip_asymmetry=np.random.randn() * 2,
                knee_asymmetry=np.random.randn() * 2,
                ankle_asymmetry=np.random.randn() * 2,
                left_hip_range=40 + np.random.randn() * 3,
                left_knee_range=70 + np.random.randn() * 5,
                left_ankle_range=30 + np.random.randn() * 3,
                right_hip_range=40 + np.random.randn() * 3,
                right_knee_range=70 + np.random.randn() * 5,
                right_ankle_range=30 + np.random.randn() * 3,
                sample_id=f"normal_{i}",
                condition_label="normal",
            )
        )

    # Stroke gait samples (asymmetric)
    for i in range(20):
        features.append(
            GaitFeatureVector(
                left_hip_mean=40 + np.random.randn() * 5,
                left_knee_mean=50 + np.random.randn() * 5,
                left_ankle_mean=15 + np.random.randn() * 3,
                right_hip_mean=50 + np.random.randn() * 5,
                right_knee_mean=65 + np.random.randn() * 5,
                right_ankle_mean=25 + np.random.randn() * 3,
                hip_asymmetry=10 + np.random.randn() * 3,
                knee_asymmetry=15 + np.random.randn() * 3,
                ankle_asymmetry=10 + np.random.randn() * 3,
                left_hip_range=30 + np.random.randn() * 3,
                left_knee_range=50 + np.random.randn() * 5,
                left_ankle_range=20 + np.random.randn() * 3,
                right_hip_range=45 + np.random.randn() * 3,
                right_knee_range=75 + np.random.randn() * 5,
                right_ankle_range=35 + np.random.randn() * 3,
                sample_id=f"stroke_{i}",
                condition_label="stroke",
            )
        )

    # Parkinson's gait samples (reduced ROM)
    for i in range(20):
        features.append(
            GaitFeatureVector(
                left_hip_mean=35 + np.random.randn() * 3,
                left_knee_mean=45 + np.random.randn() * 3,
                left_ankle_mean=12 + np.random.randn() * 2,
                right_hip_mean=35 + np.random.randn() * 3,
                right_knee_mean=45 + np.random.randn() * 3,
                right_ankle_mean=12 + np.random.randn() * 2,
                hip_asymmetry=np.random.randn() * 2,
                knee_asymmetry=np.random.randn() * 2,
                ankle_asymmetry=np.random.randn() * 2,
                left_hip_range=25 + np.random.randn() * 2,
                left_knee_range=40 + np.random.randn() * 3,
                left_ankle_range=15 + np.random.randn() * 2,
                right_hip_range=25 + np.random.randn() * 2,
                right_knee_range=40 + np.random.randn() * 3,
                right_ankle_range=15 + np.random.randn() * 2,
                sample_id=f"parkinsons_{i}",
                condition_label="parkinsons",
            )
        )

    return features


class TestRFClassifierConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RFClassifierConfig()
        assert config.n_estimators == 100
        assert config.max_depth is None
        assert config.min_samples_split == 2
        assert config.min_samples_leaf == 1
        assert config.max_features == "sqrt"
        assert config.bootstrap is True
        assert config.random_state == 42
        assert config.n_jobs == -1
        assert config.normalize_features is True
        assert config.confidence_threshold == 0.5
        assert config.class_weight == "balanced"

    def test_custom_config(self):
        """Test custom configuration."""
        config = RFClassifierConfig(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            normalize_features=False,
        )
        assert config.n_estimators == 200
        assert config.max_depth == 15
        assert config.min_samples_split == 5
        assert config.normalize_features is False


class TestRFClassifierInitialization:
    """Test classifier initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        classifier = RFGaitClassifier()
        assert classifier.config.n_estimators == 100
        assert classifier.is_trained is False
        assert classifier.classes_ is None
        assert classifier.scaler is not None

    def test_init_custom_config(self, default_config):
        """Test initialization with custom config."""
        classifier = RFGaitClassifier(default_config)
        assert classifier.config.n_estimators == 10
        assert classifier.config.random_state == 42
        assert classifier.is_trained is False

    def test_init_without_normalization(self):
        """Test initialization without feature normalization."""
        config = RFClassifierConfig(normalize_features=False)
        classifier = RFGaitClassifier(config)
        assert classifier.scaler is None


class TestRFClassifierTraining:
    """Test classifier training."""

    def test_train_basic(self, classifier, sample_features):
        """Test basic training."""
        metrics = classifier.train(sample_features, validate=False)

        assert classifier.is_trained is True
        assert classifier.classes_ is not None
        assert len(classifier.classes_) == 3
        assert "train_accuracy" in metrics
        assert metrics["train_accuracy"] > 0.5
        assert metrics["n_samples"] == 60
        assert metrics["n_features"] == 15

    def test_train_with_validation(self, classifier, sample_features):
        """Test training with cross-validation."""
        metrics = classifier.train(sample_features, validate=True)

        assert "cv_mean_accuracy" in metrics
        assert "cv_std_accuracy" in metrics
        assert 0 <= metrics["cv_mean_accuracy"] <= 1
        assert metrics["cv_std_accuracy"] >= 0

    def test_train_with_explicit_labels(self, classifier, sample_features):
        """Test training with explicit labels."""
        labels = [f.condition_label for f in sample_features]
        metrics = classifier.train(sample_features, labels=labels, validate=False)

        assert classifier.is_trained is True
        assert metrics["n_samples"] == len(labels)

    def test_train_empty_features(self, classifier):
        """Test training with empty features raises error."""
        with pytest.raises(ValueError, match="No training features provided"):
            classifier.train([])

    def test_train_mismatched_labels(self, classifier, sample_features):
        """Test training with mismatched label count raises error."""
        labels = ["normal", "stroke"]  # Too few labels
        with pytest.raises(ValueError, match="Feature count.*!= label count"):
            classifier.train(sample_features, labels=labels)

    def test_feature_importances_after_training(self, classifier, sample_features):
        """Test feature importances are computed after training."""
        classifier.train(sample_features, validate=False)

        assert classifier.feature_importances_ is not None
        assert len(classifier.feature_importances_) == 15
        assert np.sum(classifier.feature_importances_) == pytest.approx(1.0, abs=1e-6)

    def test_class_distribution_logging(self, classifier, sample_features):
        """Test class distribution is logged."""
        metrics = classifier.train(sample_features, validate=False)

        assert "class_distribution" in metrics
        dist = metrics["class_distribution"]
        assert dist["normal"] == 20
        assert dist["stroke"] == 20
        assert dist["parkinsons"] == 20


class TestRFClassifierClassification:
    """Test classification functionality."""

    def test_classify_before_training(self, classifier, sample_features):
        """Test classification before training raises error."""
        with pytest.raises(RuntimeError, match="must be trained before classification"):
            classifier.classify_gait(sample_features[0])

    def test_classify_basic(self, classifier, sample_features):
        """Test basic classification."""
        classifier.train(sample_features, validate=False)
        result = classifier.classify_gait(sample_features[0])

        assert "predicted_condition" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "tree_votes" in result
        assert "is_normal" in result
        assert "feature_vector" in result

        assert result["predicted_condition"] in ["normal", "stroke", "parkinsons"]
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 3

    def test_classify_with_dict_input(self, classifier, sample_features):
        """Test classification with dictionary input."""
        classifier.train(sample_features, validate=False)

        feature_dict = {
            "left_hip_mean": 45.0,
            "left_knee_mean": 60.0,
            "left_ankle_mean": 20.0,
            "right_hip_mean": 45.0,
            "right_knee_mean": 60.0,
            "right_ankle_mean": 20.0,
            "hip_asymmetry": 1.0,
            "knee_asymmetry": 1.0,
            "ankle_asymmetry": 1.0,
            "left_hip_range": 40.0,
            "left_knee_range": 70.0,
            "left_ankle_range": 30.0,
            "right_hip_range": 40.0,
            "right_knee_range": 70.0,
            "right_ankle_range": 30.0,
        }

        result = classifier.classify_gait(feature_dict)
        assert "predicted_condition" in result

    def test_tree_votes(self, classifier, sample_features):
        """Test tree voting information."""
        classifier.train(sample_features, validate=False)
        result = classifier.classify_gait(sample_features[0])

        tree_votes = result["tree_votes"]
        total_votes = sum(tree_votes.values())
        assert total_votes == classifier.config.n_estimators

    def test_is_normal_flag(self, classifier, sample_features):
        """Test is_normal flag is set correctly."""
        classifier.train(sample_features, validate=False)

        # Test normal sample
        normal_sample = [f for f in sample_features if f.condition_label == "normal"][0]
        result = classifier.classify_gait(normal_sample)
        # Note: May not always predict correctly, but flag should be set based on prediction
        if result["predicted_condition"].lower() in ["normal", "healthy"]:
            assert result["is_normal"] is True

    def test_get_classification_confidence(self, classifier, sample_features):
        """Test confidence extraction."""
        classifier.train(sample_features, validate=False)
        result = classifier.classify_gait(sample_features[0])

        confidence = classifier.get_classification_confidence(result)
        assert confidence == result["confidence"]
        assert 0 <= confidence <= 1


class TestFeatureImportance:
    """Test feature importance functionality."""

    def test_get_feature_importances(self, classifier, sample_features):
        """Test getting feature importances."""
        classifier.train(sample_features, validate=False)
        importances = classifier.get_feature_importances()

        assert len(importances) == 15
        assert all(isinstance(imp, FeatureImportance) for imp in importances)
        assert all(imp.rank == i + 1 for i, imp in enumerate(importances))

        # Check sorted by importance
        for i in range(len(importances) - 1):
            assert importances[i].importance >= importances[i + 1].importance

    def test_get_top_n_importances(self, classifier, sample_features):
        """Test getting top N feature importances."""
        classifier.train(sample_features, validate=False)
        top_5 = classifier.get_feature_importances(top_n=5)

        assert len(top_5) == 5
        assert top_5[0].rank == 1

    def test_feature_importance_before_training(self, classifier):
        """Test getting importances before training raises error."""
        with pytest.raises(RuntimeError, match="must be trained"):
            classifier.get_feature_importances()

    def test_feature_importance_repr(self):
        """Test FeatureImportance string representation."""
        imp = FeatureImportance(feature_name="left_hip_mean", importance=0.1234, rank=1)
        repr_str = repr(imp)
        assert "1." in repr_str
        assert "left_hip_mean" in repr_str
        assert "0.1234" in repr_str


class TestExplainClassification:
    """Test classification explanation."""

    def test_explain_basic(self, classifier, sample_features):
        """Test basic explanation generation."""
        classifier.train(sample_features, validate=False)
        result = classifier.classify_gait(sample_features[0])
        explanation = classifier.explain_classification(result)

        assert isinstance(explanation, str)
        assert "Predicted Condition:" in explanation
        assert "Confidence:" in explanation
        assert "Probability Distribution:" in explanation
        assert "Tree Voting" in explanation
        assert "Top Contributing Features:" in explanation

    def test_explanation_contains_all_classes(self, classifier, sample_features):
        """Test explanation includes all classes."""
        classifier.train(sample_features, validate=False)
        result = classifier.classify_gait(sample_features[0])
        explanation = classifier.explain_classification(result)

        for class_name in classifier.classes_:
            assert class_name in explanation


class TestEvaluation:
    """Test model evaluation."""

    def test_evaluate_basic(self, classifier, sample_features):
        """Test basic evaluation."""
        # Split data
        train_features = sample_features[:45]
        test_features = sample_features[45:]

        classifier.train(train_features, validate=False)
        metrics = classifier.evaluate(test_features)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics
        assert metrics["n_test_samples"] == 15

    def test_evaluate_before_training(self, classifier, sample_features):
        """Test evaluation before training raises error."""
        with pytest.raises(RuntimeError, match="must be trained before evaluation"):
            classifier.evaluate(sample_features)

    def test_confusion_matrix_shape(self, classifier, sample_features):
        """Test confusion matrix has correct shape."""
        train_features = sample_features[:45]
        test_features = sample_features[45:]

        classifier.train(train_features, validate=False)
        metrics = classifier.evaluate(test_features)

        conf_matrix = np.array(metrics["confusion_matrix"])
        n_classes = len(classifier.classes_)
        assert conf_matrix.shape == (n_classes, n_classes)


class TestHyperparameterTuning:
    """Test hyperparameter tuning."""

    @pytest.mark.slow
    def test_tune_default_grid(self, classifier, sample_features):
        """Test tuning with default parameter grid."""
        results = classifier.tune_hyperparameters(sample_features, cv_folds=3)

        assert "best_params" in results
        assert "best_score" in results
        assert "cv_results" in results
        assert classifier.is_trained is True
        assert 0 <= results["best_score"] <= 1

    @pytest.mark.slow
    def test_tune_custom_grid(self, classifier, sample_features):
        """Test tuning with custom parameter grid."""
        param_grid = {
            "n_estimators": [10, 20],
            "max_depth": [5, 10],
        }

        results = classifier.tune_hyperparameters(
            sample_features, param_grid=param_grid, cv_folds=3
        )

        assert "n_estimators" in results["best_params"]
        assert "max_depth" in results["best_params"]
        assert results["best_params"]["n_estimators"] in [10, 20]
        assert results["best_params"]["max_depth"] in [5, 10]


class TestModelPersistence:
    """Test model save/load functionality."""

    def test_save_untrained_model(self, classifier, tmp_path):
        """Test saving untrained model raises error."""
        filepath = tmp_path / "model.pkl"
        with pytest.raises(RuntimeError, match="Cannot save untrained classifier"):
            classifier.save(filepath)

    def test_save_and_load(self, classifier, sample_features, tmp_path):
        """Test saving and loading trained model."""
        filepath = tmp_path / "model.pkl"

        # Train and save
        classifier.train(sample_features, validate=False)
        original_prediction = classifier.classify_gait(sample_features[0])
        classifier.save(filepath)

        assert filepath.exists()

        # Load and verify
        loaded_classifier = RFGaitClassifier.load(filepath)
        assert loaded_classifier.is_trained is True
        assert loaded_classifier.config.n_estimators == classifier.config.n_estimators
        assert list(loaded_classifier.classes_) == list(classifier.classes_)

        # Verify predictions match
        loaded_prediction = loaded_classifier.classify_gait(sample_features[0])
        assert loaded_prediction["predicted_condition"] == original_prediction["predicted_condition"]
        assert loaded_prediction["confidence"] == pytest.approx(
            original_prediction["confidence"], abs=1e-6
        )

    def test_save_creates_directory(self, classifier, sample_features, tmp_path):
        """Test save creates parent directories."""
        filepath = tmp_path / "subdir" / "model.pkl"

        classifier.train(sample_features, validate=False)
        classifier.save(filepath)

        assert filepath.exists()
        assert filepath.parent.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_class_training(self, classifier):
        """Test training with single class."""
        features = [
            GaitFeatureVector(
                left_hip_mean=45,
                left_knee_mean=60,
                left_ankle_mean=20,
                right_hip_mean=45,
                right_knee_mean=60,
                right_ankle_mean=20,
                condition_label="normal",
            )
            for _ in range(10)
        ]

        # Should train but may have warnings
        metrics = classifier.train(features, validate=False)
        assert classifier.is_trained is True
        assert len(classifier.classes_) == 1

    def test_imbalanced_classes(self, classifier):
        """Test training with imbalanced classes."""
        features = []

        # 50 normal samples
        for i in range(50):
            features.append(
                GaitFeatureVector(
                    left_hip_mean=45 + np.random.randn(),
                    left_knee_mean=60 + np.random.randn(),
                    left_ankle_mean=20 + np.random.randn(),
                    right_hip_mean=45 + np.random.randn(),
                    right_knee_mean=60 + np.random.randn(),
                    right_ankle_mean=20 + np.random.randn(),
                    condition_label="normal",
                )
            )

        # 5 stroke samples
        for i in range(5):
            features.append(
                GaitFeatureVector(
                    left_hip_mean=40 + np.random.randn(),
                    left_knee_mean=50 + np.random.randn(),
                    left_ankle_mean=15 + np.random.randn(),
                    right_hip_mean=50 + np.random.randn(),
                    right_knee_mean=65 + np.random.randn(),
                    right_ankle_mean=25 + np.random.randn(),
                    hip_asymmetry=10,
                    condition_label="stroke",
                )
            )

        # Should handle imbalanced data with class_weight='balanced'
        metrics = classifier.train(features, validate=False)
        assert classifier.is_trained is True
        assert "class_distribution" in metrics

    def test_missing_feature_values(self, classifier, sample_features):
        """Test classification with missing feature values (defaults to 0)."""
        classifier.train(sample_features, validate=False)

        incomplete_dict = {
            "left_hip_mean": 45.0,
            "left_knee_mean": 60.0,
            # Missing other features
        }

        result = classifier.classify_gait(incomplete_dict)
        assert "predicted_condition" in result
