"""
Comprehensive tests for all gait classifiers.

Tests all classifiers (XGBoost, SVM, Logistic, DecisionTree, MLP, Ensemble)
with a unified test suite to ensure consistency and completeness.
"""

import pytest
import numpy as np
from pathlib import Path

from tests.ambient.classification.test_classifier_utils import (
    generate_sample_gait_features,
    assert_valid_classification_result,
    assert_valid_training_metrics,
    assert_valid_evaluation_metrics,
)

# Import all classifiers
from ambient.classification.xgboost_classifier import (
    XGBoostGaitClassifier,
    XGBoostClassifierConfig,
    XGBOOST_AVAILABLE,
)
from ambient.classification.svm_classifier import SVMGaitClassifier, SVMClassifierConfig
from ambient.classification.logistic_classifier import (
    LogisticGaitClassifier,
    LogisticClassifierConfig,
)
from ambient.classification.decisiontree_classifier import (
    DecisionTreeGaitClassifier,
    DecisionTreeClassifierConfig,
)
from ambient.classification.mlp_classifier import MLPGaitClassifier, MLPClassifierConfig
from ambient.classification.ensemble_classifier import (
    EnsembleGaitClassifier,
    EnsembleClassifierConfig,
    VotingStrategy,
)


# Classifier configurations for testing
CLASSIFIER_CONFIGS = [
    ("svm", SVMGaitClassifier, SVMClassifierConfig(C=1.0, max_iter=1000)),
    ("logistic", LogisticGaitClassifier, LogisticClassifierConfig(max_iter=1000)),
    ("decisiontree", DecisionTreeGaitClassifier, DecisionTreeClassifierConfig(max_depth=5)),
    ("mlp", MLPGaitClassifier, MLPClassifierConfig(hidden_layer_sizes=(50,), max_iter=100)),
]

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    CLASSIFIER_CONFIGS.append(
        ("xgboost", XGBoostGaitClassifier, XGBoostClassifierConfig(n_estimators=10))
    )


@pytest.fixture
def sample_features():
    """Generate sample features for testing."""
    return generate_sample_gait_features(n_normal=15, n_stroke=15, n_parkinsons=15)


class TestClassifierInitialization:
    """Test initialization of all classifiers."""

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_init_with_config(self, clf_name, clf_class, config):
        """Test initialization with custom config."""
        classifier = clf_class(config)
        assert classifier is not None
        assert classifier.is_trained is False
        assert classifier.config == config

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_init_default_config(self, clf_name, clf_class, config):
        """Test initialization with default config."""
        classifier = clf_class()
        assert classifier is not None
        assert classifier.is_trained is False


class TestClassifierTraining:
    """Test training for all classifiers."""

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_train_basic(self, clf_name, clf_class, config, sample_features):
        """Test basic training."""
        classifier = clf_class(config)
        metrics = classifier.train(sample_features, validate=False)

        assert classifier.is_trained is True
        assert_valid_training_metrics(metrics)
        assert len(classifier.classes_) == 3

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_train_with_validation(self, clf_name, clf_class, config, sample_features):
        """Test training with cross-validation."""
        classifier = clf_class(config)
        metrics = classifier.train(sample_features, validate=True)

        assert "cv_mean_accuracy" in metrics or "n_samples" in metrics
        assert classifier.is_trained is True

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_train_empty_features(self, clf_name, clf_class, config):
        """Test training with empty features raises error."""
        classifier = clf_class(config)
        with pytest.raises(ValueError, match="No training features"):
            classifier.train([])


class TestClassifierClassification:
    """Test classification for all classifiers."""

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_classify_before_training(self, clf_name, clf_class, config, sample_features):
        """Test classification before training raises error."""
        classifier = clf_class(config)
        with pytest.raises(RuntimeError, match="must be trained"):
            classifier.classify_gait(sample_features[0])

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_classify_basic(self, clf_name, clf_class, config, sample_features):
        """Test basic classification."""
        classifier = clf_class(config)
        classifier.train(sample_features, validate=False)

        result = classifier.classify_gait(sample_features[0])
        assert_valid_classification_result(result)
        assert result["predicted_condition"] in ["normal", "stroke", "parkinsons"]

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_classify_with_dict(self, clf_name, clf_class, config, sample_features):
        """Test classification with dictionary input."""
        classifier = clf_class(config)
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
        assert_valid_classification_result(result)


class TestClassifierEvaluation:
    """Test evaluation for all classifiers."""

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_evaluate_basic(self, clf_name, clf_class, config, sample_features):
        """Test basic evaluation."""
        train_features = sample_features[:35]
        test_features = sample_features[35:]

        classifier = clf_class(config)
        classifier.train(train_features, validate=False)
        metrics = classifier.evaluate(test_features)

        assert_valid_evaluation_metrics(metrics)
        assert metrics["n_test_samples"] == len(test_features)

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_evaluate_before_training(self, clf_name, clf_class, config, sample_features):
        """Test evaluation before training raises error."""
        classifier = clf_class(config)
        with pytest.raises(RuntimeError, match="must be trained"):
            classifier.evaluate(sample_features)


class TestClassifierPersistence:
    """Test save/load for all classifiers."""

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_save_and_load(self, clf_name, clf_class, config, sample_features, tmp_path):
        """Test saving and loading."""
        filepath = tmp_path / f"{clf_name}_model.pkl"

        # Train and save
        classifier = clf_class(config)
        classifier.train(sample_features, validate=False)
        original_pred = classifier.classify_gait(sample_features[0])
        classifier.save(filepath)

        assert filepath.exists()

        # Load and verify
        loaded_classifier = clf_class.load(filepath)
        assert loaded_classifier.is_trained is True
        assert list(loaded_classifier.classes_) == list(classifier.classes_)

        # Verify predictions match
        loaded_pred = loaded_classifier.classify_gait(sample_features[0])
        assert loaded_pred["predicted_condition"] == original_pred["predicted_condition"]


class TestClassifierExplanation:
    """Test explanation generation for all classifiers."""

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS)
    def test_explain_classification(self, clf_name, clf_class, config, sample_features):
        """Test explanation generation."""
        classifier = clf_class(config)
        classifier.train(sample_features, validate=False)

        result = classifier.classify_gait(sample_features[0])
        explanation = classifier.explain_classification(result)

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert result["predicted_condition"] in explanation


class TestEnsembleClassifier:
    """Specific tests for Ensemble classifier."""

    def test_ensemble_init(self):
        """Test ensemble initialization."""
        config = EnsembleClassifierConfig(
            classifiers=["logistic", "decisiontree"],
            voting_strategy=VotingStrategy.SOFT,
        )
        ensemble = EnsembleGaitClassifier(config)
        assert len(ensemble.base_classifiers) == 2

    def test_ensemble_train(self, sample_features):
        """Test ensemble training."""
        config = EnsembleClassifierConfig(
            classifiers=["logistic", "decisiontree"],
            voting_strategy=VotingStrategy.SOFT,
        )
        ensemble = EnsembleGaitClassifier(config)
        metrics = ensemble.train(sample_features, validate=False)

        assert ensemble.is_trained is True
        assert "classifiers" in metrics
        assert len(metrics["classifiers"]) == 2

    def test_ensemble_classify(self, sample_features):
        """Test ensemble classification."""
        config = EnsembleClassifierConfig(
            classifiers=["logistic", "decisiontree"],
            voting_strategy=VotingStrategy.SOFT,
        )
        ensemble = EnsembleGaitClassifier(config)
        ensemble.train(sample_features, validate=False)

        result = ensemble.classify_gait(sample_features[0])
        assert_valid_classification_result(result)
        assert "individual_predictions" in result
        assert "agreement" in result

    def test_ensemble_voting_strategies(self, sample_features):
        """Test different voting strategies."""
        for strategy in [VotingStrategy.HARD, VotingStrategy.SOFT, VotingStrategy.WEIGHTED]:
            config = EnsembleClassifierConfig(
                classifiers=["logistic", "decisiontree"],
                voting_strategy=strategy,
            )
            ensemble = EnsembleGaitClassifier(config)
            ensemble.train(sample_features, validate=False)

            result = ensemble.classify_gait(sample_features[0])
            assert_valid_classification_result(result)

    def test_ensemble_prediction_breakdown(self, sample_features):
        """Test prediction breakdown."""
        config = EnsembleClassifierConfig(
            classifiers=["logistic", "decisiontree"],
        )
        ensemble = EnsembleGaitClassifier(config)
        ensemble.train(sample_features, validate=False)

        breakdown = ensemble.get_prediction_breakdown(sample_features[0])
        assert len(breakdown) == 2
        assert all("predicted_condition" in pred for pred in breakdown.values())


@pytest.mark.slow
class TestClassifierHyperparameterTuning:
    """Test hyperparameter tuning (marked as slow)."""

    @pytest.mark.parametrize("clf_name,clf_class,config", CLASSIFIER_CONFIGS[:2])  # Test subset
    def test_tune_hyperparameters(self, clf_name, clf_class, config, sample_features):
        """Test hyperparameter tuning."""
        classifier = clf_class(config)

        # Small parameter grid for speed
        if clf_name == "svm":
            param_grid = {"C": [0.1, 1.0], "gamma": ["scale"]}
        elif clf_name == "logistic":
            param_grid = {"C": [0.1, 1.0], "penalty": ["l2"]}
        else:
            param_grid = None

        results = classifier.tune_hyperparameters(
            sample_features, param_grid=param_grid, cv_folds=3
        )

        assert "best_params" in results
        assert "best_score" in results
        assert classifier.is_trained is True


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
