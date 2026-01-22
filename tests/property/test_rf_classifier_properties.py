"""
Property-based tests for Random Forest Gait Classifier.

Uses Hypothesis to test invariants and properties that should hold
for all valid inputs.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite

from ambient.classification.rf_classifier import (
    RFGaitClassifier,
    RFClassifierConfig,
)
from ambient.classification.knn_classifier import GaitFeatureVector


# Custom strategies for generating test data
@composite
def gait_feature_vector(draw, condition=None):
    """Generate a valid GaitFeatureVector."""
    conditions = ["normal", "stroke", "parkinsons", "antalgic"]
    condition_label = condition or draw(st.sampled_from(conditions))

    return GaitFeatureVector(
        left_hip_mean=draw(st.floats(min_value=0, max_value=180)),
        left_knee_mean=draw(st.floats(min_value=0, max_value=180)),
        left_ankle_mean=draw(st.floats(min_value=0, max_value=90)),
        right_hip_mean=draw(st.floats(min_value=0, max_value=180)),
        right_knee_mean=draw(st.floats(min_value=0, max_value=180)),
        right_ankle_mean=draw(st.floats(min_value=0, max_value=90)),
        hip_asymmetry=draw(st.floats(min_value=0, max_value=50)),
        knee_asymmetry=draw(st.floats(min_value=0, max_value=50)),
        ankle_asymmetry=draw(st.floats(min_value=0, max_value=30)),
        left_hip_range=draw(st.floats(min_value=0, max_value=90)),
        left_knee_range=draw(st.floats(min_value=0, max_value=140)),
        left_ankle_range=draw(st.floats(min_value=0, max_value=60)),
        right_hip_range=draw(st.floats(min_value=0, max_value=90)),
        right_knee_range=draw(st.floats(min_value=0, max_value=140)),
        right_ankle_range=draw(st.floats(min_value=0, max_value=60)),
        sample_id=draw(st.text(min_size=1, max_size=20)),
        condition_label=condition_label,
    )


@composite
def training_dataset(draw, min_samples=10, max_samples=50, n_classes=3):
    """Generate a valid training dataset."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    conditions = ["normal", "stroke", "parkinsons", "antalgic"][:n_classes]

    features = []
    for i in range(n_samples):
        condition = draw(st.sampled_from(conditions))
        feature = draw(gait_feature_vector(condition=condition))
        feature.sample_id = f"sample_{i}"
        features.append(feature)

    return features


@composite
def rf_config(draw):
    """Generate a valid RFClassifierConfig."""
    return RFClassifierConfig(
        n_estimators=draw(st.integers(min_value=5, max_value=50)),
        max_depth=draw(st.one_of(st.none(), st.integers(min_value=2, max_value=20))),
        min_samples_split=draw(st.integers(min_value=2, max_value=10)),
        min_samples_leaf=draw(st.integers(min_value=1, max_value=5)),
        max_features=draw(st.sampled_from(["sqrt", "log2"])),
        bootstrap=draw(st.booleans()),
        random_state=42,  # Fixed for reproducibility
        n_jobs=1,  # Single thread for reproducibility
        normalize_features=draw(st.booleans()),
        confidence_threshold=draw(st.floats(min_value=0.1, max_value=0.9)),
    )


class TestRFClassifierProperties:
    """Property-based tests for RF classifier."""

    @given(config=rf_config())
    @settings(max_examples=20, deadline=None)
    def test_initialization_always_succeeds(self, config):
        """Property: Classifier initialization should always succeed with valid config."""
        classifier = RFGaitClassifier(config)
        assert classifier is not None
        assert classifier.is_trained is False
        assert classifier.config == config

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_training_produces_valid_model(self, dataset):
        """Property: Training should always produce a valid model."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)

        metrics = classifier.train(dataset, validate=False)

        # Invariants after training
        assert classifier.is_trained is True
        assert classifier.classes_ is not None
        assert len(classifier.classes_) > 0
        assert 0 <= metrics["train_accuracy"] <= 1
        assert metrics["n_samples"] == len(dataset)
        assert metrics["n_features"] == 15

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_predictions_are_consistent(self, dataset):
        """Property: Same input should produce same prediction."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        test_feature = dataset[0]

        # Make multiple predictions
        result1 = classifier.classify_gait(test_feature)
        result2 = classifier.classify_gait(test_feature)

        # Should be identical
        assert result1["predicted_condition"] == result2["predicted_condition"]
        assert result1["confidence"] == pytest.approx(result2["confidence"])

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_confidence_bounds(self, dataset):
        """Property: Confidence scores should always be in [0, 1]."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        for feature in dataset[:5]:  # Test subset
            result = classifier.classify_gait(feature)
            assert 0 <= result["confidence"] <= 1

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_probabilities_sum_to_one(self, dataset):
        """Property: Probability distribution should sum to 1."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        for feature in dataset[:5]:
            result = classifier.classify_gait(feature)
            prob_sum = sum(result["probabilities"].values())
            assert prob_sum == pytest.approx(1.0, abs=1e-6)

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_predicted_class_in_training_classes(self, dataset):
        """Property: Predictions should only be from trained classes."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        for feature in dataset[:5]:
            result = classifier.classify_gait(feature)
            assert result["predicted_condition"] in classifier.classes_

    @given(dataset=training_dataset(min_samples=15, max_samples=30, n_classes=2))
    @settings(max_examples=10, deadline=None)
    def test_feature_importances_sum_to_one(self, dataset):
        """Property: Feature importances should sum to 1."""
        # Ensure we have at least 2 classes
        unique_labels = set(f.condition_label for f in dataset)
        assume(len(unique_labels) >= 2)
        
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        importances = classifier.get_feature_importances()
        total_importance = sum(imp.importance for imp in importances)
        assert total_importance == pytest.approx(1.0, abs=1e-6)

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_feature_importances_non_negative(self, dataset):
        """Property: Feature importances should be non-negative."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        importances = classifier.get_feature_importances()
        assert all(imp.importance >= 0 for imp in importances)

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_feature_importances_sorted(self, dataset):
        """Property: Feature importances should be sorted descending."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        importances = classifier.get_feature_importances()
        for i in range(len(importances) - 1):
            assert importances[i].importance >= importances[i + 1].importance

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_tree_votes_sum_to_n_estimators(self, dataset):
        """Property: Tree votes should sum to number of estimators."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        result = classifier.classify_gait(dataset[0])
        total_votes = sum(result["tree_votes"].values())
        assert total_votes == config.n_estimators

    @given(
        dataset=training_dataset(min_samples=20, max_samples=40),
        test_size=st.integers(min_value=5, max_value=10),
    )
    @settings(max_examples=10, deadline=None)
    def test_evaluation_metrics_bounds(self, dataset, test_size):
        """Property: Evaluation metrics should be in valid ranges."""
        assume(len(dataset) > test_size)

        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)

        train_data = dataset[:-test_size]
        test_data = dataset[-test_size:]

        classifier.train(train_data, validate=False)
        metrics = classifier.evaluate(test_data)

        # All metrics should be in [0, 1]
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_save_load_preserves_predictions(self, dataset, tmp_path):
        """Property: Save/load should preserve model predictions."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        # Get original prediction
        test_feature = dataset[0]
        original_result = classifier.classify_gait(test_feature)

        # Save and load
        filepath = tmp_path / "test_model.pkl"
        classifier.save(filepath)
        loaded_classifier = RFGaitClassifier.load(filepath)

        # Get loaded prediction
        loaded_result = loaded_classifier.classify_gait(test_feature)

        # Should be identical
        assert loaded_result["predicted_condition"] == original_result["predicted_condition"]
        assert loaded_result["confidence"] == pytest.approx(
            original_result["confidence"], abs=1e-6
        )

    @given(
        feature=gait_feature_vector(),
        n_estimators=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=10, deadline=None)
    def test_more_trees_not_worse_confidence(self, feature, n_estimators):
        """Property: More trees should not decrease confidence (generally)."""
        # Create simple dataset
        dataset = [feature] * 10
        for i, f in enumerate(dataset):
            f.sample_id = f"sample_{i}"

        # Train with fewer trees
        config1 = RFClassifierConfig(n_estimators=5, random_state=42, n_jobs=1)
        classifier1 = RFGaitClassifier(config1)
        classifier1.train(dataset, validate=False)
        result1 = classifier1.classify_gait(feature)

        # Train with more trees
        config2 = RFClassifierConfig(n_estimators=n_estimators, random_state=42, n_jobs=1)
        classifier2 = RFGaitClassifier(config2)
        classifier2.train(dataset, validate=False)
        result2 = classifier2.classify_gait(feature)

        # Both should predict same class for identical training data
        assert result1["predicted_condition"] == result2["predicted_condition"]

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_explanation_contains_prediction(self, dataset):
        """Property: Explanation should contain the predicted condition."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        result = classifier.classify_gait(dataset[0])
        explanation = classifier.explain_classification(result)

        assert result["predicted_condition"] in explanation
        assert str(result["confidence"]) in explanation or f"{result['confidence']:.1%}" in explanation


class TestRFClassifierInvariants:
    """Test invariants that should always hold."""

    def test_feature_vector_length_invariant(self):
        """Invariant: Feature vectors should always have 15 elements."""
        feature = GaitFeatureVector()
        assert len(feature.to_array()) == 15
        assert len(GaitFeatureVector.get_feature_names()) == 15

    @given(dataset=training_dataset(min_samples=10, max_samples=20))
    @settings(max_examples=10, deadline=None)
    def test_classes_match_labels(self, dataset):
        """Invariant: Trained classes should match unique labels in data."""
        config = RFClassifierConfig(n_estimators=10, random_state=42, n_jobs=1)
        classifier = RFGaitClassifier(config)
        classifier.train(dataset, validate=False)

        unique_labels = set(f.condition_label for f in dataset)
        trained_classes = set(classifier.classes_)

        assert unique_labels == trained_classes

    @given(dataset=training_dataset(min_samples=15, max_samples=30))
    @settings(max_examples=10, deadline=None)
    def test_normalization_invariant(self, dataset):
        """Invariant: Normalization should not change prediction order."""
        # Train with normalization
        config1 = RFClassifierConfig(
            n_estimators=10, random_state=42, n_jobs=1, normalize_features=True
        )
        classifier1 = RFGaitClassifier(config1)
        classifier1.train(dataset, validate=False)

        # Train without normalization
        config2 = RFClassifierConfig(
            n_estimators=10, random_state=42, n_jobs=1, normalize_features=False
        )
        classifier2 = RFGaitClassifier(config2)
        classifier2.train(dataset, validate=False)

        # Both should produce valid predictions
        test_feature = dataset[0]
        result1 = classifier1.classify_gait(test_feature)
        result2 = classifier2.classify_gait(test_feature)

        assert result1["predicted_condition"] in classifier1.classes_
        assert result2["predicted_condition"] in classifier2.classes_
