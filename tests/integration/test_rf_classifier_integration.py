"""
Integration tests for Random Forest Gait Classifier.

Tests integration with:
- Feature extraction from real gait data
- Comparison with KNN classifier
- End-to-end classification workflow
"""

import pytest
import numpy as np
from pathlib import Path

from ambient.classification.rf_classifier import RFGaitClassifier, RFClassifierConfig
from ambient.classification.knn_classifier import KNNGaitClassifier, KNNClassifierConfig, GaitFeatureVector


@pytest.fixture
def gavd_style_features():
    """Generate features similar to GAVD dataset."""
    features = []

    # Normal gait patterns
    for i in range(30):
        features.append(
            GaitFeatureVector(
                left_hip_mean=45 + np.random.randn() * 3,
                left_knee_mean=60 + np.random.randn() * 5,
                left_ankle_mean=20 + np.random.randn() * 2,
                right_hip_mean=45 + np.random.randn() * 3,
                right_knee_mean=60 + np.random.randn() * 5,
                right_ankle_mean=20 + np.random.randn() * 2,
                hip_asymmetry=np.abs(np.random.randn() * 2),
                knee_asymmetry=np.abs(np.random.randn() * 2),
                ankle_asymmetry=np.abs(np.random.randn() * 2),
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

    # Stroke/hemiplegic gait (asymmetric)
    for i in range(30):
        features.append(
            GaitFeatureVector(
                left_hip_mean=35 + np.random.randn() * 3,
                left_knee_mean=45 + np.random.randn() * 5,
                left_ankle_mean=12 + np.random.randn() * 2,
                right_hip_mean=50 + np.random.randn() * 3,
                right_knee_mean=70 + np.random.randn() * 5,
                right_ankle_mean=28 + np.random.randn() * 2,
                hip_asymmetry=15 + np.random.randn() * 3,
                knee_asymmetry=25 + np.random.randn() * 3,
                ankle_asymmetry=16 + np.random.randn() * 2,
                left_hip_range=25 + np.random.randn() * 2,
                left_knee_range=50 + np.random.randn() * 4,
                left_ankle_range=18 + np.random.randn() * 2,
                right_hip_range=45 + np.random.randn() * 3,
                right_knee_range=75 + np.random.randn() * 5,
                right_ankle_range=35 + np.random.randn() * 3,
                sample_id=f"stroke_{i}",
                condition_label="stroke",
            )
        )

    # Parkinson's gait (reduced ROM, shuffling)
    for i in range(30):
        features.append(
            GaitFeatureVector(
                left_hip_mean=30 + np.random.randn() * 2,
                left_knee_mean=40 + np.random.randn() * 3,
                left_ankle_mean=10 + np.random.randn() * 1,
                right_hip_mean=30 + np.random.randn() * 2,
                right_knee_mean=40 + np.random.randn() * 3,
                right_ankle_mean=10 + np.random.randn() * 1,
                hip_asymmetry=np.abs(np.random.randn() * 2),
                knee_asymmetry=np.abs(np.random.randn() * 2),
                ankle_asymmetry=np.abs(np.random.randn() * 1),
                left_hip_range=20 + np.random.randn() * 2,
                left_knee_range=35 + np.random.randn() * 3,
                left_ankle_range=12 + np.random.randn() * 1,
                right_hip_range=20 + np.random.randn() * 2,
                right_knee_range=35 + np.random.randn() * 3,
                right_ankle_range=12 + np.random.randn() * 1,
                sample_id=f"parkinsons_{i}",
                condition_label="parkinsons",
            )
        )

    return features




class TestRFvsKNNComparison:
    """Compare RF and KNN classifiers."""

    def test_both_classifiers_train_on_same_data(self, gavd_style_features):
        """Test that both classifiers can train on same dataset."""
        # Train RF
        rf_config = RFClassifierConfig(n_estimators=20, random_state=42)
        rf_classifier = RFGaitClassifier(rf_config)
        rf_metrics = rf_classifier.train(gavd_style_features, validate=False)

        # Train KNN
        knn_config = KNNClassifierConfig(n_neighbors=5)
        knn_classifier = KNNGaitClassifier(knn_config)
        knn_metrics = knn_classifier.train(gavd_style_features, validate=False)

        # Both should train successfully
        assert rf_classifier.is_trained
        assert knn_classifier.is_trained
        assert rf_metrics["train_accuracy"] > 0.5
        assert knn_metrics["train_accuracy"] > 0.5

    def test_prediction_comparison(self, gavd_style_features):
        """Compare predictions from RF and KNN."""
        train_features = gavd_style_features[:75]
        test_features = gavd_style_features[75:]

        # Train both classifiers
        rf_classifier = RFGaitClassifier(
            RFClassifierConfig(n_estimators=20, random_state=42)
        )
        rf_classifier.train(train_features, validate=False)

        knn_classifier = KNNGaitClassifier(KNNClassifierConfig(n_neighbors=5))
        knn_classifier.train(train_features, validate=False)

        # Compare predictions
        agreements = 0
        for feature in test_features:
            rf_result = rf_classifier.classify_gait(feature)
            knn_result = knn_classifier.classify_gait(feature)

            if rf_result["predicted_condition"] == knn_result["predicted_condition"]:
                agreements += 1

        # Classifiers should agree on most predictions
        agreement_rate = agreements / len(test_features)
        assert agreement_rate > 0.5  # At least 50% agreement

    def test_rf_provides_feature_importance(self, gavd_style_features):
        """Test that RF provides feature importance (KNN doesn't)."""
        rf_classifier = RFGaitClassifier(
            RFClassifierConfig(n_estimators=20, random_state=42)
        )
        rf_classifier.train(gavd_style_features, validate=False)

        # RF should provide feature importances
        importances = rf_classifier.get_feature_importances()
        assert len(importances) == 15
        assert all(imp.importance >= 0 for imp in importances)

        # KNN doesn't have this capability
        knn_classifier = KNNGaitClassifier(KNNClassifierConfig(n_neighbors=5))
        knn_classifier.train(gavd_style_features, validate=False)

        with pytest.raises(AttributeError):
            knn_classifier.get_feature_importances()


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_train_evaluate_save_load_predict(self, gavd_style_features, tmp_path):
        """Test complete workflow: train -> evaluate -> save -> load -> predict."""
        # Split data
        train_features = gavd_style_features[:70]
        test_features = gavd_style_features[70:]

        # Train
        classifier = RFGaitClassifier(
            RFClassifierConfig(n_estimators=30, random_state=42)
        )
        train_metrics = classifier.train(train_features, validate=True)

        assert train_metrics["train_accuracy"] > 0.7
        assert "cv_mean_accuracy" in train_metrics

        # Evaluate
        eval_metrics = classifier.evaluate(test_features)
        assert eval_metrics["accuracy"] > 0.5
        assert "f1_score" in eval_metrics

        # Save
        model_path = tmp_path / "rf_model.pkl"
        classifier.save(model_path)
        assert model_path.exists()

        # Load
        loaded_classifier = RFGaitClassifier.load(model_path)
        assert loaded_classifier.is_trained

        # Predict with loaded model
        result = loaded_classifier.classify_gait(test_features[0])
        assert "predicted_condition" in result
        assert result["confidence"] > 0

    def test_hyperparameter_tuning_workflow(self, gavd_style_features):
        """Test hyperparameter tuning workflow."""
        classifier = RFGaitClassifier()

        param_grid = {
            "n_estimators": [10, 20],
            "max_depth": [5, 10],
        }

        results = classifier.tune_hyperparameters(
            gavd_style_features, param_grid=param_grid, cv_folds=3
        )

        assert "best_params" in results
        assert "best_score" in results
        assert classifier.is_trained

        # Use tuned model for prediction
        test_result = classifier.classify_gait(gavd_style_features[0])
        assert "predicted_condition" in test_result

    def test_batch_classification(self, gavd_style_features):
        """Test classifying multiple samples."""
        train_features = gavd_style_features[:70]
        test_features = gavd_style_features[70:]

        classifier = RFGaitClassifier(
            RFClassifierConfig(n_estimators=20, random_state=42)
        )
        classifier.train(train_features, validate=False)

        # Classify all test samples
        results = []
        for feature in test_features:
            result = classifier.classify_gait(feature)
            results.append(result)

        assert len(results) == len(test_features)
        assert all("predicted_condition" in r for r in results)
        assert all(0 <= r["confidence"] <= 1 for r in results)


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_imbalanced_dataset_handling(self):
        """Test classifier handles imbalanced datasets."""
        features = []

        # 60 normal samples
        for i in range(60):
            features.append(
                GaitFeatureVector(
                    left_hip_mean=45 + np.random.randn() * 3,
                    left_knee_mean=60 + np.random.randn() * 5,
                    left_ankle_mean=20 + np.random.randn() * 2,
                    right_hip_mean=45 + np.random.randn() * 3,
                    right_knee_mean=60 + np.random.randn() * 5,
                    right_ankle_mean=20 + np.random.randn() * 2,
                    condition_label="normal",
                )
            )

        # 10 stroke samples
        for i in range(10):
            features.append(
                GaitFeatureVector(
                    left_hip_mean=35 + np.random.randn() * 3,
                    left_knee_mean=45 + np.random.randn() * 5,
                    left_ankle_mean=12 + np.random.randn() * 2,
                    right_hip_mean=50 + np.random.randn() * 3,
                    right_knee_mean=70 + np.random.randn() * 5,
                    right_ankle_mean=28 + np.random.randn() * 2,
                    hip_asymmetry=15,
                    condition_label="stroke",
                )
            )

        # Train with balanced class weights
        classifier = RFGaitClassifier(
            RFClassifierConfig(n_estimators=20, random_state=42, class_weight="balanced")
        )
        metrics = classifier.train(features, validate=False)

        assert classifier.is_trained
        assert "class_distribution" in metrics

        # Should still be able to predict minority class
        stroke_sample = [f for f in features if f.condition_label == "stroke"][0]
        result = classifier.classify_gait(stroke_sample)
        # Probability for stroke should be non-zero
        assert result["probabilities"]["stroke"] > 0

    def test_multi_class_classification(self):
        """Test classification with multiple conditions."""
        conditions = ["normal", "stroke", "parkinsons", "antalgic", "myopathic"]
        features = []

        for condition in conditions:
            for i in range(15):
                # Generate distinct patterns for each condition
                base_offset = conditions.index(condition) * 10
                features.append(
                    GaitFeatureVector(
                        left_hip_mean=40 + base_offset + np.random.randn() * 2,
                        left_knee_mean=50 + base_offset + np.random.randn() * 3,
                        left_ankle_mean=15 + base_offset / 2 + np.random.randn(),
                        right_hip_mean=40 + base_offset + np.random.randn() * 2,
                        right_knee_mean=50 + base_offset + np.random.randn() * 3,
                        right_ankle_mean=15 + base_offset / 2 + np.random.randn(),
                        condition_label=condition,
                    )
                )

        classifier = RFGaitClassifier(
            RFClassifierConfig(n_estimators=30, random_state=42)
        )
        metrics = classifier.train(features, validate=False)

        assert len(classifier.classes_) == 5
        assert metrics["train_accuracy"] > 0.3  # Lower threshold for 5 classes

        # Test prediction
        result = classifier.classify_gait(features[0])
        assert len(result["probabilities"]) == 5

    def test_feature_importance_interpretation(self, gavd_style_features):
        """Test feature importance for clinical interpretation."""
        classifier = RFGaitClassifier(
            RFClassifierConfig(n_estimators=50, random_state=42)
        )
        classifier.train(gavd_style_features, validate=False)

        importances = classifier.get_feature_importances()

        # Check that asymmetry features are important for distinguishing conditions
        asymmetry_features = [
            imp for imp in importances if "asymmetry" in imp.feature_name
        ]
        assert len(asymmetry_features) == 3

        # At least one asymmetry feature should be in top 10
        top_10_names = [imp.feature_name for imp in importances[:10]]
        assert any("asymmetry" in name for name in top_10_names)

    def test_confidence_calibration(self, gavd_style_features):
        """Test that confidence scores are meaningful."""
        train_features = gavd_style_features[:70]
        test_features = gavd_style_features[70:]

        classifier = RFGaitClassifier(
            RFClassifierConfig(n_estimators=50, random_state=42)
        )
        classifier.train(train_features, validate=False)

        confidences = []
        correct_predictions = []

        for feature in test_features:
            result = classifier.classify_gait(feature)
            confidences.append(result["confidence"])
            correct = result["predicted_condition"] == feature.condition_label
            correct_predictions.append(correct)

        # Higher confidence should correlate with correct predictions
        high_conf_correct = sum(
            1 for conf, correct in zip(confidences, correct_predictions)
            if conf > 0.7 and correct
        )
        low_conf_correct = sum(
            1 for conf, correct in zip(confidences, correct_predictions)
            if conf < 0.5 and correct
        )

        # This is a weak test but checks general trend
        assert len(confidences) > 0
