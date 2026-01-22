"""
Random Forest Gait Classifier Example

This example demonstrates how to use the Random Forest classifier for gait analysis.
It shows training, evaluation, feature importance analysis, and model persistence.

Author: AlexPose Team
"""

import numpy as np
from pathlib import Path

from ambient.classification.rf_classifier import (
    RFGaitClassifier,
    RFClassifierConfig,
)
from ambient.classification.knn_classifier import GaitFeatureVector


def generate_sample_data():
    """Generate sample gait features for demonstration."""
    np.random.seed(42)
    features = []

    # Normal gait samples
    print("Generating normal gait samples...")
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

    # Stroke gait samples (asymmetric)
    print("Generating stroke gait samples...")
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

    # Parkinson's gait samples (reduced ROM)
    print("Generating Parkinson's gait samples...")
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


def example_basic_training():
    """Example 1: Basic training and classification."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Training and Classification")
    print("=" * 60)

    # Generate data
    features = generate_sample_data()

    # Configure classifier
    config = RFClassifierConfig(
        n_estimators=100,
        max_depth=20,
        random_state=42,
    )

    # Train classifier
    print("\nTraining Random Forest classifier...")
    classifier = RFGaitClassifier(config)
    metrics = classifier.train(features, validate=True)

    print(f"\nTraining Results:")
    print(f"  Training Accuracy: {metrics['train_accuracy']:.3f}")
    print(f"  CV Accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")
    print(f"  Number of Trees: {metrics['n_estimators']}")
    print(f"  Classes: {metrics['classes']}")

    # Classify a test sample
    print("\nClassifying test sample...")
    test_sample = features[0]
    result = classifier.classify_gait(test_sample)

    print(f"\nClassification Result:")
    print(f"  Predicted: {result['predicted_condition']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for condition, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {condition:15s}: {prob:.2%}")

    return classifier, features


def example_feature_importance(classifier):
    """Example 2: Analyze feature importance."""
    print("\n" + "=" * 60)
    print("Example 2: Feature Importance Analysis")
    print("=" * 60)

    # Get feature importances
    importances = classifier.get_feature_importances()

    print("\nTop 10 Most Important Features:")
    for imp in importances[:10]:
        bar = "█" * int(imp.importance * 50)
        print(f"  {imp.rank:2d}. {imp.feature_name:20s}: {imp.importance:.4f} {bar}")

    # Analyze by feature type
    print("\nFeature Importance by Type:")

    asymmetry_features = [imp for imp in importances if "asymmetry" in imp.feature_name]
    range_features = [imp for imp in importances if "range" in imp.feature_name]
    mean_features = [imp for imp in importances if "mean" in imp.feature_name]

    print(f"  Asymmetry features: {sum(imp.importance for imp in asymmetry_features):.3f}")
    print(f"  Range features: {sum(imp.importance for imp in range_features):.3f}")
    print(f"  Mean features: {sum(imp.importance for imp in mean_features):.3f}")


def example_evaluation(classifier, features):
    """Example 3: Evaluate classifier performance."""
    print("\n" + "=" * 60)
    print("Example 3: Model Evaluation")
    print("=" * 60)

    # Split data for evaluation
    train_features = features[:75]
    test_features = features[75:]

    # Retrain on training set
    print("\nRetraining on training set...")
    classifier.train(train_features, validate=False)

    # Evaluate on test set
    print("Evaluating on test set...")
    metrics = classifier.evaluate(test_features)

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"  Classes: {metrics['classes']}")
    conf_matrix = np.array(metrics['confusion_matrix'])
    for i, row in enumerate(conf_matrix):
        print(f"  {metrics['classes'][i]:15s}: {row}")


def example_explanation(classifier, features):
    """Example 4: Generate classification explanations."""
    print("\n" + "=" * 60)
    print("Example 4: Classification Explanation")
    print("=" * 60)

    # Classify a sample
    test_sample = features[35]  # Stroke sample
    result = classifier.classify_gait(test_sample)

    # Generate explanation
    explanation = classifier.explain_classification(result)

    print(f"\nDetailed Explanation:")
    print(explanation)


def example_model_persistence(classifier):
    """Example 5: Save and load model."""
    print("\n" + "=" * 60)
    print("Example 5: Model Persistence")
    print("=" * 60)

    # Save model
    model_path = Path("data/models/rf_example.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to {model_path}...")
    classifier.save(model_path)
    print(f"Model saved successfully!")

    # Load model
    print(f"\nLoading model from {model_path}...")
    loaded_classifier = RFGaitClassifier.load(model_path)
    print(f"Model loaded successfully!")

    # Verify loaded model works
    print("\nVerifying loaded model...")
    test_feature = GaitFeatureVector(
        left_hip_mean=45.0,
        left_knee_mean=60.0,
        left_ankle_mean=20.0,
        right_hip_mean=45.0,
        right_knee_mean=60.0,
        right_ankle_mean=20.0,
        hip_asymmetry=1.0,
        knee_asymmetry=1.0,
        ankle_asymmetry=1.0,
        left_hip_range=40.0,
        left_knee_range=70.0,
        left_ankle_range=30.0,
        right_hip_range=40.0,
        right_knee_range=70.0,
        right_ankle_range=30.0,
    )

    result = loaded_classifier.classify_gait(test_feature)
    print(f"Prediction: {result['predicted_condition']} (confidence: {result['confidence']:.2%})")


def example_hyperparameter_tuning(features):
    """Example 6: Hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("Example 6: Hyperparameter Tuning")
    print("=" * 60)

    # Create classifier
    classifier = RFGaitClassifier()

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
    }

    print("\nTuning hyperparameters...")
    print(f"Parameter grid: {param_grid}")

    # Tune hyperparameters
    results = classifier.tune_hyperparameters(
        features,
        param_grid=param_grid,
        cv_folds=3
    )

    print(f"\nTuning Results:")
    print(f"  Best Parameters: {results['best_params']}")
    print(f"  Best CV Score: {results['best_score']:.3f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Random Forest Gait Classifier Examples")
    print("=" * 60)

    # Example 1: Basic training
    classifier, features = example_basic_training()

    # Example 2: Feature importance
    example_feature_importance(classifier)

    # Example 3: Evaluation
    example_evaluation(classifier, features)

    # Example 4: Explanation
    example_explanation(classifier, features)

    # Example 5: Model persistence
    example_model_persistence(classifier)

    # Example 6: Hyperparameter tuning
    example_hyperparameter_tuning(features)

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
