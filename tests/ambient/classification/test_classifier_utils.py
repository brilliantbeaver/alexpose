"""
Shared test utilities for classifier tests.

Provides common fixtures and helper functions to reduce code duplication
across classifier test files.
"""

import pytest
import numpy as np
from typing import List

from ambient.classification.knn_classifier import GaitFeatureVector


def generate_sample_gait_features(
    n_normal: int = 20,
    n_stroke: int = 20,
    n_parkinsons: int = 20,
    random_seed: int = 42,
) -> List[GaitFeatureVector]:
    """
    Generate sample gait features for testing.
    
    Args:
        n_normal: Number of normal gait samples
        n_stroke: Number of stroke gait samples
        n_parkinsons: Number of Parkinson's gait samples
        random_seed: Random seed for reproducibility
        
    Returns:
        List of GaitFeatureVector objects
    """
    np.random.seed(random_seed)
    features = []

    # Normal gait samples
    for i in range(n_normal):
        features.append(
            GaitFeatureVector(
                left_hip_mean=45 + np.random.randn() * 5,
                left_knee_mean=60 + np.random.randn() * 5,
                left_ankle_mean=20 + np.random.randn() * 3,
                right_hip_mean=45 + np.random.randn() * 5,
                right_knee_mean=60 + np.random.randn() * 5,
                right_ankle_mean=20 + np.random.randn() * 3,
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
    for i in range(n_stroke):
        features.append(
            GaitFeatureVector(
                left_hip_mean=35 + np.random.randn() * 5,
                left_knee_mean=45 + np.random.randn() * 5,
                left_ankle_mean=12 + np.random.randn() * 3,
                right_hip_mean=50 + np.random.randn() * 5,
                right_knee_mean=70 + np.random.randn() * 5,
                right_ankle_mean=28 + np.random.randn() * 3,
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
    for i in range(n_parkinsons):
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


@pytest.fixture
def sample_features():
    """Fixture providing sample gait features."""
    return generate_sample_gait_features()


@pytest.fixture
def small_sample_features():
    """Fixture providing small sample for quick tests."""
    return generate_sample_gait_features(n_normal=5, n_stroke=5, n_parkinsons=5)


def assert_valid_classification_result(result: dict):
    """Assert that a classification result has valid structure."""
    assert "predicted_condition" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert isinstance(result["predicted_condition"], str)
    assert 0 <= result["confidence"] <= 1
    assert isinstance(result["probabilities"], dict)
    assert all(0 <= p <= 1 for p in result["probabilities"].values())


def assert_valid_training_metrics(metrics: dict):
    """Assert that training metrics have valid structure."""
    assert "train_accuracy" in metrics
    assert "n_samples" in metrics
    assert "n_features" in metrics
    assert "classes" in metrics
    assert 0 <= metrics["train_accuracy"] <= 1
    assert metrics["n_samples"] > 0
    assert metrics["n_features"] == 15  # Standard gait feature count


def assert_valid_evaluation_metrics(metrics: dict):
    """Assert that evaluation metrics have valid structure."""
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert all(0 <= metrics[k] <= 1 for k in ["accuracy", "precision", "recall", "f1_score"])
