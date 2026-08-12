"""
K-Nearest Neighbors (KNN) Gait Classifier

This module provides a KNN-based classifier for gait condition classification
using extracted gait features. It follows SOLID principles and integrates
seamlessly with the existing ambient analysis pipeline.

The classifier uses mean joint angles and left-right asymmetry features
to classify gait patterns into conditions such as:
- Normal/Healthy gait
- Stroke (hemiplegic gait)
- Parkinson's disease
- Antalgic gait


Ankle range of motion (ROM) is particularly important for:
- Detecting foot drop (common in stroke/hemiplegic gait)
- Identifying reduced dorsiflexion (Parkinson's shuffling gait)
- Measuring push-off power (antalgic gait compensation)
- Assessing overall gait fluidity

Author: AlexPose Team
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from loguru import logger

from ambient.classification.base_classifier import BaseGaitClassifier, BaseClassifierConfig
from ambient.classification.features import GaitFeatureVector


@dataclass
class KNNClassifierConfig(BaseClassifierConfig):
    """Configuration for KNN classifier."""

    n_neighbors: int = 5
    weights: str = "distance"  # 'uniform' or 'distance'
    metric: str = "euclidean"  # 'euclidean', 'manhattan', 'minkowski'
    algorithm: str = "auto"  # 'auto', 'ball_tree', 'kd_tree', 'brute'


class KNNGaitClassifier(BaseGaitClassifier):
    """
    K-Nearest Neighbors classifier for gait condition classification.

    This classifier uses mean joint angles and asymmetry features to classify
    gait patterns. It inherits from BaseGaitClassifier for consistency with
    other classification approaches in the system.

    Key Features:
    - Feature normalization using StandardScaler
    - Distance-weighted voting for better accuracy
    - Confidence scores based on neighbor agreement
    - Cross-validation support for model evaluation
    - Hyperparameter tuning via grid search

    Example:
        >>> from ambient.classification.knn_classifier import KNNGaitClassifier
        >>> from ambient.pose.joint_angles import get_joint_angles
        >>>
        >>> # Train classifier
        >>> classifier = KNNGaitClassifier(n_neighbors=5)
        >>> classifier.train(training_features, training_labels)
        >>>
        >>> # Classify new sample
        >>> result = classifier.classify_gait(test_features)
        >>> print(f"Predicted: {result['predicted_condition']}")
        >>> print(f"Confidence: {result['confidence']:.2f}")
    """

    def __init__(self, config: Optional[KNNClassifierConfig] = None):
        """
        Initialize KNN gait classifier.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        config = config or KNNClassifierConfig()
        super().__init__(config)
        self.config: KNNClassifierConfig = config

        logger.info(f"KNN classifier initialized with k={self.config.n_neighbors}")

    def _create_model(self):
        """Create KNN model."""
        return KNeighborsClassifier(
            n_neighbors=self.config.n_neighbors,
            weights=self.config.weights,
            metric=self.config.metric,
            algorithm=self.config.algorithm,
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters."""
        return {
            "n_neighbors": self.config.n_neighbors,
            "weights": self.config.weights,
            "metric": self.config.metric,
        }

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
        auto_remove_invalid: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the KNN classifier.

        Args:
            features: List of GaitFeatureVector objects
            labels: Optional list of condition labels (uses feature.condition_label if None)
            validate: Whether to perform cross-validation
            auto_remove_invalid: If True, automatically remove samples with NaN/Inf instead of raising error

        Returns:
            Dictionary with training metrics
            
        Raises:
            ValueError: If features contain NaN values and auto_remove_invalid=False
        """
        # Call base class train method (handles NaN validation)
        metrics = super().train(features, labels, validate, auto_remove_invalid)

        # Add KNN-specific metrics
        metrics["n_neighbors"] = self.config.n_neighbors

        return metrics

    def classify_gait(
        self,
        gait_features: Union[GaitFeatureVector, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify gait condition from features with KNN-specific information.

        Args:
            gait_features: GaitFeatureVector or dict with feature values
            context: Optional context information

        Returns:
            Dictionary with classification results including:
            - predicted_condition: Predicted condition label
            - confidence: Confidence score (0-1)
            - probabilities: Dict mapping conditions to probabilities
            - neighbors: Information about nearest neighbors
        """
        # Call base class method
        result = super().classify_gait(gait_features, context)

        # Add KNN-specific information
        if hasattr(self.model, 'kneighbors'):
            # Convert to feature vector if needed
            feature_vec = (
                gait_features
                if isinstance(gait_features, GaitFeatureVector)
                else self._dict_to_feature_vector(gait_features)
            )
            
            # Extract and normalize features
            X = feature_vec.to_array().reshape(1, -1)
            if self.scaler:
                X = self.scaler.transform(X)

            # Get nearest neighbors for explainability
            distances, indices = self.model.kneighbors(X)

            result["neighbors"] = {
                "distances": distances[0].tolist(),
                "indices": indices[0].tolist(),
                "k": self.config.n_neighbors,
            }

        return result

    def explain_classification(self, result: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of classification.

        Args:
            result: Classification result from classify_gait()

        Returns:
            Explanation string
        """
        # Get base explanation
        explanation = super().explain_classification(result)

        # Add neighbor information
        neighbors = result.get("neighbors", {})
        if neighbors:
            explanation += f"\n\nBased on {neighbors['k']} nearest neighbors\n"
            explanation += f"Average distance: {np.mean(neighbors['distances']):.3f}\n"

        return explanation

    def tune_hyperparameters(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        param_grid: Optional[Dict[str, List]] = None,
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search with cross-validation.

        Args:
            features: Training features
            labels: Training labels
            param_grid: Parameter grid for search. Uses defaults if None.

        Returns:
            Dictionary with best parameters and scores
        """
        X, y = self._prepare_features(features, fit_scaler=True)
        if labels:
            y = np.array(labels)

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            }

        logger.info("Starting hyperparameter tuning...")

        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=min(5, len(X)),
            scoring="accuracy",
            n_jobs=self.config.cv_n_jobs,
        )

        grid_search.fit(X, y)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self.classes_ = self.model.classes_

        results = {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": {
                "mean_test_scores": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_test_scores": grid_search.cv_results_["std_test_score"].tolist(),
                "params": grid_search.cv_results_["params"],
            },
        }

        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.3f}")

        return results
