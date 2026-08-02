"""
Random Forest Gait Classifier

This module provides a Random Forest-based classifier for gait condition classification
using extracted gait features. It follows SOLID principles and integrates seamlessly
with the existing ambient analysis pipeline.

The classifier uses mean joint angles, range of motion, and left-right asymmetry features
to classify gait patterns into conditions such as:
- Normal/Healthy gait
- Stroke (hemiplegic gait)
- Parkinson's disease
- Antalgic gait
- Cerebral palsy
- Myopathic gait
- Prosthetic gait

Random Forest advantages:
- Handles non-linear relationships between features
- Provides feature importance rankings
- Robust to outliers and noise
- Less prone to overfitting than single decision trees
- Can handle missing values gracefully

Author: AlexPose Team
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from loguru import logger

from ambient.classification.base_classifier import BaseGaitClassifier, BaseClassifierConfig
from ambient.classification.features import GaitFeatureVector


@dataclass
class RFClassifierConfig(BaseClassifierConfig):
    """Configuration for Random Forest classifier."""

    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"  # 'sqrt', 'log2', or float
    bootstrap: bool = True
    n_jobs: int = -1  # Use all available cores
    class_weight: Optional[str] = "balanced"  # Handle imbalanced classes


@dataclass
class FeatureImportance:
    """Feature importance information."""

    feature_name: str
    importance: float
    rank: int

    def __repr__(self) -> str:
        return f"{self.rank}. {self.feature_name}: {self.importance:.4f}"


class RFGaitClassifier(BaseGaitClassifier):
    """
    Random Forest classifier for gait condition classification.

    This classifier uses ensemble learning with multiple decision trees to classify
    gait patterns. It implements the BaseGaitClassifier interface for consistency with
    other classification approaches in the system.

    Key Features:
    - Ensemble of decision trees for robust predictions
    - Feature importance analysis for interpretability
    - Handles non-linear feature relationships
    - Built-in cross-validation and hyperparameter tuning
    - Confidence scores based on tree voting
    - Optional feature normalization
    - Balanced class weights for imbalanced datasets

    Example:
        >>> from ambient.classification.rf_classifier import RFGaitClassifier
        >>> from ambient.classification.knn_classifier import GaitFeatureVector
        >>>
        >>> # Train classifier
        >>> config = RFClassifierConfig(n_estimators=200, max_depth=10)
        >>> classifier = RFGaitClassifier(config)
        >>> classifier.train(training_features, training_labels)
        >>>
        >>> # Classify new sample
        >>> result = classifier.classify_gait(test_features)
        >>> print(f"Predicted: {result['predicted_condition']}")
        >>> print(f"Confidence: {result['confidence']:.2f}")
        >>>
        >>> # Analyze feature importance
        >>> importances = classifier.get_feature_importances()
        >>> for imp in importances[:5]:
        >>>     print(imp)
    """

    def __init__(self, config: Optional[RFClassifierConfig] = None):
        """
        Initialize Random Forest gait classifier.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        config = config or RFClassifierConfig()
        super().__init__(config)
        self.config: RFClassifierConfig = config
        self.feature_importances_ = None

        logger.info(
            f"Random Forest classifier initialized with {config.n_estimators} trees"
        )

    def _create_model(self):
        """Create Random Forest model."""
        return RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            class_weight=self.config.class_weight,
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters."""
        return {
            "feature_importances": self.feature_importances_,
            "n_estimators": self.config.n_estimators,
        }

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
        auto_remove_invalid: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the Random Forest classifier.

        Args:
            features: List of GaitFeatureVector objects
            labels: Optional list of condition labels (uses feature.condition_label if None)
            validate: Whether to perform cross-validation
            auto_remove_invalid: If True, automatically remove samples with NaN/Inf

        Returns:
            Dictionary with training metrics including:
            - train_accuracy: Training set accuracy
            - n_samples: Number of training samples
            - n_features: Number of features
            - classes: List of class labels
            - n_estimators: Number of trees in forest
            - cv_mean_accuracy: Cross-validation mean accuracy (if validate=True)
            - cv_std_accuracy: Cross-validation std accuracy (if validate=True)
            - feature_importances: Feature importance rankings
        """
        # Call base class train method (handles NaN validation)
        metrics = super().train(features, labels, validate, auto_remove_invalid)

        # Store feature importances
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_

            # Add feature importance to metrics
            importances = self.get_feature_importances()
            metrics["feature_importances"] = [
                {"feature": imp.feature_name, "importance": imp.importance, "rank": imp.rank}
                for imp in importances
            ]

            logger.info("Top 5 important features:")
            for imp in importances[:5]:
                logger.info(f"  {imp}")

        # Add RF-specific metrics
        metrics["n_estimators"] = self.config.n_estimators

        return metrics

    def classify_gait(
        self,
        gait_features: Union[GaitFeatureVector, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify gait condition from features with RF-specific information.

        Args:
            gait_features: GaitFeatureVector or dict with feature values
            context: Optional context information

        Returns:
            Dictionary with classification results including:
            - predicted_condition: Predicted condition label
            - confidence: Confidence score (0-1)
            - probabilities: Dict mapping conditions to probabilities
            - tree_votes: Voting distribution across trees
            - is_normal: Boolean indicating if gait is normal
            - feature_vector: Input feature values
        """
        # Call base class method
        result = super().classify_gait(gait_features, context)

        # Add RF-specific information
        if hasattr(self.model, 'estimators_'):
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

            # Get individual tree predictions for voting analysis
            tree_predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
            unique_votes, vote_counts = np.unique(tree_predictions, return_counts=True)
            tree_votes = {
                str(vote): int(count) for vote, count in zip(unique_votes, vote_counts)
            }

            result["tree_votes"] = tree_votes
            result["n_trees"] = self.config.n_estimators

        return result

    def explain_classification(self, result: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of classification.

        Args:
            result: Classification result from classify_gait()

        Returns:
            Explanation string with prediction details and feature importance
        """
        # Get base explanation
        explanation = super().explain_classification(result)

        # Add RF-specific information
        tree_votes = result.get("tree_votes", {})
        if tree_votes:
            explanation += "\n\nTree Voting (out of {} trees):\n".format(result.get("n_trees", 0))
            sorted_votes = sorted(tree_votes.items(), key=lambda x: x[1], reverse=True)
            for vote_class, count in sorted_votes:
                pct = count / result.get("n_trees", 1) * 100
                explanation += f"  {vote_class:15s}: {count:3d} trees ({pct:.1f}%)\n"

        # Add top feature importances
        if self.feature_importances_ is not None:
            explanation += "\nTop Contributing Features:\n"
            importances = self.get_feature_importances()
            for imp in importances[:5]:
                bar = "█" * int(imp.importance * 20)
                explanation += f"  {imp.feature_name:20s}: {imp.importance:.3f} {bar}\n"

        return explanation

    def get_feature_importances(self, top_n: Optional[int] = None) -> List[FeatureImportance]:
        """
        Get feature importance rankings.

        Args:
            top_n: Return only top N features. Returns all if None.

        Returns:
            List of FeatureImportance objects sorted by importance
        """
        if not self.is_trained or self.feature_importances_ is None:
            raise RuntimeError("Classifier must be trained to get feature importances")

        importances = [
            FeatureImportance(
                feature_name=name,
                importance=float(importance),
                rank=rank + 1,
            )
            for rank, (name, importance) in enumerate(
                sorted(
                    zip(self.feature_names, self.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        ]

        return importances[:top_n] if top_n else importances

    def tune_hyperparameters(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search with cross-validation.

        Args:
            features: Training features
            labels: Training labels
            param_grid: Parameter grid for search. Uses defaults if None.
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with best parameters and scores
        """
        X, y = self._prepare_features(features, fit_scaler=True)
        if labels:
            y = np.array(labels)

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            }

        logger.info("Starting hyperparameter tuning...")
        logger.info(f"Parameter grid: {param_grid}")

        # Use stratified k-fold for imbalanced datasets
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            RandomForestClassifier(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                class_weight=self.config.class_weight,
            ),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=self.config.cv_n_jobs,
            verbose=1,
        )

        grid_search.fit(X, y)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self.classes_ = self.model.classes_
        self.feature_importances_ = self.model.feature_importances_

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
