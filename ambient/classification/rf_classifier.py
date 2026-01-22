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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from loguru import logger

from ambient.core.interfaces import IClassifier
from ambient.classification.knn_classifier import GaitFeatureVector


@dataclass
class RFClassifierConfig:
    """Configuration for Random Forest classifier."""

    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"  # 'sqrt', 'log2', or float
    bootstrap: bool = True
    random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    normalize_features: bool = True
    confidence_threshold: float = 0.5
    class_weight: Optional[str] = "balanced"  # Handle imbalanced classes


@dataclass
class FeatureImportance:
    """Feature importance information."""

    feature_name: str
    importance: float
    rank: int

    def __repr__(self) -> str:
        return f"{self.rank}. {self.feature_name}: {self.importance:.4f}"


class RFGaitClassifier(IClassifier):
    """
    Random Forest classifier for gait condition classification.

    This classifier uses ensemble learning with multiple decision trees to classify
    gait patterns. It implements the IClassifier interface for consistency with
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
        self.config = config or RFClassifierConfig()
        self.model = RandomForestClassifier(
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
        self.scaler = StandardScaler() if self.config.normalize_features else None
        self.is_trained = False
        self.classes_ = None
        self.feature_names = GaitFeatureVector.get_feature_names()
        self.feature_importances_ = None

        logger.info(
            f"Random Forest classifier initialized with {self.config.n_estimators} trees"
        )

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the Random Forest classifier.

        Args:
            features: List of GaitFeatureVector objects
            labels: Optional list of condition labels (uses feature.condition_label if None)
            validate: Whether to perform cross-validation

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
        if not features:
            raise ValueError("No training features provided")

        # Extract feature arrays and labels
        X = np.array([f.to_array() for f in features])
        y = np.array(labels if labels else [f.condition_label for f in features])

        if len(X) != len(y):
            raise ValueError(f"Feature count ({len(X)}) != label count ({len(y)})")

        logger.info(f"Training Random Forest classifier with {len(X)} samples")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Classes: {np.unique(y)}")

        # Check for class imbalance
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_distribution}")

        # Normalize features if configured
        if self.scaler:
            X = self.scaler.fit_transform(X)
            logger.info("Features normalized using StandardScaler")

        # Train model
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        self.feature_importances_ = self.model.feature_importances_

        # Training metrics
        train_accuracy = self.model.score(X, y)
        metrics = {
            "train_accuracy": train_accuracy,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "classes": list(self.classes_),
            "n_estimators": self.config.n_estimators,
            "class_distribution": class_distribution,
        }

        # Cross-validation with stratified folds
        if validate and len(X) >= 5:
            n_splits = min(5, min(counts))  # Ensure enough samples per fold
            if n_splits >= 2:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    self.model, X, y, cv=cv, scoring="accuracy", n_jobs=-1
                )
                metrics["cv_mean_accuracy"] = float(np.mean(cv_scores))
                metrics["cv_std_accuracy"] = float(np.std(cv_scores))
                logger.info(
                    f"Cross-validation accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}"
                )

        # Feature importance
        importances = self.get_feature_importances()
        metrics["feature_importances"] = [
            {"feature": imp.feature_name, "importance": imp.importance, "rank": imp.rank}
            for imp in importances
        ]

        logger.info(f"Training complete. Accuracy: {train_accuracy:.3f}")
        logger.info("Top 5 important features:")
        for imp in importances[:5]:
            logger.info(f"  {imp}")

        return metrics

    def classify_gait(
        self,
        gait_features: Union[GaitFeatureVector, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify gait condition from features.

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
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before classification")

        # Convert to feature vector if needed
        if isinstance(gait_features, dict):
            feature_vec = self._dict_to_feature_vector(gait_features)
        else:
            feature_vec = gait_features

        # Extract and normalize features
        X = feature_vec.to_array().reshape(1, -1)
        if self.scaler:
            X = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Get individual tree predictions for voting analysis
        tree_predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
        unique_votes, vote_counts = np.unique(tree_predictions, return_counts=True)
        tree_votes = {
            str(vote): int(count) for vote, count in zip(unique_votes, vote_counts)
        }

        # Build probability dict
        prob_dict = {
            cls: float(prob) for cls, prob in zip(self.classes_, probabilities)
        }

        # Confidence is the probability of the predicted class
        confidence = float(probabilities[np.where(self.classes_ == prediction)[0][0]])

        result = {
            "predicted_condition": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "tree_votes": tree_votes,
            "n_trees": self.config.n_estimators,
            "is_normal": prediction.lower() in ["normal", "healthy"],
            "feature_vector": feature_vec.to_array().tolist(),
        }

        logger.info(f"Classification: {prediction} (confidence: {confidence:.3f})")

        return result

    def get_classification_confidence(self, result: Dict[str, Any]) -> float:
        """Get confidence score from classification result."""
        return result.get("confidence", 0.0)

    def explain_classification(self, result: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of classification.

        Args:
            result: Classification result from classify_gait()

        Returns:
            Explanation string with prediction details and feature importance
        """
        condition = result["predicted_condition"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        tree_votes = result.get("tree_votes", {})

        explanation = [
            f"Predicted Condition: {condition}",
            f"Confidence: {confidence:.1%}",
            "",
            "Probability Distribution:",
        ]

        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        for cond, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            explanation.append(f"  {cond:15s}: {prob:.1%} {bar}")

        # Add tree voting information
        if tree_votes:
            explanation.append("")
            explanation.append(f"Tree Voting (out of {result['n_trees']} trees):")
            sorted_votes = sorted(tree_votes.items(), key=lambda x: x[1], reverse=True)
            for vote_class, count in sorted_votes:
                pct = count / result["n_trees"] * 100
                explanation.append(f"  {vote_class:15s}: {count:3d} trees ({pct:.1f}%)")

        # Add top feature importances
        if self.feature_importances_ is not None:
            explanation.append("")
            explanation.append("Top Contributing Features:")
            importances = self.get_feature_importances()
            for imp in importances[:5]:
                bar = "█" * int(imp.importance * 20)
                explanation.append(f"  {imp.feature_name:20s}: {imp.importance:.3f} {bar}")

        return "\n".join(explanation)

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

    def evaluate(
        self,
        test_features: List[GaitFeatureVector],
        test_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate classifier on test data.

        Args:
            test_features: List of test feature vectors
            test_labels: Optional test labels (uses feature.condition_label if None)

        Returns:
            Dictionary with evaluation metrics including:
            - accuracy: Overall accuracy
            - precision: Macro-averaged precision
            - recall: Macro-averaged recall
            - f1_score: Macro-averaged F1 score
            - confusion_matrix: Confusion matrix
            - classification_report: Detailed per-class metrics
            - n_test_samples: Number of test samples
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before evaluation")

        X_test = np.array([f.to_array() for f in test_features])
        y_test = np.array(
            test_labels if test_labels else [f.condition_label for f in test_features]
        )

        if self.scaler:
            X_test = self.scaler.transform(X_test)

        # Predictions
        y_pred = self.model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=self.classes_)
        class_report = classification_report(
            y_test, y_pred, labels=self.classes_, output_dict=True, zero_division=0
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "n_test_samples": len(X_test),
            "classes": list(self.classes_),
        }

        logger.info(f"Test accuracy: {accuracy:.3f}")
        logger.info(f"Test F1 score: {f1:.3f}")

        return metrics

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
        X = np.array([f.to_array() for f in features])
        y = np.array(labels if labels else [f.condition_label for f in features])

        if self.scaler:
            X = self.scaler.fit_transform(X)

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
            n_jobs=-1,
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

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save trained classifier to file.

        Args:
            filepath: Path to save file
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained classifier")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
            "classes": self.classes_,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances_,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Classifier saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RFGaitClassifier":
        """
        Load trained classifier from file.

        Args:
            filepath: Path to saved classifier

        Returns:
            Loaded RFGaitClassifier instance
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        classifier = cls(config=model_data["config"])
        classifier.model = model_data["model"]
        classifier.scaler = model_data["scaler"]
        classifier.classes_ = model_data["classes"]
        classifier.feature_names = model_data["feature_names"]
        classifier.feature_importances_ = model_data["feature_importances"]
        classifier.is_trained = model_data["is_trained"]

        logger.info(f"Classifier loaded from {filepath}")

        return classifier

    def _dict_to_feature_vector(
        self, feature_dict: Dict[str, Any]
    ) -> GaitFeatureVector:
        """Convert dictionary to GaitFeatureVector."""
        return GaitFeatureVector(
            left_hip_mean=feature_dict.get("left_hip_mean", 0),
            left_knee_mean=feature_dict.get("left_knee_mean", 0),
            left_ankle_mean=feature_dict.get("left_ankle_mean", 0),
            right_hip_mean=feature_dict.get("right_hip_mean", 0),
            right_knee_mean=feature_dict.get("right_knee_mean", 0),
            right_ankle_mean=feature_dict.get("right_ankle_mean", 0),
            hip_asymmetry=feature_dict.get("hip_asymmetry", 0),
            knee_asymmetry=feature_dict.get("knee_asymmetry", 0),
            ankle_asymmetry=feature_dict.get("ankle_asymmetry", 0),
            left_hip_range=feature_dict.get("left_hip_range", 0),
            left_knee_range=feature_dict.get("left_knee_range", 0),
            left_ankle_range=feature_dict.get("left_ankle_range", 0),
            right_hip_range=feature_dict.get("right_hip_range", 0),
            right_knee_range=feature_dict.get("right_knee_range", 0),
            right_ankle_range=feature_dict.get("right_ankle_range", 0),
        )
