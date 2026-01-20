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

Design Principles:
- Single Responsibility: Focused on KNN classification
- Open/Closed: Extensible through feature extractors
- Liskov Substitution: Implements IClassifier interface
- Interface Segregation: Clean, minimal interface
- Dependency Inversion: Depends on abstractions

Author: AlexPose Team
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from loguru import logger

from ambient.core.interfaces import IClassifier
from ambient.core.data_models import (
    GaitFeatures, ClassificationResult, ConditionPrediction, TrainingDataSample
)


@dataclass
class KNNClassifierConfig:
    """Configuration for KNN classifier."""
    n_neighbors: int = 5
    weights: str = "distance"  # 'uniform' or 'distance'
    metric: str = "euclidean"  # 'euclidean', 'manhattan', 'minkowski'
    algorithm: str = "auto"  # 'auto', 'ball_tree', 'kd_tree', 'brute'
    normalize_features: bool = True
    confidence_threshold: float = 0.5


@dataclass
class GaitFeatureVector:
    """
    Feature vector for gait classification.
    
    This represents the features extracted from a gait sequence that will
    be used for classification. Features include mean joint angles and
    asymmetry measures.
    """
    # Mean joint angles (degrees)
    left_hip_mean: float = 0.0
    left_knee_mean: float = 0.0
    left_ankle_mean: float = 0.0
    right_hip_mean: float = 0.0
    right_knee_mean: float = 0.0
    right_ankle_mean: float = 0.0
    
    # Left-right asymmetry (absolute differences)
    hip_asymmetry: float = 0.0
    knee_asymmetry: float = 0.0
    ankle_asymmetry: float = 0.0
    
    # Range of motion features
    left_hip_range: float = 0.0
    left_knee_range: float = 0.0
    right_hip_range: float = 0.0
    right_knee_range: float = 0.0
    
    # Metadata
    sample_id: str = ""
    condition_label: str = ""
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for sklearn."""
        return np.array([
            self.left_hip_mean,
            self.left_knee_mean,
            self.left_ankle_mean,
            self.right_hip_mean,
            self.right_knee_mean,
            self.right_ankle_mean,
            self.hip_asymmetry,
            self.knee_asymmetry,
            self.ankle_asymmetry,
            self.left_hip_range,
            self.left_knee_range,
            self.right_hip_range,
            self.right_knee_range,
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get ordered list of feature names."""
        return [
            "left_hip_mean",
            "left_knee_mean",
            "left_ankle_mean",
            "right_hip_mean",
            "right_knee_mean",
            "right_ankle_mean",
            "hip_asymmetry",
            "knee_asymmetry",
            "ankle_asymmetry",
            "left_hip_range",
            "left_knee_range",
            "right_hip_range",
            "right_knee_range",
        ]
    
    @classmethod
    def from_joint_angles(
        cls,
        joint_angle_sequence,
        sample_id: str = "",
        condition_label: str = ""
    ) -> "GaitFeatureVector":
        """
        Create feature vector from JointAngleSequence.
        
        Args:
            joint_angle_sequence: JointAngleSequence object from joint_angles module
            sample_id: Identifier for this sample
            condition_label: Ground truth condition label
            
        Returns:
            GaitFeatureVector with computed features
        """
        # Extract statistics for each joint
        left_hip_stats = joint_angle_sequence.get_statistics("left_hip")
        left_knee_stats = joint_angle_sequence.get_statistics("left_knee")
        left_ankle_stats = joint_angle_sequence.get_statistics("left_ankle")
        right_hip_stats = joint_angle_sequence.get_statistics("right_hip")
        right_knee_stats = joint_angle_sequence.get_statistics("right_knee")
        right_ankle_stats = joint_angle_sequence.get_statistics("right_ankle")
        
        # Compute asymmetry features
        hip_asymmetry = abs(
            left_hip_stats.get("mean", 0) - right_hip_stats.get("mean", 0)
        )
        knee_asymmetry = abs(
            left_knee_stats.get("mean", 0) - right_knee_stats.get("mean", 0)
        )
        ankle_asymmetry = abs(
            left_ankle_stats.get("mean", 0) - right_ankle_stats.get("mean", 0)
        )
        
        return cls(
            left_hip_mean=left_hip_stats.get("mean", 0),
            left_knee_mean=left_knee_stats.get("mean", 0),
            left_ankle_mean=left_ankle_stats.get("mean", 0),
            right_hip_mean=right_hip_stats.get("mean", 0),
            right_knee_mean=right_knee_stats.get("mean", 0),
            right_ankle_mean=right_ankle_stats.get("mean", 0),
            hip_asymmetry=hip_asymmetry,
            knee_asymmetry=knee_asymmetry,
            ankle_asymmetry=ankle_asymmetry,
            left_hip_range=left_hip_stats.get("range", 0),
            left_knee_range=left_knee_stats.get("range", 0),
            right_hip_range=right_hip_stats.get("range", 0),
            right_knee_range=right_knee_stats.get("range", 0),
            sample_id=sample_id,
            condition_label=condition_label,
        )


class KNNGaitClassifier(IClassifier):
    """
    K-Nearest Neighbors classifier for gait condition classification.
    
    This classifier uses mean joint angles and asymmetry features to classify
    gait patterns. It implements the IClassifier interface for consistency
    with other classification approaches in the system.
    
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
        self.config = config or KNNClassifierConfig()
        self.model = KNeighborsClassifier(
            n_neighbors=self.config.n_neighbors,
            weights=self.config.weights,
            metric=self.config.metric,
            algorithm=self.config.algorithm
        )
        self.scaler = StandardScaler() if self.config.normalize_features else None
        self.is_trained = False
        self.classes_ = None
        self.feature_names = GaitFeatureVector.get_feature_names()
        
        logger.info(f"KNN classifier initialized with k={self.config.n_neighbors}")
    
    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Train the KNN classifier.
        
        Args:
            features: List of GaitFeatureVector objects
            labels: Optional list of condition labels (uses feature.condition_label if None)
            validate: Whether to perform cross-validation
            
        Returns:
            Dictionary with training metrics
        """
        if not features:
            raise ValueError("No training features provided")
        
        # Extract feature arrays and labels
        X = np.array([f.to_array() for f in features])
        y = np.array(labels if labels else [f.condition_label for f in features])
        
        if len(X) != len(y):
            raise ValueError(f"Feature count ({len(X)}) != label count ({len(y)})")
        
        logger.info(f"Training KNN classifier with {len(X)} samples")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Classes: {np.unique(y)}")
        
        # Normalize features if configured
        if self.scaler:
            X = self.scaler.fit_transform(X)
            logger.info("Features normalized using StandardScaler")
        
        # Train model
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        # Training metrics
        train_accuracy = self.model.score(X, y)
        metrics = {
            "train_accuracy": train_accuracy,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "classes": list(self.classes_),
            "n_neighbors": self.config.n_neighbors
        }
        
        # Cross-validation
        if validate and len(X) >= 5:  # Need at least 5 samples for CV
            cv_scores = cross_val_score(
                self.model, X, y, cv=min(5, len(X)), scoring='accuracy'
            )
            metrics["cv_mean_accuracy"] = float(np.mean(cv_scores))
            metrics["cv_std_accuracy"] = float(np.std(cv_scores))
            logger.info(f"Cross-validation accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")
        
        logger.info(f"Training complete. Accuracy: {train_accuracy:.3f}")
        
        return metrics
    
    def classify_gait(
        self,
        gait_features: Union[GaitFeatureVector, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
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
            - neighbors: Information about nearest neighbors
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
        
        # Get nearest neighbors for explainability
        distances, indices = self.model.kneighbors(X)
        
        # Build probability dict
        prob_dict = {
            cls: float(prob) 
            for cls, prob in zip(self.classes_, probabilities)
        }
        
        # Confidence is the probability of the predicted class
        confidence = float(probabilities[np.where(self.classes_ == prediction)[0][0]])
        
        result = {
            "predicted_condition": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "neighbors": {
                "distances": distances[0].tolist(),
                "indices": indices[0].tolist(),
                "k": self.config.n_neighbors
            },
            "is_normal": prediction.lower() in ["normal", "healthy"],
            "feature_vector": feature_vec.to_array().tolist()
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
            Explanation string
        """
        condition = result["predicted_condition"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        
        explanation = [
            f"Predicted Condition: {condition}",
            f"Confidence: {confidence:.1%}",
            "",
            "Probability Distribution:"
        ]
        
        # Sort by probability
        sorted_probs = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for cond, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            explanation.append(f"  {cond:15s}: {prob:.1%} {bar}")
        
        # Add neighbor information
        neighbors = result.get("neighbors", {})
        if neighbors:
            explanation.append("")
            explanation.append(f"Based on {neighbors['k']} nearest neighbors")
            explanation.append(f"Average distance: {np.mean(neighbors['distances']):.3f}")
        
        return "\n".join(explanation)
    
    def evaluate(
        self,
        test_features: List[GaitFeatureVector],
        test_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate classifier on test data.
        
        Args:
            test_features: List of test feature vectors
            test_labels: Optional test labels (uses feature.condition_label if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before evaluation")
        
        X_test = np.array([f.to_array() for f in test_features])
        y_test = np.array(test_labels if test_labels else [f.condition_label for f in test_features])
        
        if self.scaler:
            X_test = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=self.classes_)
        class_report = classification_report(
            y_test, y_pred, labels=self.classes_, output_dict=True
        )
        
        metrics = {
            "accuracy": float(accuracy),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "n_test_samples": len(X_test),
            "classes": list(self.classes_)
        }
        
        logger.info(f"Test accuracy: {accuracy:.3f}")
        
        return metrics
    
    def tune_hyperparameters(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        param_grid: Optional[Dict[str, List]] = None
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
        X = np.array([f.to_array() for f in features])
        y = np.array(labels if labels else [f.condition_label for f in features])
        
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        
        logger.info("Starting hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=min(5, len(X)),
            scoring='accuracy',
            n_jobs=-1
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
                "mean_test_scores": grid_search.cv_results_['mean_test_score'].tolist(),
                "std_test_scores": grid_search.cv_results_['std_test_score'].tolist(),
                "params": grid_search.cv_results_['params']
            }
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
            "is_trained": self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "KNNGaitClassifier":
        """
        Load trained classifier from file.
        
        Args:
            filepath: Path to saved classifier
            
        Returns:
            Loaded KNNGaitClassifier instance
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(config=model_data["config"])
        classifier.model = model_data["model"]
        classifier.scaler = model_data["scaler"]
        classifier.classes_ = model_data["classes"]
        classifier.feature_names = model_data["feature_names"]
        classifier.is_trained = model_data["is_trained"]
        
        logger.info(f"Classifier loaded from {filepath}")
        
        return classifier
    
    def _dict_to_feature_vector(self, feature_dict: Dict[str, Any]) -> GaitFeatureVector:
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
            right_hip_range=feature_dict.get("right_hip_range", 0),
            right_knee_range=feature_dict.get("right_knee_range", 0),
        )
