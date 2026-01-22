"""
Ensemble Gait Classifier

Meta-classifier that combines predictions from multiple base classifiers
using voting or stacking strategies. Ensemble methods typically achieve
better accuracy than individual classifiers by leveraging their diverse
strengths.

Key advantages:
- Higher accuracy through model diversity
- Reduced overfitting risk
- Robust to individual model failures
- Flexible voting strategies
- Can weight classifiers by performance

Ensemble Strategies:
- Hard Voting: Majority vote from classifiers
- Soft Voting: Average predicted probabilities
- Weighted Voting: Weight by classifier performance
- Stacking: Train meta-classifier on base predictions

Author: AlexPose Team
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import pickle
from loguru import logger

from ambient.core.interfaces import IClassifier
from ambient.classification.base_classifier import BaseClassifierConfig
from ambient.classification.knn_classifier import GaitFeatureVector, KNNGaitClassifier
from ambient.classification.rf_classifier import RFGaitClassifier
from ambient.classification.xgboost_classifier import XGBoostGaitClassifier
from ambient.classification.svm_classifier import SVMGaitClassifier
from ambient.classification.logistic_classifier import LogisticGaitClassifier
from ambient.classification.decisiontree_classifier import DecisionTreeGaitClassifier
from ambient.classification.mlp_classifier import MLPGaitClassifier


class VotingStrategy(Enum):
    """Voting strategies for ensemble."""
    HARD = "hard"  # Majority vote
    SOFT = "soft"  # Average probabilities
    WEIGHTED = "weighted"  # Weighted by performance


@dataclass
class EnsembleClassifierConfig(BaseClassifierConfig):
    """Configuration for Ensemble classifier."""

    voting_strategy: VotingStrategy = VotingStrategy.SOFT
    classifiers: List[str] = field(default_factory=lambda: [
        "knn", "rf", "xgboost", "svm", "logistic"
    ])
    classifier_weights: Optional[Dict[str, float]] = None
    use_stacking: bool = False
    meta_classifier: str = "logistic"  # For stacking
    min_agreement: float = 0.5  # Minimum agreement for high confidence


class EnsembleGaitClassifier(IClassifier):
    """
    Ensemble classifier combining multiple base classifiers.
    
    Combines predictions from diverse classifiers to achieve better
    accuracy and robustness. Supports multiple voting strategies and
    can automatically weight classifiers based on their performance.
    
    The ensemble leverages the strengths of different algorithms:
    - KNN: Local pattern matching
    - Random Forest: Feature interactions
    - XGBoost: Gradient boosting
    - SVM: Optimal boundaries
    - Logistic: Linear relationships
    
    Example:
        >>> config = EnsembleClassifierConfig(
        ...     voting_strategy=VotingStrategy.SOFT,
        ...     classifiers=["rf", "xgboost", "svm"]
        ... )
        >>> ensemble = EnsembleGaitClassifier(config)
        >>> ensemble.train(training_features)
        >>> result = ensemble.classify_gait(test_feature)
        >>> 
        >>> # Analyze individual classifier contributions
        >>> breakdown = ensemble.get_prediction_breakdown(test_feature)
        >>> for clf_name, pred in breakdown.items():
        ...     print(f"{clf_name}: {pred['predicted_condition']} ({pred['confidence']:.2f})")
    """

    def __init__(self, config: Optional[EnsembleClassifierConfig] = None):
        """Initialize Ensemble classifier."""
        config = config or EnsembleClassifierConfig()
        self.config = config
        self.base_classifiers: Dict[str, IClassifier] = {}
        self.classifier_weights: Dict[str, float] = {}
        self.meta_classifier = None
        self.is_trained = False
        self.classes_ = None
        self.feature_names = GaitFeatureVector.get_feature_names()

        # Initialize base classifiers
        self._initialize_classifiers()

        logger.info(
            f"Ensemble classifier initialized with {len(self.base_classifiers)} base classifiers"
        )
        logger.info(f"Voting strategy: {config.voting_strategy.value}")

    def _initialize_classifiers(self):
        """Initialize base classifiers based on configuration."""
        classifier_map = {
            "knn": KNNGaitClassifier,
            "rf": RFGaitClassifier,
            "xgboost": XGBoostGaitClassifier,
            "svm": SVMGaitClassifier,
            "logistic": LogisticGaitClassifier,
            "decisiontree": DecisionTreeGaitClassifier,
            "mlp": MLPGaitClassifier,
        }

        for clf_name in self.config.classifiers:
            if clf_name not in classifier_map:
                logger.warning(f"Unknown classifier: {clf_name}, skipping")
                continue

            try:
                clf_class = classifier_map[clf_name]
                self.base_classifiers[clf_name] = clf_class()
                logger.info(f"Initialized {clf_name} classifier")
            except Exception as e:
                logger.warning(f"Failed to initialize {clf_name}: {e}")

        if not self.base_classifiers:
            raise ValueError("No valid classifiers initialized")

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Train all base classifiers in the ensemble.
        
        Args:
            features: List of GaitFeatureVector objects
            labels: Optional list of condition labels
            validate: Whether to perform cross-validation
            
        Returns:
            Dictionary with training metrics for each classifier
        """
        if not features:
            raise ValueError("No training features provided")

        logger.info(f"Training ensemble with {len(features)} samples")

        metrics = {
            "n_samples": len(features),
            "n_classifiers": len(self.base_classifiers),
            "classifiers": {},
        }

        # Train each base classifier
        for clf_name, classifier in self.base_classifiers.items():
            logger.info(f"Training {clf_name}...")
            try:
                clf_metrics = classifier.train(features, labels, validate=validate)
                metrics["classifiers"][clf_name] = clf_metrics

                # Store accuracy for weighting
                self.classifier_weights[clf_name] = clf_metrics.get("train_accuracy", 1.0)

                logger.info(
                    f"{clf_name} trained: accuracy={clf_metrics['train_accuracy']:.3f}"
                )
            except Exception as e:
                logger.error(f"Failed to train {clf_name}: {e}")
                metrics["classifiers"][clf_name] = {"error": str(e)}

        # Normalize weights if using weighted voting
        if self.config.voting_strategy == VotingStrategy.WEIGHTED:
            total_weight = sum(self.classifier_weights.values())
            if total_weight > 0:
                self.classifier_weights = {
                    k: v / total_weight for k, v in self.classifier_weights.items()
                }
            else:
                # If all classifiers failed, use equal weights
                logger.warning("All classifiers failed, using equal weights")
                n_classifiers = len(self.base_classifiers)
                self.classifier_weights = {
                    k: 1.0 / n_classifiers for k in self.base_classifiers.keys()
                }

        # Override with custom weights if provided
        if self.config.classifier_weights:
            self.classifier_weights.update(self.config.classifier_weights)

        # Get classes from first successful classifier
        for classifier in self.base_classifiers.values():
            if hasattr(classifier, "classes_") and classifier.classes_ is not None:
                self.classes_ = classifier.classes_
                break

        self.is_trained = True

        # Calculate ensemble metrics
        metrics["ensemble_weights"] = self.classifier_weights
        metrics["voting_strategy"] = self.config.voting_strategy.value

        logger.info("Ensemble training complete")
        logger.info(f"Classifier weights: {self.classifier_weights}")

        return metrics

    def classify_gait(
        self,
        gait_features: Union[GaitFeatureVector, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify gait using ensemble voting.
        
        Args:
            gait_features: GaitFeatureVector or dict with feature values
            context: Optional context information
            
        Returns:
            Dictionary with ensemble classification results
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before classification")

        # Get predictions from all classifiers
        predictions = {}
        for clf_name, classifier in self.base_classifiers.items():
            try:
                pred = classifier.classify_gait(gait_features, context)
                predictions[clf_name] = pred
            except Exception as e:
                logger.warning(f"Classifier {clf_name} failed: {e}")

        if not predictions:
            raise RuntimeError("All classifiers failed to make predictions")

        # Combine predictions based on voting strategy
        if self.config.voting_strategy == VotingStrategy.HARD:
            result = self._hard_voting(predictions)
        elif self.config.voting_strategy == VotingStrategy.SOFT:
            result = self._soft_voting(predictions)
        elif self.config.voting_strategy == VotingStrategy.WEIGHTED:
            result = self._weighted_voting(predictions)
        else:
            raise ValueError(f"Unknown voting strategy: {self.config.voting_strategy}")

        # Add individual predictions
        result["individual_predictions"] = {
            clf_name: {
                "predicted_condition": pred["predicted_condition"],
                "confidence": pred["confidence"],
            }
            for clf_name, pred in predictions.items()
        }

        # Calculate agreement
        result["agreement"] = self._calculate_agreement(predictions)

        logger.info(
            f"Ensemble prediction: {result['predicted_condition']} "
            f"(confidence: {result['confidence']:.3f}, agreement: {result['agreement']:.2f})"
        )

        return result

    def _hard_voting(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Majority vote from classifiers."""
        votes = [pred["predicted_condition"] for pred in predictions.values()]
        unique, counts = np.unique(votes, return_counts=True)

        # Get majority vote
        majority_idx = np.argmax(counts)
        predicted_condition = unique[majority_idx]
        confidence = counts[majority_idx] / len(votes)

        # Build probabilities dict (binary: voted or not)
        probabilities = {str(cls): 0.0 for cls in unique}
        probabilities[str(predicted_condition)] = float(confidence)

        return {
            "predicted_condition": str(predicted_condition),
            "confidence": float(confidence),
            "probabilities": probabilities,
            "voting_strategy": "hard",
            "n_votes": int(counts[majority_idx]),
            "total_classifiers": len(votes),
            "is_normal": str(predicted_condition).lower() in ["normal", "healthy"],
        }

    def _soft_voting(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Average predicted probabilities."""
        # Collect all probabilities
        all_probs = {}
        for pred in predictions.values():
            for cls, prob in pred["probabilities"].items():
                if cls not in all_probs:
                    all_probs[cls] = []
                all_probs[cls].append(prob)

        # Average probabilities
        avg_probs = {cls: np.mean(probs) for cls, probs in all_probs.items()}

        # Get prediction
        predicted_condition = max(avg_probs, key=avg_probs.get)
        confidence = avg_probs[predicted_condition]

        return {
            "predicted_condition": predicted_condition,
            "confidence": float(confidence),
            "probabilities": {k: float(v) for k, v in avg_probs.items()},
            "voting_strategy": "soft",
            "is_normal": predicted_condition.lower() in ["normal", "healthy"],
        }

    def _weighted_voting(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Weighted average of probabilities."""
        # Collect weighted probabilities
        all_probs = {}
        total_weight = 0

        for clf_name, pred in predictions.items():
            weight = self.classifier_weights.get(clf_name, 1.0)
            total_weight += weight

            for cls, prob in pred["probabilities"].items():
                if cls not in all_probs:
                    all_probs[cls] = 0
                all_probs[cls] += prob * weight

        # Normalize by total weight
        avg_probs = {cls: prob / total_weight for cls, prob in all_probs.items()}

        # Get prediction
        predicted_condition = max(avg_probs, key=avg_probs.get)
        confidence = avg_probs[predicted_condition]

        return {
            "predicted_condition": predicted_condition,
            "confidence": float(confidence),
            "probabilities": {k: float(v) for k, v in avg_probs.items()},
            "voting_strategy": "weighted",
            "weights_used": self.classifier_weights,
            "is_normal": predicted_condition.lower() in ["normal", "healthy"],
        }

    def _calculate_agreement(self, predictions: Dict[str, Dict]) -> float:
        """Calculate agreement among classifiers."""
        predicted_conditions = [
            pred["predicted_condition"] for pred in predictions.values()
        ]
        if not predicted_conditions:
            return 0.0
        unique, counts = np.unique(predicted_conditions, return_counts=True)
        max_agreement = np.max(counts)
        return max_agreement / len(predicted_conditions)

    def get_prediction_breakdown(
        self, gait_features: Union[GaitFeatureVector, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed breakdown of predictions from each classifier.
        
        Args:
            gait_features: Feature vector to classify
            
        Returns:
            Dictionary mapping classifier names to their predictions
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained")

        breakdown = {}
        for clf_name, classifier in self.base_classifiers.items():
            try:
                pred = classifier.classify_gait(gait_features)
                breakdown[clf_name] = {
                    "predicted_condition": pred["predicted_condition"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    "weight": self.classifier_weights.get(clf_name, 1.0),
                }
            except Exception as e:
                breakdown[clf_name] = {"error": str(e)}

        return breakdown

    def get_classification_confidence(self, result: Dict[str, Any]) -> float:
        """Get confidence score from classification result."""
        return result.get("confidence", 0.0)

    def evaluate(
        self,
        test_features: List[GaitFeatureVector],
        test_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble and individual classifiers.
        
        Args:
            test_features: List of test feature vectors
            test_labels: Optional test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before evaluation")

        # Get true labels
        y_true = np.array(
            test_labels if test_labels else [f.condition_label for f in test_features]
        )

        # Get ensemble predictions
        y_pred = []
        for feature in test_features:
            result = self.classify_gait(feature)
            y_pred.append(result["predicted_condition"])

        y_pred = np.array(y_pred)

        # Calculate ensemble metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
        )

        ensemble_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=self.classes_).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=self.classes_, output_dict=True, zero_division=0
            ),
        }

        # Evaluate individual classifiers
        individual_metrics = {}
        for clf_name, classifier in self.base_classifiers.items():
            try:
                clf_metrics = classifier.evaluate(test_features, test_labels)
                individual_metrics[clf_name] = {
                    "accuracy": clf_metrics["accuracy"],
                    "f1_score": clf_metrics["f1_score"],
                }
            except Exception as e:
                individual_metrics[clf_name] = {"error": str(e)}

        logger.info(f"Ensemble accuracy: {ensemble_metrics['accuracy']:.3f}")
        logger.info(f"Ensemble F1 score: {ensemble_metrics['f1_score']:.3f}")

        return {
            "ensemble": ensemble_metrics,
            "individual_classifiers": individual_metrics,
            "n_test_samples": len(test_features),
        }

    def explain_classification(self, result: Dict[str, Any]) -> str:
        """Generate explanation with individual classifier contributions."""
        condition = result["predicted_condition"]
        confidence = result["confidence"]
        agreement = result.get("agreement", 0)

        explanation = [
            f"Ensemble Prediction: {condition}",
            f"Confidence: {confidence:.1%}",
            f"Agreement: {agreement:.1%}",
            f"Strategy: {result.get('voting_strategy', 'unknown')}",
            "",
            "Individual Classifier Predictions:",
        ]

        # Show individual predictions
        if "individual_predictions" in result:
            for clf_name, pred in result["individual_predictions"].items():
                weight = self.classifier_weights.get(clf_name, 1.0)
                explanation.append(
                    f"  {clf_name:15s}: {pred['predicted_condition']:15s} "
                    f"(conf: {pred['confidence']:.2f}, weight: {weight:.2f})"
                )

        # Show probability distribution
        if "probabilities" in result:
            explanation.append("\nEnsemble Probability Distribution:")
            sorted_probs = sorted(
                result["probabilities"].items(), key=lambda x: x[1], reverse=True
            )
            for cond, prob in sorted_probs:
                bar = "█" * int(prob * 20)
                explanation.append(f"  {cond:15s}: {prob:.1%} {bar}")

        return "\n".join(explanation)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save ensemble and all base classifiers."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained ensemble")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save base classifiers
        base_clf_dir = filepath.parent / f"{filepath.stem}_classifiers"
        base_clf_dir.mkdir(exist_ok=True)

        saved_classifiers = {}
        for clf_name, classifier in self.base_classifiers.items():
            clf_path = base_clf_dir / f"{clf_name}.pkl"
            try:
                classifier.save(clf_path)
                saved_classifiers[clf_name] = str(clf_path)
            except Exception as e:
                logger.warning(f"Failed to save {clf_name}: {e}")

        # Save ensemble metadata
        ensemble_data = {
            "config": self.config,
            "classifier_weights": self.classifier_weights,
            "classes": self.classes_,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "saved_classifiers": saved_classifiers,
        }

        with open(filepath, "wb") as f:
            pickle.dump(ensemble_data, f)

        logger.info(f"Ensemble saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EnsembleGaitClassifier":
        """Load ensemble and all base classifiers."""
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            ensemble_data = pickle.load(f)

        # Create ensemble
        ensemble = cls(config=ensemble_data["config"])
        ensemble.classifier_weights = ensemble_data["classifier_weights"]
        ensemble.classes_ = ensemble_data["classes"]
        ensemble.feature_names = ensemble_data["feature_names"]
        ensemble.is_trained = ensemble_data["is_trained"]

        # Load base classifiers
        classifier_map = {
            "knn": KNNGaitClassifier,
            "rf": RFGaitClassifier,
            "xgboost": XGBoostGaitClassifier,
            "svm": SVMGaitClassifier,
            "logistic": LogisticGaitClassifier,
            "decisiontree": DecisionTreeGaitClassifier,
            "mlp": MLPGaitClassifier,
        }

        ensemble.base_classifiers = {}
        for clf_name, clf_path in ensemble_data["saved_classifiers"].items():
            try:
                clf_class = classifier_map[clf_name]
                ensemble.base_classifiers[clf_name] = clf_class.load(clf_path)
            except Exception as e:
                logger.warning(f"Failed to load {clf_name}: {e}")

        logger.info(f"Ensemble loaded from {filepath}")

        return ensemble
