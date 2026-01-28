"""
Base Classifier Utilities

Shared utilities and base classes for all gait classifiers.
Provides common functionality to reduce code duplication (DRY principle).

Author: AlexPose Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
from ambient.classification.features import GaitFeatureVector


@dataclass
class BaseClassifierConfig:
    """Base configuration for all classifiers."""

    normalize_features: bool = True
    random_state: int = 42
    confidence_threshold: float = 0.5
    cv_n_jobs: int = 1  # Number of parallel jobs for cross-validation (1=sequential, -1=all CPUs)


class BaseGaitClassifier(IClassifier, ABC):
    """
    Abstract base class for gait classifiers.
    
    Provides common functionality:
    - Feature normalization
    - Model persistence
    - Evaluation metrics
    - Cross-validation
    - Consistent interface
    
    Subclasses must implement:
    - _create_model(): Create the underlying sklearn model
    - _get_model_params(): Get model-specific parameters for saving
    """

    def __init__(self, config: BaseClassifierConfig):
        """Initialize base classifier."""
        self.config = config
        self.model = None
        self.scaler = StandardScaler() if config.normalize_features else None
        self.label_encoder_ = None
        self.is_trained = False
        self.classes_ = None
        self.feature_names = GaitFeatureVector.get_feature_names()

    @abstractmethod
    def _create_model(self):
        """Create the underlying sklearn model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters for saving. Must be implemented by subclasses."""
        pass

    def _prepare_features(
        self, features: List[GaitFeatureVector], fit_scaler: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training/prediction.
        
        Args:
            features: List of GaitFeatureVector objects
            fit_scaler: Whether to fit the scaler (True for training)
            
        Returns:
            Tuple of (X, y) arrays
        """
        X = np.array([f.to_array() for f in features])
        y = np.array([f.condition_label for f in features])

        if self.scaler:
            if fit_scaler:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)

        return X, y

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
        auto_remove_invalid: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            features: List of GaitFeatureVector objects
            labels: Optional list of condition labels
            validate: Whether to perform cross-validation
            auto_remove_invalid: If True, automatically remove samples with NaN/Inf
            
        Returns:
            Dictionary with training metrics
            
        Raises:
            ValueError: If features contain NaN/Inf and auto_remove_invalid=False
        """
        if not features:
            raise ValueError("No training features provided")

        # Filter out None values (from failed feature extraction)
        original_count = len(features)
        features = [f for f in features if f is not None]
        
        if len(features) < original_count:
            removed_count = original_count - len(features)
            logger.warning(
                f"Removed {removed_count} None feature vectors (failed extraction). "
                f"Continuing with {len(features)} valid features."
            )
        
        if not features:
            raise ValueError(
                "No valid training features after filtering None values. "
                "All feature extractions failed. Check your input data."
            )

        # Extract features WITHOUT scaling (we'll scale after validation)
        X = np.array([f.to_array() for f in features])
        y = np.array([f.condition_label for f in features])
        if labels:
            y = np.array(labels)

        if len(X) != len(y):
            raise ValueError(f"Feature count ({len(X)}) != label count ({len(y)})")

        logger.info(f"Training {self.__class__.__name__} with {len(X)} samples")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Classes: {np.unique(y)}")

        # ========== NaN/Inf VALIDATION ==========
        nan_mask = np.isnan(X)
        inf_mask = np.isinf(X)
        invalid_mask = nan_mask | inf_mask
        
        if np.any(invalid_mask):
            invalid_samples = np.where(invalid_mask.any(axis=1))[0]
            nan_samples = np.where(nan_mask.any(axis=1))[0]
            inf_samples = np.where(inf_mask.any(axis=1))[0]
            invalid_features = np.where(invalid_mask.any(axis=0))[0]
            
            if auto_remove_invalid:
                # Get sample IDs before filtering
                removed_ids = [features[i].sample_id for i in invalid_samples if features[i].sample_id]
                
                # Automatically remove invalid samples
                valid_mask = ~invalid_mask.any(axis=1)
                X = X[valid_mask]
                y = y[valid_mask]
                features = [f for i, f in enumerate(features) if valid_mask[i]]
                
                logger.warning(
                    f"Automatically removed {len(invalid_samples)} invalid samples "
                    f"({len(nan_samples)} with NaN, {len(inf_samples)} with Inf)"
                )
                if removed_ids:
                    logger.warning(f"  Removed sample IDs: {removed_ids[:10]}")
                
                if len(X) < 5:
                    raise ValueError(
                        f"After removing {len(invalid_samples)} invalid samples, only {len(X)} remain. "
                        f"Need at least 5 samples for training. "
                        f"Check keypoint extraction and joint angle calculation."
                    )
                
                logger.info(f"Continuing training with {len(X)} valid samples")
            else:
                # Raise detailed error
                error_msg = [
                    f"Training data contains invalid values in {len(invalid_samples)} samples:",
                    f"  Samples with NaN: {len(nan_samples)}",
                    f"  Samples with Inf: {len(inf_samples)}",
                    f"  Affected sample indices: {invalid_samples[:10].tolist()}" + 
                        (" ..." if len(invalid_samples) > 10 else ""),
                    f"  Affected features (indices): {invalid_features.tolist()}",
                    f"  Feature names: {[self.feature_names[i] for i in invalid_features]}",
                    "",
                    "Common causes:",
                    "  1. Empty keypoint extraction (no frames processed)",
                    "  2. Joint angle calculation failed (missing keypoints)",
                    "  3. All angles were invalid/filtered out",
                    "  4. Division by zero in feature calculation",
                    "",
                    "Solutions:",
                    "  1. Check that videos exist and are valid",
                    "  2. Verify keypoint extraction succeeded (check logs)",
                    "  3. Ensure joint angles were calculated correctly",
                    "  4. Use auto_remove_invalid=True to automatically skip bad samples",
                    "  5. Filter samples manually before training",
                    "  6. Restart kernel to load updated code",
                ]
                
                # Log sample IDs if available
                bad_sample_ids = [
                    features[i].sample_id for i in invalid_samples[:5] 
                    if i < len(features) and features[i].sample_id
                ]
                if bad_sample_ids:
                    error_msg.append(f"  Sample IDs with invalid values: {bad_sample_ids}")
                
                logger.error("\n".join(error_msg))
                raise ValueError(
                    f"Training data contains invalid values (NaN/Inf). "
                    f"Use auto_remove_invalid=True or filter manually. See log for details."
                )
        # ========== END VALIDATION ==========

        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_distribution}")

        # Now scale features if configured
        if self.scaler:
            X = self.scaler.fit_transform(X)
            logger.info("Features normalized using StandardScaler")

        # Create and train model
        if self.model is None:
            self.model = self._create_model()

        # Encode string labels to integers for compatibility
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)

        self.model.fit(X, y_encoded)
        self.classes_ = self.label_encoder_.classes_
        self.is_trained = True

        # Training metrics - use encoded labels for scoring
        train_accuracy = self.model.score(X, y_encoded)
        metrics = {
            "train_accuracy": train_accuracy,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "classes": list(self.classes_),
            "class_distribution": class_distribution,
        }

        # Cross-validation
        if validate and len(X) >= 5:
            n_splits = min(5, min(counts))
            if n_splits >= 2:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                # Use configured n_jobs with fallback to sequential
                n_jobs = getattr(self.config, 'cv_n_jobs', 1)
                
                # Try cross-validation with fallback to sequential
                try:
                    cv_scores = cross_val_score(
                        self.model, X, y_encoded, cv=cv, scoring="accuracy", n_jobs=n_jobs
                    )
                except (OSError, RuntimeError) as e:
                    # Fallback to sequential processing if parallel fails
                    # This commonly happens on macOS due to semaphore limits
                    if n_jobs != 1:
                        logger.warning(
                            f"Parallel cross-validation failed ({type(e).__name__}: {e}). "
                            f"Falling back to sequential processing."
                        )
                        try:
                            cv_scores = cross_val_score(
                                self.model, X, y_encoded, cv=cv, scoring="accuracy", n_jobs=1
                            )
                        except Exception as e2:
                            logger.error(
                                f"Sequential cross-validation also failed: {e2}. "
                                f"Skipping cross-validation."
                            )
                            cv_scores = None
                    else:
                        logger.error(
                            f"Cross-validation failed: {e}. Skipping cross-validation."
                        )
                        cv_scores = None
                
                if cv_scores is not None:
                    metrics["cv_mean_accuracy"] = float(np.mean(cv_scores))
                    metrics["cv_std_accuracy"] = float(np.std(cv_scores))
                    logger.info(
                        f"Cross-validation accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}"
                    )

        logger.info(f"Training complete. Accuracy: {train_accuracy:.3f}")

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
            Dictionary with classification results
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
        prediction_encoded = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Decode prediction back to string label
        prediction = self.label_encoder_.inverse_transform([prediction_encoded])[0]

        # Build probability dict with string labels
        prob_dict = {
            str(cls): float(prob) for cls, prob in zip(self.classes_, probabilities)
        }

        # Confidence is the probability of the predicted class
        confidence = float(probabilities[prediction_encoded])

        result = {
            "predicted_condition": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "is_normal": prediction.lower() in ["normal", "healthy"],
            "feature_vector": feature_vec.to_array().tolist(),
        }

        logger.info(f"Classification: {prediction} (confidence: {confidence:.3f})")

        return result

    def get_classification_confidence(self, result: Dict[str, Any]) -> float:
        """Get confidence score from classification result."""
        return result.get("confidence", 0.0)

    def evaluate(
        self,
        test_features: List[GaitFeatureVector],
        test_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate classifier on test data.
        
        Args:
            test_features: List of test feature vectors
            test_labels: Optional test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before evaluation")

        X_test, y_test = self._prepare_features(test_features, fit_scaler=False)
        if test_labels:
            y_test = np.array(test_labels)

        # Encode labels
        y_test_encoded = self.label_encoder_.transform(y_test)

        # Predictions
        y_pred_encoded = self.model.predict(X_test)
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)

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

    def save(self, filepath: Union[str, Path]) -> None:
        """Save trained classifier to file."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained classifier")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder_,
            "config": self.config,
            "classes": self.classes_,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "model_params": self._get_model_params(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Classifier saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseGaitClassifier":
        """Load trained classifier from file."""
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        classifier = cls(config=model_data["config"])
        classifier.model = model_data["model"]
        classifier.scaler = model_data["scaler"]
        classifier.label_encoder_ = model_data.get("label_encoder")
        classifier.classes_ = model_data["classes"]
        classifier.feature_names = model_data["feature_names"]
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
            "Probability Distribution:",
        ]

        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        for cond, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            explanation.append(f"  {cond:15s}: {prob:.1%} {bar}")

        return "\n".join(explanation)
