"""
SVM Gait Classifier

Support Vector Machine classifier with RBF kernel for gait condition classification.
SVMs are particularly effective for gait patterns due to their ability to find
optimal decision boundaries in high-dimensional feature spaces.

Key advantages:
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors)
- Versatile (different kernel functions)
- Good for non-linear patterns with RBF kernel
- Robust to outliers

Author: AlexPose Team
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from loguru import logger

from ambient.classification.base_classifier import BaseGaitClassifier, BaseClassifierConfig
from ambient.classification.features import GaitFeatureVector


@dataclass
class SVMClassifierConfig(BaseClassifierConfig):
    """Configuration for SVM classifier."""

    kernel: str = "rbf"  # 'linear', 'poly', 'rbf', 'sigmoid'
    C: float = 1.0  # Regularization parameter
    gamma: Union[str, float] = "scale"  # Kernel coefficient ('scale', 'auto', or float)
    degree: int = 3  # Degree for poly kernel
    coef0: float = 0.0  # Independent term in kernel function
    shrinking: bool = True  # Use shrinking heuristic
    probability: bool = True  # Enable probability estimates
    tol: float = 1e-3  # Tolerance for stopping criterion
    cache_size: float = 200  # Kernel cache size in MB
    class_weight: Optional[str] = "balanced"  # Handle imbalanced classes
    max_iter: int = -1  # No limit on iterations


class SVMGaitClassifier(BaseGaitClassifier):
    """
    Support Vector Machine classifier for gait condition classification.
    
    Uses SVM with RBF (Radial Basis Function) kernel by default to capture
    non-linear relationships in gait features. Particularly effective for:
    - High-dimensional feature spaces
    - Non-linear decision boundaries
    - Small to medium-sized datasets
    - Binary and multi-class classification
    
    The RBF kernel maps features to infinite-dimensional space, allowing
    complex decision boundaries while maintaining computational efficiency.
    
    Example:
        >>> config = SVMClassifierConfig(C=10.0, gamma='scale')
        >>> classifier = SVMGaitClassifier(config)
        >>> classifier.train(training_features)
        >>> result = classifier.classify_gait(test_feature)
        >>> print(f"Predicted: {result['predicted_condition']}")
    """

    def __init__(self, config: Optional[SVMClassifierConfig] = None):
        """Initialize SVM classifier."""
        config = config or SVMClassifierConfig()
        super().__init__(config)
        self.config: SVMClassifierConfig = config
        self.support_vectors_ = None
        self.n_support_ = None

        logger.info(
            f"SVM classifier initialized with {config.kernel} kernel, C={config.C}"
        )

    def _create_model(self):
        """Create SVM model."""
        return SVC(
            kernel=self.config.kernel,
            C=self.config.C,
            gamma=self.config.gamma,
            degree=self.config.degree,
            coef0=self.config.coef0,
            shrinking=self.config.shrinking,
            probability=self.config.probability,
            tol=self.config.tol,
            cache_size=self.config.cache_size,
            class_weight=self.config.class_weight,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters."""
        return {
            "support_vectors": self.support_vectors_,
            "n_support": self.n_support_,
        }

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
        auto_remove_invalid: bool = True
    ) -> Dict[str, Any]:
        """Train SVM classifier."""
        metrics = super().train(features, labels, validate, auto_remove_invalid)

        # Store support vector information
        if hasattr(self.model, "support_vectors_"):
            self.support_vectors_ = self.model.support_vectors_
            self.n_support_ = self.model.n_support_

            # Add to metrics
            metrics["n_support_vectors"] = int(np.sum(self.n_support_))
            metrics["support_vectors_per_class"] = {
                str(cls): int(n)
                for cls, n in zip(self.classes_, self.n_support_)
            }

            logger.info(f"Support vectors: {metrics['n_support_vectors']} total")
            logger.info(f"Per class: {metrics['support_vectors_per_class']}")

        return metrics

    def classify_gait(
        self,
        gait_features: Union[GaitFeatureVector, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Classify gait with SVM-specific information."""
        result = super().classify_gait(gait_features, context)

        # Add decision function values if available
        if hasattr(self.model, "decision_function"):
            feature_vec = (
                gait_features
                if isinstance(gait_features, GaitFeatureVector)
                else self._dict_to_feature_vector(gait_features)
            )
            X = feature_vec.to_array().reshape(1, -1)
            if self.scaler:
                X = self.scaler.transform(X)

            decision_values = self.model.decision_function(X)[0]
            
            # For binary classification, decision_values is 1D
            if len(self.classes_) == 2:
                result["decision_value"] = float(decision_values)
            else:
                # For multi-class, it's one-vs-rest
                result["decision_values"] = {
                    str(cls): float(val)
                    for cls, val in zip(self.classes_, decision_values)
                }

        return result

    def tune_hyperparameters(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Tune SVM hyperparameters using grid search.
        
        Args:
            features: Training features
            labels: Training labels
            param_grid: Parameter grid for search
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and scores
        """
        X, y = self._prepare_features(features, fit_scaler=True)
        if labels:
            y = np.array(labels)

        # Default parameter grid for SVM
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "kernel": ["rbf", "poly"],
            }

        logger.info("Starting SVM hyperparameter tuning...")
        logger.info(f"Parameter grid: {param_grid}")

        # Use stratified k-fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            SVC(
                probability=True,
                random_state=self.config.random_state,
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

    def explain_classification(self, result: Dict[str, Any]) -> str:
        """Generate explanation with decision values."""
        explanation = super().explain_classification(result)

        # Add decision function information
        if "decision_values" in result:
            explanation += "\n\nDecision Function Values:\n"
            decision_values = result["decision_values"]
            sorted_decisions = sorted(
                decision_values.items(), key=lambda x: x[1], reverse=True
            )
            for cls, val in sorted_decisions:
                explanation += f"  {cls:15s}: {val:+.3f}\n"
        elif "decision_value" in result:
            explanation += f"\n\nDecision Value: {result['decision_value']:+.3f}\n"

        # Add support vector information
        if self.n_support_ is not None:
            explanation += f"\n\nModel uses {np.sum(self.n_support_)} support vectors\n"

        return explanation

    def get_support_vector_info(self) -> Dict[str, Any]:
        """
        Get information about support vectors.
        
        Returns:
            Dictionary with support vector statistics
        """
        if not self.is_trained or self.support_vectors_ is None:
            raise RuntimeError("Classifier must be trained to get support vector info")

        return {
            "n_support_vectors": int(np.sum(self.n_support_)),
            "support_vectors_per_class": {
                str(cls): int(n) for cls, n in zip(self.classes_, self.n_support_)
            },
            "support_vector_percentage": float(
                np.sum(self.n_support_) / len(self.support_vectors_) * 100
            ),
        }
