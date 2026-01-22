"""
Logistic Regression Gait Classifier

Fast, interpretable baseline classifier for gait condition classification.
Logistic regression provides probabilistic predictions and is excellent for
establishing performance baselines and understanding linear relationships.

Key advantages:
- Very fast training and prediction
- Probabilistic interpretation
- Feature coefficients show importance
- Low memory footprint
- Good baseline for comparison
- Regularization prevents overfitting

Author: AlexPose Team
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from loguru import logger

from ambient.classification.base_classifier import BaseGaitClassifier, BaseClassifierConfig
from ambient.classification.knn_classifier import GaitFeatureVector


@dataclass
class LogisticClassifierConfig(BaseClassifierConfig):
    """Configuration for Logistic Regression classifier."""

    penalty: str = "l2"  # 'l1', 'l2', 'elasticnet', 'none'
    C: float = 1.0  # Inverse of regularization strength
    solver: str = "lbfgs"  # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    max_iter: int = 1000  # Maximum iterations
    class_weight: Optional[str] = "balanced"  # Handle imbalanced classes
    tol: float = 1e-4  # Tolerance for stopping criteria
    n_jobs: int = -1  # Use all CPU cores


class LogisticGaitClassifier(BaseGaitClassifier):
    """
    Logistic Regression classifier for gait condition classification.
    
    Provides fast, interpretable baseline predictions using linear models.
    Despite its simplicity, logistic regression often performs surprisingly
    well and serves as an excellent baseline for comparison.
    
    Particularly useful for:
    - Quick prototyping and baseline establishment
    - Understanding linear feature relationships
    - Real-time prediction requirements
    - Interpretable clinical decisions
    - Feature selection and analysis
    
    The classifier uses L2 regularization by default to prevent overfitting
    and handles multi-class problems using one-vs-rest or multinomial approaches.
    
    Example:
        >>> config = LogisticClassifierConfig(C=10.0, max_iter=2000)
        >>> classifier = LogisticGaitClassifier(config)
        >>> classifier.train(training_features)
        >>> result = classifier.classify_gait(test_feature)
        >>> 
        >>> # Analyze feature coefficients
        >>> coefs = classifier.get_feature_coefficients()
        >>> print("Most important features:", coefs[:5])
    """

    def __init__(self, config: Optional[LogisticClassifierConfig] = None):
        """Initialize Logistic Regression classifier."""
        config = config or LogisticClassifierConfig()
        super().__init__(config)
        self.config: LogisticClassifierConfig = config
        self.coef_ = None
        self.intercept_ = None

        logger.info(
            f"Logistic Regression classifier initialized with {config.penalty} penalty, C={config.C}"
        )

    def _create_model(self):
        """Create Logistic Regression model."""
        return LogisticRegression(
            penalty=self.config.penalty,
            C=self.config.C,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            tol=self.config.tol,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters."""
        return {
            "coef": self.coef_,
            "intercept": self.intercept_,
        }

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Train Logistic Regression classifier."""
        metrics = super().train(features, labels, validate)

        # Store coefficients
        if hasattr(self.model, "coef_"):
            self.coef_ = self.model.coef_
            self.intercept_ = self.model.intercept_

            # Add coefficient statistics to metrics
            metrics["n_coefficients"] = int(np.prod(self.coef_.shape))
            metrics["coef_l1_norm"] = float(np.sum(np.abs(self.coef_)))
            metrics["coef_l2_norm"] = float(np.sqrt(np.sum(self.coef_ ** 2)))

            # Get top features by absolute coefficient value
            coef_importance = self.get_feature_coefficients()
            metrics["top_features"] = [
                {"feature": feat.feature_name, "coefficient": feat.coefficient}
                for feat in coef_importance[:5]
            ]

            logger.info("Top 5 features by coefficient magnitude:")
            for feat in coef_importance[:5]:
                logger.info(f"  {feat}")

        return metrics

    def get_feature_coefficients(self, top_n: Optional[int] = None):
        """
        Get feature coefficients ranked by absolute value.
        
        For multi-class problems, returns average absolute coefficient
        across all one-vs-rest classifiers.
        
        Args:
            top_n: Return only top N features
            
        Returns:
            List of FeatureCoefficient objects
        """
        if not self.is_trained or self.coef_ is None:
            raise RuntimeError("Classifier must be trained to get coefficients")

        from dataclasses import dataclass

        @dataclass
        class FeatureCoefficient:
            feature_name: str
            coefficient: float
            abs_coefficient: float
            rank: int

            def __repr__(self) -> str:
                sign = "+" if self.coefficient >= 0 else ""
                return f"{self.rank}. {self.feature_name}: {sign}{self.coefficient:.4f}"

        # For multi-class, average absolute coefficients across classes
        if len(self.coef_.shape) > 1 and self.coef_.shape[0] > 1:
            avg_abs_coef = np.mean(np.abs(self.coef_), axis=0)
        else:
            avg_abs_coef = np.abs(self.coef_.flatten())

        coefficients = [
            FeatureCoefficient(
                feature_name=name,
                coefficient=float(np.mean(self.coef_[:, i]) if len(self.coef_.shape) > 1 else self.coef_[i]),
                abs_coefficient=float(abs_coef),
                rank=rank + 1,
            )
            for rank, (i, (name, abs_coef)) in enumerate(
                sorted(
                    enumerate(zip(self.feature_names, avg_abs_coef)),
                    key=lambda x: x[1][1],
                    reverse=True,
                )
            )
        ]

        return coefficients[:top_n] if top_n else coefficients

    def tune_hyperparameters(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Tune Logistic Regression hyperparameters.
        
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

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],  # Support both L1 and L2
            }

        logger.info("Starting Logistic Regression hyperparameter tuning...")
        logger.info(f"Parameter grid: {param_grid}")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            LogisticRegression(
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                class_weight=self.config.class_weight,
                n_jobs=self.config.n_jobs,
            ),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X, y)

        # Update model
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self.classes_ = self.model.classes_
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_

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
        """Generate explanation with feature coefficients."""
        explanation = super().explain_classification(result)

        # Add coefficient information
        if self.coef_ is not None:
            explanation += "\n\nTop Contributing Features (by coefficient):\n"
            coefficients = self.get_feature_coefficients()
            for coef in coefficients[:5]:
                sign = "+" if coef.coefficient >= 0 else ""
                bar = "█" * int(coef.abs_coefficient * 10)
                explanation += f"  {coef.feature_name:20s}: {sign}{coef.coefficient:.3f} {bar}\n"

        return explanation

    def get_model_interpretation(self) -> Dict[str, Any]:
        """
        Get detailed model interpretation.
        
        Returns:
            Dictionary with model coefficients and statistics
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained")

        interpretation = {
            "coefficients": {},
            "intercepts": {},
            "feature_importance": [],
        }

        # Per-class coefficients (for multi-class)
        if len(self.coef_.shape) > 1:
            for i, cls in enumerate(self.classes_):
                interpretation["coefficients"][str(cls)] = {
                    name: float(coef)
                    for name, coef in zip(self.feature_names, self.coef_[i])
                }
                interpretation["intercepts"][str(cls)] = float(self.intercept_[i])
        else:
            # Binary classification
            interpretation["coefficients"]["binary"] = {
                name: float(coef)
                for name, coef in zip(self.feature_names, self.coef_[0])
            }
            interpretation["intercepts"]["binary"] = float(self.intercept_[0])

        # Feature importance
        coefficients = self.get_feature_coefficients()
        interpretation["feature_importance"] = [
            {
                "feature": coef.feature_name,
                "coefficient": coef.coefficient,
                "abs_coefficient": coef.abs_coefficient,
                "rank": coef.rank,
            }
            for coef in coefficients
        ]

        return interpretation
