"""
XGBoost Gait Classifier

Gradient boosting classifier using XGBoost for gait condition classification.
XGBoost often outperforms Random Forest through sequential tree building
and advanced regularization techniques.

Key advantages:
- State-of-the-art accuracy
- Built-in regularization (L1/L2)
- Handles missing values
- Feature importance
- Fast training with GPU support

Author: AlexPose Team
"""

import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from loguru import logger

from ambient.classification.base_classifier import BaseGaitClassifier, BaseClassifierConfig
from ambient.classification.features import GaitFeatureVector


@dataclass
class XGBoostClassifierConfig(BaseClassifierConfig):
    """Configuration for XGBoost classifier."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    min_child_weight: int = 1
    gamma: float = 0.0
    scale_pos_weight: float = 1.0
    use_label_encoder: bool = False
    eval_metric: str = "mlogloss"
    early_stopping_rounds: Optional[int] = None
    n_jobs: int = -1


class XGBoostGaitClassifier(BaseGaitClassifier):
    """
    XGBoost classifier for gait condition classification.
    
    Uses gradient boosting with decision trees to achieve state-of-the-art
    accuracy on gait classification tasks. Particularly effective for:
    - Complex, non-linear patterns
    - Imbalanced datasets
    - Feature interaction modeling
    
    Example:
        >>> config = XGBoostClassifierConfig(n_estimators=200, max_depth=5)
        >>> classifier = XGBoostGaitClassifier(config)
        >>> classifier.train(training_features)
        >>> result = classifier.classify_gait(test_feature)
    """

    def __init__(self, config: Optional[XGBoostClassifierConfig] = None):
        """Initialize XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )

        config = config or XGBoostClassifierConfig()
        super().__init__(config)
        self.config: XGBoostClassifierConfig = config
        self.feature_importances_ = None

        logger.info(
            f"XGBoost classifier initialized with {config.n_estimators} estimators"
        )

    def _create_model(self):
        """Create XGBoost model."""
        return xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            scale_pos_weight=self.config.scale_pos_weight,
            random_state=self.config.random_state,
            use_label_encoder=self.config.use_label_encoder,
            eval_metric=self.config.eval_metric,
            n_jobs=self.config.n_jobs,
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters."""
        params = {
            "feature_importances": self.feature_importances_,
        }
        # Add n_estimators if model is trained
        if self.model is not None and hasattr(self.model, 'n_estimators'):
            params["n_estimators"] = self.model.n_estimators
        return params

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
        auto_remove_invalid: bool=True
    ) -> Dict[str, Any]:
        """Train XGBoost classifier with optional early stopping."""
        metrics = super().train(features, labels, validate, auto_remove_invalid)

        # Store feature importances
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_

            # Add to metrics
            importances = self.get_feature_importances()
            metrics["feature_importances"] = [
                {
                    "feature": imp.feature_name,
                    "importance": imp.importance,
                    "rank": imp.rank,
                }
                for imp in importances
            ]

            logger.info("Top 5 important features:")
            for imp in importances[:5]:
                logger.info(f"  {imp}")

        # Add n_estimators to metrics
        if hasattr(self.model, 'n_estimators'):
            metrics["n_estimators"] = self.model.n_estimators

        return metrics

    def get_feature_importances(self, top_n: Optional[int] = None):
        """Get feature importance rankings."""
        if not self.is_trained or self.feature_importances_ is None:
            raise RuntimeError("Classifier must be trained to get feature importances")

        from ambient.classification.rf_classifier import FeatureImportance

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
        Tune XGBoost hyperparameters using grid search.
        
        Args:
            features: Training features
            labels: Training labels
            param_grid: Parameter grid for search
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and scores
        """
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        X, y = self._prepare_features(features, fit_scaler=True)
        if labels:
            y = np.array(labels)

        # Encode labels for XGBoost compatibility
        from sklearn.preprocessing import LabelEncoder
        if self.label_encoder_ is None:
            self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)

        # Default parameter grid for XGBoost
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }

        logger.info("Starting XGBoost hyperparameter tuning...")
        logger.info(f"Parameter grid: {param_grid}")

        # Use stratified k-fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            xgb.XGBClassifier(
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric="mlogloss",
                n_jobs=self.config.n_jobs,
            ),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=self.config.cv_n_jobs,
            verbose=1,
        )

        grid_search.fit(X, y_encoded)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self.classes_ = self.label_encoder_.classes_

        # Store feature importances
        if hasattr(self.model, "feature_importances_"):
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

    def explain_classification(self, result: Dict[str, Any]) -> str:
        """Generate explanation with feature importance."""
        explanation = super().explain_classification(result)

        # Add feature importance if available
        if self.feature_importances_ is not None:
            explanation += "\n\nTop Contributing Features:\n"
            importances = self.get_feature_importances()
            for imp in importances[:5]:
                bar = "█" * int(imp.importance * 20)
                explanation += f"  {imp.feature_name:20s}: {imp.importance:.3f} {bar}\n"

        return explanation
