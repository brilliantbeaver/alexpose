"""
Decision Tree Gait Classifier

Single decision tree classifier for highly interpretable gait condition classification.
Decision trees provide clear, rule-based explanations that are easy for clinicians
to understand and validate.

Key advantages:
- Maximum interpretability (visual tree structure)
- No feature scaling required
- Handles non-linear relationships
- Fast training and prediction
- Easy to explain to non-technical users
- Can export decision rules

Author: AlexPose Team
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from loguru import logger

from ambient.classification.base_classifier import BaseGaitClassifier, BaseClassifierConfig
from ambient.classification.features import GaitFeatureVector


@dataclass
class DecisionTreeClassifierConfig(BaseClassifierConfig):
    """Configuration for Decision Tree classifier."""

    criterion: str = "gini"  # 'gini' or 'entropy'
    splitter: str = "best"  # 'best' or 'random'
    max_depth: Optional[int] = None  # Maximum tree depth
    min_samples_split: int = 2  # Min samples to split node
    min_samples_leaf: int = 1  # Min samples in leaf
    max_features: Optional[Union[str, int, float]] = None  # Features per split
    max_leaf_nodes: Optional[int] = None  # Maximum leaf nodes
    min_impurity_decrease: float = 0.0  # Min impurity decrease for split
    class_weight: Optional[str] = "balanced"  # Handle imbalanced classes
    ccp_alpha: float = 0.0  # Complexity parameter for pruning


class DecisionTreeGaitClassifier(BaseGaitClassifier):
    """
    Decision Tree classifier for gait condition classification.
    
    Provides maximum interpretability through a tree structure that can be
    visualized and converted to simple if-then rules. Each decision node
    represents a test on a single feature, making the classification process
    completely transparent.
    
    Particularly useful for:
    - Clinical decision support (explainable AI)
    - Feature interaction discovery
    - Rule extraction for documentation
    - Educational purposes
    - Regulatory compliance (explainability requirements)
    
    Example:
        >>> config = DecisionTreeClassifierConfig(max_depth=5, min_samples_leaf=3)
        >>> classifier = DecisionTreeGaitClassifier(config)
        >>> classifier.train(training_features)
        >>> 
        >>> # Get decision rules
        >>> rules = classifier.get_decision_rules()
        >>> print(rules)
        >>> 
        >>> # Visualize tree
        >>> classifier.export_tree_visualization("tree.png")
    """

    def __init__(self, config: Optional[DecisionTreeClassifierConfig] = None):
        """Initialize Decision Tree classifier."""
        config = config or DecisionTreeClassifierConfig()
        super().__init__(config)
        self.config: DecisionTreeClassifierConfig = config
        self.feature_importances_ = None
        self.tree_ = None

        logger.info(
            f"Decision Tree classifier initialized with {config.criterion} criterion"
        )

    def _create_model(self):
        """Create Decision Tree model."""
        return DecisionTreeClassifier(
            criterion=self.config.criterion,
            splitter=self.config.splitter,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_impurity_decrease=self.config.min_impurity_decrease,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            ccp_alpha=self.config.ccp_alpha,
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters."""
        return {
            "feature_importances": self.feature_importances_,
            "tree_structure": self.get_tree_structure() if self.tree_ else None,
        }

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
        auto_remove_invalid: bool = True,
    ) -> Dict[str, Any]:
        # Call base class train method (handles NaN validation)
        metrics = super().train(features, labels, validate, auto_remove_invalid)

        # Store tree information
        if hasattr(self.model, "tree_"):
            self.tree_ = self.model.tree_
            self.feature_importances_ = self.model.feature_importances_

            # Add tree statistics to metrics
            metrics["tree_depth"] = int(self.model.get_depth())
            metrics["n_leaves"] = int(self.model.get_n_leaves())
            metrics["n_nodes"] = int(self.tree_.node_count)

            # Feature importance
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

            metrics["feature_importances"] = [
                {
                    "feature": imp.feature_name,
                    "importance": imp.importance,
                    "rank": imp.rank,
                }
                for imp in importances
            ]

            logger.info(f"Tree depth: {metrics['tree_depth']}")
            logger.info(f"Number of leaves: {metrics['n_leaves']}")
            logger.info("Top 5 important features:")
            for imp in importances[:5]:
                logger.info(f"  {imp}")

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

    def get_decision_rules(self, max_depth: Optional[int] = None) -> str:
        """
        Get decision tree rules as text.
        
        Args:
            max_depth: Maximum depth to display
            
        Returns:
            String representation of decision rules
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained to get decision rules")

        return export_text(
            self.model,
            feature_names=self.feature_names,
            max_depth=max_depth,
            decimals=2,
            show_weights=True,
        )

    def get_tree_structure(self) -> Dict[str, Any]:
        """
        Get tree structure as dictionary.
        
        Returns:
            Dictionary with tree structure information
        """
        if not self.is_trained or self.tree_ is None:
            raise RuntimeError("Classifier must be trained to get tree structure")

        return {
            "n_nodes": int(self.tree_.node_count),
            "n_leaves": int(self.model.get_n_leaves()),
            "max_depth": int(self.model.get_depth()),
            "n_features": int(self.tree_.n_features),
            "n_classes": int(self.tree_.n_classes[0]),
            "n_outputs": int(self.tree_.n_outputs),
        }

    def get_decision_path(
        self, gait_features: Union[GaitFeatureVector, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get the decision path for a specific sample.
        
        Args:
            gait_features: Feature vector to analyze
            
        Returns:
            Dictionary with decision path information
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained")

        # Prepare features
        feature_vec = (
            gait_features
            if isinstance(gait_features, GaitFeatureVector)
            else self._dict_to_feature_vector(gait_features)
        )
        X = feature_vec.to_array().reshape(1, -1)
        if self.scaler:
            X = self.scaler.transform(X)

        # Get decision path
        decision_path = self.model.decision_path(X)
        node_indicator = decision_path.toarray()[0]
        node_indices = np.where(node_indicator)[0]

        # Build path description
        path_description = []
        for node_id in node_indices:
            if self.tree_.feature[node_id] != -2:  # Not a leaf
                feature_name = self.feature_names[self.tree_.feature[node_id]]
                threshold = self.tree_.threshold[node_id]
                feature_value = X[0, self.tree_.feature[node_id]]

                if feature_value <= threshold:
                    comparison = "<="
                else:
                    comparison = ">"

                path_description.append({
                    "node_id": int(node_id),
                    "feature": feature_name,
                    "threshold": float(threshold),
                    "value": float(feature_value),
                    "comparison": comparison,
                    "description": f"{feature_name} {comparison} {threshold:.2f} (value: {feature_value:.2f})",
                })

        return {
            "path": path_description,
            "n_nodes_visited": len(node_indices),
            "leaf_node": int(node_indices[-1]),
        }

    def tune_hyperparameters(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Tune Decision Tree hyperparameters."""
        X, y = self._prepare_features(features, fit_scaler=True)
        if labels:
            y = np.array(labels)

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
            }

        logger.info("Starting Decision Tree hyperparameter tuning...")
        logger.info(f"Parameter grid: {param_grid}")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            DecisionTreeClassifier(
                random_state=self.config.random_state,
                class_weight=self.config.class_weight,
            ),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=self.config.cv_n_jobs,
            verbose=1,
        )

        grid_search.fit(X, y)

        # Update model
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self.classes_ = self.model.classes_
        self.tree_ = self.model.tree_
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
        """Generate explanation with decision path."""
        explanation = super().explain_classification(result)

        # Add feature importance
        if self.feature_importances_ is not None:
            explanation += "\n\nTop Contributing Features:\n"
            importances = self.get_feature_importances()
            for imp in importances[:5]:
                bar = "█" * int(imp.importance * 20)
                explanation += f"  {imp.feature_name:20s}: {imp.importance:.3f} {bar}\n"

        # Add tree structure info
        if self.tree_ is not None:
            explanation += f"\n\nTree Structure:\n"
            explanation += f"  Depth: {self.model.get_depth()}\n"
            explanation += f"  Leaves: {self.model.get_n_leaves()}\n"

        return explanation

    def export_tree_visualization(
        self, filepath: Union[str, Path], **kwargs
    ) -> None:
        """
        Export tree visualization to file.
        
        Args:
            filepath: Path to save visualization
            **kwargs: Additional arguments for plot_tree
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained to export visualization")

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=kwargs.pop("figsize", (20, 10)))
            plot_tree(
                self.model,
                feature_names=self.feature_names,
                class_names=[str(c) for c in self.classes_],
                filled=True,
                rounded=True,
                **kwargs,
            )
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Tree visualization saved to {filepath}")
        except ImportError:
            logger.warning("matplotlib not available for tree visualization")
