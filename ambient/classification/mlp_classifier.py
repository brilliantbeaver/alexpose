"""
MLP Neural Network Gait Classifier

Multi-Layer Perceptron (MLP) neural network for gait condition classification.
MLPs can learn complex non-linear patterns and are particularly effective
when sufficient training data is available.

Key advantages:
- Learns complex non-linear patterns
- Flexible architecture (multiple hidden layers)
- Good with large datasets
- Can capture feature interactions
- Supports online learning

Author: AlexPose Team
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from loguru import logger

from ambient.classification.base_classifier import BaseGaitClassifier, BaseClassifierConfig
from ambient.classification.knn_classifier import GaitFeatureVector


@dataclass
class MLPClassifierConfig(BaseClassifierConfig):
    """Configuration for MLP Neural Network classifier."""

    hidden_layer_sizes: Tuple[int, ...] = (100, 50)  # Neurons per hidden layer
    activation: str = "relu"  # 'identity', 'logistic', 'tanh', 'relu'
    solver: str = "adam"  # 'lbfgs', 'sgd', 'adam'
    alpha: float = 0.0001  # L2 regularization parameter
    batch_size: Union[str, int] = "auto"  # Batch size for SGD/Adam
    learning_rate: str = "constant"  # 'constant', 'invscaling', 'adaptive'
    learning_rate_init: float = 0.001  # Initial learning rate
    max_iter: int = 200  # Maximum iterations
    shuffle: bool = True  # Shuffle samples in each iteration
    tol: float = 1e-4  # Tolerance for optimization
    momentum: float = 0.9  # Momentum for SGD
    early_stopping: bool = False  # Use validation set for early stopping
    validation_fraction: float = 0.1  # Fraction for validation
    beta_1: float = 0.9  # Adam parameter
    beta_2: float = 0.999  # Adam parameter
    epsilon: float = 1e-8  # Adam parameter
    n_iter_no_change: int = 10  # Iterations with no improvement for early stopping


class MLPGaitClassifier(BaseGaitClassifier):
    """
    Multi-Layer Perceptron classifier for gait condition classification.
    
    Implements a feedforward neural network with backpropagation for learning
    complex non-linear relationships in gait features. The default architecture
    uses two hidden layers (100, 50 neurons) with ReLU activation.
    
    Particularly effective for:
    - Large datasets (>200 samples)
    - Complex feature interactions
    - Non-linear decision boundaries
    - Continuous learning scenarios
    
    The MLP requires more data than simpler models but can achieve higher
    accuracy when sufficient training examples are available. Feature
    normalization is highly recommended (enabled by default).
    
    Example:
        >>> config = MLPClassifierConfig(
        ...     hidden_layer_sizes=(128, 64, 32),
        ...     learning_rate_init=0.001,
        ...     early_stopping=True
        ... )
        >>> classifier = MLPGaitClassifier(config)
        >>> classifier.train(training_features)
        >>> 
        >>> # Monitor training progress
        >>> history = classifier.get_training_history()
        >>> print(f"Final loss: {history['loss'][-1]:.4f}")
    """

    def __init__(self, config: Optional[MLPClassifierConfig] = None):
        """Initialize MLP classifier."""
        config = config or MLPClassifierConfig()
        super().__init__(config)
        self.config: MLPClassifierConfig = config
        self.loss_curve_ = None
        self.validation_scores_ = None

        logger.info(
            f"MLP classifier initialized with layers {config.hidden_layer_sizes}"
        )

    def _create_model(self):
        """Create MLP model."""
        return MLPClassifier(
            hidden_layer_sizes=self.config.hidden_layer_sizes,
            activation=self.config.activation,
            solver=self.config.solver,
            alpha=self.config.alpha,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            learning_rate_init=self.config.learning_rate_init,
            max_iter=self.config.max_iter,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state,
            tol=self.config.tol,
            momentum=self.config.momentum,
            early_stopping=self.config.early_stopping,
            validation_fraction=self.config.validation_fraction,
            beta_1=self.config.beta_1,
            beta_2=self.config.beta_2,
            epsilon=self.config.epsilon,
            n_iter_no_change=self.config.n_iter_no_change,
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters."""
        return {
            "loss_curve": self.loss_curve_,
            "validation_scores": self.validation_scores_,
            "n_layers": self.model.n_layers_ if hasattr(self.model, "n_layers_") else None,
            "n_iter": self.model.n_iter_ if hasattr(self.model, "n_iter_") else None,
        }

    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Train MLP classifier."""
        metrics = super().train(features, labels, validate)

        # Store training history
        if hasattr(self.model, "loss_curve_"):
            self.loss_curve_ = self.model.loss_curve_
            metrics["final_loss"] = float(self.loss_curve_[-1])
            metrics["n_iterations"] = int(self.model.n_iter_)
            metrics["n_layers"] = int(self.model.n_layers_)

            logger.info(f"Training iterations: {metrics['n_iterations']}")
            logger.info(f"Final loss: {metrics['final_loss']:.4f}")

        # Store validation scores if early stopping was used
        if hasattr(self.model, "validation_scores_") and self.model.validation_scores_:
            self.validation_scores_ = self.model.validation_scores_
            metrics["best_validation_score"] = float(max(self.validation_scores_))
            logger.info(f"Best validation score: {metrics['best_validation_score']:.3f}")

        return metrics

    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history including loss curve.
        
        Returns:
            Dictionary with training history
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained to get training history")

        history = {
            "n_iterations": int(self.model.n_iter_),
            "n_layers": int(self.model.n_layers_),
        }

        if self.loss_curve_ is not None:
            history["loss"] = [float(loss) for loss in self.loss_curve_]
            history["final_loss"] = float(self.loss_curve_[-1])

        if self.validation_scores_ is not None:
            history["validation_scores"] = [
                float(score) for score in self.validation_scores_
            ]
            history["best_validation_score"] = float(max(self.validation_scores_))

        return history

    def get_network_architecture(self) -> Dict[str, Any]:
        """
        Get neural network architecture details.
        
        Returns:
            Dictionary with architecture information
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained")

        architecture = {
            "n_layers": int(self.model.n_layers_),
            "hidden_layer_sizes": self.config.hidden_layer_sizes,
            "activation": self.config.activation,
            "solver": self.config.solver,
            "n_outputs": int(self.model.n_outputs_),
        }

        # Count total parameters
        total_params = 0
        if hasattr(self.model, "coefs_"):
            for coef_matrix in self.model.coefs_:
                total_params += np.prod(coef_matrix.shape)
            for intercept_vector in self.model.intercepts_:
                total_params += len(intercept_vector)

        architecture["total_parameters"] = total_params

        # Layer-wise parameter counts
        architecture["parameters_per_layer"] = []
        if hasattr(self.model, "coefs_"):
            for i, (coef, intercept) in enumerate(
                zip(self.model.coefs_, self.model.intercepts_)
            ):
                layer_params = np.prod(coef.shape) + len(intercept)
                architecture["parameters_per_layer"].append({
                    "layer": i,
                    "weights_shape": coef.shape,
                    "n_parameters": int(layer_params),
                })

        return architecture

    def tune_hyperparameters(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 3,  # Fewer folds for MLP (slower training)
    ) -> Dict[str, Any]:
        """Tune MLP hyperparameters."""
        X, y = self._prepare_features(features, fit_scaler=True)
        if labels:
            y = np.array(labels)

        # Default parameter grid (smaller for MLP due to training time)
        if param_grid is None:
            param_grid = {
                "hidden_layer_sizes": [(50,), (100,), (100, 50), (128, 64)],
                "learning_rate_init": [0.001, 0.01],
                "alpha": [0.0001, 0.001, 0.01],
            }

        logger.info("Starting MLP hyperparameter tuning...")
        logger.info(f"Parameter grid: {param_grid}")
        logger.warning("MLP tuning may take several minutes...")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            MLPClassifier(
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                early_stopping=True,  # Speed up tuning
            ),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,  # MLP doesn't parallelize well
            verbose=1,
        )

        grid_search.fit(X, y)

        # Update model
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self.classes_ = self.model.classes_
        self.loss_curve_ = self.model.loss_curve_ if hasattr(self.model, "loss_curve_") else None

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
        """Generate explanation with network information."""
        explanation = super().explain_classification(result)

        # Add network architecture info
        if self.is_trained:
            arch = self.get_network_architecture()
            explanation += f"\n\nNeural Network Architecture:\n"
            explanation += f"  Layers: {arch['n_layers']}\n"
            explanation += f"  Hidden units: {arch['hidden_layer_sizes']}\n"
            explanation += f"  Total parameters: {arch['total_parameters']}\n"

        # Add training info
        if self.loss_curve_ is not None:
            explanation += f"\n\nTraining Information:\n"
            explanation += f"  Iterations: {self.model.n_iter_}\n"
            explanation += f"  Final loss: {self.loss_curve_[-1]:.4f}\n"

        return explanation

    def partial_fit(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Incrementally train on new data (online learning).
        
        Args:
            features: New training features
            labels: New training labels
            
        Returns:
            Dictionary with update metrics
        """
        if not features:
            raise ValueError("No features provided for partial fit")

        X, y = self._prepare_features(features, fit_scaler=False)
        if labels:
            y = np.array(labels)

        # Initialize model if needed
        if self.model is None:
            self.model = self._create_model()

        # Partial fit
        if not self.is_trained:
            # First call - need to specify classes
            unique_classes = np.unique(y)
            self.model.partial_fit(X, y, classes=unique_classes)
            self.classes_ = unique_classes
            self.is_trained = True
        else:
            self.model.partial_fit(X, y)

        # Update loss curve
        if hasattr(self.model, "loss_curve_"):
            self.loss_curve_ = self.model.loss_curve_

        metrics = {
            "n_samples_added": len(X),
            "current_loss": float(self.loss_curve_[-1]) if self.loss_curve_ else None,
            "n_iterations": int(self.model.n_iter_),
        }

        logger.info(f"Partial fit completed: {metrics['n_samples_added']} samples added")

        return metrics
