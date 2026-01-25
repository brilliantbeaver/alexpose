#!/usr/bin/env python
"""
Script to add Decision Tree, Logistic Regression, MLP, and Ensemble classifier
sections to the notebook systematically.

This follows SOLID principles, DRY, and ensures consistency across all classifier sections.
"""

import json
from pathlib import Path


def create_classifier_section(
    classifier_name: str,
    classifier_class: str,
    config_class: str,
    import_statement: str,
    config_params: dict,
    description: str,
    key_params_description: dict,
    section_number: int,
    has_feature_importance: bool = False,
    has_special_metrics: dict = None,
) -> list:
    """
    Create a complete classifier training section following DRY principles.
    
    Args:
        classifier_name: Display name (e.g., "Decision Tree")
        classifier_class: Class name (e.g., "DecisionTreeGaitClassifier")
        config_class: Config class name (e.g., "DecisionTreeClassifierConfig")
        import_statement: Import statement
        config_params: Dictionary of configuration parameters
        description: Classifier description
        key_params_description: Dictionary describing key parameters
        section_number: Section number in notebook
        has_feature_importance: Whether classifier has feature importance
        has_special_metrics: Dictionary of special metrics to display
    
    Returns:
        List of notebook cells
    """
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"## {section_number}. Train & Eval {classifier_name} Classifier\n",
            "\n",
            f"{description}"
        ]
    })
    
    # Import cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [import_statement]
    })
    
    # Configuration and training cell
    config_code = [
        f"# Configure and train {classifier_name} classifier\n",
        f"# {description}\n",
        "#\n",
        "# Key parameters:\n",
    ]
    
    for param, desc in key_params_description.items():
        config_code.append(f"#   - {param}: {desc}\n")
    
    config_code.append(f"{classifier_name.lower().replace(' ', '_')}_config = {config_class}(\n")
    for param, value in config_params.items():
        if isinstance(value, str):
            config_code.append(f"    {param}='{value}',\n")
        else:
            config_code.append(f"    {param}={value},\n")
    config_code.append(")\n\n")
    
    config_code.append(f"# Initialize and train the {classifier_name} classifier\n")
    config_code.append(f"{classifier_name.lower().replace(' ', '_')}_classifier = {classifier_class}({classifier_name.lower().replace(' ', '_')}_config)\n")
    config_code.append(f'print("✓ {classifier_name} classifier created")\n')
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": config_code
    })
    
    # Training cell
    training_code = [
        f"# Train {classifier_name} classifier\n",
        f"{classifier_name.lower().replace(' ', '_')}_metrics = {classifier_name.lower().replace(' ', '_')}_classifier.train(train_features, validate=True)\n\n",
        "# Display training results\n",
        'print("\\n" + "="*70)\n',
        f'print("{classifier_name.upper()} TRAINING RESULTS")\n',
        'print("="*70)\n',
        f'print(f"Training Accuracy:     {{{classifier_name.lower().replace(" ", "_")}_metrics[\'train_accuracy\']:.3f}}")\n',
        f'print(f"Number of Samples:     {{{classifier_name.lower().replace(" ", "_")}_metrics[\'n_samples\']}}")\n',
        f'print(f"Number of Features:    {{{classifier_name.lower().replace(" ", "_")}_metrics[\'n_features\']}}")\n',
    ]
    
    # Add special metrics
    if has_special_metrics:
        for metric_key, metric_label in has_special_metrics.items():
            training_code.append(
                f'if \'{metric_key}\' in {classifier_name.lower().replace(" ", "_")}_metrics:\n'
                f'    print(f"{metric_label}: {{{classifier_name.lower().replace(" ", "_")}_metrics[\'{metric_key}\']:.3f}}")\n'
            )
    
    training_code.append(f'print(f"Classes:               {{{classifier_name.lower().replace(" ", "_")}_metrics[\'classes\']}}")\n\n')
    
    # Cross-validation results
    training_code.extend([
        "# Display cross-validation results if available\n",
        f'if \'cv_mean_accuracy\' in {classifier_name.lower().replace(" ", "_")}_metrics:\n',
        '    print(f"\\nCross-Validation Results:")\n',
        f'    print(f"  Mean Accuracy:       {{{classifier_name.lower().replace(" ", "_")}_metrics[\'cv_mean_accuracy\']:.3f}}")\n',
        f'    print(f"  Std Deviation:       {{{classifier_name.lower().replace(" ", "_")}_metrics[\'cv_std_accuracy\']:.3f}}")\n\n',
    ])
    
    # Class distribution
    training_code.extend([
        "# Display class distribution\n",
        f'if \'class_distribution\' in {classifier_name.lower().replace(" ", "_")}_metrics:\n',
        '    print(f"\\nClass Distribution:")\n',
        f'    for cls, count in {classifier_name.lower().replace(" ", "_")}_metrics[\'class_distribution\'].items():\n',
        '        print(f"  {cls:15s}: {count:3d} samples")\n\n',
    ])
    
    # Feature importance
    if has_feature_importance:
        training_code.extend([
            "# Display top 5 most important features\n",
            f'if \'feature_importances\' in {classifier_name.lower().replace(" ", "_")}_metrics:\n',
            '    print(f"\\nTop 5 Most Important Features:")\n',
            f'    for feat_info in {classifier_name.lower().replace(" ", "_")}_metrics[\'feature_importances\'][:5]:\n',
            '        print(f"  {feat_info[\'rank\']}. {feat_info[\'feature\']:20s}: {feat_info[\'importance\']:.4f}")\n\n',
        ])
    
    training_code.append('print("="*70)\n')
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": training_code
    })
    
    # Evaluation cell
    eval_code = [
        f"# Evaluate {classifier_name} on test set\n",
        f"{classifier_name.lower().replace(' ', '_')}_eval_metrics = {classifier_name.lower().replace(' ', '_')}_classifier.evaluate(test_features)\n\n",
        f'print(f"Test Accuracy: {{{classifier_name.lower().replace(" ", "_")}_eval_metrics[\'accuracy\']:.3f}}")\n',
        'print(f"\\nClassification Report:")\n',
        f'{classifier_name.lower().replace(" ", "_")}_report = {classifier_name.lower().replace(" ", "_")}_eval_metrics[\'classification_report\']\n',
        f'for class_name in {classifier_name.lower().replace(" ", "_")}_eval_metrics[\'classes\']:\n',
        f'    if class_name in {classifier_name.lower().replace(" ", "_")}_report:\n',
        f'        cm = {classifier_name.lower().replace(" ", "_")}_report[class_name]\n',
        '        print(f"\\n{class_name}:")\n',
        '        print(f"  Precision: {cm[\'precision\']:.3f}")\n',
        '        print(f"  Recall: {cm[\'recall\']:.3f}")\n',
        '        print(f"  F1-Score: {cm[\'f1-score\']:.3f}")\n',
    ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": eval_code
    })
    
    # Confusion matrix visualization
    viz_code = [
        f"# Plot {classifier_name} confusion matrix\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n\n",
        "plt.figure(figsize=(10, 8))\n",
        "sns.heatmap(\n",
        f"    {classifier_name.lower().replace(' ', '_')}_eval_metrics['confusion_matrix'],\n",
        "    annot=True,\n",
        "    fmt='d',\n",
        "    cmap='Purples',\n",
        f"    xticklabels={classifier_name.lower().replace(' ', '_')}_eval_metrics['classes'],\n",
        f"    yticklabels={classifier_name.lower().replace(' ', '_')}_eval_metrics['classes']\n",
        ")\n",
        f"plt.title('{classifier_name} Classifier - Confusion Matrix', fontsize=14, fontweight='bold')\n",
        "plt.ylabel('True Label', fontsize=12)\n",
        "plt.xlabel('Predicted Label', fontsize=12)\n",
        "plt.tight_layout()\n",
        "plt.show()\n\n",
        f'print("\\n✓ {classifier_name} classifier trained and evaluated")\n',
    ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": viz_code
    })
    
    return cells


def main():
    """Generate and add classifier sections to notebook."""
    
    # Define classifiers to add
    classifiers = [
        {
            "name": "Decision Tree",
            "class": "DecisionTreeGaitClassifier",
            "config_class": "DecisionTreeClassifierConfig",
            "import": [
                "from ambient.classification.decisiontree_classifier import (\n",
                "    DecisionTreeGaitClassifier,\n",
                "    DecisionTreeClassifierConfig\n",
                ")\n"
            ],
            "config_params": {
                "criterion": "gini",
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "normalize_features": True,
                "random_state": 42,
            },
            "description": "Decision trees provide maximum interpretability through a tree structure that can be\nvisualized and converted to simple if-then rules. Each decision node represents a test\non a single feature, making the classification process completely transparent.",
            "key_params": {
                "criterion": "Splitting criterion ('gini' or 'entropy')",
                "max_depth": "Maximum tree depth (prevents overfitting)",
                "min_samples_split": "Minimum samples required to split a node",
                "min_samples_leaf": "Minimum samples required at leaf node",
                "class_weight": "'balanced' automatically adjusts for imbalanced datasets",
                "random_state": "Seed for reproducibility",
            },
            "has_feature_importance": True,
            "special_metrics": {
                "tree_depth": "Tree Depth:            ",
                "n_leaves": "Number of Leaves:      ",
            },
        },
        {
            "name": "Logistic Regression",
            "class": "LogisticGaitClassifier",
            "config_class": "LogisticClassifierConfig",
            "import": [
                "from ambient.classification.logistic_classifier import (\n",
                "    LogisticGaitClassifier,\n",
                "    LogisticClassifierConfig\n",
                ")\n"
            ],
            "config_params": {
                "penalty": "l2",
                "C": 10.0,
                "solver": "lbfgs",
                "max_iter": 2000,
                "class_weight": "balanced",
                "normalize_features": True,
                "random_state": 42,
            },
            "description": "Logistic Regression provides fast, interpretable baseline predictions using linear models.\nDespite its simplicity, it often performs surprisingly well and serves as an excellent\nbaseline for comparison. Feature coefficients show linear relationships.",
            "key_params": {
                "penalty": "Regularization type ('l1', 'l2', 'elasticnet', 'none')",
                "C": "Inverse of regularization strength (higher = less regularization)",
                "solver": "Optimization algorithm ('lbfgs', 'liblinear', 'saga')",
                "max_iter": "Maximum iterations for convergence",
                "class_weight": "'balanced' automatically adjusts for imbalanced datasets",
                "random_state": "Seed for reproducibility",
            },
            "has_feature_importance": False,
            "special_metrics": {},
        },
        {
            "name": "MLP Neural Network",
            "class": "MLPGaitClassifier",
            "config_class": "MLPClassifierConfig",
            "import": [
                "from ambient.classification.mlp_classifier import (\n",
                "    MLPGaitClassifier,\n",
                "    MLPClassifierConfig\n",
                ")\n"
            ],
            "config_params": {
                "hidden_layer_sizes": (100, 50),
                "activation": "relu",
                "solver": "adam",
                "learning_rate_init": 0.001,
                "max_iter": 300,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "normalize_features": True,
                "random_state": 42,
            },
            "description": "Multi-Layer Perceptron (MLP) neural network learns complex non-linear patterns through\nmultiple hidden layers. Particularly effective with larger datasets and can capture\ncomplex feature interactions that simpler models might miss.",
            "key_params": {
                "hidden_layer_sizes": "Neurons per hidden layer (e.g., (100, 50) = 2 layers)",
                "activation": "Activation function ('relu', 'tanh', 'logistic')",
                "solver": "Optimization algorithm ('adam', 'sgd', 'lbfgs')",
                "learning_rate_init": "Initial learning rate",
                "max_iter": "Maximum training iterations",
                "early_stopping": "Stop training when validation score stops improving",
                "random_state": "Seed for reproducibility",
            },
            "has_feature_importance": False,
            "special_metrics": {
                "n_iterations": "Training Iterations:   ",
                "final_loss": "Final Loss:            ",
            },
        },
        {
            "name": "Ensemble",
            "class": "EnsembleGaitClassifier",
            "config_class": "EnsembleClassifierConfig",
            "import": [
                "from ambient.classification.ensemble_classifier import (\n",
                "    EnsembleGaitClassifier,\n",
                "    EnsembleClassifierConfig,\n",
                "    VotingStrategy\n",
                ")\n"
            ],
            "config_params": {
                "voting_strategy": "VotingStrategy.SOFT",
                "classifiers": ["knn", "rf", "xgboost", "svm", "logistic"],
                "normalize_features": True,
                "random_state": 42,
            },
            "description": "Ensemble classifier combines predictions from multiple base classifiers using voting\nstrategies. Typically achieves better accuracy than individual classifiers by leveraging\ntheir diverse strengths. Supports hard voting, soft voting, and weighted voting.",
            "key_params": {
                "voting_strategy": "Voting method (HARD, SOFT, or WEIGHTED)",
                "classifiers": "List of base classifiers to combine",
                "normalize_features": "Standardize features before training",
                "random_state": "Seed for reproducibility",
            },
            "has_feature_importance": False,
            "special_metrics": {},
        },
    ]
    
    # Generate cells for each classifier
    all_cells = []
    for i, clf_info in enumerate(classifiers, start=11):  # Start from section 11
        cells = create_classifier_section(
            classifier_name=clf_info["name"],
            classifier_class=clf_info["class"],
            config_class=clf_info["config_class"],
            import_statement=clf_info["import"],
            config_params=clf_info["config_params"],
            description=clf_info["description"],
            key_params_description=clf_info["key_params"],
            section_number=i,
            has_feature_importance=clf_info["has_feature_importance"],
            has_special_metrics=clf_info.get("special_metrics", {}),
        )
        all_cells.extend(cells)
    
    # Add final comparison section
    comparison_cells = create_final_comparison_section(len(classifiers) + 11)
    all_cells.extend(comparison_cells)
    
    # Save cells to JSON for manual insertion
    output_file = Path("new_classifier_cells.json")
    with open(output_file, "w") as f:
        json.dump(all_cells, f, indent=2)
    
    print(f"✓ Generated {len(all_cells)} cells")
    print(f"✓ Saved to {output_file}")
    print("\nTo add to notebook:")
    print("1. Open the notebook in Jupyter")
    print("2. Insert cells after section 9 (XGBoost)")
    print("3. Copy cell content from new_classifier_cells.json")


def create_final_comparison_section(section_number: int) -> list:
    """Create final comprehensive comparison section."""
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"## {section_number}. Comprehensive Classifier Comparison\n",
            "\n",
            "Let's compare the performance of all classifiers side by side."
        ]
    })
    
    comparison_code = [
        "# Comprehensive classifier performance comparison\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n\n",
        "# Collect metrics from all classifiers\n",
        "all_classifiers = {\n",
        "    'KNN': eval_metrics,\n",
        "    'Random Forest': eval_metrics,\n",
        "    'SVM': svm_eval_metrics,\n",
        "    'XGBoost': xgb_eval_metrics,\n",
        "    'Decision Tree': decision_tree_eval_metrics,\n",
        "    'Logistic Regression': logistic_regression_eval_metrics,\n",
        "    'MLP Neural Network': mlp_neural_network_eval_metrics,\n",
        "    'Ensemble': ensemble_eval_metrics['ensemble'] if 'ensemble' in ensemble_eval_metrics else ensemble_eval_metrics,\n",
        "}\n\n",
        "# Create comparison DataFrame\n",
        "comparison_data = {\n",
        "    'Classifier': list(all_classifiers.keys()),\n",
        "    'Accuracy': [metrics['accuracy'] for metrics in all_classifiers.values()],\n",
        "    'Precision': [metrics['precision'] for metrics in all_classifiers.values()],\n",
        "    'Recall': [metrics['recall'] for metrics in all_classifiers.values()],\n",
        "    'F1-Score': [metrics['f1_score'] for metrics in all_classifiers.values()],\n",
        "}\n\n",
        "df_comparison = pd.DataFrame(comparison_data)\n",
        "df_comparison = df_comparison.sort_values('Accuracy', ascending=False)\n\n",
        'print("\\n" + "="*80)\n',
        'print("COMPREHENSIVE CLASSIFIER PERFORMANCE COMPARISON")\n',
        'print("="*80)\n',
        "print(df_comparison.to_string(index=False))\n",
        'print("="*80)\n\n',
        "# Visualize comparison\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "fig.suptitle('Classifier Performance Comparison', fontsize=16, fontweight='bold')\n\n",
        "# Bar chart for all metrics\n",
        "ax1 = axes[0]\n",
        "x = range(len(df_comparison))\n",
        "width = 0.2\n",
        "ax1.bar([i - 1.5*width for i in x], df_comparison['Accuracy'], width, label='Accuracy', alpha=0.8)\n",
        "ax1.bar([i - 0.5*width for i in x], df_comparison['Precision'], width, label='Precision', alpha=0.8)\n",
        "ax1.bar([i + 0.5*width for i in x], df_comparison['Recall'], width, label='Recall', alpha=0.8)\n",
        "ax1.bar([i + 1.5*width for i in x], df_comparison['F1-Score'], width, label='F1-Score', alpha=0.8)\n",
        "ax1.set_xlabel('Classifier', fontsize=12)\n",
        "ax1.set_ylabel('Score', fontsize=12)\n",
        "ax1.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')\n",
        "ax1.set_xticks(x)\n",
        "ax1.set_xticklabels(df_comparison['Classifier'], rotation=45, ha='right')\n",
        "ax1.legend()\n",
        "ax1.set_ylim([0, 1.0])\n",
        "ax1.grid(axis='y', alpha=0.3)\n\n",
        "# Accuracy ranking\n",
        "ax2 = axes[1]\n",
        "colors = plt.cm.viridis(np.linspace(0, 1, len(df_comparison)))\n",
        "bars = ax2.barh(df_comparison['Classifier'], df_comparison['Accuracy'], color=colors)\n",
        "ax2.set_xlabel('Accuracy', fontsize=12)\n",
        "ax2.set_title('Accuracy Ranking', fontsize=13, fontweight='bold')\n",
        "ax2.set_xlim([0, 1.0])\n",
        "ax2.grid(axis='x', alpha=0.3)\n\n",
        "# Add value labels\n",
        "for i, (bar, acc) in enumerate(zip(bars, df_comparison['Accuracy'])):\n",
        "    ax2.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontweight='bold')\n\n",
        "plt.tight_layout()\n",
        "plt.show()\n\n",
        "# Statistical summary\n",
        'print("\\nStatistical Summary:")\n',
        'print(f"Best Accuracy:  {df_comparison[\'Classifier\'].iloc[0]} ({df_comparison[\'Accuracy\'].iloc[0]:.3f})")\n',
        'print(f"Best F1-Score:  {df_comparison.loc[df_comparison[\'F1-Score\'].idxmax(), \'Classifier\']} ({df_comparison[\'F1-Score\'].max():.3f})")\n',
        'print(f"Mean Accuracy:  {df_comparison[\'Accuracy\'].mean():.3f} ± {df_comparison[\'Accuracy\'].std():.3f}")\n',
        'print(f"Mean F1-Score:  {df_comparison[\'F1-Score\'].mean():.3f} ± {df_comparison[\'F1-Score\'].std():.3f}")\n\n',
        'print("\\n✓ Comprehensive classifier comparison complete")\n',
    ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": comparison_code
    })
    
    return cells


if __name__ == "__main__":
    main()
