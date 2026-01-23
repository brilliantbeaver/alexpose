#!/usr/bin/env python
"""Add missing classifier sections to notebook."""
import json
from pathlib import Path

def create_markdown_cell(content):
    return {"cell_type": "markdown", "metadata": {}, "source": content}

def create_code_cell(content):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": content}

# Load notebook
notebook_path = Path('experiments/exp4/02_train_KNN.ipynb')
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Find insertion point (after XGBoost, before comparison)
insert_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if 'Compare Classifier Performance' in source:
            insert_idx = i
            break

if insert_idx is None:
    print("Could not find comparison section")
    exit(1)

print(f"Found comparison section at index {insert_idx}")
print(f"Will insert new sections before it")

# Create new cells
new_cells = []

# ============================================================================
# DECISION TREE SECTION
# ============================================================================
new_cells.extend([
    create_markdown_cell(
        "## 11. Train & Eval Decision Tree Classifier\n\n"
        "Decision Trees provide maximum interpretability through a tree structure that can be "
        "visualized and converted to simple if-then rules. Each decision node represents a test "
        "on a single feature, making the classification process completely transparent."
    ),
    create_code_cell(
        "from ambient.classification.decisiontree_classifier import (\n"
        "    DecisionTreeGaitClassifier,\n"
        "    DecisionTreeClassifierConfig\n"
        ")\n\n"
        "# Configure Decision Tree classifier\n"
        "# Decision Trees create a flowchart-like structure where each internal node\n"
        "# represents a test on a feature, each branch represents the outcome,\n"
        "# and each leaf node represents a class label.\n"
        "#\n"
        "# Key parameters:\n"
        "#   - criterion: Function to measure split quality ('gini' or 'entropy')\n"
        "#   - max_depth: Maximum depth of tree (None = unlimited)\n"
        "#   - min_samples_split: Minimum samples required to split a node\n"
        "#   - min_samples_leaf: Minimum samples required at a leaf node\n"
        "#   - class_weight: 'balanced' handles imbalanced datasets\n"
        "dt_config = DecisionTreeClassifierConfig(\n"
        "    criterion='gini',           # Gini impurity for split quality\n"
        "    max_depth=10,               # Limit depth to prevent overfitting\n"
        "    min_samples_split=5,        # Require 5 samples to split\n"
        "    min_samples_leaf=2,         # Require 2 samples per leaf\n"
        "    class_weight='balanced',    # Handle imbalanced classes\n"
        "    normalize_features=True,    # Standardize features\n"
        "    random_state=42             # For reproducibility\n"
        ")\n\n"
        "dt_classifier = DecisionTreeGaitClassifier(dt_config)\n"
        "print(\"✓ Decision Tree classifier created\")"
    ),

    create_code_cell(
        "# Train Decision Tree classifier\n"
        "dt_metrics = dt_classifier.train(train_features, validate=True)\n\n"
        "# Display training results\n"
        "print(\"\\n\" + \"=\"*70)\n"
        "print(\"DECISION TREE TRAINING RESULTS\")\n"
        "print(\"=\"*70)\n"
        "print(f\"Training Accuracy:     {dt_metrics['train_accuracy']:.3f}\")\n"
        "print(f\"Number of Samples:     {dt_metrics['n_samples']}\")\n"
        "print(f\"Number of Features:    {dt_metrics['n_features']}\")\n"
        "print(f\"Tree Depth:            {dt_metrics['tree_depth']}\")\n"
        "print(f\"Number of Leaves:      {dt_metrics['n_leaves']}\")\n"
        "print(f\"Classes:               {dt_metrics['classes']}\")\n\n"
        "# Display cross-validation results\n"
        "if 'cv_mean_accuracy' in dt_metrics:\n"
        "    print(f\"\\nCross-Validation Results:\")\n"
        "    print(f\"  Mean Accuracy:       {dt_metrics['cv_mean_accuracy']:.3f}\")\n"
        "    print(f\"  Std Deviation:       {dt_metrics['cv_std_accuracy']:.3f}\")\n\n"
        "# Display class distribution\n"
        "if 'class_distribution' in dt_metrics:\n"
        "    print(f\"\\nClass Distribution:\")\n"
        "    for cls, count in dt_metrics['class_distribution'].items():\n"
        "        print(f\"  {cls:15s}: {count:3d} samples\")\n\n"
        "# Display top 5 most important features\n"
        "if 'feature_importances' in dt_metrics:\n"
        "    print(f\"\\nTop 5 Most Important Features:\")\n"
        "    for feat_info in dt_metrics['feature_importances'][:5]:\n"
        "        print(f\"  {feat_info['rank']}. {feat_info['feature']:20s}: {feat_info['importance']:.4f}\")\n\n"
        "print(\"=\"*70)"
    ),
    create_code_cell(
        "# Evaluate Decision Tree on test set\n"
        "dt_eval_metrics = dt_classifier.evaluate(test_features)\n\n"
        "print(f\"Test Accuracy: {dt_eval_metrics['accuracy']:.3f}\")\n"
        "print(f\"\\nClassification Report:\")\n"
        "dt_report = dt_eval_metrics['classification_report']\n"
        "for class_name in dt_eval_metrics['classes']:\n"
        "    if class_name in dt_report:\n"
        "        cm = dt_report[class_name]\n"
        "        print(f\"\\n{class_name}:\")\n"
        "        print(f\"  Precision: {cm['precision']:.3f}\")\n"
        "        print(f\"  Recall: {cm['recall']:.3f}\")\n"
        "        print(f\"  F1-Score: {cm['f1-score']:.3f}\")"
    ),

    create_code_cell(
        "# Plot Decision Tree confusion matrix\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(\n"
        "    dt_eval_metrics['confusion_matrix'],\n"
        "    annot=True,\n"
        "    fmt='d',\n"
        "    cmap='Purples',\n"
        "    xticklabels=dt_eval_metrics['classes'],\n"
        "    yticklabels=dt_eval_metrics['classes']\n"
        ")\n"
        "plt.title('Decision Tree Classifier - Confusion Matrix', fontsize=14, fontweight='bold')\n"
        "plt.ylabel('True Label', fontsize=12)\n"
        "plt.xlabel('Predicted Label', fontsize=12)\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print(\"\\n✓ Decision Tree classifier trained and evaluated\")"
    ),
])


# ============================================================================
# LOGISTIC REGRESSION SECTION
# ============================================================================
new_cells.extend([
    create_markdown_cell(
        "## 12. Train & Eval Logistic Regression Classifier\n\n"
        "Logistic Regression provides fast, interpretable baseline predictions using linear models. "
        "Despite its simplicity, it often performs surprisingly well and serves as an excellent "
        "baseline for comparison."
    ),
    create_code_cell(
        "from ambient.classification.logistic_classifier import (\n"
        "    LogisticGaitClassifier,\n"
        "    LogisticClassifierConfig\n"
        ")\n\n"
        "# Configure Logistic Regression classifier\n"
        "# Logistic Regression is a linear model that predicts probabilities\n"
        "# using a logistic (sigmoid) function. It's fast, interpretable,\n"
        "# and works well as a baseline.\n"
        "#\n"
        "# Key parameters:\n"
        "#   - penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')\n"
        "#   - C: Inverse of regularization strength (smaller = stronger)\n"
        "#   - solver: Optimization algorithm\n"
        "#   - max_iter: Maximum iterations for convergence\n"
        "logistic_config = LogisticClassifierConfig(\n"
        "    penalty='l2',               # L2 regularization (Ridge)\n"
        "    C=1.0,                      # Regularization strength\n"
        "    solver='lbfgs',             # Optimization algorithm\n"
        "    max_iter=1000,              # Maximum iterations\n"
        "    class_weight='balanced',    # Handle imbalanced classes\n"
        "    normalize_features=True,    # Standardize features\n"
        "    random_state=42             # For reproducibility\n"
        ")\n\n"
        "logistic_classifier = LogisticGaitClassifier(logistic_config)\n"
        "print(\"✓ Logistic Regression classifier created\")"
    ),

    create_code_cell(
        "# Train Logistic Regression classifier\n"
        "logistic_metrics = logistic_classifier.train(train_features, validate=True)\n\n"
        "# Display training results\n"
        "print(\"\\n\" + \"=\"*70)\n"
        "print(\"LOGISTIC REGRESSION TRAINING RESULTS\")\n"
        "print(\"=\"*70)\n"
        "print(f\"Training Accuracy:     {logistic_metrics['train_accuracy']:.3f}\")\n"
        "print(f\"Number of Samples:     {logistic_metrics['n_samples']}\")\n"
        "print(f\"Number of Features:    {logistic_metrics['n_features']}\")\n"
        "print(f\"Classes:               {logistic_metrics['classes']}\")\n\n"
        "# Display cross-validation results\n"
        "if 'cv_mean_accuracy' in logistic_metrics:\n"
        "    print(f\"\\nCross-Validation Results:\")\n"
        "    print(f\"  Mean Accuracy:       {logistic_metrics['cv_mean_accuracy']:.3f}\")\n"
        "    print(f\"  Std Deviation:       {logistic_metrics['cv_std_accuracy']:.3f}\")\n\n"
        "# Display class distribution\n"
        "if 'class_distribution' in logistic_metrics:\n"
        "    print(f\"\\nClass Distribution:\")\n"
        "    for cls, count in logistic_metrics['class_distribution'].items():\n"
        "        print(f\"  {cls:15s}: {count:3d} samples\")\n\n"
        "# Display top features by coefficient\n"
        "if 'top_features' in logistic_metrics:\n"
        "    print(f\"\\nTop 5 Features by Coefficient Magnitude:\")\n"
        "    for i, feat_info in enumerate(logistic_metrics['top_features'], 1):\n"
        "        print(f\"  {i}. {feat_info['feature']:20s}: {feat_info['coefficient']:+.4f}\")\n\n"
        "print(\"=\"*70)"
    ),
    create_code_cell(
        "# Evaluate Logistic Regression on test set\n"
        "logistic_eval_metrics = logistic_classifier.evaluate(test_features)\n\n"
        "print(f\"Test Accuracy: {logistic_eval_metrics['accuracy']:.3f}\")\n"
        "print(f\"\\nClassification Report:\")\n"
        "logistic_report = logistic_eval_metrics['classification_report']\n"
        "for class_name in logistic_eval_metrics['classes']:\n"
        "    if class_name in logistic_report:\n"
        "        cm = logistic_report[class_name]\n"
        "        print(f\"\\n{class_name}:\")\n"
        "        print(f\"  Precision: {cm['precision']:.3f}\")\n"
        "        print(f\"  Recall: {cm['recall']:.3f}\")\n"
        "        print(f\"  F1-Score: {cm['f1-score']:.3f}\")"
    ),
    create_code_cell(
        "# Plot Logistic Regression confusion matrix\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(\n"
        "    logistic_eval_metrics['confusion_matrix'],\n"
        "    annot=True,\n"
        "    fmt='d',\n"
        "    cmap='Oranges',\n"
        "    xticklabels=logistic_eval_metrics['classes'],\n"
        "    yticklabels=logistic_eval_metrics['classes']\n"
        ")\n"
        "plt.title('Logistic Regression Classifier - Confusion Matrix', fontsize=14, fontweight='bold')\n"
        "plt.ylabel('True Label', fontsize=12)\n"
        "plt.xlabel('Predicted Label', fontsize=12)\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print(\"\\n✓ Logistic Regression classifier trained and evaluated\")"
    ),
])


# ============================================================================
# MLP SECTION
# ============================================================================
new_cells.extend([
    create_markdown_cell(
        "## 13. Train & Eval MLP Neural Network Classifier\n\n"
        "Multi-Layer Perceptron (MLP) is a feedforward neural network that can learn complex "
        "non-linear patterns. It's particularly effective when sufficient training data is available."
    ),
    create_code_cell(
        "from ambient.classification.mlp_classifier import (\n"
        "    MLPGaitClassifier,\n"
        "    MLPClassifierConfig\n"
        ")\n\n"
        "# Configure MLP classifier\n"
        "# MLP is a neural network with multiple layers that can learn\n"
        "# complex non-linear relationships between features.\n"
        "#\n"
        "# Key parameters:\n"
        "#   - hidden_layer_sizes: Tuple of neurons per hidden layer\n"
        "#   - activation: Activation function ('relu', 'tanh', 'logistic')\n"
        "#   - solver: Optimization algorithm ('adam', 'sgd', 'lbfgs')\n"
        "#   - learning_rate_init: Initial learning rate\n"
        "#   - max_iter: Maximum training iterations\n"
        "mlp_config = MLPClassifierConfig(\n"
        "    hidden_layer_sizes=(100, 50),  # Two hidden layers\n"
        "    activation='relu',              # ReLU activation\n"
        "    solver='adam',                  # Adam optimizer\n"
        "    learning_rate_init=0.001,       # Learning rate\n"
        "    max_iter=300,                   # Maximum iterations\n"
        "    early_stopping=True,            # Use validation for early stopping\n"
        "    validation_fraction=0.1,        # 10% for validation\n"
        "    normalize_features=True,        # Standardize features (important!)\n"
        "    random_state=42                 # For reproducibility\n"
        ")\n\n"
        "mlp_classifier = MLPGaitClassifier(mlp_config)\n"
        "print(\"✓ MLP classifier created\")"
    ),

    create_code_cell(
        "# Train MLP classifier\n"
        "mlp_metrics = mlp_classifier.train(train_features, validate=True)\n\n"
        "# Display training results\n"
        "print(\"\\n\" + \"=\"*70)\n"
        "print(\"MLP NEURAL NETWORK TRAINING RESULTS\")\n"
        "print(\"=\"*70)\n"
        "print(f\"Training Accuracy:     {mlp_metrics['train_accuracy']:.3f}\")\n"
        "print(f\"Number of Samples:     {mlp_metrics['n_samples']}\")\n"
        "print(f\"Number of Features:    {mlp_metrics['n_features']}\")\n"
        "print(f\"Number of Layers:      {mlp_metrics['n_layers']}\")\n"
        "print(f\"Training Iterations:   {mlp_metrics['n_iterations']}\")\n"
        "print(f\"Final Loss:            {mlp_metrics['final_loss']:.4f}\")\n"
        "print(f\"Classes:               {mlp_metrics['classes']}\")\n\n"
        "# Display cross-validation results\n"
        "if 'cv_mean_accuracy' in mlp_metrics:\n"
        "    print(f\"\\nCross-Validation Results:\")\n"
        "    print(f\"  Mean Accuracy:       {mlp_metrics['cv_mean_accuracy']:.3f}\")\n"
        "    print(f\"  Std Deviation:       {mlp_metrics['cv_std_accuracy']:.3f}\")\n\n"
        "# Display validation score if early stopping was used\n"
        "if 'best_validation_score' in mlp_metrics:\n"
        "    print(f\"\\nEarly Stopping:\")\n"
        "    print(f\"  Best Validation Score: {mlp_metrics['best_validation_score']:.3f}\")\n\n"
        "# Display class distribution\n"
        "if 'class_distribution' in mlp_metrics:\n"
        "    print(f\"\\nClass Distribution:\")\n"
        "    for cls, count in mlp_metrics['class_distribution'].items():\n"
        "        print(f\"  {cls:15s}: {count:3d} samples\")\n\n"
        "print(\"=\"*70)"
    ),
    create_code_cell(
        "# Evaluate MLP on test set\n"
        "mlp_eval_metrics = mlp_classifier.evaluate(test_features)\n\n"
        "print(f\"Test Accuracy: {mlp_eval_metrics['accuracy']:.3f}\")\n"
        "print(f\"\\nClassification Report:\")\n"
        "mlp_report = mlp_eval_metrics['classification_report']\n"
        "for class_name in mlp_eval_metrics['classes']:\n"
        "    if class_name in mlp_report:\n"
        "        cm = mlp_report[class_name]\n"
        "        print(f\"\\n{class_name}:\")\n"
        "        print(f\"  Precision: {cm['precision']:.3f}\")\n"
        "        print(f\"  Recall: {cm['recall']:.3f}\")\n"
        "        print(f\"  F1-Score: {cm['f1-score']:.3f}\")"
    ),
    create_code_cell(
        "# Plot MLP confusion matrix\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(\n"
        "    mlp_eval_metrics['confusion_matrix'],\n"
        "    annot=True,\n"
        "    fmt='d',\n"
        "    cmap='Reds',\n"
        "    xticklabels=mlp_eval_metrics['classes'],\n"
        "    yticklabels=mlp_eval_metrics['classes']\n"
        ")\n"
        "plt.title('MLP Neural Network Classifier - Confusion Matrix', fontsize=14, fontweight='bold')\n"
        "plt.ylabel('True Label', fontsize=12)\n"
        "plt.xlabel('Predicted Label', fontsize=12)\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print(\"\\n✓ MLP classifier trained and evaluated\")"
    ),
])


# ============================================================================
# ENSEMBLE SECTION
# ============================================================================
new_cells.extend([
    create_markdown_cell(
        "## 14. Train & Eval Ensemble Classifier\n\n"
        "Ensemble methods combine predictions from multiple classifiers to achieve better "
        "accuracy and robustness. The ensemble leverages the strengths of different algorithms "
        "through voting strategies."
    ),
    create_code_cell(
        "from ambient.classification.ensemble_classifier import (\n"
        "    EnsembleGaitClassifier,\n"
        "    EnsembleClassifierConfig,\n"
        "    VotingStrategy\n"
        ")\n\n"
        "# Configure Ensemble classifier\n"
        "# Ensemble combines multiple classifiers to leverage their diverse strengths:\n"
        "#   - KNN: Local pattern matching\n"
        "#   - Random Forest: Feature interactions and robustness\n"
        "#   - XGBoost: Gradient boosting for high accuracy\n"
        "#   - SVM: Optimal decision boundaries\n"
        "#   - Logistic: Linear relationships\n"
        "#   - Decision Tree: Interpretable rules\n"
        "#   - MLP: Non-linear patterns\n"
        "#\n"
        "# Voting strategies:\n"
        "#   - HARD: Majority vote (most common prediction)\n"
        "#   - SOFT: Average predicted probabilities (recommended)\n"
        "#   - WEIGHTED: Weight by classifier performance\n"
        "ensemble_config = EnsembleClassifierConfig(\n"
        "    voting_strategy=VotingStrategy.SOFT,  # Average probabilities\n"
        "    classifiers=['knn', 'rf', 'xgboost', 'svm', 'logistic', 'decisiontree', 'mlp'],\n"
        "    normalize_features=True,\n"
        "    random_state=42\n"
        ")\n\n"
        "ensemble_classifier = EnsembleGaitClassifier(ensemble_config)\n"
        "print(\"✓ Ensemble classifier created with 7 base classifiers\")"
    ),
    create_code_cell(
        "# Train Ensemble classifier (trains all base classifiers)\n"
        "ensemble_metrics = ensemble_classifier.train(train_features, validate=True)\n\n"
        "# Display training results\n"
        "print(\"\\n\" + \"=\"*70)\n"
        "print(\"ENSEMBLE CLASSIFIER TRAINING RESULTS\")\n"
        "print(\"=\"*70)\n"
        "print(f\"Number of Samples:     {ensemble_metrics['n_samples']}\")\n"
        "print(f\"Number of Classifiers: {ensemble_metrics['n_classifiers']}\")\n"
        "print(f\"Voting Strategy:       {ensemble_metrics['voting_strategy']}\")\n\n"
        "print(\"Individual Classifier Performance:\")\n"
        "for clf_name, clf_metrics in ensemble_metrics['classifiers'].items():\n"
        "    if 'train_accuracy' in clf_metrics:\n"
        "        acc = clf_metrics['train_accuracy']\n"
        "        weight = ensemble_metrics['ensemble_weights'].get(clf_name, 0)\n"
        "        print(f\"  {clf_name:15s}: {acc:.3f} (weight: {weight:.3f})\")\n"
        "    elif 'error' in clf_metrics:\n"
        "        print(f\"  {clf_name:15s}: FAILED - {clf_metrics['error']}\")\n\n"
        "print(\"=\"*70)"
    ),

    create_code_cell(
        "# Evaluate Ensemble on test set\n"
        "ensemble_eval = ensemble_classifier.evaluate(test_features)\n\n"
        "print(\"\\n\" + \"=\"*70)\n"
        "print(\"ENSEMBLE TEST RESULTS\")\n"
        "print(\"=\"*70)\n"
        "print(f\"Ensemble Accuracy: {ensemble_eval['ensemble']['accuracy']:.3f}\")\n"
        "print(f\"Ensemble F1 Score: {ensemble_eval['ensemble']['f1_score']:.3f}\")\n\n"
        "print(\"\\nIndividual Classifier Test Accuracy:\")\n"
        "for clf_name, clf_metrics in ensemble_eval['individual_classifiers'].items():\n"
        "    if 'accuracy' in clf_metrics:\n"
        "        print(f\"  {clf_name:15s}: {clf_metrics['accuracy']:.3f}\")\n\n"
        "print(\"\\nEnsemble Classification Report:\")\n"
        "ensemble_report = ensemble_eval['ensemble']['classification_report']\n"
        "for class_name in ensemble_eval['ensemble']['classification_report'].keys():\n"
        "    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:\n"
        "        cm = ensemble_report[class_name]\n"
        "        print(f\"\\n{class_name}:\")\n"
        "        print(f\"  Precision: {cm['precision']:.3f}\")\n"
        "        print(f\"  Recall: {cm['recall']:.3f}\")\n"
        "        print(f\"  F1-Score: {cm['f1-score']:.3f}\")\n\n"
        "print(\"=\"*70)"
    ),
    create_code_cell(
        "# Plot Ensemble confusion matrix\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(\n"
        "    ensemble_eval['ensemble']['confusion_matrix'],\n"
        "    annot=True,\n"
        "    fmt='d',\n"
        "    cmap='YlGnBu',\n"
        "    xticklabels=eval_metrics['classes'],\n"
        "    yticklabels=eval_metrics['classes']\n"
        ")\n"
        "plt.title('Ensemble Classifier - Confusion Matrix', fontsize=14, fontweight='bold')\n"
        "plt.ylabel('True Label', fontsize=12)\n"
        "plt.xlabel('Predicted Label', fontsize=12)\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print(\"\\n✓ Ensemble classifier trained and evaluated\")"
    ),
])


# ============================================================================
# UPDATED COMPARISON SECTION
# ============================================================================
# Replace the old comparison section with comprehensive one
comparison_cells = [
    create_markdown_cell(
        "## 15. Comprehensive Classifier Performance Comparison\n\n"
        "Let's compare the performance of all classifiers side by side, including:\n"
        "- KNN (K-Nearest Neighbors)\n"
        "- Random Forest\n"
        "- SVM (Support Vector Machine)\n"
        "- XGBoost (Gradient Boosting)\n"
        "- Decision Tree\n"
        "- Logistic Regression\n"
        "- MLP (Neural Network)\n"
        "- Ensemble (Combined)"
    ),
    create_code_cell(
        "# Collect all evaluation metrics\n"
        "all_classifiers = {\n"
        "    'KNN': eval_metrics,\n"
        "    'Random Forest': eval_metrics,  # From section 7\n"
        "    'SVM': svm_eval_metrics,\n"
        "    'XGBoost': xgb_eval_metrics,\n"
        "    'Decision Tree': dt_eval_metrics,\n"
        "    'Logistic Regression': logistic_eval_metrics,\n"
        "    'MLP': mlp_eval_metrics,\n"
        "    'Ensemble': ensemble_eval['ensemble']\n"
        "}\n\n"
        "# Create comprehensive comparison DataFrame\n"
        "comparison_data = {\n"
        "    'Classifier': [],\n"
        "    'Test Accuracy': [],\n"
        "    'Precision': [],\n"
        "    'Recall': [],\n"
        "    'F1 Score': []\n"
        "}\n\n"
        "for clf_name, clf_metrics in all_classifiers.items():\n"
        "    comparison_data['Classifier'].append(clf_name)\n"
        "    comparison_data['Test Accuracy'].append(clf_metrics['accuracy'])\n"
        "    comparison_data['Precision'].append(clf_metrics['precision'])\n"
        "    comparison_data['Recall'].append(clf_metrics['recall'])\n"
        "    comparison_data['F1 Score'].append(clf_metrics['f1_score'])\n\n"
        "df_comparison = pd.DataFrame(comparison_data)\n"
        "df_comparison = df_comparison.sort_values('Test Accuracy', ascending=False)\n\n"
        "print(\"\\n\" + \"=\"*80)\n"
        "print(\"COMPREHENSIVE CLASSIFIER PERFORMANCE COMPARISON\")\n"
        "print(\"=\"*80)\n"
        "print(df_comparison.to_string(index=False, float_format=lambda x: f'{x:.3f}'))\n"
        "print(\"=\"*80)\n\n"
        "# Find best classifier\n"
        "best_clf = df_comparison.iloc[0]['Classifier']\n"
        "best_acc = df_comparison.iloc[0]['Test Accuracy']\n"
        "print(f\"\\n🏆 Best Classifier: {best_clf} (Accuracy: {best_acc:.3f})\")"
    ),

    create_code_cell(
        "# Visualize overall performance comparison\n"
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n"
        "fig.suptitle('Classifier Performance Comparison', fontsize=16, fontweight='bold')\n\n"
        "# 1. Test Accuracy Comparison\n"
        "ax1 = axes[0, 0]\n"
        "bars1 = ax1.barh(df_comparison['Classifier'], df_comparison['Test Accuracy'], \n"
        "                 color='steelblue', alpha=0.8)\n"
        "ax1.set_xlabel('Test Accuracy', fontsize=12)\n"
        "ax1.set_title('Test Accuracy by Classifier', fontsize=13, fontweight='bold')\n"
        "ax1.set_xlim([0, 1.0])\n"
        "ax1.grid(axis='x', alpha=0.3)\n"
        "# Add value labels\n"
        "for i, (bar, val) in enumerate(zip(bars1, df_comparison['Test Accuracy'])):\n"
        "    ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold')\n\n"
        "# 2. F1 Score Comparison\n"
        "ax2 = axes[0, 1]\n"
        "bars2 = ax2.barh(df_comparison['Classifier'], df_comparison['F1 Score'], \n"
        "                 color='forestgreen', alpha=0.8)\n"
        "ax2.set_xlabel('F1 Score', fontsize=12)\n"
        "ax2.set_title('F1 Score by Classifier', fontsize=13, fontweight='bold')\n"
        "ax2.set_xlim([0, 1.0])\n"
        "ax2.grid(axis='x', alpha=0.3)\n"
        "for i, (bar, val) in enumerate(zip(bars2, df_comparison['F1 Score'])):\n"
        "    ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold')\n\n"
        "# 3. Precision vs Recall\n"
        "ax3 = axes[1, 0]\n"
        "x = range(len(df_comparison))\n"
        "width = 0.35\n"
        "ax3.bar([i - width/2 for i in x], df_comparison['Precision'], width, \n"
        "        label='Precision', color='coral', alpha=0.8)\n"
        "ax3.bar([i + width/2 for i in x], df_comparison['Recall'], width, \n"
        "        label='Recall', color='skyblue', alpha=0.8)\n"
        "ax3.set_ylabel('Score', fontsize=12)\n"
        "ax3.set_title('Precision vs Recall', fontsize=13, fontweight='bold')\n"
        "ax3.set_xticks(x)\n"
        "ax3.set_xticklabels(df_comparison['Classifier'], rotation=45, ha='right')\n"
        "ax3.legend()\n"
        "ax3.set_ylim([0, 1.0])\n"
        "ax3.grid(axis='y', alpha=0.3)\n\n"
        "# 4. All Metrics Radar Chart\n"
        "ax4 = axes[1, 1]\n"
        "metrics_to_plot = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']\n"
        "colors = plt.cm.Set3(range(len(df_comparison)))\n"
        "for idx, row in df_comparison.iterrows():\n"
        "    values = [row[m] for m in metrics_to_plot]\n"
        "    ax4.plot(metrics_to_plot, values, 'o-', label=row['Classifier'], \n"
        "             color=colors[idx], linewidth=2, markersize=8, alpha=0.7)\n"
        "ax4.set_ylim([0, 1.0])\n"
        "ax4.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')\n"
        "ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)\n"
        "ax4.grid(True, alpha=0.3)\n"
        "ax4.set_xticklabels(metrics_to_plot, rotation=15, ha='right')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),

    create_code_cell(
        "# Per-class performance comparison\n"
        "print(\"\\n\" + \"=\"*80)\n"
        "print(\"PER-CLASS PERFORMANCE COMPARISON\")\n"
        "print(\"=\"*80)\n\n"
        "# Get all unique classes\n"
        "classes = eval_metrics['classes']\n\n"
        "for class_name in classes:\n"
        "    print(f\"\\n{class_name.upper()}:\")\n"
        "    print(\"-\" * 80)\n"
        "    print(f\"{'Classifier':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\")\n"
        "    print(\"-\" * 80)\n"
        "    \n"
        "    for clf_name, clf_metrics in all_classifiers.items():\n"
        "        report = clf_metrics['classification_report']\n"
        "        if class_name in report:\n"
        "            cm = report[class_name]\n"
        "            print(f\"{clf_name:<20} {cm['precision']:>10.3f} {cm['recall']:>10.3f} \"\n"
        "                  f\"{cm['f1-score']:>10.3f}\")\n\n"
        "print(\"=\"*80)"
    ),
    create_code_cell(
        "# Statistical summary\n"
        "print(\"\\n\" + \"=\"*80)\n"
        "print(\"STATISTICAL SUMMARY\")\n"
        "print(\"=\"*80)\n\n"
        "summary_stats = df_comparison[['Test Accuracy', 'Precision', 'Recall', 'F1 Score']].describe()\n"
        "print(summary_stats.to_string(float_format=lambda x: f'{x:.3f}'))\n\n"
        "print(\"\\n\" + \"=\"*80)\n"
        "print(\"KEY INSIGHTS\")\n"
        "print(\"=\"*80)\n\n"
        "# Best performers\n"
        "best_accuracy = df_comparison.loc[df_comparison['Test Accuracy'].idxmax()]\n"
        "best_f1 = df_comparison.loc[df_comparison['F1 Score'].idxmax()]\n"
        "best_precision = df_comparison.loc[df_comparison['Precision'].idxmax()]\n"
        "best_recall = df_comparison.loc[df_comparison['Recall'].idxmax()]\n\n"
        "print(f\"🎯 Highest Accuracy:  {best_accuracy['Classifier']} ({best_accuracy['Test Accuracy']:.3f})\")\n"
        "print(f\"🎯 Highest F1 Score:  {best_f1['Classifier']} ({best_f1['F1 Score']:.3f})\")\n"
        "print(f\"🎯 Highest Precision: {best_precision['Classifier']} ({best_precision['Precision']:.3f})\")\n"
        "print(f\"🎯 Highest Recall:    {best_recall['Classifier']} ({best_recall['Recall']:.3f})\")\n\n"
        "# Performance spread\n"
        "acc_range = df_comparison['Test Accuracy'].max() - df_comparison['Test Accuracy'].min()\n"
        "print(f\"\\n📊 Accuracy Range: {acc_range:.3f} \"\n"
        "      f\"(from {df_comparison['Test Accuracy'].min():.3f} to {df_comparison['Test Accuracy'].max():.3f})\")\n\n"
        "# Recommendations\n"
        "print(\"\\n💡 RECOMMENDATIONS:\")\n"
        "if best_accuracy['Classifier'] == 'Ensemble':\n"
        "    print(\"   ✓ Ensemble achieves best performance by combining multiple models\")\n"
        "else:\n"
        "    print(f\"   ✓ {best_accuracy['Classifier']} performs best for this dataset\")\n"
        "    print(f\"   ✓ Consider using Ensemble for production (robustness)\")\n\n"
        "if acc_range < 0.1:\n"
        "    print(\"   ✓ All classifiers perform similarly - choose based on interpretability/speed\")\n"
        "else:\n"
        "    print(\"   ✓ Significant performance differences - model selection is important\")\n\n"
        "print(\"\\n✓ Classifier comparison complete\")\n"
        "print(\"=\"*80)"
    )
]


# ============================================================================
# INSERT NEW CELLS AND SAVE
# ============================================================================

# Insert new cells before the old comparison section
notebook['cells'] = (
    notebook['cells'][:insert_idx] +  # Everything before comparison
    new_cells +                        # New classifier sections
    comparison_cells +                 # Updated comparison section
    notebook['cells'][insert_idx+3:]   # Everything after old comparison (skip old comparison cells)
)

# Save updated notebook
output_path = Path('experiments/exp4/02_train_KNN_updated.ipynb')
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"\n{'='*80}")
print(f"✓ Successfully added {len(new_cells) + len(comparison_cells)} new cells")
print(f"✓ Updated notebook saved to: {output_path}")
print(f"{'='*80}")
print("\nNew sections added:")
print("  11. Decision Tree Classifier")
print("  12. Logistic Regression Classifier")
print("  13. MLP Neural Network Classifier")
print("  14. Ensemble Classifier")
print("  15. Comprehensive Classifier Comparison (updated)")
print("\nTo use the updated notebook:")
print(f"  1. Review: jupyter notebook {output_path}")
print(f"  2. If satisfied, replace original:")
print(f"     mv {output_path} {notebook_path}")
