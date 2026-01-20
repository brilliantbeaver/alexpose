"""
KNN Classifier Training Script

This script trains a K-Nearest Neighbors classifier for gait condition classification.

Author: AlexPose Team
Date: January 19, 2026
"""

import os
import sys
import contextlib
import string
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

# Suppress TensorFlow and MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_stderrthreshold'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_VMODULE'] = 'inference_feedback_manager=0,gl_context=0'


from ambient.gavd import GAVDDataLoader
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from ambient.pose.joint_angles import get_joint_angles

from ambient.classification.knn_classifier import (
    KNNGaitClassifier,
    KNNClassifierConfig,
    GaitFeatureVector
)
from ambient.utils.log_config import get_logger

logger = get_logger()
logger.remove()
logger.add(sys.stderr, level="INFO")

from ambient.utils.path_utils import get_project_root

project_root = get_project_root()

def setup_paths() -> Tuple[Path, Path, Path, Path]:
    """Setup project paths."""
    video_base_path = project_root / "data" / "youtube"
    data_root = project_root / "data" / "gavd"
    output_dir = project_root / "experiments" / "exp2" / "models"
    
    assert video_base_path.exists(), f"video_base_path does not exist: {video_base_path}"
    assert data_root.exists(), f"data_root does not exist: {data_root}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return project_root, video_base_path, data_root, output_dir


def get_condition_paths(data_root: Path) -> List[Path]:
    """Get paths to condition directories."""
    condition_paths = [
        p for p in data_root.iterdir() 
        if p.is_dir() and p.name[0] in string.ascii_letters
    ]
    
    logger.info(f"Found {len(condition_paths)} condition directories")
    for path in condition_paths:
        logger.info(f"  - {path.name}")
    
    return condition_paths


def extract_features_from_sequence(
    sequence_df: pd.DataFrame,
    video_base_path: Path,
    seq_id: str,
    condition_name: str
) -> GaitFeatureVector:
    """Extract gait features from a sequence."""
    extractor = SequenceKeypointExtractor()
    
    keypoints_array = extractor.extract_from_sequence(
        sequence_df,
        video_base_path=video_base_path,
        verbose=False
    )
    
    if not keypoints_array:
        logger.warning(f"No keypoints extracted for {seq_id}")
        return None
    
    joint_angles = get_joint_angles(
        keypoints_array=keypoints_array,
        keypoint_format="BLAZEPOSE_33",
        fps=30.0,
        confidence_threshold=0.3,
        sequence_id=seq_id
    )
    
    if len(joint_angles.frames) == 0:
        logger.warning(f"No joint angle frames computed for {seq_id}")
        return None
    
    if len(joint_angles.frames[0].angles) == 0:
        logger.warning(f"First frame has no joint angles for {seq_id}")
        return None
    
    feature_vector = GaitFeatureVector.from_joint_angles(
        joint_angles,
        sample_id=seq_id,
        condition_label=condition_name
    )
    
    return feature_vector


def load_training_data(
    condition_paths: List[Path],
    video_base_path: Path
) -> Tuple[List[GaitFeatureVector], Dict[str, int]]:
    """Load training data from all condition directories."""
    all_features = []
    condition_counts = defaultdict(int)
    gavd_loader = GAVDDataLoader()
    
    for condition_path in condition_paths:
        condition_name = condition_path.name
        logger.info(f"\nProcessing condition: {condition_name}")
        
        csv_files = list(condition_path.glob("*.csv"))
        logger.info(f"  Found {len(csv_files)} CSV files")
        
        for csv_path in csv_files:
            try:
                df = gavd_loader.load_gavd_data(str(csv_path))
                sequences = gavd_loader.organize_by_sequence(df)
                
                logger.info(f"  Processing {csv_path.name}: {len(sequences)} sequences")
                
                for seq_id in sequences:
                    sequence_df = sequences[seq_id]
                    
                    try:
                        feature_vector = extract_features_from_sequence(
                            sequence_df,
                            video_base_path,
                            seq_id,
                            condition_name
                        )
                        
                        if feature_vector is not None:
                            all_features.append(feature_vector)
                            condition_counts[condition_name] += 1
                            logger.info(f"    ✓ {seq_id}: extracted features")
                        else:
                            logger.warning(f"    ✗ {seq_id}: feature extraction failed")
                    
                    except Exception as e:
                        logger.error(f"    ✗ {seq_id}: {str(e)}")
                        continue
            
            except Exception as e:
                logger.error(f"  Cannot process {csv_path.name}: {str(e)}")
                continue
    
    logger.info(f"\nTotal features extracted: {len(all_features)}")
    logger.info("Condition distribution:")
    for condition, count in sorted(condition_counts.items()):
        logger.info(f"  {condition}: {count} samples")
    
    return all_features, dict(condition_counts)



def train_classifier(
    features: List[GaitFeatureVector],
    config: KNNClassifierConfig
) -> Tuple[KNNGaitClassifier, Dict]:
    """Train KNN classifier."""
    logger.info("\n" + "="*60)
    logger.info("Training KNN Classifier")
    logger.info("="*60)
    
    classifier = KNNGaitClassifier(config=config)
    metrics = classifier.train(features, validate=True)
    
    logger.info("\nTraining Results:")
    logger.info(f"  Training Accuracy: {metrics['train_accuracy']:.3f}")
    if 'cv_mean_accuracy' in metrics:
        logger.info(f"  CV Accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")
    logger.info(f"  Number of Samples: {metrics['n_samples']}")
    logger.info(f"  Number of Features: {metrics['n_features']}")
    logger.info(f"  Classes: {metrics['classes']}")
    
    return classifier, metrics


def evaluate_classifier(
    classifier: KNNGaitClassifier,
    test_features: List[GaitFeatureVector]
) -> Dict:
    """Evaluate classifier on test data."""
    logger.info("\n" + "="*60)
    logger.info("Evaluating Classifier")
    logger.info("="*60)
    
    metrics = classifier.evaluate(test_features)
    
    logger.info(f"\nTest Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"Number of Test Samples: {metrics['n_test_samples']}")
    
    logger.info("\nClassification Report:")
    report = metrics['classification_report']
    for class_name in metrics['classes']:
        if class_name in report:
            class_metrics = report[class_name]
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {class_metrics['precision']:.3f}")
            logger.info(f"    Recall: {class_metrics['recall']:.3f}")
            logger.info(f"    F1-Score: {class_metrics['f1-score']:.3f}")
    
    return metrics


def save_results(
    classifier: KNNGaitClassifier,
    training_metrics: Dict,
    evaluation_metrics: Dict,
    condition_counts: Dict,
    output_dir: Path
) -> None:
    """Save classifier and results."""
    model_path = output_dir / "knn_classifier.pkl"
    classifier.save(model_path)
    logger.info(f"\nClassifier saved to: {model_path}")
    
    results_dir = output_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = results_dir / "training_metrics.json"
    all_metrics = {
        "training": training_metrics,
        "evaluation": evaluation_metrics,
        "data_distribution": condition_counts
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_path}")
    
    report_path = results_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("KNN Gait Classifier - Evaluation Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("Training Metrics:\n")
        f.write(f"  Training Accuracy: {training_metrics['train_accuracy']:.3f}\n")
        if 'cv_mean_accuracy' in training_metrics:
            f.write(f"  CV Accuracy: {training_metrics['cv_mean_accuracy']:.3f} ± {training_metrics['cv_std_accuracy']:.3f}\n")
        f.write(f"  Number of Samples: {training_metrics['n_samples']}\n")
        f.write(f"  Number of Features: {training_metrics['n_features']}\n\n")
        
        f.write("Test Metrics:\n")
        f.write(f"  Test Accuracy: {evaluation_metrics['accuracy']:.3f}\n")
        f.write(f"  Number of Test Samples: {evaluation_metrics['n_test_samples']}\n\n")
        
        f.write("Classification Report:\n")
        report = evaluation_metrics['classification_report']
        for class_name in evaluation_metrics['classes']:
            if class_name in report:
                class_metrics = report[class_name]
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {class_metrics['precision']:.3f}\n")
                f.write(f"    Recall: {class_metrics['recall']:.3f}\n")
                f.write(f"    F1-Score: {class_metrics['f1-score']:.3f}\n")
        
        f.write("\nData Distribution:\n")
        for condition, count in sorted(condition_counts.items()):
            f.write(f"  {condition}: {count} samples\n")
    
    logger.info(f"Report saved to: {report_path}")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("KNN Gait Classifier Training Pipeline")
    logger.info("="*60)
    
    project_root, video_base_path, data_root, output_dir = setup_paths()
    condition_paths = get_condition_paths(data_root)
    
    if not condition_paths:
        logger.error("No condition directories found!")
        return
    
    logger.info("\n" + "="*60)
    logger.info("Loading Training Data")
    logger.info("="*60)
    
    all_features, condition_counts = load_training_data(
        condition_paths,
        video_base_path
    )
    
    if not all_features:
        logger.error("No features extracted! Cannot train classifier.")
        return
    
    np.random.seed(42)
    indices = np.random.permutation(len(all_features))
    split_idx = int(0.8 * len(all_features))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_features = [all_features[i] for i in train_indices]
    test_features = [all_features[i] for i in test_indices]
    
    logger.info(f"\nTrain/Test Split:")
    logger.info(f"  Training samples: {len(train_features)}")
    logger.info(f"  Test samples: {len(test_features)}")
    
    config = KNNClassifierConfig(
        n_neighbors=5,
        weights="distance",
        metric="euclidean",
        normalize_features=True
    )
    
    classifier, training_metrics = train_classifier(train_features, config)
    
    if test_features:
        evaluation_metrics = evaluate_classifier(classifier, test_features)
    else:
        logger.warning("No test samples available for evaluation")
        evaluation_metrics = {}
    
    save_results(
        classifier,
        training_metrics,
        evaluation_metrics,
        condition_counts,
        output_dir
    )
    
    logger.info("\n" + "="*60)
    logger.info("Training Pipeline Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
