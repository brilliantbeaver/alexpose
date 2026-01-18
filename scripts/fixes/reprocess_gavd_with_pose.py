#!/usr/bin/env python3
"""
Reprocess GAVD dataset with real pose estimation.

This script reprocesses an existing GAVD dataset to generate real pose keypoints
instead of placeholder grid keypoints.

Usage:
    python scripts/reprocess_gavd_with_pose.py <dataset_id> [--estimator mediapipe]
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.gavd import GAVDProcessor, PoseDataConverter
from ambient.pose import get_pose_estimator
from server.services.gavd_service import GAVDService
from ambient.core.config import ConfigurationManager


def reprocess_dataset(dataset_id: str, pose_estimator: str = "mediapipe"):
    """
    Reprocess a GAVD dataset with real pose estimation.
    
    Args:
        dataset_id: Dataset ID to reprocess
        pose_estimator: Pose estimator to use (mediapipe, openpose, ultralytics)
    """
    logger.info(f"Reprocessing dataset {dataset_id} with {pose_estimator}")
    
    # Load configuration
    config_manager = ConfigurationManager()
    
    # Create GAVD service
    gavd_service = GAVDService(config_manager)
    
    # Get dataset metadata
    metadata = gavd_service.get_dataset_metadata(dataset_id)
    if not metadata:
        logger.error(f"Dataset {dataset_id} not found")
        return False
    
    logger.info(f"Dataset: {metadata['original_filename']}")
    logger.info(f"Status: {metadata['status']}")
    logger.info(f"Sequences: {metadata.get('sequence_count', 'unknown')}")
    
    # Get CSV file path
    csv_file = metadata['file_path']
    if not Path(csv_file).exists():
        logger.error(f"CSV file not found: {csv_file}")
        return False
    
    # Create pose estimator
    try:
        logger.info(f"Initializing {pose_estimator} pose estimator...")
        estimator = get_pose_estimator(pose_estimator)
        logger.info(f"✓ Pose estimator initialized: {estimator.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to initialize pose estimator: {e}")
        logger.info("Falling back to placeholder keypoints")
        estimator = None
    
    # Create converter with estimator
    converter = PoseDataConverter(estimator=estimator)
    
    # Create processor
    processor = GAVDProcessor(data_converter=converter)
    
    # Reprocess the dataset
    try:
        logger.info("Starting pose estimation processing...")
        logger.info("This may take several minutes depending on video length...")
        
        # Process with pose estimation
        gavd_service.process_dataset(
            dataset_id,
            max_sequences=None,  # Process all sequences
            pose_estimator=pose_estimator
        )
        
        logger.success(f"✓ Dataset {dataset_id} reprocessed successfully")
        logger.info(f"Pose data saved to: data/training/gavd/results/{dataset_id}_pose_data.json")
        return True
        
    except Exception as e:
        logger.error(f"Failed to reprocess dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Reprocess GAVD dataset with real pose estimation"
    )
    parser.add_argument(
        "dataset_id",
        help="Dataset ID to reprocess"
    )
    parser.add_argument(
        "--estimator",
        default="mediapipe",
        choices=["mediapipe", "openpose", "ultralytics"],
        help="Pose estimator to use (default: mediapipe)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    if args.list:
        # List available datasets
        config_manager = ConfigurationManager()
        gavd_service = GAVDService(config_manager)
        datasets = gavd_service.list_datasets(limit=100)
        
        logger.info(f"Found {len(datasets)} datasets:")
        for ds in datasets:
            logger.info(f"  {ds['dataset_id']}: {ds['original_filename']} ({ds['status']})")
        return
    
    # Reprocess dataset
    success = reprocess_dataset(args.dataset_id, args.estimator)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
