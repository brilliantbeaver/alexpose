"""
Script to reset stuck GAVD datasets from 'processing' to 'uploaded' status.

This is useful when datasets get stuck in processing state due to:
- Long processing times
- Server restarts
- Crashes during processing
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


def reset_stuck_datasets():
    """Reset all datasets stuck in 'processing' status."""
    
    metadata_dir = project_root / "data" / "training" / "gavd" / "metadata"
    
    if not metadata_dir.exists():
        logger.error(f"Metadata directory not found: {metadata_dir}")
        return
    
    logger.info(f"Checking for stuck datasets in: {metadata_dir}")
    
    stuck_datasets = []
    
    # Find all metadata files
    for metadata_file in metadata_dir.glob("*.json"):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            status = metadata.get("status")
            dataset_id = metadata.get("dataset_id")
            filename = metadata.get("original_filename")
            
            if status == "processing":
                stuck_datasets.append({
                    "dataset_id": dataset_id,
                    "filename": filename,
                    "metadata_file": metadata_file,
                    "metadata": metadata
                })
                logger.warning(f"Found stuck dataset: {filename} ({dataset_id})")
                
        except Exception as e:
            logger.error(f"Error reading {metadata_file}: {e}")
    
    if not stuck_datasets:
        logger.info("No stuck datasets found!")
        return
    
    logger.info(f"\nFound {len(stuck_datasets)} stuck dataset(s)")
    
    # Ask for confirmation
    print("\nThe following datasets will be reset:")
    for ds in stuck_datasets:
        print(f"  - {ds['filename']} ({ds['dataset_id']})")
    
    response = input("\nReset these datasets? (yes/no): ").strip().lower()
    
    if response != "yes":
        logger.info("Reset cancelled")
        return
    
    # Reset each dataset
    for ds in stuck_datasets:
        try:
            metadata = ds["metadata"]
            metadata["status"] = "uploaded"
            metadata["progress"] = "Ready to process"
            metadata["progress_percent"] = 0
            metadata["frames_processed"] = 0
            metadata["reset_at"] = datetime.utcnow().isoformat()
            metadata["previous_status"] = "processing"
            
            # Save updated metadata
            with open(ds["metadata_file"], 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.success(f"Reset: {ds['filename']}")
            
        except Exception as e:
            logger.error(f"Failed to reset {ds['filename']}: {e}")
    
    logger.success(f"\nReset complete! {len(stuck_datasets)} dataset(s) reset to 'uploaded' status")
    logger.info("You can now process them again from the frontend")


if __name__ == "__main__":
    reset_stuck_datasets()
