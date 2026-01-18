#!/usr/bin/env python3
"""
Clean GAVD dataset metadata by removing sequence-level fields.

This script removes 'seq' and 'gait_pat' fields from dataset metadata files
since these are sequence-level properties, not dataset-level properties.
"""

import json
from pathlib import Path
from loguru import logger

def clean_metadata_files(metadata_dir: Path) -> None:
    """
    Remove seq and gait_pat fields from all GAVD dataset metadata files.
    
    Args:
        metadata_dir: Path to metadata directory
    """
    if not metadata_dir.exists():
        logger.warning(f"Metadata directory does not exist: {metadata_dir}")
        return
    
    metadata_files = list(metadata_dir.glob("*.json"))
    logger.info(f"Found {len(metadata_files)} metadata files")
    
    cleaned_count = 0
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if problematic fields exist
            has_seq = 'seq' in metadata
            has_gait_pat = 'gait_pat' in metadata
            
            if has_seq or has_gait_pat:
                # Remove the fields
                if has_seq:
                    del metadata['seq']
                    logger.info(f"Removed 'seq' from {metadata_file.name}")
                if has_gait_pat:
                    del metadata['gait_pat']
                    logger.info(f"Removed 'gait_pat' from {metadata_file.name}")
                
                # Write back
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                cleaned_count += 1
        except Exception as e:
            logger.error(f"Error processing {metadata_file}: {str(e)}")
    
    logger.info(f"Cleaned {cleaned_count} metadata files")

if __name__ == "__main__":
    # Default path - adjust if needed
    metadata_dir = Path("data/training/gavd/metadata")
    
    logger.info("Starting GAVD metadata cleanup...")
    clean_metadata_files(metadata_dir)
    logger.info("Cleanup complete!")
