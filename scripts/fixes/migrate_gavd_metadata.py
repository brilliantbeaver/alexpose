"""
Migration script to add seq and gait_pat fields to existing GAVD dataset metadata.

This script reads existing GAVD dataset CSV files and updates their metadata
to include the first sequence's seq and gait_pat values.
"""

import json
import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.core.config import ConfigurationManager


def migrate_gavd_metadata():
    """Update existing GAVD dataset metadata with seq and gait_pat values."""
    
    # Load configuration
    config_manager = ConfigurationManager()
    config = config_manager.config
    
    # Get GAVD directories
    training_dir = Path(getattr(config.storage, 'training_directory', 'data/training'))
    gavd_dir = training_dir / 'gavd'
    metadata_dir = gavd_dir / 'metadata'
    
    if not metadata_dir.exists():
        logger.warning(f"Metadata directory does not exist: {metadata_dir}")
        return
    
    # Find all metadata files
    metadata_files = list(metadata_dir.glob("*.json"))
    logger.info(f"Found {len(metadata_files)} metadata files to process")
    
    updated_count = 0
    error_count = 0
    
    for metadata_file in metadata_files:
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if already has seq and gait_pat
            if metadata.get('seq') and metadata.get('gait_pat'):
                logger.debug(f"Skipping {metadata_file.name} - already has seq and gait_pat")
                continue
            
            # Get CSV file path
            csv_path = Path(metadata.get('file_path', ''))
            if not csv_path.exists():
                logger.warning(f"CSV file not found for {metadata_file.name}: {csv_path}")
                error_count += 1
                continue
            
            # Read first row from CSV
            logger.info(f"Processing {metadata_file.name} - reading {csv_path}")
            df = pd.read_csv(csv_path)
            
            if len(df) == 0:
                logger.warning(f"Empty CSV file: {csv_path}")
                error_count += 1
                continue
            
            # Extract first row values
            first_row = df.iloc[0]
            seq = first_row['seq'] if 'seq' in df.columns else None
            gait_pat = first_row['gait_pat'] if 'gait_pat' in df.columns else None
            
            if seq is None or gait_pat is None:
                logger.warning(f"Missing seq or gait_pat columns in {csv_path}")
                error_count += 1
                continue
            
            # Update metadata
            metadata['seq'] = str(seq)
            metadata['gait_pat'] = str(gait_pat)
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.success(f"Updated {metadata_file.name}: seq={seq}, gait_pat={gait_pat}")
            updated_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {metadata_file.name}: {str(e)}")
            error_count += 1
    
    logger.info(f"\nMigration complete:")
    logger.info(f"  Updated: {updated_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Total: {len(metadata_files)}")


if __name__ == "__main__":
    logger.info("Starting GAVD metadata migration...")
    migrate_gavd_metadata()
    logger.info("Migration finished!")
