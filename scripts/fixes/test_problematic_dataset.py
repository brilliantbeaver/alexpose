"""Test the problematic dataset with frames 205-355."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ambient.gavd.gavd_processor import create_gavd_processor
from loguru import logger

csv_file = "data/training/gavd/470be6bd-f4b4-446f-a140-42e9d36bdceb.csv"

logger.info(f"Processing {csv_file}")
processor = create_gavd_processor()
results = processor.process_gavd_file(
    csv_file_path=csv_file,
    max_sequences=None,
    include_metadata=True,
    verbose=True
)

total_frames = results["summary"]["total_frames"]
logger.info(f"Total frames processed: {total_frames} out of 151")

if total_frames == 151:
    logger.info("✓ SUCCESS: All 151 frames processed!")
else:
    logger.error(f"❌ FAIL: Only {total_frames}/151 frames processed")
    sys.exit(1)
