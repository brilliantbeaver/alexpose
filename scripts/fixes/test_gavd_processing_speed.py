"""
Test script to verify GAVD processing speed improvements.

This tests the batch video processing optimization.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from ambient.gavd.gavd_processor import GAVDProcessor, create_gavd_processor


def test_processing_speed():
    """Test GAVD processing speed with batch optimization."""
    
    # Find a GAVD CSV file
    gavd_dir = project_root / "data" / "gavd"
    csv_files = list(gavd_dir.rglob("*.csv"))
    
    if not csv_files:
        logger.error("No GAVD CSV files found")
        return
    
    csv_file = csv_files[0]
    logger.info(f"Testing with: {csv_file}")
    
    # Create processor (no estimator - uses batch fallback)
    processor = create_gavd_processor()
    
    # Process with timing
    logger.info("Starting processing...")
    start_time = time.time()
    
    try:
        results = processor.process_gavd_file(
            csv_file_path=str(csv_file),
            max_sequences=1,  # Just test 1 sequence
            include_metadata=True,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        
        logger.success(f"\n{'='*60}")
        logger.success(f"Processing completed in {elapsed:.2f} seconds")
        logger.success(f"{'='*60}")
        logger.info(f"Sequences processed: {results['total_sequences']}")
        logger.info(f"Total frames: {results['summary']['total_frames']}")
        logger.info(f"Average time per frame: {elapsed / results['summary']['total_frames']:.3f}s")
        
        # Check if we got real keypoints
        if results['sequences']:
            first_seq = list(results['sequences'].values())[0]
            if first_seq['pose_data']:
                first_frame = first_seq['pose_data'][0]
                keypoints = first_frame.get('pose_keypoints_2d', [])
                logger.info(f"Keypoints per frame: {len(keypoints)}")
                
                if len(keypoints) == 33:
                    logger.success("✓ Real MediaPipe keypoints detected!")
                elif len(keypoints) == 25:
                    logger.warning("⚠ Grid placeholder keypoints detected")
                else:
                    logger.info(f"Keypoint count: {len(keypoints)}")
        
        # Performance expectations
        frames = results['summary']['total_frames']
        time_per_frame = elapsed / frames if frames > 0 else 0
        
        logger.info(f"\n{'='*60}")
        logger.info("Performance Analysis:")
        logger.info(f"{'='*60}")
        
        if time_per_frame < 0.5:
            logger.success(f"✓ EXCELLENT: {time_per_frame:.3f}s per frame (batch optimized)")
        elif time_per_frame < 1.0:
            logger.info(f"✓ GOOD: {time_per_frame:.3f}s per frame")
        elif time_per_frame < 2.0:
            logger.warning(f"⚠ SLOW: {time_per_frame:.3f}s per frame")
        else:
            logger.error(f"✗ VERY SLOW: {time_per_frame:.3f}s per frame (not optimized)")
        
        logger.info(f"\nExpected times:")
        logger.info(f"  - 50 frames: ~{50 * time_per_frame:.1f}s")
        logger.info(f"  - 150 frames: ~{150 * time_per_frame:.1f}s")
        logger.info(f"  - 500 frames: ~{500 * time_per_frame:.1f}s")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_processing_speed()
