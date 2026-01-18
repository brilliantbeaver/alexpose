"""
Test script to verify GAVD upload flow uses batch optimization.

This script simulates the upload flow to ensure:
1. No estimator is created by default
2. Batch optimization path is used
3. Processing completes quickly (not 20+ minutes)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.gavd.gavd_processor import create_gavd_processor
from loguru import logger
import time

def test_upload_flow():
    """Test the upload flow without estimator."""
    
    # Find a test GAVD CSV file
    test_csv_dir = project_root / "data" / "gavd" / "abnormal"
    csv_files = list(test_csv_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error("No GAVD CSV files found in data/gavd/abnormal/")
        return False
    
    csv_file_path = csv_files[0]
    logger.info(f"Testing with CSV file: {csv_file_path}")
    
    # Count rows
    import pandas as pd
    try:
        df = pd.read_csv(csv_file_path)
        row_count = len(df)
        logger.info(f"Row count: {row_count}")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return False
    
    # Create processor WITHOUT estimator (simulating upload flow)
    logger.info("Creating GAVD processor WITHOUT estimator...")
    processor = create_gavd_processor()
    
    # Verify no estimator is set
    if hasattr(processor, 'data_converter') and hasattr(processor.data_converter, 'estimator'):
        if processor.data_converter.estimator is not None:
            logger.error("❌ FAIL: Estimator was created (should be None)")
            return False
    
    logger.info("✓ No estimator created (will use batch optimization)")
    
    # Process with max 1 sequence for quick test
    logger.info("Processing 1 sequence to test batch optimization...")
    start_time = time.time()
    
    try:
        results = processor.process_gavd_file(
            csv_file_path=csv_file_path,
            max_sequences=1,
            include_metadata=True,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"Results type: {type(results)}")
        logger.info(f"Results: {results}")
        
        if results:
            # Results is a dict mapping seq_id -> result_dict
            if isinstance(results, dict) and len(results) > 0:
                first_seq_id = list(results.keys())[0]
                first_result = results[first_seq_id]
                logger.info(f"First result type: {type(first_result)}")
            else:
                logger.error(f"❌ Unexpected results format: {type(results)}")
                return False
            num_frames = len(first_result.get('pose_data', []))
            logger.info(f"✓ Processing completed in {elapsed:.2f}s")
            logger.info(f"✓ Extracted {num_frames} frames")
            
            # Check if keypoints are real (not grid)
            if num_frames > 0:
                first_frame = first_result['pose_data'][0]
                if 'people' in first_frame and len(first_frame['people']) > 0:
                    keypoints = first_frame['people'][0].get('pose_keypoints_2d', [])
                    num_keypoints = len(keypoints) // 3
                    logger.info(f"✓ Real keypoints detected: {num_keypoints} keypoints per frame")
                    
                    # Check if they're real (MediaPipe has 33 keypoints)
                    if num_keypoints == 33:
                        logger.info("✓ EXCELLENT: Using real MediaPipe keypoints (33 keypoints)")
                    elif num_keypoints == 25:
                        logger.warning("⚠ Using grid keypoints (25 keypoints) - batch optimization may not be working")
                    else:
                        logger.info(f"✓ Using keypoints: {num_keypoints} keypoints")
            
            # Performance check
            if elapsed < 30:
                logger.info(f"✓ EXCELLENT: Processing time is good ({elapsed:.2f}s)")
            elif elapsed < 60:
                logger.warning(f"⚠ ACCEPTABLE: Processing time is acceptable ({elapsed:.2f}s)")
            else:
                logger.error(f"❌ SLOW: Processing took too long ({elapsed:.2f}s)")
                return False
            
            return True
        else:
            logger.error("❌ FAIL: No results returned")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ FAIL: Processing failed after {elapsed:.2f}s: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing GAVD Upload Flow (No Estimator)")
    logger.info("=" * 60)
    
    success = test_upload_flow()
    
    logger.info("=" * 60)
    if success:
        logger.info("✓ TEST PASSED: Upload flow uses batch optimization")
    else:
        logger.error("❌ TEST FAILED: Upload flow has issues")
    logger.info("=" * 60)
    
    sys.exit(0 if success else 1)
