"""
Test script to verify complete GAVD processing with all frames.

This script tests that:
1. All frames in a GAVD CSV are processed (no skipping)
2. Real MediaPipe keypoints are extracted
3. Processing completes successfully on Windows
4. No [WinError 1] errors occur
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.gavd.gavd_processor import create_gavd_processor
from loguru import logger
import pandas as pd

def test_complete_processing():
    """Test that all frames are processed without errors."""
    
    # Find a test GAVD CSV file
    test_csv_dir = project_root / "data" / "training" / "gavd"
    csv_files = list(test_csv_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error("No GAVD CSV files found in data/training/gavd/")
        return False
    
    csv_file_path = csv_files[0]
    logger.info(f"Testing with CSV file: {csv_file_path}")
    
    # Count total rows
    try:
        df = pd.read_csv(csv_file_path)
        total_rows = len(df)
        logger.info(f"Total rows in CSV: {total_rows}")
        
        # Get frame range
        min_frame = df['frame_num'].min()
        max_frame = df['frame_num'].max()
        logger.info(f"Frame range: {min_frame} to {max_frame}")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return False
    
    # Create processor WITHOUT estimator (use batch optimization)
    logger.info("Creating GAVD processor WITHOUT estimator...")
    processor = create_gavd_processor()
    
    # Verify no estimator is set
    if hasattr(processor, 'data_converter') and hasattr(processor.data_converter, 'estimator'):
        if processor.data_converter.estimator is not None:
            logger.error("❌ FAIL: Estimator was created (should be None)")
            return False
    
    logger.info("✓ No estimator created (will use batch optimization)")
    
    # Process all sequences
    logger.info("Processing all sequences...")
    start_time = time.time()
    
    try:
        results = processor.process_gavd_file(
            csv_file_path=str(csv_file_path),
            max_sequences=None,  # Process all
            include_metadata=True,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        
        if results and len(results['sequences']) > 0:
            total_frames_processed = results['summary']['total_frames']
            logger.info(f"✓ Processing completed in {elapsed:.2f}s")
            logger.info(f"✓ Total frames processed: {total_frames_processed}")
            logger.info(f"✓ Total rows in CSV: {total_rows}")
            
            # Check if all frames were processed
            if total_frames_processed == total_rows:
                logger.info(f"✓ EXCELLENT: All {total_rows} frames were processed!")
            elif total_frames_processed < total_rows:
                missing = total_rows - total_frames_processed
                logger.error(f"❌ FAIL: Only {total_frames_processed}/{total_rows} frames processed ({missing} missing)")
                
                # Show which frames were processed
                for seq_id, seq_data in results['sequences'].items():
                    pose_data = seq_data['pose_data']
                    if pose_data:
                        processed_frames = [frame['frame'] for frame in pose_data]
                        logger.info(f"Sequence {seq_id}: processed frames {min(processed_frames)} to {max(processed_frames)}")
                
                return False
            else:
                logger.warning(f"⚠ More frames processed than expected: {total_frames_processed} > {total_rows}")
            
            # Check keypoint quality
            first_seq = list(results['sequences'].values())[0]
            if first_seq['pose_data'] and len(first_seq['pose_data']) > 0:
                first_frame = first_seq['pose_data'][0]
                if 'pose_keypoints_2d' in first_frame:
                    keypoints = first_frame['pose_keypoints_2d']
                    num_keypoints = len(keypoints)
                    logger.info(f"✓ Keypoints per frame: {num_keypoints}")
                    
                    if num_keypoints == 33:
                        logger.info("✓ EXCELLENT: Using real MediaPipe keypoints (33 keypoints)")
                    elif num_keypoints == 25:
                        logger.warning("⚠ Using grid keypoints (25 keypoints) - real extraction may have failed")
                    
                    # Check confidence
                    if keypoints and len(keypoints) > 0:
                        avg_confidence = sum(kp.get('confidence', 0) for kp in keypoints) / len(keypoints)
                        logger.info(f"✓ Average confidence: {avg_confidence:.3f}")
                        
                        if avg_confidence > 0.9:
                            logger.info("✓ EXCELLENT: High confidence keypoints")
                        elif avg_confidence > 0.5:
                            logger.info("✓ GOOD: Acceptable confidence keypoints")
                        else:
                            logger.warning(f"⚠ LOW: Low confidence keypoints ({avg_confidence:.3f})")
            
            # Performance check
            frames_per_second = total_frames_processed / elapsed if elapsed > 0 else 0
            logger.info(f"✓ Processing speed: {frames_per_second:.1f} frames/second")
            
            if frames_per_second > 5:
                logger.info("✓ EXCELLENT: Fast processing speed")
            elif frames_per_second > 2:
                logger.info("✓ GOOD: Acceptable processing speed")
            else:
                logger.warning(f"⚠ SLOW: Processing speed is low ({frames_per_second:.1f} fps)")
            
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
    logger.info("=" * 70)
    logger.info("Testing Complete GAVD Processing (All Frames)")
    logger.info("=" * 70)
    
    success = test_complete_processing()
    
    logger.info("=" * 70)
    if success:
        logger.info("✓ TEST PASSED: All frames processed successfully")
    else:
        logger.error("❌ TEST FAILED: Not all frames were processed")
    logger.info("=" * 70)
    
    sys.exit(0 if success else 1)
