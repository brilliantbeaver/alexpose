"""
Test script to verify real keypoint extraction is working in GAVD processor.

This script tests the complete pipeline from GAVD CSV to real pose keypoints.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
import pandas as pd
from ambient.gavd.gavd_processor import (
    GAVDDataLoader,
    PoseDataConverter,
    PoseKeypointExtractor,
)


def test_keypoint_extractor():
    """Test PoseKeypointExtractor initialization."""
    logger.info("=" * 60)
    logger.info("Test 1: PoseKeypointExtractor Initialization")
    logger.info("=" * 60)
    
    extractor = PoseKeypointExtractor()
    logger.info(f"✓ Extractor created: {extractor}")
    logger.info(f"  - Has sequence_extractor: {extractor.sequence_extractor is not None}")
    logger.info(f"  - Has bbox_processor: {extractor.bbox_processor is not None}")
    logger.info(f"  - Has keypoint_generator: {extractor.keypoint_generator is not None}")
    
    # Test lazy initialization
    seq_ext = extractor._ensure_sequence_extractor()
    logger.info(f"  - Sequence extractor initialized: {seq_ext is not None}")
    
    return extractor


def test_pose_data_converter():
    """Test PoseDataConverter initialization."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: PoseDataConverter Initialization")
    logger.info("=" * 60)
    
    converter = PoseDataConverter()
    logger.info(f"✓ Converter created: {converter}")
    logger.info(f"  - Has keypoint_extractor: {converter.keypoint_extractor is not None}")
    logger.info(f"  - Has estimator: {converter.estimator is not None}")
    logger.info(f"  - Video cache dir: {converter.video_cache_dir}")
    
    return converter


def test_with_sample_data():
    """Test with sample GAVD data."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Processing Sample GAVD Data")
    logger.info("=" * 60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'seq': ['test_seq_001'] * 3,
        'frame_num': [1757, 1758, 1759],
        'bbox': [
            {'left': 100, 'top': 50, 'width': 200, 'height': 400},
            {'left': 105, 'top': 52, 'width': 198, 'height': 398},
            {'left': 110, 'top': 54, 'width': 196, 'height': 396},
        ],
        'url': ['https://www.youtube.com/watch?v=dQw4w9WgXcQ'] * 3,
        'gait_pat': ['normal'] * 3,
        'cam_view': ['front'] * 3,
        'gait_event': ['walking'] * 3,
        'dataset': ['GAVD'] * 3,
        'vid_info': [{}] * 3,
    })
    
    logger.info(f"Sample data created: {len(sample_data)} frames")
    logger.info(f"  - Sequence: {sample_data['seq'].iloc[0]}")
    logger.info(f"  - Frame range: {sample_data['frame_num'].min()} - {sample_data['frame_num'].max()}")
    logger.info(f"  - URL: {sample_data['url'].iloc[0]}")
    
    # Create converter
    converter = PoseDataConverter()
    
    # Check if video is cached
    video_path = converter._resolve_cached_video_path(sample_data['url'].iloc[0])
    logger.info(f"  - Video cached: {video_path is not None}")
    if video_path:
        logger.info(f"  - Video path: {video_path}")
    
    # Try to convert (will use fallback if video not available)
    logger.info("\nAttempting to convert sequence to pose format...")
    try:
        pose_data = converter.convert_sequence_to_pose_format(
            sample_data,
            include_metadata=True,
            num_keypoints=25,
        )
        
        logger.info(f"\n✓ Conversion successful!")
        logger.info(f"  - Generated {len(pose_data)} pose frames")
        
        if pose_data:
            first_frame = pose_data[0]
            keypoints = first_frame.get('pose_keypoints_2d', [])
            logger.info(f"  - First frame has {len(keypoints)} keypoints")
            
            if keypoints:
                first_kp = keypoints[0]
                logger.info(f"  - First keypoint: x={first_kp.get('x', 0):.2f}, "
                          f"y={first_kp.get('y', 0):.2f}, "
                          f"confidence={first_kp.get('confidence', 0):.3f}")
                
                # Check if these are real keypoints or grid
                # Real keypoints from MediaPipe have 33 keypoints
                # Grid keypoints have 25 by default
                if len(keypoints) == 33:
                    logger.info("  ✓ REAL KEYPOINTS DETECTED (33 MediaPipe keypoints)")
                elif len(keypoints) == 25:
                    logger.warning("  ⚠ GRID KEYPOINTS DETECTED (25 placeholder keypoints)")
                    logger.warning("    This means real extraction failed or video not available")
                else:
                    logger.info(f"  - Keypoint count: {len(keypoints)}")
        
        return pose_data
        
    except Exception as e:
        logger.error(f"✗ Conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def test_with_real_gavd_file():
    """Test with actual GAVD CSV file if available."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Processing Real GAVD File")
    logger.info("=" * 60)
    
    # Look for GAVD CSV files
    gavd_dir = project_root / "data" / "gavd"
    if not gavd_dir.exists():
        logger.warning("GAVD directory not found, skipping real file test")
        return None
    
    # Find first CSV file
    csv_files = list(gavd_dir.rglob("*.csv"))
    if not csv_files:
        logger.warning("No GAVD CSV files found, skipping real file test")
        return None
    
    csv_file = csv_files[0]
    logger.info(f"Found GAVD file: {csv_file}")
    
    try:
        # Load data
        loader = GAVDDataLoader()
        df = loader.load_gavd_data(str(csv_file), verbose=False)
        logger.info(f"✓ Loaded {len(df)} rows")
        
        # Get first sequence
        sequences = loader.organize_by_sequence(df, verbose=False)
        if not sequences:
            logger.warning("No sequences found")
            return None
        
        first_seq_id = list(sequences.keys())[0]
        first_seq = sequences[first_seq_id]
        logger.info(f"✓ Processing sequence: {first_seq_id} ({len(first_seq)} frames)")
        
        # Take only first 3 frames for testing
        test_seq = first_seq.head(3)
        
        # Convert to pose format
        converter = PoseDataConverter()
        pose_data = converter.convert_sequence_to_pose_format(
            test_seq,
            include_metadata=True,
        )
        
        logger.info(f"\n✓ Conversion successful!")
        logger.info(f"  - Generated {len(pose_data)} pose frames")
        
        if pose_data:
            first_frame = pose_data[0]
            keypoints = first_frame.get('pose_keypoints_2d', [])
            logger.info(f"  - First frame has {len(keypoints)} keypoints")
            
            if len(keypoints) == 33:
                logger.info("  ✓✓✓ REAL KEYPOINTS DETECTED (33 MediaPipe keypoints)")
                logger.info("  SUCCESS: Real pose extraction is working!")
            elif len(keypoints) == 25:
                logger.warning("  ⚠⚠⚠ GRID KEYPOINTS DETECTED (25 placeholder keypoints)")
                logger.warning("  ISSUE: Real extraction is not working - still using placeholders")
            
            # Show keypoint details
            if keypoints:
                logger.info(f"\n  Keypoint sample:")
                for i, kp in enumerate(keypoints[:5]):
                    logger.info(f"    [{i}] x={kp.get('x', 0):.1f}, "
                              f"y={kp.get('y', 0):.1f}, "
                              f"conf={kp.get('confidence', 0):.3f}")
        
        return pose_data
        
    except Exception as e:
        logger.error(f"✗ Real file test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Run all tests."""
    logger.info("Starting Real Keypoint Extraction Tests")
    logger.info("=" * 60)
    
    # Test 1: Extractor initialization
    extractor = test_keypoint_extractor()
    
    # Test 2: Converter initialization
    converter = test_pose_data_converter()
    
    # Test 3: Sample data
    sample_result = test_with_sample_data()
    
    # Test 4: Real GAVD file
    real_result = test_with_real_gavd_file()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✓ Extractor initialization: PASSED")
    logger.info(f"✓ Converter initialization: PASSED")
    logger.info(f"{'✓' if sample_result else '✗'} Sample data processing: {'PASSED' if sample_result else 'FAILED'}")
    logger.info(f"{'✓' if real_result else '⚠'} Real file processing: {'PASSED' if real_result else 'SKIPPED/FAILED'}")
    
    if real_result:
        keypoints = real_result[0].get('pose_keypoints_2d', [])
        if len(keypoints) == 33:
            logger.info("\n🎉 SUCCESS: Real keypoint extraction is working!")
        else:
            logger.warning("\n⚠ WARNING: Still using placeholder grid keypoints")
            logger.warning("Check logs above for error details")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
