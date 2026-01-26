"""
Test script to verify realtime data flow and keypoint format.
Run this to see what data is actually being sent from backend to frontend.
"""

import asyncio
import base64
import json
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ambient.realtime.stream_processor import StreamProcessor
from ambient.realtime.interfaces import ProcessingMode


async def test_pose_detection():
    """Test pose detection with a sample image."""
    
    print("=" * 80)
    print("REALTIME POSE DETECTION TEST")
    print("=" * 80)
    
    # Create stream processor
    processor = StreamProcessor(
        processing_mode=ProcessingMode.BALANCED,
        buffer_size=30,
        enable_tracking=False
    )
    
    # Start processing
    processor.start_processing()
    print("\n✓ Stream processor started")
    
    # Create a test image (solid color with some noise)
    # In real scenario, this would be from webcam
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Convert to JPEG and base64 (simulating frontend)
    _, buffer = cv2.imencode('.jpg', test_image)
    base64_data = base64.b64encode(buffer).decode('utf-8')
    
    print(f"\n✓ Created test image: {test_image.shape}")
    print(f"✓ Encoded to base64: {len(base64_data)} characters")
    
    # Process frame
    print("\n" + "-" * 80)
    print("PROCESSING FRAME...")
    print("-" * 80)
    
    result = await processor.process_frame(base64_data)
    
    # Analyze result
    print("\n" + "=" * 80)
    print("RESULT ANALYSIS")
    print("=" * 80)
    
    print(f"\nSuccess: {result.get('success', False)}")
    
    if result.get('success'):
        pose = result.get('pose', {})
        
        print(f"\nPose Data Structure:")
        print(f"  - Has 'pose' key: {bool(pose)}")
        print(f"  - Pose keys: {list(pose.keys()) if pose else 'N/A'}")
        
        keypoints = pose.get('keypoints', [])
        print(f"\nKeypoints:")
        print(f"  - Type: {type(keypoints)}")
        print(f"  - Count: {len(keypoints) if keypoints else 0}")
        
        if keypoints:
            print(f"  - First keypoint type: {type(keypoints[0])}")
            print(f"  - First keypoint: {keypoints[0]}")
            print(f"\n  Sample of first 3 keypoints:")
            for i, kp in enumerate(keypoints[:3]):
                print(f"    [{i}] {kp}")
        else:
            print("  - WARNING: No keypoints detected!")
            print("  - This could mean:")
            print("    1. No person in frame (expected for test image)")
            print("    2. MediaPipe model not loaded")
            print("    3. Processing error")
        
        confidence_scores = pose.get('confidence_scores', [])
        print(f"\nConfidence Scores:")
        print(f"  - Count: {len(confidence_scores) if confidence_scores else 0}")
        if confidence_scores:
            print(f"  - Average: {sum(confidence_scores) / len(confidence_scores):.3f}")
            print(f"  - Min: {min(confidence_scores):.3f}")
            print(f"  - Max: {max(confidence_scores):.3f}")
        
        print(f"\nProcessing Time: {pose.get('processing_time_ms', 0):.2f}ms")
        print(f"Frame ID: {pose.get('frame_id', 'N/A')}")
        print(f"Timestamp: {pose.get('timestamp', 'N/A')}")
        
        estimator_info = pose.get('estimator_info', {})
        print(f"\nEstimator Info:")
        for key, value in estimator_info.items():
            print(f"  - {key}: {value}")
    
    else:
        error = result.get('error', 'Unknown error')
        print(f"\n✗ Processing failed: {error}")
    
    # Test with actual person image if available
    print("\n" + "=" * 80)
    print("TESTING WITH REAL IMAGE (if available)")
    print("=" * 80)
    
    test_image_path = Path("data/test_data/real")
    if test_image_path.exists():
        image_files = list(test_image_path.glob("*.jpg")) + list(test_image_path.glob("*.png"))
        if image_files:
            real_image_path = image_files[0]
            print(f"\n✓ Found test image: {real_image_path}")
            
            # Load and process real image
            real_image = cv2.imread(str(real_image_path))
            if real_image is not None:
                print(f"✓ Loaded image: {real_image.shape}")
                
                # Encode to base64
                _, buffer = cv2.imencode('.jpg', real_image)
                base64_data = base64.b64encode(buffer).decode('utf-8')
                
                # Process
                result = await processor.process_frame(base64_data)
                
                if result.get('success'):
                    pose = result.get('pose', {})
                    keypoints = pose.get('keypoints', [])
                    print(f"\n✓ Detected {len(keypoints)} keypoints from real image")
                    
                    if keypoints:
                        print(f"\nFirst 3 keypoints from real image:")
                        for i, kp in enumerate(keypoints[:3]):
                            print(f"  [{i}] {kp}")
                else:
                    print(f"\n✗ Failed to process real image: {result.get('error')}")
        else:
            print("\n⚠ No test images found in data/test_data/real/")
    else:
        print("\n⚠ Test data directory not found: data/test_data/real/")
    
    # Stop processor
    processor.stop_processing()
    print("\n✓ Stream processor stopped")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    # Print summary
    print("\nSUMMARY:")
    print("--------")
    if result.get('success'):
        pose = result.get('pose', {})
        keypoints = pose.get('keypoints', [])
        if keypoints:
            print(f"✓ Pose detection working")
            print(f"✓ Keypoint format: {type(keypoints[0])}")
            print(f"✓ Keypoint structure: {list(keypoints[0].keys()) if isinstance(keypoints[0], dict) else 'Not a dict'}")
        else:
            print("⚠ Pose detection working but no keypoints detected")
            print("  (This is expected for random test image)")
    else:
        print("✗ Pose detection failed")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(test_pose_detection())
    
    # Print final JSON structure that would be sent to frontend
    print("\n" + "=" * 80)
    print("JSON STRUCTURE SENT TO FRONTEND")
    print("=" * 80)
    
    # Simulate WebSocket message
    websocket_message = {
        "type": "pose_result",
        "data": result
    }
    
    print(json.dumps(websocket_message, indent=2, default=str))
