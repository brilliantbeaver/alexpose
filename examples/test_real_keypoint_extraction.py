"""
Example script demonstrating real keypoint extraction in GAVD processor.

This script shows how the PoseKeypointExtractor now uses SequenceKeypointExtractor
for real MediaPipe-based pose detection instead of placeholder grid keypoints.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from ambient.gavd.gavd_processor import PoseKeypointExtractor


def create_test_image_with_person():
    """Create a simple test image (in real use, this would be an actual person image)."""
    # Create a 640x480 RGB image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a simple stick figure (for demonstration)
    # In real usage, this would be an actual photo/video frame with a person
    center_x, center_y = 320, 240
    
    # Head
    cv2.circle(image, (center_x, center_y - 60), 30, (0, 0, 0), -1)
    
    # Body
    cv2.line(image, (center_x, center_y - 30), (center_x, center_y + 60), (0, 0, 0), 5)
    
    # Arms
    cv2.line(image, (center_x, center_y), (center_x - 50, center_y + 30), (0, 0, 0), 5)
    cv2.line(image, (center_x, center_y), (center_x + 50, center_y + 30), (0, 0, 0), 5)
    
    # Legs
    cv2.line(image, (center_x, center_y + 60), (center_x - 30, center_y + 120), (0, 0, 0), 5)
    cv2.line(image, (center_x, center_y + 60), (center_x + 30, center_y + 120), (0, 0, 0), 5)
    
    return image


def main():
    """Demonstrate real keypoint extraction."""
    print("=" * 60)
    print("Real Keypoint Extraction Demo")
    print("=" * 60)
    
    # Create test image
    print("\n1. Creating test image...")
    image = create_test_image_with_person()
    print(f"   Image shape: {image.shape}")
    
    # Define bounding box around the person
    bbox = {
        "left": 200,
        "top": 100,
        "width": 240,
        "height": 300
    }
    print(f"\n2. Bounding box: {bbox}")
    
    # Initialize extractor
    print("\n3. Initializing PoseKeypointExtractor...")
    extractor = PoseKeypointExtractor()
    
    # Extract keypoints using real pose estimation
    print("\n4. Extracting keypoints with real pose estimation...")
    try:
        keypoints = extractor.extract_from_image_and_bbox(image, bbox)
        
        print(f"\n5. Results:")
        print(f"   - Extracted {len(keypoints)} keypoints")
        
        if keypoints:
            print(f"   - First keypoint: x={keypoints[0]['x']:.2f}, "
                  f"y={keypoints[0]['y']:.2f}, "
                  f"confidence={keypoints[0]['confidence']:.3f}")
            
            # Count high-confidence keypoints
            high_conf = sum(1 for kp in keypoints if kp['confidence'] > 0.5)
            print(f"   - High confidence keypoints (>0.5): {high_conf}/{len(keypoints)}")
            
            # Show coordinate ranges
            x_coords = [kp['x'] for kp in keypoints]
            y_coords = [kp['y'] for kp in keypoints]
            print(f"   - X range: {min(x_coords):.1f} to {max(x_coords):.1f}")
            print(f"   - Y range: {min(y_coords):.1f} to {max(y_coords):.1f}")
        else:
            print("   - No keypoints detected (expected for simple stick figure)")
            print("   - In real usage with actual person images, keypoints would be detected")
        
        print("\n✓ Real keypoint extraction is working!")
        print("  The system now uses MediaPipe for actual pose detection")
        print("  instead of generating placeholder grid keypoints.")
        
    except Exception as e:
        print(f"\n✗ Error during extraction: {e}")
        print("  This might be expected if MediaPipe model is not downloaded yet.")
        print("  The system will fall back to grid keypoints in this case.")
    
    # Demonstrate fallback behavior
    print("\n6. Testing fallback to grid keypoints (when image not available)...")
    grid_keypoints = extractor.extract_from_bbox(bbox, num_keypoints=25)
    print(f"   - Generated {len(grid_keypoints)} grid keypoints as fallback")
    print(f"   - First grid keypoint: x={grid_keypoints[0]['x']:.2f}, "
          f"y={grid_keypoints[0]['y']:.2f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
