# Pose Overlay Fix - "Show Pose Overlay" Not Working

## Problem Description

The "Show Pose Overlay" checkbox in the GAVD video player was checked but no skeletal keypoints were being displayed on the video frames. The bounding box overlay was working correctly, but the pose skeleton was completely missing.

## Root Cause Analysis

After deep investigation, I found two main issues:

### 1. **Missing `keypoint_id` Field**
- **Issue**: The API was returning keypoints without the required `keypoint_id` field
- **Impact**: The frontend skeleton drawing code couldn't map keypoints to skeleton connections
- **Data Structure Problem**:
  ```json
  // What API returned:
  {"x": 260.0, "y": 363.5, "confidence": 0.8}
  
  // What frontend expected:
  {"x": 260.0, "y": 363.5, "confidence": 0.8, "keypoint_id": 0}
  ```

### 2. **Skeleton Format Mismatch**
- **Issue**: Frontend was configured for MediaPipe's 33-point format, but data had 25 keypoints (OpenPose BODY_25 format)
- **Impact**: Skeleton connections were trying to connect non-existent keypoints
- **Format Mismatch**:
  - **Data Format**: OpenPose BODY_25 (25 keypoints)
  - **Frontend Expected**: MediaPipe Pose (33 keypoints)

## Solution Implementation

### 1. **Backend Fix: Add Missing `keypoint_id`**

**File**: `server/routers/gavd.py`

Added logic to ensure each keypoint has a `keypoint_id` field:

```python
# Ensure each keypoint has a keypoint_id for frontend skeleton drawing
# Add keypoint_id if missing (using array index)
processed_keypoints = []
for i, kp in enumerate(keypoints):
    if isinstance(kp, dict):
        # Create a copy to avoid modifying original data
        processed_kp = kp.copy()
        if 'keypoint_id' not in processed_kp:
            processed_kp['keypoint_id'] = i
        processed_keypoints.append(processed_kp)
    else:
        # Handle malformed keypoint data
        processed_keypoints.append({
            'x': 0.0,
            'y': 0.0,
            'confidence': 0.0,
            'keypoint_id': i
        })
```

### 2. **Frontend Fix: Dynamic Skeleton Format Detection**

**File**: `frontend/components/GAVDVideoPlayer.tsx`

Added automatic detection of keypoint format and appropriate skeleton connections:

```typescript
// Determine the keypoint format based on the number of keypoints
const numKeypoints = normalizedKeypoints.length;

if (numKeypoints === 33) {
  // MediaPipe format (33 keypoints)
  connections = [/* MediaPipe connections */];
} else if (numKeypoints === 25) {
  // OpenPose BODY_25 format (25 keypoints)
  connections = [/* OpenPose connections */];
} else {
  // Unknown format - create basic connections
  console.warn(`Unknown keypoint format with ${numKeypoints} keypoints`);
  connections = [/* Basic fallback connections */];
}
```

### 3. **Enhanced Debugging**

Added comprehensive logging to help diagnose future issues:

```typescript
console.log(`Drawing ${keypoints.length} keypoints`);
console.log(`Normalized keypoints sample:`, normalizedKeypoints.slice(0, 2));
```

## OpenPose BODY_25 Skeleton Connections

The fix includes proper skeleton connections for OpenPose BODY_25 format:

```typescript
// OpenPose BODY_25 format (25 keypoints)
connections = [
  // Head and neck
  [0, 1],   // Nose to Neck
  [1, 2], [1, 5],   // Neck to shoulders
  [2, 3], [3, 4],   // Right shoulder to elbow to wrist
  [5, 6], [6, 7],   // Left shoulder to elbow to wrist
  
  // Torso
  [1, 8],   // Neck to MidHip
  [8, 9], [8, 12],  // MidHip to hips
  
  // Right leg
  [9, 10], [10, 11],  // Right hip to knee to ankle
  [11, 22], [11, 24], // Right ankle to foot
  [22, 23],  // Right heel to big toe
  
  // Left leg
  [12, 13], [13, 14], // Left hip to knee to ankle
  [14, 19], [14, 21], // Left ankle to foot
  [19, 20],  // Left heel to big toe
  
  // Face (if available)
  [0, 15], [15, 17],  // Nose to right eye to right ear
  [0, 16], [16, 18],  // Nose to left eye to left ear
];
```

## Testing and Verification

### Backend API Test
```bash
# Test the fixed API endpoint
curl "http://localhost:8000/api/v1/gavd/sequence/8bfa3a9c-5e97-44ec-860d-e7de291f34ab/cljan9b4p00043n6ligceanyp/frame/1757/pose"

# Expected result: keypoints with keypoint_id field
{
  "success": true,
  "pose_keypoints": [
    {"x": 260.0, "y": 363.5, "confidence": 0.8, "keypoint_id": 0},
    {"x": 265.0, "y": 363.5, "confidence": 0.8, "keypoint_id": 1},
    ...
  ]
}
```

### Frontend Verification
1. Open the GAVD dataset page
2. Navigate to the "Visualization" tab
3. Select a sequence with pose data
4. Check the "Show Pose Overlay" checkbox
5. **Expected Result**: Green skeleton lines and red keypoint dots should appear on the person in the video

## Files Modified

1. **`server/routers/gavd.py`** - Added `keypoint_id` field to API response
2. **`frontend/components/GAVDVideoPlayer.tsx`** - Added dynamic skeleton format detection and OpenPose BODY_25 support

## Impact

- ✅ **Fixed**: "Show Pose Overlay" now displays skeletal keypoints correctly
- ✅ **Robust**: Supports both MediaPipe (33 points) and OpenPose (25 points) formats
- ✅ **Backward Compatible**: Existing data continues to work
- ✅ **Future-Proof**: Handles unknown keypoint formats gracefully
- ✅ **Debuggable**: Enhanced logging for troubleshooting

## Expected Behavior After Fix

When "Show Pose Overlay" is checked:
- **Green lines** connect keypoints to form a skeleton
- **Red dots** mark individual keypoints
- **Confidence threshold** filters out low-confidence keypoints (< 0.3)
- **Dynamic scaling** adjusts for different video resolutions
- **Format detection** automatically uses correct skeleton connections

The pose overlay should now work seamlessly alongside the existing bounding box overlay functionality.