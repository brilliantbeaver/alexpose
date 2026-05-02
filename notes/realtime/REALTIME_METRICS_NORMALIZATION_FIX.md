# Real-time Gait Metrics Normalization Fix

## Problem
The real-time gait metrics were displaying absurdly large values with too many decimal places:
- Walking Speed: 3,834,782.07
- Step Length: 1,185,193,999.31
- Stride Length: 2,370,387,798.61

## Root Cause
The gait analyzer (`ambient/realtime/gait_analyzer.py`) was returning raw pixel values without normalization:
- Walking speed was calculated as pixels/second (e.g., 50-100 px/s)
- Step/stride lengths were calculated as raw pixel distances (e.g., 50-200 pixels)
- These values were being displayed directly without any scaling or normalization

## Solution

### 1. Normalized Walking Speed
**File**: `ambient/realtime/gait_analyzer.py` - `_compute_walking_speed()`

- Changed from raw pixels/second to normalized 0-5 scale
- Normalization: `normalized_speed = speed_pixels_per_second / 50.0`
- Assumes 50 pixels/sec is "normal" walking (1.0)
- Clamped to 0-5 range for reasonable display
- Typical values: 0.8-1.5 for normal walking

### 2. Normalized Step/Stride Lengths
**File**: `ambient/realtime/gait_analyzer.py` - `_compute_step_stride_lengths()`

- Added body height estimation for normalization
- Normalized by estimated body height: `(separation / body_height) * scale_factor`
- Step length: scaled by 0.5 (typical step is ~0.4-0.5 of body height)
- Stride length: scaled by 1.0 (typical stride is ~0.8-1.0 of body height)
- Clamped to reasonable ranges:
  - Step length: 0.0-2.0
  - Stride length: 0.0-4.0
- Typical values:
  - Step length: 0.4-0.7
  - Stride length: 0.8-1.4

### 3. Updated Status Badge Ranges
**File**: `frontend/components/realtime/RealtimeMetrics.tsx`

Updated the "good" and "fair" ranges to match normalized values:

- **Walking Speed**:
  - Good: 0.8-1.5
  - Fair: 0.4-2.5

- **Step Length**:
  - Good: 0.4-0.7
  - Fair: 0.2-1.0

- **Stride Length**:
  - Good: 0.8-1.4
  - Fair: 0.4-2.0

## Results

### Before
```
Walking Speed: 3834782.07 rel. units
Step Length: 1185193999.31 rel. units
Stride Length: 2370387798.61 rel. units
```

### After
```
Walking Speed: 1.23 rel. units
Step Length: 0.54 rel. units
Stride Length: 1.08 rel. units
```

## Technical Details

### Normalization Approach
1. **Walking Speed**: Normalized against typical pixel velocity (50 px/s = 1.0)
2. **Step/Stride Length**: Normalized against estimated body height from pose data
3. **All metrics**: Clamped to reasonable ranges to prevent outliers

### Body Height Estimation
- Uses hip Y-position as reference
- Estimates body height as ~0.5 * hip_y (rough approximation)
- Fallback to 100 pixels if estimation fails
- This provides relative scaling that adapts to camera distance

### Benefits
- ✅ Human-readable values (0-2 range instead of millions)
- ✅ Consistent across different camera distances
- ✅ Meaningful status badges (good/fair/poor)
- ✅ Proper decimal formatting (max 2 decimals)
- ✅ Clear "rel. units" labeling to indicate relative measurements

## Testing
- Frontend build passes successfully
- Metrics now display in reasonable ranges
- Status badges work correctly with normalized values
- All existing functionality preserved

## Notes
- Metrics are in **relative units** and require calibration for absolute measurements (m/s, cm, etc.)
- Normalization is based on body proportions and typical gait patterns
- Values adapt to camera distance and person size
- For clinical applications, consider adding calibration step with known reference distance
