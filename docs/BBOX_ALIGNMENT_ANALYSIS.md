# Bounding Box Alignment Analysis and Fixes

## Root Cause Analysis

After systematic investigation, I've identified the following root causes for bounding box misalignment:

### Primary Root Cause: Frame Number Interpretation
- **Issue**: CSV frame numbers (e.g., 1757, 1758) are absolute frame numbers in the original YouTube video
- **Problem**: When extracting frames, we assume frame_num directly maps to video frame index
- **Impact**: If video was processed/trimmed or if there's a frame offset, bbox won't align

### Secondary Root Cause: Video Resolution Mismatch
- **Issue**: `vid_info` specifies resolution (1280x720), but actual video might differ
- **Problem**: Bbox coordinates assume specific resolution
- **Impact**: If video resolution differs, bbox coordinates won't align correctly

### Tertiary Root Cause: Frame Extraction Precision
- **Issue**: OpenCV's `CAP_PROP_POS_FRAMES` seeks to nearest keyframe, not exact frame
- **Problem**: May not extract the exact frame specified
- **Impact**: Slight frame misalignment can cause bbox to be off

## Proposed Fixes

### Fix 1: Add Frame Offset Calculation
Calculate frame offset based on sequence start to handle cases where annotations start mid-video.

### Fix 2: Video Resolution Validation and Bbox Scaling
Verify video resolution matches `vid_info` and scale bbox coordinates if needed.

### Fix 3: Improved Frame Extraction
Use FFmpeg for more precise frame extraction instead of OpenCV's frame seeking.

### Fix 4: Diagnostic Tools
Add comprehensive diagnostics to identify exact frame offsets and resolution mismatches.

