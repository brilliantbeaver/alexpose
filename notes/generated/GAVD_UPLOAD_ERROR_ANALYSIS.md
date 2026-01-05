# GAVD Dataset Upload Error - Root Cause Analysis

## Executive Summary

The "Upload GAVD Dataset" button fails because **the backend FastAPI server is not running**. Additionally, there are several implementation gaps in the GAVD processing pipeline that need to be addressed.

## Root Causes

### 1. **Backend Server Not Running** (CRITICAL)
- **Issue**: The FastAPI backend server at `http://localhost:8000` is not running
- **Evidence**: 
  - `curl http://localhost:8000/health` returns "Unable to connect to the remote server"
  - `netstat -ano | findstr ":8000"` shows no process listening on port 8000
- **Impact**: Frontend cannot communicate with backend, causing all upload requests to fail
- **Solution**: Start the backend server with `uv run uvicorn server.main:app --reload --host 127.0.0.1 --port 8000`

### 2. **Incomplete GAVD Processing Implementation**
- **Issue**: The `PoseDataConverter.convert_sequence_to_pose_format()` method is truncated
- **Location**: `ambient/gavd/gavd_processor.py` line 847+ (file has 1179 lines but only 847 were read)
- **Impact**: Even if the server runs, GAVD processing may fail or produce incomplete results
- **Solution**: Complete the implementation following the notebook's processing steps

### 3. **Missing Process Management**
- **Issue**: No clear instructions or scripts to start both frontend and backend servers
- **Impact**: Users don't know how to properly start the application
- **Solution**: Create startup scripts and documentation

## Expected GAVD Processing Flow (from explore2.ipynb)

Based on the notebook, the GAVD processing should follow these steps:

### Step 1: Load GAVD CSV Data
```python
loader = GAVDDataLoader()
df = loader.load_gavd_data(csv_path)
sequences = loader.organize_by_sequence(df)
```

### Step 2: Download YouTube Videos
- Extract video IDs from URLs
- Cache videos locally in `data/youtube/`
- Handle cookies for age-restricted content

### Step 3: Extract Frames
- For each sequence, extract frames at specified frame numbers
- Use FFmpeg for precise frame extraction
- Scale bounding boxes if video resolution differs from annotations

### Step 4: Run Pose Estimation
- Use MediaPipe or OpenPose to extract keypoints
- Apply bounding box constraints
- Generate pose data in standardized format

### Step 5: Organize and Save Results
- Group by sequences
- Save metadata and pose data
- Generate statistics and summaries

## Implementation Gaps

### 1. **Incomplete `convert_sequence_to_pose_format()` Method**

The method is cut off at line 847. It should:
- Iterate through each frame in the sequence
- Extract video frames using FFmpeg
- Run pose estimation on each frame
- Handle bounding box scaling
- Generate pose data in OpenPose format
- Include metadata (frame_num, bbox, vid_info, etc.)

### 2. **Missing Error Handling**
- No handling for missing YouTube videos
- No validation of video frame counts vs. CSV frame numbers
- No graceful degradation if pose estimation fails

### 3. **Incomplete CSV Validation**
The `validate_csv_file()` function in `server/utils/file_validation.py` is also truncated at line 250.

### 4. **No Progress Tracking**
- Background processing has no progress updates
- Users can't see which sequence is being processed
- No ETA or completion percentage

## Recommended Solutions

### Immediate Actions (Critical)

1. **Start the Backend Server**
   ```bash
   # In terminal 1
   cd C:\Users\theod\dev\alex\alexpose
   uv run uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
   ```

2. **Start the Frontend Server**
   ```bash
   # In terminal 2
   cd C:\Users\theod\dev\alex\alexpose\frontend
   npm run dev
   ```

3. **Test the Upload Flow**
   - Navigate to http://localhost:3000/training/gavd
   - Upload a GAVD CSV file
   - Monitor backend logs for errors

### Short-term Fixes

1. **Complete the `convert_sequence_to_pose_format()` Implementation**
   - Read the rest of `ambient/gavd/gavd_processor.py` (lines 847-1179)
   - Ensure it follows the notebook's processing steps
   - Add proper error handling

2. **Complete the `validate_csv_file()` Implementation**
   - Read the rest of `server/utils/file_validation.py` (line 250+)
   - Add validation for:
     - Row count
     - Sequence count
     - URL format
     - Bbox structure
     - Vid_info structure

3. **Add Progress Tracking**
   - Implement WebSocket or SSE for real-time updates
   - Track processing stages: downloading, extracting, estimating, saving
   - Show progress per sequence

### Long-term Improvements

1. **Create Startup Scripts**
   ```bash
   # scripts/start-dev.sh
   #!/bin/bash
   # Start backend
   uv run uvicorn server.main:app --reload --host 127.0.0.1 --port 8000 &
   
   # Start frontend
   cd frontend && npm run dev &
   
   # Wait for both
   wait
   ```

2. **Add Health Checks**
   - Frontend should check backend health before allowing uploads
   - Display connection status in UI
   - Show helpful error messages if backend is down

3. **Improve Error Messages**
   - Replace generic "Upload failed" with specific errors:
     - "Backend server not running"
     - "Invalid CSV format"
     - "Missing required columns"
     - "YouTube video download failed"
     - "Pose estimation failed"

4. **Add Retry Logic**
   - Retry failed YouTube downloads
   - Retry failed pose estimations
   - Save partial results for resume capability

5. **Add Validation Preview**
   - Show CSV preview before upload
   - Display detected sequences and frame counts
   - Warn about potential issues (missing videos, invalid URLs, etc.)

## Testing Checklist

After implementing fixes, test the following:

- [ ] Backend server starts without errors
- [ ] Frontend server starts without errors
- [ ] Health check endpoint responds: `curl http://localhost:8000/health`
- [ ] Upload page loads: http://localhost:3000/training/gavd
- [ ] CSV file can be selected/dropped
- [ ] Upload request reaches backend (check logs)
- [ ] CSV validation passes
- [ ] YouTube videos download successfully
- [ ] Frames extract correctly
- [ ] Pose estimation runs (or placeholder keypoints generate)
- [ ] Results save to database/files
- [ ] Status polling works
- [ ] Results page displays correctly

## Files to Review/Fix

1. `ambient/gavd/gavd_processor.py` (lines 847-1179) - Complete implementation
2. `server/utils/file_validation.py` (line 250+) - Complete validation
3. `server/services/gavd_service.py` - Review error handling
4. `frontend/app/training/gavd/page.tsx` - Add connection status check
5. Create `scripts/start-dev.sh` - Startup script
6. Create `scripts/start-dev.ps1` - Windows startup script

## Next Steps

1. Start both servers manually to test current functionality
2. Complete the truncated implementations
3. Add comprehensive error handling
4. Implement progress tracking
5. Create startup scripts
6. Add health checks and connection status
7. Test end-to-end with sample GAVD CSV file

## Sample GAVD CSV Structure

Based on the notebook, a valid GAVD CSV should have:

```csv
seq,frame_num,cam_view,gait_event,dataset,gait_pat,bbox,vid_info,id,url
cljan9b4p00043n6ligceanyp,1757,front,heel_strike,GAVD,normal,"{""left"":156,""top"":125,""width"":238,""height"":500}","{""width"":1280,""height"":720}",abc123,https://www.youtube.com/watch?v=B5hrxKe2nP8
```

Key requirements:
- `seq`: Unique sequence identifier
- `frame_num`: Frame number in video (1-based)
- `bbox`: JSON object with left, top, width, height
- `vid_info`: JSON object with width, height
- `url`: YouTube video URL

## Conclusion

The primary issue is that the backend server is not running. Once started, there are several implementation gaps that need to be addressed to ensure the complete GAVD processing pipeline works as designed in the notebook.

The processing flow should be:
1. Upload CSV → 2. Validate → 3. Download videos → 4. Extract frames → 5. Run pose estimation → 6. Save results → 7. Display analysis

Each step needs proper error handling, progress tracking, and user feedback.
