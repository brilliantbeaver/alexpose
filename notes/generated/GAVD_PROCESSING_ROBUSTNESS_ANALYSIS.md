# GAVD Processing Robustness Analysis and Fixes

## Issue Summary

User uploaded a GAVD CSV file but cannot see processing progress. The server logs show:
1. Processing starts successfully
2. MediaPipe warnings appear
3. Processing appears to hang during video processing
4. No status updates visible to user

## Root Cause Analysis

### Critical Issue #1: Async/Sync Mismatch (BLOCKING)
**Location**: `server/services/gavd_service.py:109`

```python
async def process_dataset(self, dataset_id: str, ...) -> None:
    # This is async but calls blocking operations!
    results = processor.process_gavd_file(...)  # BLOCKS EVENT LOOP
```

**Problem**:
- Method is declared `async` but calls synchronous blocking operations
- `processor.process_gavd_file()` processes videos synchronously
- MediaPipe video processing is CPU-intensive and blocks for minutes
- Blocks FastAPI event loop, preventing status updates
- User sees no progress because server can't respond to status requests

**Impact**: HIGH - Server appears frozen during processing

### Critical Issue #2: No Progress Reporting During Video Processing
**Location**: `ambient/gavd/pose_estimators.py:337`

```python
def estimate_video_keypoints(self, video_path: Path, ...) -> List[List[Keypoint]]:
    logger.info(f"Processing video: {frame_count} frames at {fps} fps")
    # Processes 6176 frames with no progress updates!
    for frame_idx in range(frame_count):
        # Process frame... (no progress callback)
```

**Problem**:
- Processes thousands of frames with no progress updates
- User has no idea if processing is working or stuck
- No way to estimate completion time
- No intermediate status updates to database

**Impact**: HIGH - Poor user experience, appears frozen

### Issue #3: MediaPipe Warnings Not Handled
**Warnings in logs**:
```
W0000 00:00:1767515773.328706 inference_feedback_manager.cc:121] 
Feedback manager requires a model with a single signature inference.

W0000 00:00:1767515774.188671 landmark_projection_calculator.cc:81] 
Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI.
```

**Problem**:
- MediaPipe TensorFlow Lite warnings are normal but alarming
- No explanation to user that these are expected
- Could indicate configuration issues

**Impact**: MEDIUM - Confusing logs, potential performance issues

### Issue #4: No Timeout Protection
**Location**: `server/services/gavd_service.py:109`

**Problem**:
- No timeout for video processing
- Large videos (6176 frames) could process for hours
- No way to cancel long-running operations
- Could exhaust server resources

**Impact**: MEDIUM - Resource exhaustion risk

### Issue #5: No Error Recovery
**Location**: `server/services/gavd_service.py:260`

**Problem**:
- If processing fails mid-way, partial data may be saved
- No cleanup of partial results
- Status may be stuck in "processing"
- No retry mechanism

**Impact**: MEDIUM - Data inconsistency

### Issue #6: Memory Issues with Large Videos
**Location**: `ambient/gavd/pose_estimators.py:300`

**Problem**:
- Loads entire video into memory for processing
- 6176 frames at 640x360 = ~4GB uncompressed
- No frame batching or streaming
- Could cause OOM errors

**Impact**: MEDIUM - Memory exhaustion on large videos

## Proposed Solutions

### Solution 1: Run Processing in Thread Pool (CRITICAL)

**Change**: Use `asyncio.to_thread()` to run blocking operations in thread pool

```python
# server/services/gavd_service.py
async def process_dataset(self, dataset_id: str, ...) -> None:
    try:
        # ... setup code ...
        
        # Run blocking operation in thread pool
        results = await asyncio.to_thread(
            processor.process_gavd_file,
            csv_file_path=csv_file_path,
            max_sequences=max_sequences,
            include_metadata=True,
            verbose=True
        )
        
        # ... save results ...
    except Exception as e:
        # ... error handling ...
```

**Benefits**:
- Doesn't block event loop
- Server can respond to status requests
- Better resource utilization
- Standard FastAPI pattern

### Solution 2: Add Progress Callback System

**Change**: Add progress callback to video processing

```python
# ambient/gavd/pose_estimators.py
def estimate_video_keypoints(
    self,
    video_path: Path,
    model: str = "BODY_25",
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[List[Keypoint]]:
    
    for frame_idx in range(frame_count):
        # Process frame...
        
        # Report progress every 100 frames
        if progress_callback and frame_idx % 100 == 0:
            progress_callback(frame_idx, frame_count)
```

**Benefits**:
- Real-time progress updates
- User sees processing is active
- Can estimate completion time
- Better UX

### Solution 3: Add Timeout Configuration

**Change**: Add configurable timeout for processing

```python
# config/alexpose.yaml
gavd:
  processing:
    timeout_seconds: 3600  # 1 hour max
    frame_batch_size: 100
    max_video_frames: 10000  # Reject videos larger than this
```

```python
# server/services/gavd_service.py
async def process_dataset(self, dataset_id: str, ...) -> None:
    timeout = self.config.gavd.processing.timeout_seconds
    
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(processor.process_gavd_file, ...),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Processing timeout after {timeout}s")
        self.update_dataset_metadata(dataset_id, {
            "status": "error",
            "error": f"Processing timeout after {timeout} seconds"
        })
```

**Benefits**:
- Prevents infinite hangs
- Protects server resources
- Clear error messages
- Configurable limits

### Solution 4: Add Frame Batching

**Change**: Process video in batches to reduce memory usage

```python
# ambient/gavd/pose_estimators.py
def estimate_video_keypoints_batched(
    self,
    video_path: Path,
    batch_size: int = 100,
    progress_callback: Optional[Callable] = None
) -> List[List[Keypoint]]:
    
    all_keypoints = []
    
    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(batch_start + batch_size, frame_count)
        
        # Process batch
        batch_keypoints = self._process_frame_batch(
            cap, batch_start, batch_end
        )
        all_keypoints.extend(batch_keypoints)
        
        # Report progress
        if progress_callback:
            progress_callback(batch_end, frame_count)
    
    return all_keypoints
```

**Benefits**:
- Reduced memory usage
- Better progress granularity
- Can pause/resume processing
- More scalable

### Solution 5: Suppress MediaPipe Warnings

**Change**: Configure MediaPipe to suppress expected warnings

```python
# ambient/gavd/pose_estimators.py
import os
import warnings

# Suppress TensorFlow Lite warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')

# Or add to logger
logger.info("MediaPipe warnings about feedback manager and NORM_RECT are expected and can be ignored")
```

**Benefits**:
- Cleaner logs
- Less user confusion
- Focus on real errors

### Solution 6: Add Robust Error Handling

**Change**: Comprehensive error handling with cleanup

```python
# server/services/gavd_service.py
async def process_dataset(self, dataset_id: str, ...) -> None:
    results_file = None
    pose_data_file = None
    
    try:
        # ... processing ...
        
    except asyncio.TimeoutError:
        logger.error(f"Processing timeout for {dataset_id}")
        self.update_dataset_metadata(dataset_id, {
            "status": "error",
            "error": "Processing timeout - video too large or complex"
        })
        
    except MemoryError:
        logger.error(f"Out of memory processing {dataset_id}")
        self.update_dataset_metadata(dataset_id, {
            "status": "error",
            "error": "Out of memory - video too large"
        })
        
    except Exception as e:
        logger.error(f"Error processing {dataset_id}: {str(e)}")
        self.update_dataset_metadata(dataset_id, {
            "status": "error",
            "error": str(e)
        })
        
    finally:
        # Cleanup partial results on error
        if self.get_dataset_metadata(dataset_id).get("status") == "error":
            if results_file and results_file.exists():
                results_file.unlink()
            if pose_data_file and pose_data_file.exists():
                pose_data_file.unlink()
```

**Benefits**:
- Clean error states
- No partial data
- Clear error messages
- Proper cleanup

## Implementation Priority

### Phase 1: Critical Fixes (Immediate)
1. ✅ Fix async/sync mismatch with `asyncio.to_thread()`
2. ✅ Add timeout protection
3. ✅ Add basic error handling

### Phase 2: User Experience (Short-term)
4. ✅ Add progress callback system
5. ✅ Update frontend to show progress
6. ✅ Suppress MediaPipe warnings

### Phase 3: Scalability (Medium-term)
7. ⏳ Add frame batching
8. ⏳ Add memory limits
9. ⏳ Add processing queue

## Testing Strategy

### Test Cases

1. **Test Async Processing**
   ```python
   async def test_process_dataset_non_blocking():
       # Start processing
       task = asyncio.create_task(service.process_dataset(dataset_id))
       
       # Should be able to query status while processing
       await asyncio.sleep(1)
       status = service.get_dataset_metadata(dataset_id)
       assert status["status"] == "processing"
       
       # Wait for completion
       await task
   ```

2. **Test Timeout**
   ```python
   async def test_process_dataset_timeout():
       # Set very short timeout
       with pytest.raises(asyncio.TimeoutError):
           await asyncio.wait_for(
               service.process_dataset(dataset_id),
               timeout=1.0
           )
   ```

3. **Test Progress Updates**
   ```python
   def test_progress_callback():
       progress_updates = []
       
       def callback(current, total):
           progress_updates.append((current, total))
       
       estimator.estimate_video_keypoints(
           video_path,
           progress_callback=callback
       )
       
       assert len(progress_updates) > 0
       assert progress_updates[-1][0] == progress_updates[-1][1]
   ```

4. **Test Error Recovery**
   ```python
   async def test_error_cleanup():
       # Cause processing error
       with patch('processor.process_gavd_file', side_effect=Exception("Test error")):
           await service.process_dataset(dataset_id)
       
       # Check status is error
       metadata = service.get_dataset_metadata(dataset_id)
       assert metadata["status"] == "error"
       
       # Check no partial files
       assert not results_file.exists()
       assert not pose_data_file.exists()
   ```

## Configuration Changes

### Add to `config/alexpose.yaml`

```yaml
# GAVD Processing Configuration
gavd:
  processing:
    # Maximum processing time in seconds (1 hour default)
    timeout_seconds: 3600
    
    # Frame batch size for memory efficiency
    frame_batch_size: 100
    
    # Maximum frames per video (reject larger videos)
    max_video_frames: 10000
    
    # Progress update interval (frames)
    progress_interval: 100
    
    # Enable/disable MediaPipe warnings
    suppress_mediapipe_warnings: true
```

## Monitoring and Logging

### Add Structured Logging

```python
# server/services/gavd_service.py
logger.info(
    "GAVD processing started",
    extra={
        "dataset_id": dataset_id,
        "sequences": metadata.get("sequence_count"),
        "frames": metadata.get("row_count"),
        "estimator": pose_estimator
    }
)

# During processing
logger.info(
    "Video processing progress",
    extra={
        "dataset_id": dataset_id,
        "video_id": video_id,
        "frames_processed": current_frame,
        "total_frames": total_frames,
        "percent_complete": (current_frame / total_frames) * 100
    }
)
```

## User Communication

### Frontend Status Display

```typescript
// frontend/app/training/gavd/page.tsx
interface ProcessingStatus {
  status: 'uploaded' | 'processing' | 'completed' | 'error';
  progress?: string;
  percent_complete?: number;
  current_video?: string;
  frames_processed?: number;
  total_frames?: number;
  estimated_time_remaining?: number;
}

// Poll for status updates
useEffect(() => {
  if (status === 'processing') {
    const interval = setInterval(async () => {
      const response = await fetch(`/api/v1/gavd/status/${datasetId}`);
      const data = await response.json();
      setProcessingStatus(data);
    }, 2000); // Poll every 2 seconds
    
    return () => clearInterval(interval);
  }
}, [status]);
```

## Summary

### Current Issues
1. ❌ Async/sync mismatch blocks event loop
2. ❌ No progress reporting during video processing
3. ❌ MediaPipe warnings confuse users
4. ❌ No timeout protection
5. ❌ Poor error handling
6. ❌ Memory issues with large videos

### Proposed Fixes
1. ✅ Use `asyncio.to_thread()` for blocking operations
2. ✅ Add progress callback system
3. ✅ Suppress expected MediaPipe warnings
4. ✅ Add timeout configuration
5. ✅ Improve error handling and cleanup
6. ✅ Add frame batching for memory efficiency

### Expected Improvements
- ✅ Server remains responsive during processing
- ✅ Users see real-time progress updates
- ✅ Clear error messages and recovery
- ✅ Protection against resource exhaustion
- ✅ Better scalability for large videos
- ✅ Improved user experience

## Next Steps

1. Implement Phase 1 critical fixes
2. Test with user's GAVD file
3. Monitor processing performance
4. Implement Phase 2 UX improvements
5. Add comprehensive logging
6. Document processing limits and best practices
