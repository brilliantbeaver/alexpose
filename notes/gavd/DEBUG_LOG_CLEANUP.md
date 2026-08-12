# DEBUG Log Cleanup - windows_ffmpeg_handler.py

## Issue
During GAVD processing, the server logs were cluttered with DEBUG-level log messages from `ambient.pose.windows_ffmpeg_handler`, making it difficult to track important information.

## Solution
Removed all DEBUG log statements from `ambient/pose/windows_ffmpeg_handler.py` to provide cleaner server output during development.

## Changes Made

### Removed Debug Logs:

1. **WindowsTempFileManager**
   - ❌ "Created temporary file: {temp_path}"
   - ❌ "Successfully cleaned up temporary file: {temp_path}"
   - ❌ "Cleanup attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"

2. **WindowsFFmpegExtractor**
   - ❌ "FFmpeg is available and working"
   - ❌ "FFmpeg command: {' '.join(cmd)}"
   - ❌ "Successfully extracted frame: {frame.shape} from {image_path} ({file_size} bytes)"

3. **WindowsVideoFrameExtractor**
   - ❌ "FFmpeg failed for frame {frame_number}, trying OpenCV fallback"
   - ❌ "OpenCV failed for frame {frame_number}, trying FFmpeg fallback"
   - ❌ "FFmpeg extraction failed for frame {frame_number}: {e}"
   - ❌ "OpenCV extraction failed for frame {frame_number}: {e}"

### Kept Important Logs:

✅ **Warnings** (still logged):
- "FFmpeg is installed but not working properly"
- "FFmpeg not available: {e}"
- "Failed to cleanup temporary file after {max_retries} attempts: {temp_path}"
- "Unexpected FFmpeg error for frame {frame_number}: {e}"

✅ **Errors** (still raised as exceptions):
- All FFmpegExtractionError exceptions
- All FFmpegNotFoundError exceptions

## Impact

### Before:
```
DEBUG | Created temporary file: /tmp/alexpose_abc123.jpg
DEBUG | FFmpeg command: ffmpeg -i video.mp4 -vf select=eq(n\,100) ...
DEBUG | Successfully extracted frame: (720, 1280, 3) from /tmp/alexpose_abc123.jpg (45678 bytes)
DEBUG | Successfully cleaned up temporary file: /tmp/alexpose_abc123.jpg
... (repeated for every frame)
```

### After:
```
(Clean output - only warnings and errors shown)
```

## Benefits

1. **Cleaner Logs**: Server logs are now much easier to read during development
2. **Better Performance**: Slightly reduced I/O overhead from logging
3. **Focused Debugging**: Important warnings and errors stand out
4. **Maintained Functionality**: All error handling and warnings preserved

## Backward Compatibility

- The `verbose` parameter still exists and can be used if needed
- All functionality remains unchanged
- Only logging output was affected

## Testing

No functional changes were made - only logging statements were removed. The frame extraction logic remains identical.

## File Modified

- `ambient/pose/windows_ffmpeg_handler.py` - Removed 11 debug log statements
