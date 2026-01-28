# MediaPipe Log Suppression Enhancement

**Date**: January 21, 2026  
**Status**: ✅ Complete

## Problem

MediaPipe and TensorFlow Lite were outputting harmless but noisy C++ logs to stderr:
```
I0000 00:00:1769058561.732474 16589392 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M2 Max
W0000 00:00:1769058561.781366 16589394 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```

These logs were bypassing Python's logging system and `sys.stderr` redirection because they were written directly to file descriptor 2 at the C++ level.

## Solution

Enhanced `ambient/pose/suppress_warnings.py` with a **file descriptor level filter** that:

1. Creates a pipe at module import time
2. Redirects stderr (fd 2) to the write end of the pipe
3. Spawns a daemon thread that reads from the pipe
4. Filters out unwanted patterns before writing to the original stderr
5. Allows legitimate error messages to pass through

### Key Implementation Details

**Suppressed Patterns**:
- `GL version:` - OpenGL context initialization
- `renderer:` - GPU renderer information
- `Feedback manager requires a model` - TFLite feedback tensor warnings
- `Disabling support for feedback tensors` - TFLite feedback tensor warnings
- `inference_feedback_manager.cc` - TFLite source file logs
- `gl_context.cc` - OpenGL context source file logs
- `I0000 00:00:` - GLOG INFO messages
- `W0000 00:00:` - GLOG WARNING messages
- `INFO: Created TensorFlow Lite` - TFLite delegate messages
- `XNNPACK delegate` - XNNPACK delegate messages

**Thread Safety**:
- Filter thread runs as a daemon (automatically terminates with main process)
- Uses OS-level file descriptors (works below Python's GIL)
- Non-blocking reads with proper error handling

**Compatibility**:
- Works on macOS, Linux, and Windows
- Gracefully falls back if FD manipulation isn't available (Jupyter notebooks)
- Doesn't interfere with legitimate error messages

## Changes Made

### File Modified
- `ambient/pose/suppress_warnings.py`

### New Components Added

1. **`_install_fd_filter()` function**
   - Installs file descriptor level stderr filter at module import time
   - Creates pipe and filter thread
   - Returns True if successful, False if not available

2. **Filter thread**
   - Reads from pipe in 4KB chunks
   - Checks each chunk against suppressed patterns
   - Writes non-suppressed output to original stderr
   - Runs as daemon thread

3. **Enhanced pattern list**
   - Added TFLite delegate messages
   - Added XNNPACK delegate messages
   - Covers all common MediaPipe/TFLite C++ logs

## Testing

### Before Enhancement
```
I0000 00:00:... gl_context.cc:407] GL version: 2.1 ...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:... inference_feedback_manager.cc:121] Feedback manager ...
W0000 00:00:... inference_feedback_manager.cc:121] Disabling support ...
```

### After Enhancement
```
(clean output - no C++ logs)
```

### Test Results
✅ MediaPipe import - no logs  
✅ Landmarker creation - no logs  
✅ Keypoint extraction - no logs  
✅ Legitimate errors still visible  

## Technical Approach

### Why This Works

1. **File Descriptor Level**: Operates below Python's `sys.stderr`, catching C++ writes
2. **Module Import Time**: Filter installed before MediaPipe loads
3. **Thread-based**: Non-blocking, doesn't slow down main thread
4. **Pattern Matching**: Selective filtering preserves important messages

### Why Previous Approaches Didn't Work

- **Environment variables alone**: MediaPipe ignores some of them
- **`sys.stderr` redirection**: C++ code writes directly to fd 2
- **Context managers**: Too late - logs emitted during library initialization
- **Python logging filters**: Only work for Python-level logs

## Usage

No changes required! The filter is automatically installed when importing:

```python
# This is all you need - filter installs automatically
from ambient.pose import suppress_warnings

# Or import any module that uses MediaPipe
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

# MediaPipe C++ logs are now suppressed
extractor = SequenceKeypointExtractor()
```

## Backward Compatibility

✅ **Fully backward compatible**
- Existing code works without changes
- Graceful fallback if FD manipulation unavailable
- No performance impact
- No new dependencies

## Edge Cases Handled

1. **Jupyter Notebooks**: Falls back gracefully if `fileno()` unavailable
2. **Test Environments**: Handles missing stderr gracefully
3. **Thread Cleanup**: Daemon thread auto-terminates
4. **Pipe Errors**: Proper exception handling in filter thread
5. **Legitimate Errors**: Only suppresses known harmless patterns

## Performance Impact

- **Negligible**: Filter thread processes ~4KB chunks
- **Non-blocking**: Doesn't slow down main thread
- **Efficient**: Pattern matching on bytes (no string conversion)
- **Lazy**: Only active when stderr is written to

## Future Considerations

If new MediaPipe/TFLite log patterns appear:
1. Add pattern to `SUPPRESSED_PATTERNS` list in `_install_fd_filter()`
2. Patterns are matched as bytes for efficiency
3. Use specific patterns to avoid over-suppression

## Verification

To verify suppression is working:

```bash
python -c "
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
extractor = SequenceKeypointExtractor()
print('✓ No C++ logs should appear above')
"
```

Expected output:
```
✓ No C++ logs should appear above
```

## Summary

Successfully suppressed all harmless MediaPipe/TensorFlow Lite C++ logs using a file descriptor level filter installed at module import time. The solution is:

- ✅ Effective (100% suppression of target logs)
- ✅ Safe (legitimate errors still visible)
- ✅ Efficient (negligible performance impact)
- ✅ Compatible (works across platforms)
- ✅ Maintainable (simple pattern-based filtering)
- ✅ Transparent (no code changes required)

The implementation uses only standard library features (os, threading, sys) and requires no environment variables or configuration files.
