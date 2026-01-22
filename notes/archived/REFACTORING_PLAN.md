# Process Isolation Removal - Refactoring Plan

## Objective
Remove Windows-specific process isolation complexity from the codebase since we're running on macOS/Linux.

## Analysis Summary
- **Platform**: macOS (darwin) - no Windows threading issues
- **Current complexity**: ~500 lines of multiprocessing code
- **Performance impact**: 2x overhead (0.18s → 0.39s per frame)
- **Usage**: Only in `keypoint_extractor.py` and `gavd_processor.py`

## Files to Modify

### 1. Core Library Files (CRITICAL)
- `ambient/pose/keypoint_extractor.py` - Remove process isolation logic
- `ambient/gavd/gavd_processor.py` - Remove Windows-specific checks

### 2. Files to Delete
- `ambient/pose/process_isolated_extractor.py` - Entire file (~500 lines)

### 3. Test/Script Files to Delete (Windows-specific)
- `scripts/fixes/test_process_isolation_fix.py`
- `scripts/fixes/test_gavd_immediate_process_isolation.py`
- `scripts/fixes/test_gavd_windows_optimization.py`
- `scripts/fixes/test_gavd_end_to_end.py`

### 4. Test/Script Files to Update (Remove use_process_isolation parameter)
- `scripts/test_pose_dimension_capture.py`
- `scripts/test_debug_log_suppression.py`
- `scripts/test_complete_solution.py`

### 5. Documentation to Update/Archive
- `notes/mediapipe/WINDOWS_MEDIAPIPE_THREADING_SOLUTION.md` - Archive
- `notes/gavd/GAVD_IMMEDIATE_PROCESS_ISOLATION_SOLUTION.md` - Archive

## Refactoring Steps

### Step 1: Simplify `keypoint_extractor.py`
**Remove:**
- `use_process_isolation` parameter from `__init__`
- `_use_process_isolation`, `_process_extractor` instance variables
- `_threading_failures`, `_max_threading_failures` tracking
- `_should_use_process_isolation()` method
- `_get_process_extractor()` method
- Process isolation fallback logic in `extract_from_image()`
- Windows-specific error handling for "WinError 1"

**Keep:**
- MediaPipe singleton pattern (still valuable for memory management)
- Core extraction methods
- Retry logic with reset
- All sequence processing methods

### Step 2: Simplify `gavd_processor.py`
**Remove:**
- Windows platform checks (`os.name == 'nt'`)
- `use_process_isolation` parameter passing
- Windows-specific logging messages

**Keep:**
- All core GAVD processing logic
- Standard SequenceKeypointExtractor usage

### Step 3: Delete Process Isolation Module
- Delete `ambient/pose/process_isolated_extractor.py`

### Step 4: Update Test Scripts
- Remove `use_process_isolation=False` parameters
- Remove Windows-specific test files

### Step 5: Archive Documentation
- Move Windows-specific docs to `notes/archived/`

## Expected Benefits
1. **Simpler codebase**: ~600 lines removed
2. **Better performance**: No IPC overhead
3. **Easier debugging**: No cross-process errors
4. **Clearer intent**: Code does what it needs, nothing more
5. **Easier maintenance**: Fewer edge cases to handle

## Backward Compatibility
- API remains the same (just remove optional parameter)
- Existing code continues to work
- No database changes needed
- No breaking changes to public interfaces

## Testing Strategy
1. Run existing unit tests for keypoint extraction
2. Test GAVD processing end-to-end
3. Verify memory management still works (singleton pattern)
4. Check that extraction performance improves

## Rollback Plan
If issues arise:
1. Git revert the changes
2. Process isolation code is preserved in git history
3. Can be restored if Windows support needed later
