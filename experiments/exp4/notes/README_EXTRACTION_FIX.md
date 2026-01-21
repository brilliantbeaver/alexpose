# Fix for extract_from_sequence Returning Empty Array

## Problem
`extractor.extract_from_sequence()` returns `[]` with 0 frames processed for sequence `cljo30lnz001q3n6lopfty7q5`.

## Solution

### Option 1: Use the Safe Wrapper (Recommended)

Add this cell to your notebook BEFORE the extraction:

```python
# Load helper functions
exec(open('notebook_helper.py').read())

# Use safe extraction with built-in diagnostics
normal_keypoints_array = safe_extract_keypoints(
    normal_df, 
    video_base_path, 
    extractor,
    verbose=True
)
```

This will:
1. Run comprehensive diagnostics
2. Show exactly what's wrong if it fails
3. Proceed with extraction if everything is OK

### Option 2: Add Diagnostic Cell

Add this cell BEFORE your extraction to see what's wrong:

```python
# Load diagnostic function
exec(open('notebook_helper.py').read())

# Run diagnostics
diagnose_extraction_issue(normal_df, video_base_path, extractor)
```

Then fix any issues it identifies and run your normal extraction code.

### Option 3: Manual Validation

Add this before extraction:

```python
# Check validation
is_valid, message = extractor.validate_sequence_data_verbose(normal_df, video_base_path)

if not is_valid:
    print(f"❌ Validation failed: {message}")
    print(f"   DataFrame shape: {normal_df.shape}")
    print(f"   Columns: {list(normal_df.columns)}")
else:
    print(f"✅ Validation passed - proceeding with extraction")
    
    # Add verbose=True to see progress
    normal_keypoints_array = extractor.extract_from_sequence(
        sequence_data=normal_df,
        video_base_path=video_base_path,
        verbose=True  # ← Shows frame-by-frame progress
    )
    
    print(f"\nExtracted {len(normal_keypoints_array)} frames")
```

## Common Issues & Fixes

### Issue 1: Missing Columns
**Error:** `Missing columns: ['frame_num']` or `['url']`

**Fix:**
```python
# Check what columns you have
print(normal_df.columns)

# Rename if needed
normal_df = normal_df.rename(columns={
    'frame': 'frame_num',  # if using 'frame'
    'video_url': 'url'      # if using 'video_url'
})
```

### Issue 2: Video File Not Found
**Error:** `Sample video file not found`

**Fix:**
```python
# Videos should auto-download when using GAVDDataLoader
gavd_loader = GAVDDataLoader()
normal_df = gavd_loader.load_gavd_data(normal_csv)  # This downloads videos

# Verify video exists
from ambient.utils.youtube_cache import extract_video_id
video_id = extract_video_id(normal_df['url'].iloc[0])
video_path = video_base_path / f"{video_id}.mp4"
print(f"Video exists: {video_path.exists()}")
```

### Issue 3: Wrong video_base_path
**Error:** `Video base path does not exist`

**Fix:**
```python
# Ensure correct path
video_base_path = project_root / "data" / "youtube"
print(f"Path: {video_base_path}")
print(f"Exists: {video_base_path.exists()}")
```

## Testing

To test if the fix works, run the diagnostic script:

```bash
cd experiments/exp4
python debug_specific_sequence.py
```

This will test extraction on the exact sequence you're having trouble with.

## What Changed

The `extract_from_sequence` method was refactored to:

1. **Better validation** - New `validate_sequence_data_verbose()` method provides detailed error messages
2. **Graceful degradation** - Individual frame failures don't stop the entire sequence
3. **Better logging** - Clear messages about what failed and why
4. **Partial results** - Returns frames that succeeded even if some failed

## Need More Help?

See `TROUBLESHOOTING.md` for comprehensive debugging guide.
