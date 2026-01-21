# Troubleshooting: extract_from_sequence Returns Empty Array

## Problem
`extractor.extract_from_sequence()` returns an empty array `[]` with 0 frames processed.

## Root Causes & Solutions

### 1. Missing Required Columns ⚠️ MOST COMMON
**Symptom:** Method returns `[]` immediately without processing any frames.

**Check:**
```python
print("Columns:", list(normal_df.columns))
print("Has 'frame_num':", 'frame_num' in normal_df.columns)
print("Has 'url':", 'url' in normal_df.columns)
```

**Solution:** Ensure DataFrame has `frame_num` and `url` columns. If using different names, rename them:
```python
# If your CSV uses 'frame' instead of 'frame_num'
normal_df = normal_df.rename(columns={'frame': 'frame_num'})

# If your CSV uses 'video_url' instead of 'url'
normal_df = normal_df.rename(columns={'video_url': 'url'})
```

### 2. Video Files Not Found
**Symptom:** Validation passes but all frames fail during processing.

**Check:**
```python
from ambient.utils.youtube_cache import extract_video_id

sample_url = normal_df['url'].iloc[0]
video_id = extract_video_id(sample_url)
video_path = video_base_path / f"{video_id}.mp4"

print(f"Video path: {video_path}")
print(f"Exists: {video_path.exists()}")
```

**Solution:** 
- Ensure videos are downloaded to `video_base_path`
- Use `GAVDDataLoader.load_gavd_data()` which auto-downloads videos
- Check `video_base_path` is correct (should be `project_root / "data" / "youtube"`)

### 3. Empty DataFrame
**Symptom:** Returns `[]` immediately.

**Check:**
```python
print(f"DataFrame shape: {normal_df.shape}")
print(f"Is empty: {normal_df.empty}")
```

**Solution:** Verify CSV file loaded correctly and contains data.

### 4. MediaPipe Model Not Loaded
**Symptom:** Error message about landmarker initialization.

**Check:** Look for ERROR logs mentioning "Failed to initialize landmarker"

**Solution:**
```python
# Force model download
from ambient.pose.model_management import MediaPipeModelManager
manager = MediaPipeModelManager()
model_path = manager.ensure_model_available()
print(f"Model path: {model_path}")
```

### 5. Incorrect video_base_path
**Symptom:** Validation fails with "Video base path does not exist"

**Check:**
```python
print(f"video_base_path: {video_base_path}")
print(f"Exists: {video_base_path.exists()}")
print(f"Is directory: {video_base_path.is_dir()}")
```

**Solution:** Ensure path points to directory containing `.mp4` files:
```python
video_base_path = project_root / "data" / "youtube"
```

## Diagnostic Code for Notebook

Add this cell BEFORE calling `extract_from_sequence`:

```python
# === DIAGNOSTIC CELL ===
print("🔍 Pre-flight checks:")
print()

# 1. DataFrame validation
print(f"1. DataFrame: {normal_df.shape}")
print(f"   Columns: {list(normal_df.columns)}")
print(f"   Empty: {normal_df.empty}")

# 2. Required columns
required = ['frame_num', 'url']
missing = [c for c in required if c not in normal_df.columns]
if missing:
    print(f"   ❌ Missing columns: {missing}")
    print(f"   → FIX: Rename columns or check CSV file")
else:
    print(f"   ✅ Has required columns")

# 3. Video path
print(f"\n2. Video path: {video_base_path}")
print(f"   Exists: {video_base_path.exists()}")

# 4. Sample video file
if 'url' in normal_df.columns and not normal_df.empty:
    from ambient.utils.youtube_cache import extract_video_id
    sample_url = normal_df['url'].iloc[0]
    video_id = extract_video_id(sample_url)
    video_path = video_base_path / f"{video_id}.mp4"
    print(f"\n3. Sample video: {video_id}.mp4")
    print(f"   Exists: {video_path.exists()}")
    if not video_path.exists():
        print(f"   ❌ Video file not found!")
        print(f"   → FIX: Run GAVDDataLoader.load_gavd_data() to download")

# 5. Validation check
print(f"\n4. Validation:")
is_valid, message = extractor.validate_sequence_data_verbose(normal_df, video_base_path)
print(f"   Status: {'✅ PASS' if is_valid else '❌ FAIL'}")
print(f"   Message: {message}")

if not is_valid:
    print(f"\n❌ VALIDATION FAILED - extraction will return empty array!")
    print(f"   Fix the issue above before proceeding")
else:
    print(f"\n✅ All checks passed - ready for extraction")
```

## Quick Fix Template

If you're getting empty arrays, try this:

```python
# 1. Verify data loaded correctly
print(f"DataFrame shape: {normal_df.shape}")
print(f"Columns: {list(normal_df.columns)}")

# 2. Check validation
is_valid, message = extractor.validate_sequence_data_verbose(normal_df, video_base_path)
if not is_valid:
    print(f"❌ Validation failed: {message}")
    # Fix the issue based on the message
else:
    # 3. Try with verbose=True to see what's happening
    normal_keypoints_array = extractor.extract_from_sequence(
        sequence_data=normal_df,
        video_base_path=video_base_path,
        verbose=True  # ← Add this to see progress
    )
    
    print(f"\nResult: {len(normal_keypoints_array)} frames")
    if normal_keypoints_array:
        successful = sum(1 for k in normal_keypoints_array if k is not None)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {len(normal_keypoints_array) - successful}")
```

## Still Having Issues?

Run the diagnostic script:
```bash
cd experiments/exp4
python debug_specific_sequence.py
```

This will show exactly what's wrong with your specific sequence.
