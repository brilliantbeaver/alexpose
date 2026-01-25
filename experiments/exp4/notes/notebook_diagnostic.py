"""
Diagnostic to run inside the notebook to identify the issue.
Copy and paste this into a notebook cell.
"""

# First, let's check what's actually in normal_df
print("="*70)
print("NOTEBOOK ENVIRONMENT DIAGNOSTIC")
print("="*70)
print()

print("1. DataFrame Info:")
print(f"   Shape: {normal_df.shape}")
print(f"   Columns: {list(normal_df.columns)}")
print(f"   Memory usage: {normal_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print()

print("2. Required Columns Check:")
required = ['frame_num', 'url']
for col in required:
    exists = col in normal_df.columns
    print(f"   {col}: {'✅' if exists else '❌'}")
    if exists:
        print(f"      Type: {normal_df[col].dtype}")
        print(f"      Nulls: {normal_df[col].isnull().sum()}")
        print(f"      Sample: {normal_df[col].iloc[0]}")
print()

print("3. Sequence ID:")
if 'seq' in normal_df.columns:
    seq_id = normal_df['seq'].iloc[0]
    print(f"   {seq_id}")
else:
    print("   ❌ No 'seq' column")
print()

print("4. Video Path Check:")
print(f"   video_base_path: {video_base_path}")
print(f"   Exists: {video_base_path.exists()}")
print()

if 'url' in normal_df.columns:
    from ambient.utils.youtube_cache import extract_video_id
    sample_url = normal_df['url'].iloc[0]
    print(f"5. Sample Video File:")
    print(f"   URL: {sample_url}")
    try:
        video_id = extract_video_id(sample_url)
        video_path = video_base_path / f"{video_id}.mp4"
        print(f"   Video ID: {video_id}")
        print(f"   Path: {video_path}")
        print(f"   Exists: {video_path.exists()}")
        if video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"   ❌ Error: {e}")
print()

print("6. Extractor Validation:")
try:
    is_valid, message = extractor.validate_sequence_data_verbose(normal_df, video_base_path)
    print(f"   Valid: {'✅' if is_valid else '❌'}")
    print(f"   Message: {message}")
except Exception as e:
    print(f"   ❌ Error calling validation: {e}")
print()

print("7. Try extraction with first 2 frames:")
try:
    test_df = normal_df.head(2)
    print(f"   Test DataFrame shape: {test_df.shape}")
    
    test_result = extractor.extract_from_sequence(
        sequence_data=test_df,
        video_base_path=video_base_path,
        verbose=True
    )
    
    print(f"   Result length: {len(test_result)}")
    if test_result:
        successful = sum(1 for k in test_result if k is not None)
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {len(test_result) - successful}")
    else:
        print(f"   ❌ EMPTY ARRAY!")
        print(f"   Check the logs above for ERROR messages")
except Exception as e:
    print(f"   ❌ Exception: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
