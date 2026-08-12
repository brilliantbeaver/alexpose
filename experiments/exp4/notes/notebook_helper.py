"""
Helper functions for notebook - copy these into a notebook cell.
"""

def diagnose_extraction_issue(df, video_base_path, extractor):
    """
    Comprehensive diagnostic for extract_from_sequence issues.
    
    Usage in notebook:
        diagnose_extraction_issue(normal_df, video_base_path, extractor)
    """
    print("="*70)
    print("EXTRACTION DIAGNOSTIC")
    print("="*70)
    print()
    
    # 1. DataFrame check
    print("📊 DataFrame:")
    print(f"   Shape: {df.shape}")
    print(f"   Empty: {df.empty}")
    print(f"   Columns: {list(df.columns)}")
    
    if df.empty:
        print("\n❌ PROBLEM: DataFrame is empty!")
        print("   → Check CSV file loading")
        return False
    
    # 2. Required columns
    print("\n📋 Required Columns:")
    required = ['frame_num', 'url']
    missing = []
    for col in required:
        exists = col in df.columns
        status = "✅" if exists else "❌"
        print(f"   {status} {col}")
        if not exists:
            missing.append(col)
    
    if missing:
        print(f"\n❌ PROBLEM: Missing columns: {missing}")
        print("   → Available columns:", list(df.columns))
        
        # Suggest fixes
        if 'frame_num' in missing:
            frame_cols = [c for c in df.columns if 'frame' in c.lower()]
            if frame_cols:
                print(f"   → Try: df.rename(columns={{'{frame_cols[0]}': 'frame_num'}})")
        
        if 'url' in missing:
            url_cols = [c for c in df.columns if 'url' in c.lower() or 'video' in c.lower()]
            if url_cols:
                print(f"   → Try: df.rename(columns={{'{url_cols[0]}': 'url'}})")
        
        return False
    
    # 3. Video path
    print(f"\n📁 Video Path:")
    print(f"   Path: {video_base_path}")
    print(f"   Exists: {video_base_path.exists()}")
    
    if not video_base_path.exists():
        print(f"\n❌ PROBLEM: Video directory doesn't exist!")
        print(f"   → Create directory or check path")
        return False
    
    # 4. Sample video file
    print(f"\n🎥 Sample Video:")
    try:
        from ambient.utils.youtube_cache import extract_video_id
        sample_url = df['url'].iloc[0]
        video_id = extract_video_id(sample_url)
        video_path = video_base_path / f"{video_id}.mp4"
        
        print(f"   URL: {sample_url}")
        print(f"   Video ID: {video_id}")
        print(f"   File: {video_path.name}")
        print(f"   Exists: {video_path.exists()}")
        
        if video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")
        else:
            print(f"\n❌ PROBLEM: Video file not found!")
            print(f"   → Video needs to be downloaded")
            print(f"   → GAVDDataLoader.load_gavd_data() should auto-download")
            return False
            
    except Exception as e:
        print(f"\n❌ PROBLEM: Error checking video: {e}")
        return False
    
    # 5. Extractor validation
    print(f"\n✓ Validation:")
    try:
        is_valid, message = extractor.validate_sequence_data_verbose(df, video_base_path)
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"   {status}")
        print(f"   Message: {message}")
        
        if not is_valid:
            print(f"\n❌ PROBLEM: Validation failed!")
            print(f"   → Fix the issue above")
            return False
            
    except Exception as e:
        print(f"\n❌ PROBLEM: Validation error: {e}")
        return False
    
    # 6. Test extraction
    print(f"\n🧪 Test Extraction (2 frames):")
    try:
        test_df = df.head(2)
        test_result = extractor.extract_from_sequence(
            sequence_data=test_df,
            video_base_path=video_base_path,
            verbose=False
        )
        
        print(f"   Returned: {len(test_result)} frames")
        
        if test_result:
            successful = sum(1 for k in test_result if k is not None)
            failed = len(test_result) - successful
            print(f"   ✅ Successful: {successful}")
            if failed > 0:
                print(f"   ❌ Failed: {failed}")
            
            if successful == 0:
                print(f"\n❌ PROBLEM: All frames failed!")
                print(f"   → Check logs above for errors")
                print(f"   → Video file might be corrupted")
                return False
        else:
            print(f"\n❌ PROBLEM: Empty array returned!")
            print(f"   → This shouldn't happen if validation passed")
            print(f"   → Check error logs above")
            return False
            
    except Exception as e:
        print(f"\n❌ PROBLEM: Extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED - Ready for full extraction")
    print("="*70)
    return True


def safe_extract_keypoints(df, video_base_path, extractor, verbose=True):
    """
    Safe wrapper for extract_from_sequence with diagnostics.
    
    Usage in notebook:
        keypoints = safe_extract_keypoints(normal_df, video_base_path, extractor)
    """
    # Run diagnostics first
    if not diagnose_extraction_issue(df, video_base_path, extractor):
        print("\n❌ Diagnostic failed - cannot proceed with extraction")
        return []
    
    print(f"\n🔄 Extracting keypoints from {len(df)} frames...")
    print()
    
    result = extractor.extract_from_sequence(
        sequence_data=df,
        video_base_path=video_base_path,
        verbose=verbose
    )
    
    print()
    print(f"✅ Extraction complete: {len(result)} frames")
    if result:
        successful = sum(1 for k in result if k is not None)
        failed = len(result) - successful
        print(f"   Successful: {successful}")
        if failed > 0:
            print(f"   Failed: {failed}")
    
    return result
