#!/usr/bin/env python3
"""
Ambient Batch Upload Script
Uploads all videos and CSVs to Gemini and caches references locally.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

import glob
import os

try:
    import google.genai as genai
except ImportError:
    genai = None

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    def find_dotenv():
        return None
    def load_dotenv(path):
        return False

from .upload_manager import AmbientGeminiFileManager

# Load environment variables
_ = load_dotenv(find_dotenv())


def upload_all_files():
    """Upload all videos and CSVs to Gemini in batch."""
    if genai is None:
        print("ERROR: google-generativeai package is required. Install it with: pip install google-generativeai")
        return

    # Configure Gemini
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        print("ERROR: GOOGLE_API_KEY environment variable is not set!")
        return

    genai.configure(api_key=gemini_api_key)

    # Initialize file manager
    manager = AmbientGeminiFileManager()

    # Configuration - read from environment variables
    videos_dir = os.getenv("VIDEOS_DIR")
    openpose_dir = os.getenv("OPENPOSE_OUTPUTS_DIR")

    # Validate environment variables
    if not videos_dir:
        print("Please set VIDEOS_DIR in your .env file")
        return

    if not openpose_dir:
        print("Please set OPENPOSE_OUTPUTS_DIR in your .env file")
        return

    print(f"==> Videos directory: {videos_dir}", flush=True)
    print(f"==> OpenPose directory: {openpose_dir}", flush=True)

    # Verify directories exist
    if not os.path.exists(videos_dir):
        print(f"ERROR: Videos directory does not exist: {videos_dir}")
        return

    if not os.path.exists(openpose_dir):
        print(f"ERROR: OpenPose directory does not exist: {openpose_dir}")
        return

    # Find all video files
    video_pattern = os.path.join(videos_dir, "*-bottom.mp4")
    video_files = glob.glob(video_pattern)

    print(f"Found {len(video_files)} video files to process")

    if len(video_files) == 0:
        print("No video files found. Check the path and file pattern.")
        return

    uploaded_count = 0
    failed_count = 0

    for video_path in video_files:
        record_id = os.path.basename(video_path).replace("-bottom.mp4", "")

        print(f"\nProcessing {record_id}...")

        # Upload video
        video_ref = manager.upload_video(video_path)
        if not video_ref:
            print(f"Failed to upload video: {video_path}")
            failed_count += 1
            continue

        # Find corresponding CSV
        csv_path = os.path.join(
            openpose_dir, f"{record_id}", f"{record_id}-bottom-gait.csv"
        )
        if os.path.exists(csv_path):
            csv_ref = manager.upload_csv(csv_path)
            if csv_ref:
                uploaded_count += 1
                print(f"Successfully processed: {record_id}")
            else:
                print(f"Failed to upload CSV for: {record_id}")
                failed_count += 1
        else:
            print(f"CSV not found for: {record_id}")
            print(f"Expected CSV path: {csv_path}")
            failed_count += 1

    print(f"\n=== Upload Summary ===")
    print(f"Successfully uploaded: {uploaded_count} video-CSV pairs")
    print(f"Failed: {failed_count}")
    print(f"Cache saved to: {manager.cache_file}")


def list_cached_files():
    """List all cached file references."""
    manager = AmbientGeminiFileManager()

    if not manager.cache:
        print("No cached files found.")
        return

    print(f"Cached files in {manager.cache_file}:")
    for key, data in manager.cache.items():
        print(f"  {key}: {data['reference']['display_name']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_cached_files()
    else:
        upload_all_files()
