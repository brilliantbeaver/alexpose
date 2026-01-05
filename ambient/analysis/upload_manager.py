"""
File State Documentation
=======================

The Google Generative AI SDK uses enum values for file states. Understanding these states
is crucial for proper file handling and avoiding infinite loops.

File States:
- STATE_UNSPECIFIED = 0: Initial state when file is first uploaded
- PROCESSING = 1: File is being processed by Gemini (not ready for use)
- ACTIVE = 2: File is ready for use in analysis
- FAILED = 10: File processing failed

Important Notes:
1. Files start in PROCESSING state (1) and transition to ACTIVE (2)
2. CSV files often become ACTIVE immediately after upload
3. Video files may take longer to process and transition from PROCESSING to ACTIVE
4. Always check for ACTIVE state (2) before using files in analysis
5. Handle both enum values and string representations for robustness

Example Usage:
    file_obj = genai.upload_file("video.mp4")
    if file_obj.state == 2 or file_obj.state.name == 'ACTIVE':
        # File is ready for use
        pass

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import google.genai as genai
except ImportError:
    genai = None


class AmbientGeminiFileManager:
    """Manages file uploads to Gemini and caches references locally."""

    def __init__(self, cache_filepath: Optional[str] = None):
        """
        Initialize the file manager.

        Args:
            cache_filepath: Path to the cache file. If None, uses default.
        """
        if cache_filepath is None:
            cache_filepath = "config/ambient_gemini_cache.json"
        self.cache_filepath = cache_filepath
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cached file references from disk."""
        if os.path.exists(self.cache_filepath):
            with open(self.cache_filepath, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save file references to disk."""
        with open(self.cache_filepath, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to detect changes."""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def upload_video(self, video_path: str) -> Optional[object]:
        """
        Upload video file to Gemini, cache reference.

        File States (Google Generative AI SDK):
        - STATE_UNSPECIFIED = 0: Initial state
        - PROCESSING = 1: File is being processed
        - ACTIVE = 2: File is ready for use
        - FAILED = 10: File processing failed

        Args:
            video_path: Path to the video file to upload

        Returns:
            Gemini file object if successful, None if failed
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package is required. Install it with: pip install google-generativeai"
            )
        file_hash = self._get_file_hash(video_path)
        filename = os.path.basename(video_path)
        cache_key = f"video_{filename}"

        # Check if already cached and file hasn't changed
        cached_entry = None
        if cache_key in self.cache and self.cache[cache_key]["hash"] == file_hash:
            cached_entry = self.cache[cache_key]
            print(f"Using cached video reference: {filename}")

        if cached_entry:
            # Try to get the file object from cache first
            try:
                cached_ref = cached_entry["reference"]
                print(f"DEBUG: Cached reference: {cached_ref}")

                # Try to retrieve the file object using the cached name
                video_ref = genai.get_file(cached_ref["name"])
                print(
                    f"DEBUG: Retrieved file object from cache, state: {getattr(video_ref, 'state', None)}"
                )

                # Check if the file is already active (enum value 2 or string 'ACTIVE')
                if (
                    getattr(video_ref, "state", None) == "ACTIVE"
                    or getattr(video_ref, "state", None) == 2
                    or (
                        hasattr(getattr(video_ref, "state", None), "name")
                        and getattr(video_ref, "state", None).name == "ACTIVE"
                    )
                ):
                    print(f"DEBUG: File is already ACTIVE, using cached reference")
                    return video_ref
                else:
                    print(f"DEBUG: File is not ACTIVE, will re-upload")
                    # Fall through to re-upload
            except Exception as e:
                print(f"DEBUG: Error retrieving cached file: {e}")
                # Fall through to re-upload

        # Upload to Gemini
        try:
            print(f"DEBUG: Uploading video file: {video_path}")
            video_ref = genai.upload_file(video_path)
            print(
                f"DEBUG: Upload successful, file state: {getattr(video_ref, 'state', None)}"
            )

            self.cache[cache_key] = {
                "hash": file_hash,
                "reference": {
                    "name": video_ref.name,
                    "uri": video_ref.uri,
                    "display_name": video_ref.display_name,
                },
            }
            self._save_cache()
            print(f"Uploaded and cached video: {filename}")
            return video_ref
        except Exception as e:
            print(f"Error uploading video {video_path}: {e}")
            return None

    def upload_csv(self, csv_path: str) -> Optional[object]:
        """
        Upload CSV file to Gemini, cache reference.

        File States (Google Generative AI SDK):
        - STATE_UNSPECIFIED = 0: Initial state
        - PROCESSING = 1: File is being processed
        - ACTIVE = 2: File is ready for use
        - FAILED = 10: File processing failed

        Args:
            csv_path: Path to the CSV file to upload

        Returns:
            Gemini file object if successful, None if failed
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package is required. Install it with: pip install google-generativeai"
            )
        file_hash = self._get_file_hash(csv_path)
        filename = os.path.basename(csv_path)
        cache_key = f"csv_{filename}"

        # Check if already cached and file hasn't changed
        cached_entry = None
        if cache_key in self.cache and self.cache[cache_key]["hash"] == file_hash:
            cached_entry = self.cache[cache_key]
            print(f"Using cached CSV reference: {filename}")

        if cached_entry:
            # Try to get the file object from cache first
            try:
                cached_ref = cached_entry["reference"]
                print(f"DEBUG: Cached CSV reference: {cached_ref}")

                # Try to retrieve the file object using the cached name
                csv_ref = genai.get_file(cached_ref["name"])
                print(
                    f"DEBUG: Retrieved CSV file object from cache, state: {getattr(csv_ref, 'state', None)}"
                )

                # Check if the file is already active (enum value 2 or string 'ACTIVE')
                if (
                    getattr(csv_ref, "state", None) == "ACTIVE"
                    or getattr(csv_ref, "state", None) == 2
                    or (
                        hasattr(getattr(csv_ref, "state", None), "name")
                        and getattr(csv_ref, "state", None).name == "ACTIVE"
                    )
                ):
                    print(f"DEBUG: CSV file is already ACTIVE, using cached reference")
                    return csv_ref
                else:
                    print(f"DEBUG: CSV file is not ACTIVE, will re-upload")
                    # Fall through to re-upload
            except Exception as e:
                print(f"DEBUG: Error retrieving cached CSV file: {e}")
                # Fall through to re-upload

        try:
            print(f"DEBUG: Uploading CSV file: {csv_path}")
            csv_ref = genai.upload_file(csv_path, mime_type="text/csv")
            print(
                f"DEBUG: CSV upload successful, file state: {getattr(csv_ref, 'state', None)}"
            )

            self.cache[cache_key] = {
                "hash": file_hash,
                "reference": {
                    "name": csv_ref.name,
                    "uri": csv_ref.uri,
                    "display_name": csv_ref.display_name,
                },
            }
            self._save_cache()
            print(f"Uploaded and cached CSV: {filename}")
            return csv_ref
        except Exception as e:
            print(f"Error uploading CSV {csv_path}: {e}")
            return None

    def get_cached_reference(self, file_path: str) -> Optional[Dict]:
        """Get cached reference for a file."""
        file_hash = self._get_file_hash(file_path)
        filename = os.path.basename(file_path)
        cache_key = f"{'video' if file_path.endswith('.mp4') else 'csv'}_{filename}"

        # Check if cached and file hasn't changed
        if cache_key in self.cache and self.cache[cache_key]["hash"] == file_hash:
            return self.cache[cache_key]["reference"]
        return None

    def clear_cache(self):
        """Clear all cached references."""
        self.cache = {}
        self._save_cache()
        print("Cache cleared")

    def remove_expired(self):
        """Remove references for files that no longer exist."""
        expired_keys = []
        for key, data in self.cache.items():
            # Extract filename from cache key
            if key.startswith("video_"):
                filename = key.replace("video_", "")
                # Check if the file exists in current directory or common paths
                if not self._file_exists_anywhere(filename):
                    expired_keys.append(key)
            elif key.startswith("csv_"):
                filename = key.replace("csv_", "")
                # Check if the file exists in current directory or common paths
                if not self._file_exists_anywhere(filename):
                    expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self._save_cache()
            print(f"Removed {len(expired_keys)} expired references")

    def _file_exists_anywhere(self, filename: str) -> bool:
        """Check if a file exists in current directory or common data paths."""
        # Check current directory
        if os.path.exists(filename):
            return True

        # Check common data directories
        common_paths = ["data", "data/toronto-gait-outputs", "tests/data", "examples"]

        for path in common_paths:
            if os.path.exists(os.path.join(path, filename)):
                return True

        return False

    def upload_directory(self, dir_path: str, pattern: str = "*.mp4"):
        """Upload all files matching pattern in directory."""
        import glob

        files = glob.glob(os.path.join(dir_path, pattern))
        for file_path in files:
            if file_path.endswith(".mp4"):
                self.upload_video(file_path)
            elif file_path.endswith(".csv"):
                self.upload_csv(file_path)
