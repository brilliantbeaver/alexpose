"""
YouTube video handler for downloading and processing YouTube videos.

This module provides functionality to download YouTube videos using yt-dlp
and integrate them with the existing video processing pipeline.

Author: AlexPose Team
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, List
from urllib.parse import urlparse, parse_qs
from loguru import logger

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

from ambient.core.frame import FrameSequence


class YouTubeHandler:
    """
    Handler for YouTube video downloading and processing.
    
    This class manages YouTube video downloads, caching, and metadata extraction
    using yt-dlp while providing integration with the existing video processing
    pipeline.
    """
    
    def __init__(
        self,
        download_dir: Optional[Path] = None,
        quality: str = "best[height<=720]",
        audio: bool = False,
        max_duration: Optional[int] = None
    ):
        """
        Initialize YouTube handler.
        
        Args:
            download_dir: Directory to store downloaded videos
            quality: Video quality selector for yt-dlp
            audio: Whether to download audio
            max_duration: Maximum video duration in seconds
        """
        if not YT_DLP_AVAILABLE:
            raise ImportError(
                "yt-dlp is not installed. Please install it with: "
                "pip install yt-dlp"
            )
        
        self.download_dir = download_dir or Path("data/youtube")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality = quality
        self.audio = audio
        self.max_duration = max_duration
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': quality,
            'outtmpl': str(self.download_dir / '%(id)s.%(ext)s'),
            'writeinfojson': True,
            'writedescription': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': False,
            'no_warnings': False,
            'extractflat': False,
            'writethumbnail': False,
            'writewebvtt': False,
        }
        
        if not audio:
            self.ydl_opts['format'] = f"{quality}/best[vcodec!=none]"
        
        if max_duration:
            self.ydl_opts['match_filter'] = self._duration_filter
        
        logger.info(f"YouTube handler initialized with download dir: {self.download_dir}")
    
    def _duration_filter(self, info_dict):
        """Filter videos by duration."""
        duration = info_dict.get('duration')
        if duration and self.max_duration and duration > self.max_duration:
            return f"Video too long: {duration}s > {self.max_duration}s"
        return None
    
    def is_youtube_url(self, url: str) -> bool:
        """
        Check if a URL is a valid YouTube URL.
        
        Supports multiple YouTube URL formats:
        - Standard watch URLs: youtube.com/watch?v=VIDEO_ID
        - Short URLs: youtu.be/VIDEO_ID
        - Embed URLs: youtube.com/embed/VIDEO_ID
        - Direct video URLs: youtube.com/v/VIDEO_ID
        - Shorts URLs: youtube.com/shorts/VIDEO_ID
        - Live URLs: youtube.com/live/VIDEO_ID
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is a YouTube URL
        """
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/live/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in youtube_patterns:
            if re.match(pattern, url):
                return True
        
        return False
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract YouTube video ID from URL.
        
        Supports multiple YouTube URL formats including shorts, live streams,
        embeds, and standard watch URLs.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None if not found
        """
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/live/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in youtube_patterns:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get video information without downloading.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video information dictionary or None if failed
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'duration': info.get('duration'),
                    'upload_date': info.get('upload_date'),
                    'uploader': info.get('uploader'),
                    'view_count': info.get('view_count'),
                    'width': info.get('width'),
                    'height': info.get('height'),
                    'fps': info.get('fps'),
                    'format': info.get('format'),
                    'filesize': info.get('filesize'),
                    'thumbnail': info.get('thumbnail')
                }
        
        except Exception as e:
            logger.error(f"Failed to get video info for {url}: {e}")
            return None
    
    def download_video(self, url: str, force_redownload: bool = False) -> Optional[Path]:
        """
        Download YouTube video.
        
        Args:
            url: YouTube URL
            force_redownload: Force redownload even if file exists
            
        Returns:
            Path to downloaded video file or None if failed
        """
        if not self.is_youtube_url(url):
            raise ValueError(f"Invalid YouTube URL: {url}")
        
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {url}")
        
        # Check if video already exists
        existing_files = list(self.download_dir.glob(f"{video_id}.*"))
        video_files = [f for f in existing_files if f.suffix in ['.mp4', '.webm', '.mkv', '.avi']]
        
        if video_files and not force_redownload:
            logger.info(f"Video {video_id} already exists: {video_files[0]}")
            return video_files[0]
        
        try:
            # Download video
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                logger.info(f"Downloading YouTube video: {url}")
                ydl.download([url])
            
            # Find downloaded file
            new_files = list(self.download_dir.glob(f"{video_id}.*"))
            video_files = [f for f in new_files if f.suffix in ['.mp4', '.webm', '.mkv', '.avi']]
            
            if video_files:
                logger.info(f"Successfully downloaded: {video_files[0]}")
                return video_files[0]
            else:
                logger.error(f"No video file found after download for {video_id}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to download video {url}: {e}")
            return None
    
    def get_cached_video(self, url: str) -> Optional[Path]:
        """
        Get cached video file if it exists.
        
        Args:
            url: YouTube URL
            
        Returns:
            Path to cached video file or None if not found
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            return None
        
        # Look for existing video files
        existing_files = list(self.download_dir.glob(f"{video_id}.*"))
        video_files = [f for f in existing_files if f.suffix in ['.mp4', '.webm', '.mkv', '.avi']]
        
        return video_files[0] if video_files else None
    
    def get_or_download_video(self, url: str) -> Optional[Path]:
        """
        Get cached video or download if not exists.
        
        Args:
            url: YouTube URL
            
        Returns:
            Path to video file or None if failed
        """
        # Check cache first
        cached_path = self.get_cached_video(url)
        if cached_path and cached_path.exists():
            return cached_path
        
        # Download if not cached
        return self.download_video(url)
    
    def get_video_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata from info JSON file.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video metadata dictionary or None if not found
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            return None
        
        info_file = self.download_dir / f"{video_id}.info.json"
        if not info_file.exists():
            # Try to get info without downloading
            return self.get_video_info(url)
        
        try:
            import json
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read metadata file {info_file}: {e}")
            return None
    
    def clean_cache(self, max_age_days: int = 30, max_size_gb: float = 10.0):
        """
        Clean old cached videos.
        
        Args:
            max_age_days: Maximum age of cached videos in days
            max_size_gb: Maximum total cache size in GB
        """
        import time
        import os
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        # Get all files with sizes and ages
        files_info = []
        total_size = 0
        
        for file_path in self.download_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                age = current_time - stat.st_mtime
                size = stat.st_size
                
                files_info.append({
                    'path': file_path,
                    'age': age,
                    'size': size
                })
                total_size += size
        
        # Remove old files
        removed_count = 0
        for file_info in files_info:
            if file_info['age'] > max_age_seconds:
                try:
                    file_info['path'].unlink()
                    total_size -= file_info['size']
                    removed_count += 1
                    logger.info(f"Removed old cached file: {file_info['path']}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_info['path']}: {e}")
        
        # Remove largest files if still over size limit
        if total_size > max_size_bytes:
            # Sort by size (largest first)
            files_info = [f for f in files_info if f['path'].exists()]
            files_info.sort(key=lambda x: x['size'], reverse=True)
            
            for file_info in files_info:
                if total_size <= max_size_bytes:
                    break
                
                try:
                    file_info['path'].unlink()
                    total_size -= file_info['size']
                    removed_count += 1
                    logger.info(f"Removed large cached file: {file_info['path']}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_info['path']}: {e}")
        
        logger.info(f"Cache cleanup complete: removed {removed_count} files, "
                   f"total size now {total_size / (1024*1024*1024):.2f} GB")
    
    def list_cached_videos(self) -> List[Dict[str, Any]]:
        """
        List all cached videos with metadata.
        
        Returns:
            List of cached video information
        """
        cached_videos = []
        
        # Find all video files
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi']
        for file_path in self.download_dir.iterdir():
            if file_path.suffix.lower() in video_extensions:
                video_id = file_path.stem
                
                # Get file info
                stat = file_path.stat()
                
                video_info = {
                    'video_id': video_id,
                    'file_path': file_path,
                    'file_size': stat.st_size,
                    'modified_time': stat.st_mtime,
                    'extension': file_path.suffix
                }
                
                # Try to get metadata
                metadata = self.get_video_metadata(f"https://youtube.com/watch?v={video_id}")
                if metadata:
                    video_info.update({
                        'title': metadata.get('title'),
                        'duration': metadata.get('duration'),
                        'uploader': metadata.get('uploader'),
                        'upload_date': metadata.get('upload_date')
                    })
                
                cached_videos.append(video_info)
        
        return cached_videos
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for cached videos.
        
        Returns:
            Storage statistics dictionary
        """
        total_size = 0
        file_count = 0
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi']
        
        for file_path in self.download_dir.iterdir():
            if file_path.is_file():
                total_size += file_path.stat().st_size
                if file_path.suffix.lower() in video_extensions:
                    file_count += 1
        
        return {
            'download_dir': str(self.download_dir),
            'total_files': len(list(self.download_dir.iterdir())),
            'video_files': file_count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024)
        }


# Global YouTube handler instance
_youtube_handler = None


def get_youtube_handler(
    download_dir: Optional[Path] = None,
    quality: str = "best[height<=720]",
    **kwargs
) -> YouTubeHandler:
    """
    Get global YouTube handler instance.
    
    Args:
        download_dir: Directory to store downloaded videos
        quality: Video quality selector
        **kwargs: Additional arguments for YouTubeHandler
        
    Returns:
        Global YouTubeHandler instance
    """
    global _youtube_handler
    if _youtube_handler is None:
        _youtube_handler = YouTubeHandler(
            download_dir=download_dir,
            quality=quality,
            **kwargs
        )
    return _youtube_handler


def download_youtube_video(url: str, **kwargs) -> Optional[Path]:
    """
    Convenience function to download a YouTube video.
    
    Args:
        url: YouTube URL
        **kwargs: Additional arguments for YouTubeHandler
        
    Returns:
        Path to downloaded video file or None if failed
    """
    handler = get_youtube_handler(**kwargs)
    return handler.get_or_download_video(url)


def is_youtube_url(url: str) -> bool:
    """
    Convenience function to check if URL is a YouTube URL.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is a YouTube URL
    """
    handler = get_youtube_handler()
    return handler.is_youtube_url(url)