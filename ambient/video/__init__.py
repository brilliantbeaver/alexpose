"""
Video processing module for AlexPose.

This module provides video processing capabilities including YouTube support,
frame extraction, and video metadata handling.

Author: AlexPose Team
"""

from ambient.video.processor import VideoProcessor
from ambient.video.youtube_handler import YouTubeHandler

__all__ = [
    "VideoProcessor",
    "YouTubeHandler"
]