"""
Video streaming endpoints for AlexPose FastAPI application.

Provides endpoints for streaming cached YouTube videos with range request support
for efficient video playback in the browser.
"""

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional
from pathlib import Path
import os
from loguru import logger

from ambient.utils.youtube_cache import extract_video_id

router = APIRouter(prefix="/api/v1/video", tags=["video"])


def get_video_path(video_id: str, youtube_dir: Path) -> Optional[Path]:
    """
    Find cached video file for a given video ID.
    
    Args:
        video_id: YouTube video ID
        youtube_dir: Directory containing cached videos
        
    Returns:
        Path to video file or None if not found
    """
    for ext in ['.mp4', '.webm', '.mkv', '.mov']:
        video_path = youtube_dir / f"{video_id}{ext}"
        if video_path.exists() and video_path.is_file():
            return video_path
    return None


def range_requests_response(
    file_path: Path,
    range_header: Optional[str] = None,
    content_type: str = "video/mp4"
):
    """
    Create a streaming response with range request support.
    
    This enables seeking in video players and efficient bandwidth usage.
    
    Args:
        file_path: Path to video file
        range_header: HTTP Range header value
        content_type: MIME type of the video
        
    Returns:
        StreamingResponse with appropriate headers
    """
    file_size = file_path.stat().st_size
    
    # Parse range header
    start = 0
    end = file_size - 1
    
    if range_header:
        # Range header format: "bytes=start-end"
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else file_size - 1
    
    # Ensure valid range
    start = max(0, start)
    end = min(end, file_size - 1)
    content_length = end - start + 1
    
    # Create streaming generator
    def iterfile():
        with open(file_path, 'rb') as f:
            f.seek(start)
            remaining = content_length
            chunk_size = 8192  # 8KB chunks
            
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
    
    # Prepare headers
    headers = {
        'Content-Range': f'bytes {start}-{end}/{file_size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(content_length),
        'Content-Type': content_type,
        'Cache-Control': 'public, max-age=3600',
    }
    
    # Return 206 Partial Content if range was requested, otherwise 200
    status_code = 206 if range_header else 200
    
    return StreamingResponse(
        iterfile(),
        status_code=status_code,
        headers=headers,
        media_type=content_type
    )


@router.get("/stream/{video_id}")
async def stream_video(
    request: Request,
    video_id: str,
    range: Optional[str] = Header(None)
):
    """
    Stream a cached YouTube video with range request support.
    
    This endpoint enables efficient video playback in the browser with seeking support.
    
    Args:
        video_id: YouTube video ID
        range: HTTP Range header for partial content requests
        
    Returns:
        StreamingResponse with video data
        
    Example:
        GET /api/v1/video/stream/B5hrxKe2nP8
        Range: bytes=0-1023
    """
    try:
        # Get configuration
        config_manager = request.app.state.config
        youtube_dir = Path(getattr(config_manager.config.storage, 'youtube_directory', 'data/youtube'))
        
        # Find video file
        video_path = get_video_path(video_id, youtube_dir)
        
        if not video_path:
            raise HTTPException(
                status_code=404,
                detail=f"Video not found: {video_id}. Video may not be cached yet."
            )
        
        # Determine content type based on extension
        content_type_map = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.mov': 'video/quicktime'
        }
        content_type = content_type_map.get(video_path.suffix.lower(), 'video/mp4')
        
        logger.debug(f"Streaming video: {video_path} (Range: {range})")
        
        return range_requests_response(video_path, range, content_type)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming video {video_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream video: {str(e)}"
        )


@router.get("/info/{video_id}")
async def get_video_info(
    request: Request,
    video_id: str
):
    """
    Get information about a cached video.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Video metadata including file size, format, and availability
    """
    try:
        config_manager = request.app.state.config
        youtube_dir = Path(getattr(config_manager.config.storage, 'youtube_directory', 'data/youtube'))
        
        video_path = get_video_path(video_id, youtube_dir)
        
        if not video_path:
            return {
                "success": False,
                "video_id": video_id,
                "available": False,
                "message": "Video not cached"
            }
        
        file_size = video_path.stat().st_size
        
        return {
            "success": True,
            "video_id": video_id,
            "available": True,
            "file_path": str(video_path),
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024**2), 2),
            "format": video_path.suffix[1:],  # Remove the dot
            "stream_url": f"/api/v1/video/stream/{video_id}"
        }
        
    except Exception as e:
        logger.error(f"Error getting video info for {video_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video info: {str(e)}"
        )


@router.get("/url-to-id")
async def url_to_video_id(url: str):
    """
    Extract video ID from a YouTube URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID and stream URL
        
    Example:
        GET /api/v1/video/url-to-id?url=https://www.youtube.com/watch?v=B5hrxKe2nP8
    """
    try:
        video_id = extract_video_id(url)
        
        if not video_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid YouTube URL"
            )
        
        return {
            "success": True,
            "url": url,
            "video_id": video_id,
            "stream_url": f"/api/v1/video/stream/{video_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting video ID from URL: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract video ID: {str(e)}"
        )
