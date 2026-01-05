"""
Video upload endpoints for AlexPose FastAPI application.

Provides endpoints for uploading video files and YouTube URLs for gait analysis.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from pathlib import Path
import uuid
import shutil
from datetime import datetime
from loguru import logger

from server.services.upload_service import UploadService
from server.utils.file_validation import validate_video_file, validate_youtube_url

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])


@router.post("/video")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Upload a video file for gait analysis.
    
    Args:
        file: Video file to upload
        description: Optional description of the video
        metadata: Optional JSON metadata
        
    Returns:
        Upload confirmation with file ID and details
    """
    logger.info(f"Received video upload request: {file.filename}")
    
    # Validate file
    try:
        validation_result = await validate_video_file(file)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video file: {validation_result['error']}"
            )
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")
    
    # Get configuration
    config_manager = request.app.state.config
    videos_dir = Path(getattr(config_manager.config.storage, 'videos_directory', 'data/videos'))
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = videos_dir / f"{file_id}{file_extension}"
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Video saved successfully: {file_path}")
        
        # Create upload service
        upload_service = UploadService(config_manager)
        
        # Store upload metadata
        upload_metadata = {
            "file_id": file_id,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "content_type": file.content_type,
            "description": description,
            "uploaded_at": datetime.utcnow().isoformat(),
            "status": "uploaded",
            "validation": validation_result
        }
        
        # Save metadata
        upload_service.save_upload_metadata(file_id, upload_metadata)
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "file_size": upload_metadata["file_size"],
            "uploaded_at": upload_metadata["uploaded_at"],
            "message": "Video uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error saving video file: {str(e)}")
        # Clean up partial file if it exists
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")


@router.post("/youtube")
async def upload_youtube_url(
    request: Request,
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    description: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Process a YouTube URL for gait analysis.
    
    Args:
        url: YouTube video URL
        description: Optional description
        
    Returns:
        Upload confirmation with video ID and download status
    """
    logger.info(f"Received YouTube URL: {url}")
    
    # Validate YouTube URL
    try:
        validation_result = validate_youtube_url(url)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid YouTube URL: {validation_result['error']}"
            )
    except Exception as e:
        logger.error(f"URL validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"URL validation failed: {str(e)}")
    
    # Get configuration
    config_manager = request.app.state.config
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Create upload service
    upload_service = UploadService(config_manager)
    
    # Store initial metadata
    upload_metadata = {
        "video_id": video_id,
        "youtube_url": url,
        "description": description,
        "uploaded_at": datetime.utcnow().isoformat(),
        "status": "pending_download",
        "validation": validation_result
    }
    
    upload_service.save_upload_metadata(video_id, upload_metadata)
    
    # Schedule background download
    background_tasks.add_task(
        upload_service.download_youtube_video,
        video_id,
        url
    )
    
    return {
        "success": True,
        "video_id": video_id,
        "youtube_url": url,
        "status": "pending_download",
        "message": "YouTube video download initiated. Check status endpoint for progress."
    }


@router.get("/status/{file_id}")
async def get_upload_status(
    request: Request,
    file_id: str
) -> Dict[str, Any]:
    """
    Get upload status and metadata.
    
    Args:
        file_id: File or video ID
        
    Returns:
        Upload status and metadata
    """
    config_manager = request.app.state.config
    upload_service = UploadService(config_manager)
    
    try:
        metadata = upload_service.get_upload_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return {
            "success": True,
            "file_id": file_id,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error retrieving upload status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve status: {str(e)}")


@router.delete("/{file_id}")
async def delete_upload(
    request: Request,
    file_id: str
) -> Dict[str, Any]:
    """
    Delete an uploaded video file.
    
    Args:
        file_id: File or video ID to delete
        
    Returns:
        Deletion confirmation
    """
    config_manager = request.app.state.config
    upload_service = UploadService(config_manager)
    
    try:
        result = upload_service.delete_upload(file_id)
        if not result:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return {
            "success": True,
            "file_id": file_id,
            "message": "Upload deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete upload: {str(e)}")


@router.get("/list")
async def list_uploads(
    request: Request,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    List all uploaded videos.
    
    Args:
        limit: Maximum number of results
        offset: Offset for pagination
        
    Returns:
        List of uploads with metadata
    """
    config_manager = request.app.state.config
    upload_service = UploadService(config_manager)
    
    try:
        uploads = upload_service.list_uploads(limit=limit, offset=offset)
        
        return {
            "success": True,
            "count": len(uploads),
            "limit": limit,
            "offset": offset,
            "uploads": uploads
        }
    except Exception as e:
        logger.error(f"Error listing uploads: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list uploads: {str(e)}")
