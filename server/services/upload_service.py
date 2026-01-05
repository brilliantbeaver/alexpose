"""
Upload service for managing video file uploads and YouTube downloads.

Handles file storage, metadata management, and YouTube video processing.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import shutil
from datetime import datetime
from loguru import logger

from ambient.video.youtube_handler import YouTubeHandler
from ambient.core.config import ConfigurationManager


class UploadService:
    """
    Service for managing video uploads and YouTube downloads.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize upload service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.videos_dir = Path(getattr(self.config.storage, 'videos_directory', 'data/videos'))
        self.youtube_dir = Path(getattr(self.config.storage, 'youtube_directory', 'data/youtube'))
        self.metadata_dir = Path(getattr(self.config.storage, 'data_directory', 'data')) / 'uploads_metadata'
        
        # Create directories
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.youtube_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YouTube handler
        self.youtube_handler = YouTubeHandler(str(self.youtube_dir))
    
    def save_upload_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """
        Save upload metadata to JSON file.
        
        Args:
            file_id: Unique file identifier
            metadata: Metadata dictionary
        """
        metadata_file = self.metadata_dir / f"{file_id}.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata for {file_id}")
        except Exception as e:
            logger.error(f"Error saving metadata for {file_id}: {str(e)}")
            raise
    
    def get_upload_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve upload metadata.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_file = self.metadata_dir / f"{file_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata for {file_id}: {str(e)}")
            return None
    
    def update_upload_metadata(self, file_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update upload metadata.
        
        Args:
            file_id: Unique file identifier
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self.get_upload_metadata(file_id)
        if not metadata:
            return False
        
        metadata.update(updates)
        metadata['updated_at'] = datetime.utcnow().isoformat()
        
        self.save_upload_metadata(file_id, metadata)
        return True
    
    def delete_upload(self, file_id: str) -> bool:
        """
        Delete an upload and its metadata.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            True if successful, False if not found
        """
        metadata = self.get_upload_metadata(file_id)
        if not metadata:
            return False
        
        # Delete video file
        if 'file_path' in metadata:
            file_path = Path(metadata['file_path'])
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted video file: {file_path}")
        
        # Delete YouTube directory if applicable
        if 'youtube_video_id' in metadata:
            youtube_video_dir = self.youtube_dir / metadata['youtube_video_id']
            if youtube_video_dir.exists():
                shutil.rmtree(youtube_video_dir)
                logger.info(f"Deleted YouTube directory: {youtube_video_dir}")
        
        # Delete metadata file
        metadata_file = self.metadata_dir / f"{file_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info(f"Deleted metadata file: {metadata_file}")
        
        return True
    
    def list_uploads(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all uploads with pagination.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of upload metadata dictionaries
        """
        metadata_files = sorted(
            self.metadata_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        uploads = []
        for metadata_file in metadata_files[offset:offset + limit]:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    uploads.append(metadata)
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_file}: {str(e)}")
        
        return uploads
    
    async def download_youtube_video(self, video_id: str, url: str) -> None:
        """
        Download YouTube video in background.
        
        Args:
            video_id: Unique video identifier
            url: YouTube URL
        """
        try:
            logger.info(f"Starting YouTube download for {video_id}: {url}")
            
            # Update status
            self.update_upload_metadata(video_id, {"status": "downloading"})
            
            # Download video
            result = self.youtube_handler.download_video(url)
            
            if result['success']:
                # Update metadata with download results
                self.update_upload_metadata(video_id, {
                    "status": "downloaded",
                    "youtube_video_id": result['video_id'],
                    "file_path": result['video_path'],
                    "file_size": Path(result['video_path']).stat().st_size,
                    "title": result.get('title'),
                    "duration": result.get('duration'),
                    "downloaded_at": datetime.utcnow().isoformat()
                })
                logger.info(f"YouTube download completed for {video_id}")
            else:
                # Update with error
                self.update_upload_metadata(video_id, {
                    "status": "download_failed",
                    "error": result.get('error', 'Unknown error')
                })
                logger.error(f"YouTube download failed for {video_id}: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error downloading YouTube video {video_id}: {str(e)}")
            self.update_upload_metadata(video_id, {
                "status": "download_failed",
                "error": str(e)
            })
