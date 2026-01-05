"""
File validation utilities for video uploads and GAVD dataset uploads.

Provides validation for video files, YouTube URLs, and GAVD CSV files.
"""

from fastapi import UploadFile
from typing import Dict, Any
import re
import csv
import io
from pathlib import Path
from loguru import logger


# Supported video formats
SUPPORTED_VIDEO_FORMATS = {
    '.mp4': 'video/mp4',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
    '.webm': 'video/webm',
    '.mkv': 'video/x-matroska',
    '.flv': 'video/x-flv'
}

# Maximum file size (500 MB by default)
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


async def validate_video_file(file: UploadFile) -> Dict[str, Any]:
    """
    Validate uploaded video file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Validation result dictionary with 'valid' boolean and optional 'error' message
    """
    # Check if file exists
    if not file:
        return {
            "valid": False,
            "error": "No file provided"
        }
    
    # Check filename
    if not file.filename:
        return {
            "valid": False,
            "error": "No filename provided"
        }
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_VIDEO_FORMATS:
        return {
            "valid": False,
            "error": f"Unsupported file format: {file_extension}. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS.keys())}"
        }
    
    # Check content type
    expected_content_type = SUPPORTED_VIDEO_FORMATS[file_extension]
    if file.content_type and not file.content_type.startswith('video/'):
        logger.warning(f"Unexpected content type: {file.content_type} for {file.filename}")
    
    # Check file size
    try:
        # Read file to check size
        file_content = await file.read()
        file_size = len(file_content)
        
        # Reset file pointer for later reading
        await file.seek(0)
        
        if file_size > MAX_FILE_SIZE_BYTES:
            return {
                "valid": False,
                "error": f"File size ({file_size / (1024**2):.2f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"
            }
        
        if file_size == 0:
            return {
                "valid": False,
                "error": "File is empty"
            }
        
        return {
            "valid": True,
            "file_extension": file_extension,
            "content_type": file.content_type,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024**2), 2)
        }
        
    except Exception as e:
        logger.error(f"Error validating file: {str(e)}")
        return {
            "valid": False,
            "error": f"Error reading file: {str(e)}"
        }


def validate_youtube_url(url: str) -> Dict[str, Any]:
    """
    Validate YouTube URL.
    
    Args:
        url: YouTube URL to validate
        
    Returns:
        Validation result dictionary with 'valid' boolean and optional 'error' message
    """
    if not url:
        return {
            "valid": False,
            "error": "No URL provided"
        }
    
    # YouTube URL patterns
    youtube_patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(https?://)?(www\.)?youtu\.be/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/embed/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/v/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/shorts/[\w-]+'
    ]
    
    # Check if URL matches any YouTube pattern
    is_valid = any(re.match(pattern, url) for pattern in youtube_patterns)
    
    if not is_valid:
        return {
            "valid": False,
            "error": "Invalid YouTube URL format. Please provide a valid YouTube video URL."
        }
    
    # Extract video ID
    video_id = None
    if 'watch?v=' in url:
        video_id = url.split('watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        video_id = url.split('youtu.be/')[1].split('?')[0]
    elif '/embed/' in url:
        video_id = url.split('/embed/')[1].split('?')[0]
    elif '/v/' in url:
        video_id = url.split('/v/')[1].split('?')[0]
    elif '/shorts/' in url:
        video_id = url.split('/shorts/')[1].split('?')[0]
    
    return {
        "valid": True,
        "url": url,
        "video_id": video_id
    }


def get_supported_formats() -> Dict[str, str]:
    """
    Get dictionary of supported video formats.
    
    Returns:
        Dictionary mapping file extensions to MIME types
    """
    return SUPPORTED_VIDEO_FORMATS.copy()


def get_max_file_size() -> Dict[str, Any]:
    """
    Get maximum allowed file size.
    
    Returns:
        Dictionary with size in bytes and MB
    """
    return {
        "bytes": MAX_FILE_SIZE_BYTES,
        "megabytes": MAX_FILE_SIZE_MB
    }


async def validate_csv_file(file: UploadFile) -> Dict[str, Any]:
    """
    Validate uploaded GAVD CSV file.
    
    Args:
        file: Uploaded CSV file object
        
    Returns:
        Validation result dictionary with 'valid' boolean and optional 'error' message
    """
    # Check if file exists
    if not file:
        return {
            "valid": False,
            "error": "No file provided"
        }
    
    # Check filename
    if not file.filename:
        return {
            "valid": False,
            "error": "No filename provided"
        }
    
    # Check file extension
    if not file.filename.endswith('.csv'):
        return {
            "valid": False,
            "error": "File must be a CSV file"
        }
    
    # Required GAVD columns
    required_columns = ['seq', 'frame_num', 'bbox', 'url']
    optional_columns = ['cam_view', 'gait_event', 'dataset', 'gait_pat', 'vid_info', 'id']
    
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Reset file pointer for later reading
        await file.seek(0)
        
        if file_size == 0:
            return {
                "valid": False,
                "error": "File is empty"
            }
        
        # Parse CSV to validate structure
        content_str = file_content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(content_str))
        
        # Get headers
        headers = csv_reader.fieldnames
        if not headers:
            return {
                "valid": False,
                "error": "CSV file has no headers"
            }
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in headers]
        if missing_columns:
            return {
                "valid": False,
                "error": f"Missing required columns: {', '.join(missing_columns)}"
            }
        
        # Count rows and unique sequences
        row_count = 0
        sequences = set()
        
        for row in csv_reader:
            row_count += 1
            if 'seq' in row and row['seq']:
                sequences.add(row['seq'])
            
            # Limit validation to first 1000 rows for performance
            if row_count >= 1000:
                break
        
        # Reset file pointer again
        await file.seek(0)
        
        return {
            "valid": True,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024**2), 2),
            "row_count": row_count,
            "sequence_count": len(sequences),
            "headers": list(headers),
            "has_all_required_columns": True
        }
        
    except UnicodeDecodeError:
        return {
            "valid": False,
            "error": "File encoding error. Please ensure the file is UTF-8 encoded."
        }
    except csv.Error as e:
        return {
            "valid": False,
            "error": f"CSV parsing error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error validating CSV file: {str(e)}")
        return {
            "valid": False,
            "error": f"Error reading file: {str(e)}"
        }
