"""
GAVD dataset upload and processing endpoints for AlexPose FastAPI application.

Provides endpoints for uploading GAVD training dataset CSV files and processing them
for gait analysis training and evaluation.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
from pathlib import Path
import uuid
import shutil
from datetime import datetime
from loguru import logger

from server.services.gavd_service import GAVDService
from server.utils.file_validation import validate_csv_file

router = APIRouter(prefix="/api/v1/gavd", tags=["gavd"])


@router.post("/upload")
async def upload_gavd_dataset(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    process_immediately: Optional[bool] = Form(True)
) -> Dict[str, Any]:
    """
    Upload a GAVD dataset CSV file for processing.
    
    The CSV file should contain columns: seq, frame_num, cam_view, gait_event, 
    dataset, gait_pat, bbox, vid_info, id, url
    
    Args:
        file: GAVD CSV file to upload
        description: Optional description of the dataset
        process_immediately: Whether to start processing immediately
        
    Returns:
        Upload confirmation with dataset ID and processing status
    """
    logger.info(f"Received GAVD dataset upload request: {file.filename}")
    
    # Validate file is CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV files are accepted for GAVD datasets."
        )
    
    # Validate CSV structure
    try:
        validation_result = await validate_csv_file(file)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid CSV file: {validation_result['error']}"
            )
    except Exception as e:
        logger.error(f"CSV validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"CSV validation failed: {str(e)}")
    
    # Get configuration
    config_manager = request.app.state.config
    training_dir = Path(getattr(config_manager.config.storage, 'training_directory', 'data/training'))
    gavd_dir = training_dir / 'gavd'
    gavd_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique dataset ID
    dataset_id = str(uuid.uuid4())
    file_path = gavd_dir / f"{dataset_id}.csv"
    
    # Save file
    try:
        # Reset file pointer after validation
        await file.seek(0)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"GAVD dataset saved successfully: {file_path}")
        
        # Create GAVD service
        gavd_service = GAVDService(config_manager)
        
        # Store dataset metadata
        dataset_metadata = {
            "dataset_id": dataset_id,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "description": description,
            "uploaded_at": datetime.utcnow().isoformat(),
            "status": "uploaded",
            "validation": validation_result,
            "row_count": validation_result.get("row_count", 0),
            "sequence_count": validation_result.get("sequence_count", 0)
        }
        
        # Save metadata
        gavd_service.save_dataset_metadata(dataset_id, dataset_metadata)
        
        # Schedule background processing if requested
        if process_immediately:
            background_tasks.add_task(
                gavd_service.process_dataset,
                dataset_id
            )
            dataset_metadata["status"] = "processing"
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "filename": file.filename,
            "file_size": dataset_metadata["file_size"],
            "row_count": dataset_metadata["row_count"],
            "sequence_count": dataset_metadata["sequence_count"],
            "uploaded_at": dataset_metadata["uploaded_at"],
            "status": dataset_metadata["status"],
            "message": "GAVD dataset uploaded successfully" + 
                      (" and processing started" if process_immediately else "")
        }
        
    except Exception as e:
        logger.error(f"Error saving GAVD dataset file: {str(e)}")
        # Clean up partial file if it exists
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save dataset: {str(e)}")


@router.post("/process/{dataset_id}")
async def process_gavd_dataset(
    request: Request,
    background_tasks: BackgroundTasks,
    dataset_id: str,
    max_sequences: Optional[int] = Form(None),
    pose_estimator: Optional[str] = Form("mediapipe")
) -> Dict[str, Any]:
    """
    Process a GAVD dataset to extract pose data and prepare for training.
    
    Args:
        dataset_id: Dataset ID to process
        max_sequences: Maximum number of sequences to process (None for all)
        pose_estimator: Pose estimator to use (mediapipe, openpose, etc.)
        
    Returns:
        Processing job ID and status
    """
    logger.info(f"Starting GAVD dataset processing: {dataset_id}")
    
    # Get configuration
    config_manager = request.app.state.config
    gavd_service = GAVDService(config_manager)
    
    # Verify dataset exists
    metadata = gavd_service.get_dataset_metadata(dataset_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Check if already processing
    if metadata.get("status") == "processing":
        raise HTTPException(status_code=400, detail="Dataset is already being processed")
    
    try:
        # Update status
        gavd_service.update_dataset_metadata(dataset_id, {
            "status": "processing",
            "processing_started_at": datetime.utcnow().isoformat(),
            "max_sequences": max_sequences,
            "pose_estimator": pose_estimator
        })
        
        # Schedule background processing
        background_tasks.add_task(
            gavd_service.process_dataset,
            dataset_id,
            max_sequences=max_sequences,
            pose_estimator=pose_estimator
        )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "status": "processing",
            "message": "GAVD dataset processing started. Use the status endpoint to monitor progress."
        }
        
    except Exception as e:
        logger.error(f"Error starting GAVD dataset processing: {str(e)}")
        gavd_service.update_dataset_metadata(dataset_id, {
            "status": "error",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


@router.get("/status/{dataset_id}")
async def get_dataset_status(
    request: Request,
    dataset_id: str
) -> Dict[str, Any]:
    """
    Get GAVD dataset processing status and metadata.
    
    Args:
        dataset_id: Dataset ID
        
    Returns:
        Dataset status and metadata
    """
    config_manager = request.app.state.config
    gavd_service = GAVDService(config_manager)
    
    try:
        metadata = gavd_service.get_dataset_metadata(dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "metadata": metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving dataset status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve status: {str(e)}")


@router.get("/results/{dataset_id}")
async def get_dataset_results(
    request: Request,
    dataset_id: str
) -> Dict[str, Any]:
    """
    Get processed GAVD dataset results.
    
    Args:
        dataset_id: Dataset ID
        
    Returns:
        Processed dataset results including sequences and statistics
    """
    config_manager = request.app.state.config
    gavd_service = GAVDService(config_manager)
    
    try:
        results = gavd_service.get_dataset_results(dataset_id)
        if not results:
            raise HTTPException(status_code=404, detail="Dataset results not found")
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving dataset results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")


@router.get("/list")
async def list_datasets(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all GAVD datasets with pagination and filtering.
    
    Args:
        limit: Maximum number of results
        offset: Offset for pagination
        status_filter: Filter by status (uploaded, processing, completed, error)
        
    Returns:
        List of datasets with metadata
    """
    config_manager = request.app.state.config
    gavd_service = GAVDService(config_manager)
    
    try:
        datasets = gavd_service.list_datasets(
            limit=limit,
            offset=offset,
            status_filter=status_filter
        )
        
        return {
            "success": True,
            "count": len(datasets),
            "limit": limit,
            "offset": offset,
            "status_filter": status_filter,
            "datasets": datasets
        }
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.delete("/{dataset_id}")
async def delete_dataset(
    request: Request,
    dataset_id: str
) -> Dict[str, Any]:
    """
    Delete a GAVD dataset and ALL its associated data.
    
    This completely eradicates:
    - Original CSV file
    - Processing results
    - Pose data
    - Downloaded YouTube videos
    - All metadata
    
    Args:
        dataset_id: Dataset ID to delete
        
    Returns:
        Deletion confirmation with details
    """
    config_manager = request.app.state.config
    gavd_service = GAVDService(config_manager)
    
    try:
        logger.info(f"Deleting dataset {dataset_id} and all associated data...")
        result = gavd_service.delete_dataset(dataset_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        logger.info(f"Successfully deleted dataset {dataset_id}")
        return {
            "success": True,
            "dataset_id": dataset_id,
            "message": "Dataset and all associated data deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@router.get("/sequences/{dataset_id}")
async def get_dataset_sequences(
    request: Request,
    dataset_id: str,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get sequences from a processed GAVD dataset.
    
    Args:
        dataset_id: Dataset ID
        limit: Maximum number of sequences to return
        offset: Offset for pagination
        
    Returns:
        List of sequences with pose data
    """
    config_manager = request.app.state.config
    gavd_service = GAVDService(config_manager)
    
    try:
        sequences = gavd_service.get_dataset_sequences(
            dataset_id,
            limit=limit,
            offset=offset
        )
        
        if sequences is None:
            raise HTTPException(status_code=404, detail="Dataset not found or not processed")
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "count": len(sequences),
            "limit": limit,
            "offset": offset,
            "sequences": sequences
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving dataset sequences: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sequences: {str(e)}")


@router.get("/sequence/{dataset_id}/{sequence_id}/frames")
async def get_sequence_frames(
    request: Request,
    dataset_id: str,
    sequence_id: str
) -> Dict[str, Any]:
    """
    Get all frames for a specific sequence with metadata.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        
    Returns:
        List of frames with bbox, vid_info, and other metadata
    """
    config_manager = request.app.state.config
    gavd_service = GAVDService(config_manager)
    
    try:
        frames = gavd_service.get_sequence_frames(dataset_id, sequence_id)
        
        if frames is None:
            raise HTTPException(status_code=404, detail="Sequence not found")
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "sequence_id": sequence_id,
            "frame_count": len(frames),
            "frames": frames
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving sequence frames: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve frames: {str(e)}")


@router.get("/sequence/{dataset_id}/{sequence_id}/frame/{frame_num}/pose")
async def get_frame_pose_data(
    request: Request,
    dataset_id: str,
    sequence_id: str,
    frame_num: int
) -> Dict[str, Any]:
    """
    Get pose keypoints for a specific frame.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        frame_num: Frame number
        
    Returns:
        Pose keypoints data with source video dimensions for proper scaling
    """
    config_manager = request.app.state.config
    gavd_service = GAVDService(config_manager)
    
    try:
        pose_data = gavd_service.get_frame_pose_data(dataset_id, sequence_id, frame_num)
        
        if pose_data is None:
            raise HTTPException(status_code=404, detail="Pose data not found")
        
        # Handle both old format (list) and new format (dict with metadata)
        if isinstance(pose_data, dict):
            keypoints = pose_data.get('keypoints', [])
            source_width = pose_data.get('source_width')
            source_height = pose_data.get('source_height')
        else:
            # Old format - list of keypoints
            keypoints = pose_data
            source_width = None
            source_height = None
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "sequence_id": sequence_id,
            "frame_num": frame_num,
            "pose_keypoints": keypoints,
            "source_video_width": source_width,
            "source_video_height": source_height
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving pose data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pose data: {str(e)}")


@router.get("/sequence/{dataset_id}/{sequence_id}/frame/{frame_num}/image")
async def get_frame_image(
    request: Request,
    dataset_id: str,
    sequence_id: str,
    frame_num: int,
    show_bbox: bool = True,
    show_pose: bool = False
) -> Dict[str, Any]:
    """
    Get a specific frame image with optional bbox and pose overlay.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        frame_num: Frame number
        show_bbox: Whether to draw bounding box
        show_pose: Whether to draw pose keypoints
        
    Returns:
        Base64 encoded image with overlays
    """
    config_manager = request.app.state.config
    gavd_service = GAVDService(config_manager)
    
    try:
        image_data = gavd_service.get_frame_image(
            dataset_id,
            sequence_id,
            frame_num,
            show_bbox=show_bbox,
            show_pose=show_pose
        )
        
        if image_data is None:
            raise HTTPException(status_code=404, detail="Frame not found")
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "sequence_id": sequence_id,
            "frame_num": frame_num,
            "image": image_data["image"],
            "width": image_data["width"],
            "height": image_data["height"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving frame image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve frame image: {str(e)}")
