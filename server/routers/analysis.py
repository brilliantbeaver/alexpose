"""
Analysis endpoints for AlexPose FastAPI application.

Provides endpoints for triggering gait analysis, monitoring progress,
and retrieving results.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from loguru import logger

from server.services.analysis_service import AnalysisService

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


class AnalysisRequest(BaseModel):
    """Request model for analysis."""
    file_id: str
    pose_estimator: Optional[str] = "mediapipe"
    frame_rate: Optional[float] = 30.0
    use_llm_classification: Optional[bool] = True
    llm_model: Optional[str] = "gpt-4o-mini"
    options: Optional[Dict[str, Any]] = {}


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    file_ids: list[str]
    pose_estimator: Optional[str] = "mediapipe"
    frame_rate: Optional[float] = 30.0
    use_llm_classification: Optional[bool] = True
    llm_model: Optional[str] = "gpt-4o-mini"
    options: Optional[Dict[str, Any]] = {}


@router.post("/start")
async def start_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    analysis_request: AnalysisRequest
) -> Dict[str, Any]:
    """
    Start gait analysis for an uploaded video.
    
    Args:
        analysis_request: Analysis configuration
        
    Returns:
        Analysis ID and status
    """
    logger.info(f"Starting analysis for file: {analysis_request.file_id}")
    
    # Get configuration
    config_manager = request.app.state.config
    
    # Create analysis service
    analysis_service = AnalysisService(config_manager)
    
    try:
        # Create analysis job
        analysis_id = analysis_service.create_analysis_job(
            file_id=analysis_request.file_id,
            pose_estimator=analysis_request.pose_estimator,
            frame_rate=analysis_request.frame_rate,
            use_llm_classification=analysis_request.use_llm_classification,
            llm_model=analysis_request.llm_model,
            options=analysis_request.options
        )
        
        # Schedule background analysis
        background_tasks.add_task(
            analysis_service.run_analysis,
            analysis_id
        )
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "status": "pending",
            "message": "Analysis started. Use the status endpoint to monitor progress."
        }
        
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")


@router.post("/batch")
async def start_batch_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_request: BatchAnalysisRequest
) -> Dict[str, Any]:
    """
    Start batch analysis for multiple videos.
    
    Args:
        batch_request: Batch analysis configuration
        
    Returns:
        Batch ID and analysis IDs
    """
    logger.info(f"Starting batch analysis for {len(batch_request.file_ids)} files")
    
    # Get configuration
    config_manager = request.app.state.config
    
    # Create analysis service
    analysis_service = AnalysisService(config_manager)
    
    try:
        # Create batch job
        batch_id, analysis_ids = analysis_service.create_batch_analysis(
            file_ids=batch_request.file_ids,
            pose_estimator=batch_request.pose_estimator,
            frame_rate=batch_request.frame_rate,
            use_llm_classification=batch_request.use_llm_classification,
            llm_model=batch_request.llm_model,
            options=batch_request.options
        )
        
        # Schedule background analyses
        for analysis_id in analysis_ids:
            background_tasks.add_task(
                analysis_service.run_analysis,
                analysis_id
            )
        
        return {
            "success": True,
            "batch_id": batch_id,
            "analysis_ids": analysis_ids,
            "count": len(analysis_ids),
            "status": "pending",
            "message": "Batch analysis started. Use the batch status endpoint to monitor progress."
        }
        
    except Exception as e:
        logger.error(f"Error starting batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start batch analysis: {str(e)}")


@router.get("/status/{analysis_id}")
async def get_analysis_status(
    request: Request,
    analysis_id: str
) -> Dict[str, Any]:
    """
    Get analysis status and progress.
    
    Args:
        analysis_id: Analysis job ID
        
    Returns:
        Analysis status and progress information
    """
    config_manager = request.app.state.config
    analysis_service = AnalysisService(config_manager)
    
    try:
        status = analysis_service.get_analysis_status(analysis_id)
        if not status:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "status": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve status: {str(e)}")


@router.get("/results/{analysis_id}")
async def get_analysis_results(
    request: Request,
    analysis_id: str,
    format: Optional[str] = "json"
) -> Dict[str, Any]:
    """
    Get analysis results.
    
    Args:
        analysis_id: Analysis job ID
        format: Result format (json, csv, xml)
        
    Returns:
        Analysis results
    """
    config_manager = request.app.state.config
    analysis_service = AnalysisService(config_manager)
    
    try:
        results = analysis_service.get_analysis_results(analysis_id, format=format)
        if not results:
            raise HTTPException(status_code=404, detail="Analysis results not found")
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "format": format,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")


@router.get("/batch/status/{batch_id}")
async def get_batch_status(
    request: Request,
    batch_id: str
) -> Dict[str, Any]:
    """
    Get batch analysis status.
    
    Args:
        batch_id: Batch job ID
        
    Returns:
        Batch status with individual analysis statuses
    """
    config_manager = request.app.state.config
    analysis_service = AnalysisService(config_manager)
    
    try:
        status = analysis_service.get_batch_status(batch_id)
        if not status:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return {
            "success": True,
            "batch_id": batch_id,
            "status": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve batch status: {str(e)}")


@router.delete("/{analysis_id}")
async def delete_analysis(
    request: Request,
    analysis_id: str
) -> Dict[str, Any]:
    """
    Delete an analysis and its results.
    
    Args:
        analysis_id: Analysis job ID
        
    Returns:
        Deletion confirmation
    """
    config_manager = request.app.state.config
    analysis_service = AnalysisService(config_manager)
    
    try:
        result = analysis_service.delete_analysis(analysis_id)
        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "message": "Analysis deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")


@router.get("/list")
async def list_analyses(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all analyses with pagination and filtering.
    
    Args:
        limit: Maximum number of results
        offset: Offset for pagination
        status_filter: Filter by status (pending, running, completed, failed)
        
    Returns:
        List of analyses
    """
    config_manager = request.app.state.config
    analysis_service = AnalysisService(config_manager)
    
    try:
        analyses = analysis_service.list_analyses(
            limit=limit,
            offset=offset,
            status_filter=status_filter
        )
        
        return {
            "success": True,
            "count": len(analyses),
            "limit": limit,
            "offset": offset,
            "status_filter": status_filter,
            "analyses": analyses
        }
    except Exception as e:
        logger.error(f"Error listing analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list analyses: {str(e)}")


@router.get("/statistics")
async def get_dashboard_statistics(
    request: Request
) -> Dict[str, Any]:
    """
    Get unified dashboard statistics for overview.
    
    Returns:
        Dashboard statistics including:
        - Total analyses count (both GAVD and full gait analyses)
        - Normal/abnormal pattern counts
        - Average confidence
        - Recent analyses from both sources
        - GAVD dataset statistics
    """
    config_manager = request.app.state.config
    analysis_service = AnalysisService(config_manager)
    
    try:
        # Get full gait analysis statistics
        gait_stats = analysis_service.get_dashboard_statistics()
        
        # Get GAVD dataset statistics
        from server.services.gavd_service import GAVDService
        gavd_service = GAVDService(config_manager)
        gavd_datasets = gavd_service.list_datasets(limit=10, offset=0)
        
        # Count GAVD datasets by status
        gavd_status_counts = {
            "uploaded": 0,
            "processing": 0,
            "completed": 0,
            "error": 0
        }
        
        gavd_recent = []
        total_gavd_sequences = 0
        total_gavd_frames = 0
        
        for dataset in gavd_datasets:
            status = dataset.get("status", "unknown")
            if status in gavd_status_counts:
                gavd_status_counts[status] += 1
            
            # Add to recent analyses list
            gavd_recent.append({
                "type": "gavd_dataset",
                "dataset_id": dataset.get("dataset_id", ""),
                "filename": dataset.get("original_filename", "Unknown"),
                "status": status,
                "uploaded_at": dataset.get("uploaded_at", ""),
                "completed_at": dataset.get("processing_completed_at"),
                "sequence_count": dataset.get("sequence_count", 0),
                "row_count": dataset.get("row_count", 0),
                "total_sequences_processed": dataset.get("total_sequences_processed", 0),
                "total_frames_processed": dataset.get("total_frames_processed", 0),
                "progress": dataset.get("progress", "")
            })
            
            # Accumulate totals
            if status == "completed":
                total_gavd_sequences += dataset.get("total_sequences_processed", 0)
                total_gavd_frames += dataset.get("total_frames_processed", 0)
        
        # Combine gait analysis recent items with GAVD datasets
        combined_recent = []
        
        # Add GAVD datasets
        for gavd_item in gavd_recent:
            combined_recent.append(gavd_item)
        
        # Add full gait analyses
        for gait_item in gait_stats.get("recent_analyses", []):
            combined_recent.append({
                "type": "gait_analysis",
                **gait_item
            })
        
        # Sort by date (most recent first)
        combined_recent.sort(
            key=lambda x: x.get("completed_at") or x.get("uploaded_at") or x.get("created_at") or "",
            reverse=True
        )
        
        # Limit to 10 most recent
        combined_recent = combined_recent[:10]
        
        # Calculate combined statistics
        total_analyses = gait_stats["total_analyses"] + len(gavd_datasets)
        
        combined_stats = {
            # Overall counts
            "total_analyses": total_analyses,
            "total_gait_analyses": gait_stats["total_analyses"],
            "total_gavd_datasets": len(gavd_datasets),
            
            # Gait analysis specific
            "normal_patterns": gait_stats["normal_patterns"],
            "abnormal_patterns": gait_stats["abnormal_patterns"],
            "normal_percentage": gait_stats["normal_percentage"],
            "abnormal_percentage": gait_stats["abnormal_percentage"],
            "avg_confidence": gait_stats["avg_confidence"],
            
            # GAVD specific
            "gavd_completed": gavd_status_counts["completed"],
            "gavd_processing": gavd_status_counts["processing"],
            "gavd_uploaded": gavd_status_counts["uploaded"],
            "gavd_error": gavd_status_counts["error"],
            "total_gavd_sequences": total_gavd_sequences,
            "total_gavd_frames": total_gavd_frames,
            
            # Combined recent items
            "recent_analyses": combined_recent,
            
            # Status breakdown
            "status_breakdown": {
                "gait_analysis": gait_stats["status_breakdown"],
                "gavd_datasets": gavd_status_counts
            },
            
            "completed_count": gait_stats["completed_count"] + gavd_status_counts["completed"]
        }
        
        return {
            "success": True,
            "statistics": combined_stats
        }
    except Exception as e:
        logger.error(f"Error retrieving dashboard statistics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")
