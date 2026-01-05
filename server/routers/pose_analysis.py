"""
Pose analysis endpoints for AlexPose FastAPI application.

Provides REST API endpoints for analyzing gait sequences using pose estimation data.

Author: AlexPose Team
Date: January 4, 2026
"""

from fastapi import APIRouter, HTTPException, Request, Query
from typing import Dict, Any, Optional
from loguru import logger

from server.services.pose_analysis_service import PoseAnalysisServiceAPI

router = APIRouter(prefix="/api/v1/pose-analysis", tags=["pose-analysis"])

# Cache the service instance to avoid re-initialization on every request
_service_cache: Dict[str, PoseAnalysisServiceAPI] = {}


def _get_service(config_manager) -> PoseAnalysisServiceAPI:
    """
    Get or create a cached service instance.
    
    This prevents re-initialization of analyzers on every request,
    improving performance and reducing memory usage.
    """
    cache_key = "default"
    if cache_key not in _service_cache:
        logger.debug("Creating new PoseAnalysisServiceAPI instance")
        _service_cache[cache_key] = PoseAnalysisServiceAPI(config_manager)
    return _service_cache[cache_key]


@router.get("/sequence/{dataset_id}/{sequence_id}")
async def get_sequence_analysis(
    request: Request,
    dataset_id: str,
    sequence_id: str,
    use_cache: bool = Query(True, description="Use cached results if available"),
    force_refresh: bool = Query(False, description="Force re-analysis even if cached")
) -> Dict[str, Any]:
    """
    Get complete pose analysis for a sequence.
    
    This endpoint provides comprehensive gait analysis including:
    - Feature extraction (kinematic, joint angles, temporal, stride, symmetry, stability)
    - Gait cycle detection (heel strikes, toe-offs, phases)
    - Symmetry analysis (left-right comparison, movement correlation)
    - Summary assessment (overall scores, recommendations)
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        use_cache: Whether to use cached results (default: True)
        force_refresh: Force re-analysis even if cached (default: False)
        
    Returns:
        Complete analysis results including features, cycles, symmetry, and summary
        
    Raises:
        HTTPException 400: Invalid request parameters
        HTTPException 404: Sequence not found or no pose data
        HTTPException 500: Analysis failed
        
    Example:
        GET /api/v1/pose-analysis/sequence/abc123/seq_001
        
        Response:
        {
            "success": true,
            "dataset_id": "abc123",
            "sequence_id": "seq_001",
            "analysis": {
                "metadata": {...},
                "features": {...},
                "gait_cycles": [...],
                "timing_analysis": {...},
                "symmetry_analysis": {...},
                "summary": {...}
            }
        }
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        logger.info(f"Received analysis request for {dataset_id}/{sequence_id}")
        
        results = service.get_sequence_analysis(
            dataset_id, 
            sequence_id,
            use_cache=use_cache,
            force_refresh=force_refresh
        )
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="Sequence not found or no pose data available"
            )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "sequence_id": sequence_id,
            "analysis": results
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/sequence/{dataset_id}/{sequence_id}/status")
async def check_analysis_status(
    request: Request,
    dataset_id: str,
    sequence_id: str
) -> Dict[str, Any]:
    """
    Check if pose analysis exists for a sequence without triggering computation.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        
    Returns:
        Status information about analysis availability
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        exists = service.check_analysis_exists(dataset_id, sequence_id)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "sequence_id": sequence_id,
            "analysis_exists": exists,
            "message": "Analysis available" if exists else "Analysis not computed yet"
        }
        
    except Exception as e:
        logger.error(f"Error checking analysis status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check analysis status: {str(e)}"
        )


@router.delete("/sequence/{dataset_id}/{sequence_id}")
async def delete_analysis(
    request: Request,
    dataset_id: str,
    sequence_id: str
) -> Dict[str, Any]:
    """
    Delete pose analysis for a sequence from both database and cache.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        
    Returns:
        Deletion status
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        deleted = service.delete_analysis(dataset_id, sequence_id)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "sequence_id": sequence_id,
            "deleted": deleted,
            "message": "Analysis deleted" if deleted else "Analysis not found"
        }
        
    except Exception as e:
        logger.error(f"Error deleting analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete analysis: {str(e)}"
        )


@router.get("/stats")
async def get_analysis_stats(request: Request) -> Dict[str, Any]:
    """
    Get comprehensive analysis statistics.
    
    Returns:
        Analysis statistics including database and cache info
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        stats = service.get_analysis_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting analysis stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analysis stats: {str(e)}"
        )
async def get_sequence_analysis(
    request: Request,
    dataset_id: str,
    sequence_id: str,
    use_cache: bool = Query(True, description="Use cached results if available"),
    force_refresh: bool = Query(False, description="Force re-analysis even if cached")
) -> Dict[str, Any]:
    """
    Get complete pose analysis for a sequence.
    
    This endpoint provides comprehensive gait analysis including:
    - Feature extraction (kinematic, joint angles, temporal, stride, symmetry, stability)
    - Gait cycle detection (heel strikes, toe-offs, phases)
    - Symmetry analysis (left-right comparison, movement correlation)
    - Summary assessment (overall scores, recommendations)
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        use_cache: Whether to use cached results (default: True)
        force_refresh: Force re-analysis even if cached (default: False)
        
    Returns:
        Complete analysis results including features, cycles, symmetry, and summary
        
    Raises:
        HTTPException 400: Invalid request parameters
        HTTPException 404: Sequence not found or no pose data
        HTTPException 500: Analysis failed
        
    Example:
        GET /api/v1/pose-analysis/sequence/abc123/seq_001
        
        Response:
        {
            "success": true,
            "dataset_id": "abc123",
            "sequence_id": "seq_001",
            "analysis": {
                "metadata": {...},
                "features": {...},
                "gait_cycles": [...],
                "timing_analysis": {...},
                "symmetry_analysis": {...},
                "summary": {...}
            }
        }
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        logger.info(f"Received analysis request for {dataset_id}/{sequence_id}")
        
        results = service.get_sequence_analysis(
            dataset_id, 
            sequence_id,
            use_cache=use_cache,
            force_refresh=force_refresh
        )
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="Analysis results not found. Sequence may not have pose data."
            )
        
        # Check if results contain an error
        if "error" in results:
            if results["error"] == "no_pose_data":
                raise HTTPException(
                    status_code=404,
                    detail=results.get("message", "No pose data available for this sequence")
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=results.get("message", "Analysis failed")
                )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "sequence_id": sequence_id,
            "analysis": results
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in pose analysis endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/sequence/{dataset_id}/{sequence_id}/features")
async def get_sequence_features(
    request: Request,
    dataset_id: str,
    sequence_id: str
) -> Dict[str, Any]:
    """
    Get extracted features only (subset of full analysis).
    
    Returns only the feature extraction results without gait cycles,
    symmetry analysis, or summary assessment.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        
    Returns:
        Features dictionary including kinematic, joint angles, temporal, stride, etc.
        
    Example:
        GET /api/v1/pose-analysis/sequence/abc123/seq_001/features
        
        Response:
        {
            "success": true,
            "dataset_id": "abc123",
            "sequence_id": "seq_001",
            "features": {
                "velocity_mean": 5.2,
                "left_knee_mean": 145.3,
                ...
            }
        }
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        results = service.get_sequence_features(dataset_id, sequence_id)
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="Features not found. Sequence may not have pose data."
            )
        
        return {
            "success": True,
            **results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in features endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get features: {str(e)}"
        )


@router.get("/sequence/{dataset_id}/{sequence_id}/cycles")
async def get_sequence_cycles(
    request: Request,
    dataset_id: str,
    sequence_id: str
) -> Dict[str, Any]:
    """
    Get detected gait cycles only (subset of full analysis).
    
    Returns gait cycle detection results including heel strikes, toe-offs,
    and timing analysis.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        
    Returns:
        Gait cycles and timing analysis
        
    Example:
        GET /api/v1/pose-analysis/sequence/abc123/seq_001/cycles
        
        Response:
        {
            "success": true,
            "dataset_id": "abc123",
            "sequence_id": "seq_001",
            "gait_cycles": [
                {
                    "cycle_id": 0,
                    "start_frame": 10,
                    "end_frame": 40,
                    "duration_frames": 30,
                    "duration_seconds": 1.0,
                    "foot": "left",
                    "type": "heel_strike_cycle"
                },
                ...
            ],
            "timing_analysis": {
                "cycle_duration_mean": 1.05,
                "cadence_steps_per_minute": 114.3,
                ...
            }
        }
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        results = service.get_sequence_cycles(dataset_id, sequence_id)
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="Gait cycles not found. Sequence may not have pose data."
            )
        
        return {
            "success": True,
            **results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in cycles endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get gait cycles: {str(e)}"
        )


@router.get("/sequence/{dataset_id}/{sequence_id}/symmetry")
async def get_sequence_symmetry(
    request: Request,
    dataset_id: str,
    sequence_id: str
) -> Dict[str, Any]:
    """
    Get symmetry analysis only (subset of full analysis).
    
    Returns left-right symmetry analysis including positional, movement,
    angular, and temporal symmetry metrics.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        
    Returns:
        Symmetry analysis results
        
    Example:
        GET /api/v1/pose-analysis/sequence/abc123/seq_001/symmetry
        
        Response:
        {
            "success": true,
            "dataset_id": "abc123",
            "sequence_id": "seq_001",
            "symmetry_analysis": {
                "overall_symmetry_index": 0.15,
                "symmetry_classification": "mildly_asymmetric",
                "knee_symmetry_index": 0.12,
                "hip_symmetry_index": 0.18,
                ...
            }
        }
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        results = service.get_sequence_symmetry(dataset_id, sequence_id)
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="Symmetry analysis not found. Sequence may not have pose data."
            )
        
        return {
            "success": True,
            **results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in symmetry endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get symmetry analysis: {str(e)}"
        )


@router.delete("/cache/{dataset_id}/{sequence_id}")
async def clear_sequence_cache(
    request: Request,
    dataset_id: str,
    sequence_id: str
) -> Dict[str, Any]:
    """
    Clear cached analysis results for a specific sequence.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        
    Returns:
        Deletion confirmation
        
    Example:
        DELETE /api/v1/pose-analysis/cache/abc123/seq_001
        
        Response:
        {
            "success": true,
            "message": "Cache cleared for sequence seq_001",
            "files_deleted": 1
        }
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        deleted_count = service.clear_cache(dataset_id, sequence_id)
        
        return {
            "success": True,
            "message": f"Cache cleared for sequence {sequence_id}",
            "files_deleted": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.delete("/cache/{dataset_id}")
async def clear_dataset_cache(
    request: Request,
    dataset_id: str
) -> Dict[str, Any]:
    """
    Clear cached analysis results for all sequences in a dataset.
    
    Args:
        dataset_id: Dataset ID
        
    Returns:
        Deletion confirmation
        
    Example:
        DELETE /api/v1/pose-analysis/cache/abc123
        
        Response:
        {
            "success": true,
            "message": "Cache cleared for dataset abc123",
            "files_deleted": 5
        }
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        deleted_count = service.clear_cache(dataset_id)
        
        return {
            "success": True,
            "message": f"Cache cleared for dataset {dataset_id}",
            "files_deleted": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/cache/stats")
async def get_cache_stats(
    request: Request
) -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns information about cached analysis results including
    total files, total size, etc.
    
    Returns:
        Cache statistics
        
    Example:
        GET /api/v1/pose-analysis/cache/stats
        
        Response:
        {
            "success": true,
            "cache_directory": "/path/to/cache",
            "total_files": 10,
            "total_size_bytes": 1048576,
            "total_size_mb": 1.0
        }
    """
    config_manager = request.app.state.config
    service = _get_service(config_manager)
    
    try:
        stats = service.get_cache_stats()
        
        return {
            "success": True,
            **stats
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get cache stats: {str(e)}"
        )


@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Health check endpoint for pose analysis service.
    
    Returns:
        Health status
        
    Example:
        GET /api/v1/pose-analysis/health
        
        Response:
        {
            "success": true,
            "status": "healthy",
            "service": "pose-analysis",
            "version": "1.0.0"
        }
    """
    return {
        "success": True,
        "status": "healthy",
        "service": "pose-analysis",
        "version": "1.0.0"
    }
