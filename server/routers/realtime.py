"""
Realtime gait analysis API endpoints.

This module provides WebSocket and HTTP endpoints for realtime pose estimation
and gait analysis using webcam streams.
"""

import asyncio
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from server.services.realtime_service import RealtimeService
from ambient.realtime.interfaces import ProcessingMode


router = APIRouter(prefix="/api/realtime", tags=["realtime"])

# Global service instance (in production, use dependency injection)
_realtime_service: Optional[RealtimeService] = None


def get_realtime_service() -> RealtimeService:
    """Get or create realtime service instance."""
    global _realtime_service
    if _realtime_service is None:
        _realtime_service = RealtimeService()
    return _realtime_service


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for realtime pose estimation stream.
    
    This endpoint handles bidirectional communication for:
    - Receiving webcam frames from frontend
    - Sending pose estimation results back to frontend
    - Configuration updates
    - Statistics and metrics
    """
    await websocket.accept()
    service = get_realtime_service()
    session_id = None
    
    logger.info("WebSocket connection established for realtime stream")
    
    try:
        # Start processing session
        session_id = service.start_session()
        
        # Send initial configuration
        await websocket.send_json({
            "type": "session_started",
            "session_id": session_id,
            "config": service.get_current_config()
        })
        
        # Main processing loop
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_json()
                
                # Handle different message types
                if message.get("type") == "frame":
                    # Process frame data
                    frame_data = message.get("data")
                    if frame_data:
                        result = await service.handle_frame(frame_data)
                        await websocket.send_json({
                            "type": "pose_result",
                            "data": result
                        })
                
                elif message.get("type") == "config_update":
                    # Update configuration
                    config = message.get("config", {})
                    service.configure_analysis(config)
                    await websocket.send_json({
                        "type": "config_updated",
                        "config": service.get_current_config()
                    })
                
                elif message.get("type") == "get_stats":
                    # Send current statistics
                    stats = await service.get_current_metrics()
                    await websocket.send_json({
                        "type": "statistics",
                        "data": stats
                    })
                
                elif message.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    })
                
                else:
                    logger.warning(f"Unknown message type: {message.get('type')}")
            
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await websocket.send_json({"type": "keepalive"})
                except:
                    # WebSocket might be closing
                    break
                continue
            
            except WebSocketDisconnect:
                # Client disconnected
                break
            
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                try:
                    # Only send error if WebSocket is still open
                    if websocket.client_state.name == "CONNECTED":
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                except:
                    # WebSocket is closing, break the loop
                    break
    
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Don't try to send error message if WebSocket is closed
    
    finally:
        # Clean up session
        if session_id:
            try:
                summary = service.end_session(session_id)
                logger.info(f"Session {session_id} ended: {summary}")
            except Exception as e:
                logger.error(f"Error ending session {session_id}: {e}")


@router.get("/config")
async def get_config() -> JSONResponse:
    """
    Get current realtime analysis configuration.
    
    Returns:
        Current configuration settings
    """
    try:
        service = get_realtime_service()
        config = service.get_current_config()
        
        return JSONResponse(content={
            "success": True,
            "config": config
        })
    
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config")
async def update_config(config: Dict[str, Any]) -> JSONResponse:
    """
    Update realtime analysis configuration.
    
    Args:
        config: Configuration parameters to update
        
    Returns:
        Updated configuration
    """
    try:
        service = get_realtime_service()
        service.configure_analysis(config)
        
        updated_config = service.get_current_config()
        
        return JSONResponse(content={
            "success": True,
            "config": updated_config
        })
    
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_statistics() -> JSONResponse:
    """
    Get current processing statistics.
    
    Returns:
        Processing statistics and metrics
    """
    try:
        service = get_realtime_service()
        stats = await service.get_current_metrics()
        
        return JSONResponse(content={
            "success": True,
            "statistics": stats
        })
    
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint for realtime service.
    
    Returns:
        Service health status
    """
    try:
        service = get_realtime_service()
        
        # Check if service is ready
        is_ready = service.is_ready()
        
        # Get basic status
        status = {
            "service": "realtime",
            "status": "healthy" if is_ready else "not_ready",
            "ready": is_ready,
            "timestamp": service.get_current_timestamp()
        }
        
        return JSONResponse(content={
            "success": True,
            "health": status
        })
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "health": {
                    "service": "realtime",
                    "status": "error",
                    "error": str(e)
                }
            }
        )


@router.post("/test-frame")
async def test_frame_processing(frame_data: Dict[str, Any]) -> JSONResponse:
    """
    Test endpoint for processing a single frame.
    
    This endpoint is useful for testing pose estimation without WebSocket.
    
    Args:
        frame_data: Frame data for processing
        
    Returns:
        Pose estimation result
    """
    try:
        service = get_realtime_service()
        
        # Start temporary session if needed
        if not service.has_active_session():
            session_id = service.start_session()
        
        # Process frame
        result = await service.handle_frame(frame_data.get("data", ""))
        
        return JSONResponse(content={
            "success": True,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Test frame processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing-modes")
async def get_processing_modes() -> JSONResponse:
    """
    Get available processing modes.
    
    Returns:
        List of available processing modes with descriptions
    """
    try:
        modes = [
            {
                "value": ProcessingMode.FAST.value,
                "label": "Fast",
                "description": "Optimized for speed, lower accuracy",
                "target_fps": 30,
                "cpu_usage": "Low"
            },
            {
                "value": ProcessingMode.BALANCED.value,
                "label": "Balanced",
                "description": "Balance between speed and accuracy",
                "target_fps": 25,
                "cpu_usage": "Medium"
            },
            {
                "value": ProcessingMode.ACCURATE.value,
                "label": "Accurate",
                "description": "Highest accuracy, slower processing",
                "target_fps": 20,
                "cpu_usage": "High"
            }
        ]
        
        return JSONResponse(content={
            "success": True,
            "modes": modes
        })
    
    except Exception as e:
        logger.error(f"Failed to get processing modes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_model_info() -> JSONResponse:
    """
    Get information about the pose estimation model.
    
    Returns:
        Model information and capabilities
    """
    try:
        service = get_realtime_service()
        model_info = service.get_model_info()
        
        return JSONResponse(content={
            "success": True,
            "model": model_info
        })
    
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))