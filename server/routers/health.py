"""
Health check and status endpoints for AlexPose FastAPI application.

Provides endpoints for monitoring system health, readiness, and status.
"""

from fastapi import APIRouter, Request
from datetime import datetime
from typing import Dict, Any
import psutil
import sys
from pathlib import Path

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0"
    }


@router.get("/health/detailed")
async def detailed_health_check(request: Request) -> Dict[str, Any]:
    """
    Detailed health check with system metrics.
    
    Returns:
        Detailed health status including system metrics
    """
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Get configuration from app state
    config_manager = getattr(request.app.state, 'config', None)
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": disk.percent
            },
            "python_version": sys.version
        },
        "configuration": {
            "loaded": config_manager is not None,
            "environment": getattr(config_manager.config, 'environment', 'unknown') if config_manager else 'unknown'
        }
    }


@router.get("/health/ready")
async def readiness_check(request: Request) -> Dict[str, Any]:
    """
    Readiness check for load balancers and orchestrators.
    
    Checks if the application is ready to accept traffic.
    
    Returns:
        Readiness status
    """
    # Check if configuration is loaded
    config_manager = getattr(request.app.state, 'config', None)
    if not config_manager:
        return {
            "ready": False,
            "reason": "Configuration not loaded",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Check if required directories exist
    try:
        data_dir = Path(getattr(config_manager.config.storage, 'data_directory', 'data'))
        if not data_dir.exists():
            return {
                "ready": False,
                "reason": "Data directory not found",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "ready": False,
            "reason": f"Directory check failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "ready": True,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check for orchestrators.
    
    Simple check to verify the application is running.
    
    Returns:
        Liveness status
    """
    return {
        "alive": "true",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/status")
async def system_status(request: Request) -> Dict[str, Any]:
    """
    Comprehensive system status endpoint.
    
    Returns:
        Detailed system status information
    """
    config_manager = getattr(request.app.state, 'config', None)
    
    # Get directory status
    directories_status = {}
    if config_manager:
        directories = {
            "data": getattr(config_manager.config.storage, 'data_directory', 'data'),
            "videos": getattr(config_manager.config.storage, 'videos_directory', 'data/videos'),
            "youtube": getattr(config_manager.config.storage, 'youtube_directory', 'data/youtube'),
            "analysis": getattr(config_manager.config.storage, 'analysis_directory', 'data/analysis'),
            "models": getattr(config_manager.config.storage, 'models_directory', 'data/models'),
            "logs": getattr(config_manager.config.storage, 'logs_directory', 'logs')
        }
        
        for name, path in directories.items():
            dir_path = Path(path)
            directories_status[name] = {
                "exists": dir_path.exists(),
                "path": str(dir_path),
                "writable": dir_path.exists() and os.access(dir_path, os.W_OK)
            }
    
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "configuration": {
            "loaded": config_manager is not None
        },
        "directories": directories_status,
        "uptime_seconds": psutil.Process().create_time()
    }


import os
