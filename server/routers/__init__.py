"""
API routers for AlexPose FastAPI application.

This package contains all API endpoint routers organized by functionality.
"""

from .health import router as health_router
from .upload import router as upload_router
from .analysis import router as analysis_router
from .gavd import router as gavd_router
from .video import router as video_router
from .models import router as models_router
from . import pose_analysis

__all__ = ["health_router", "upload_router", "analysis_router", "gavd_router", "video_router", "models_router", "pose_analysis"]
