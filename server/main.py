"""
AlexPose FastAPI Server Entry Point

This module provides the main FastAPI application entry point for the AlexPose
Gait Analysis System. It sets up the server with all necessary middleware,
routers, and configuration.

Usage:
    uvicorn server.main:app --reload
    or
    uv run server/main.py

Author: AlexPose Team
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Load environment variables from .env file
try:
    from dotenv import find_dotenv, load_dotenv
    
    # When running from server directory, .env is in parent directory
    current_dir = Path(__file__).parent
    if current_dir.name == "server":
        env_file = current_dir.parent / ".env"
    else:
        env_file = current_dir / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        # Fallback to find_dotenv for automatic discovery
        load_dotenv(find_dotenv())
        logger.info("Loaded environment variables using automatic discovery")
        
except ImportError:
    logger.warning("python-dotenv not available, environment variables from .env file will not be loaded")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration and logging
from ambient.core.config import ConfigurationManager
from ambient.utils.logging import (
    setup_logging, 
    create_component_logger,
    log_system_event,
    log_error_with_context,
    configure_production_logging,
    configure_development_logging
)

# Import middleware and routers
from server.middleware.auth import AuthMiddleware
from server.middleware.logging import LoggingMiddleware
from server.middleware.cors import setup_cors
from server.routers import health_router, upload_router, analysis_router, gavd_router, video_router, models_router, pose_analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Create server logger
    server_logger = create_component_logger("server")
    
    # Startup
    server_logger.info("Starting AlexPose FastAPI server", operation="startup")
    
    # Initialize configuration with correct path
    # When running from server directory, config is in parent directory
    current_dir = Path.cwd()
    if current_dir.name == "server":
        config_dir = current_dir.parent / "config"
    else:
        config_dir = current_dir / "config"
    
    config_manager = ConfigurationManager(config_dir=config_dir)
    app.state.config = config_manager
    server_logger.info("Configuration manager initialized", operation="startup",
                      config_loaded=True, config_dir=str(config_dir))
    
    # Setup logging based on configuration
    try:
        setup_logging(
            log_level=getattr(config_manager.config.logging, 'level', 'INFO'),
            log_dir=str(getattr(config_manager.config.storage, 'logs_directory', 'logs')),
            rotation=getattr(config_manager.config.logging, 'rotation', '1 day'),
            retention=getattr(config_manager.config.logging, 'retention', '30 days'),
            compression=getattr(config_manager.config.logging, 'compression', 'zip'),
            format_type=getattr(config_manager.config.logging, 'format', 'structured')
        )
        log_system_event("logging_configured", "Structured logging system configured", 
                        "server", {"format": "structured"})
    except AttributeError:
        # Fallback to default logging if config attributes don't exist
        setup_logging(log_level="INFO", log_dir="logs", format_type="structured")
        server_logger.info("Using default logging configuration", operation="startup",
                          reason="config_attributes_missing")
    
    # Validate configuration
    if not config_manager.validate_configuration():
        server_logger.warning("Configuration validation failed, but continuing", 
                             operation="startup", validation_passed=False)
    else:
        server_logger.info("Configuration validation passed", operation="startup",
                          validation_passed=True)
    
    # Create necessary directories
    directories = [
        getattr(config_manager.config.storage, 'data_directory', 'data'),
        getattr(config_manager.config.storage, 'videos_directory', 'data/videos'),
        getattr(config_manager.config.storage, 'youtube_directory', 'data/youtube'),
        getattr(config_manager.config.storage, 'analysis_directory', 'data/analysis'),
        getattr(config_manager.config.storage, 'models_directory', 'data/models'),
        getattr(config_manager.config.storage, 'training_directory', 'data/training'),
        getattr(config_manager.config.storage, 'cache_directory', 'data/cache'),
        getattr(config_manager.config.storage, 'exports_directory', 'data/exports'),
        getattr(config_manager.config.storage, 'logs_directory', 'logs')
    ]
    
    created_dirs = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))
    
    server_logger.info("Data directories initialized", operation="startup",
                      directories_created=len(created_dirs), 
                      total_directories=len(directories))
    
    log_system_event("server_startup_complete", "AlexPose server startup completed",
                    "server", {"directories_created": len(created_dirs)})
    
    yield
    
    # Shutdown
    server_logger.info("Shutting down AlexPose FastAPI server", operation="shutdown")
    log_system_event("server_shutdown", "AlexPose server shutdown initiated", "server")


# Create FastAPI application
app = FastAPI(
    title="AlexPose Gait Analysis System",
    description="AI-powered gait analysis for health condition identification",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware (order matters - first added is outermost)
# 1. Trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "testserver", "*.alexpose.com", "*.herokuapp.com"]
)

# 2. CORS middleware with configuration
setup_cors(app)

# 3. Logging middleware for request/response logging
app.add_middleware(LoggingMiddleware, log_request_body=False, log_response_body=False)

# 4. Authentication middleware (optional, can be enabled later)
# app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(gavd_router)
app.include_router(video_router)
app.include_router(models_router)
app.include_router(pose_analysis.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.
    """
    log_error_with_context(
        exc, 
        "server", 
        {
            "request_url": str(request.url),
            "request_method": request.method,
            "request_headers": dict(request.headers),
            "request_id": getattr(request.state, "request_id", None)
        },
        "global_exception_handler"
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.get("/")
async def root():
    """
    Root endpoint providing basic system information.
    """
    return {
        "message": "AlexPose Gait Analysis System",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    """
    return {
        "status": "healthy",
        "timestamp": "2025-01-01T00:00:00Z",
        "version": "0.1.0"
    }


def main():
    """
    Main entry point for running the server directly.
    """
    import uvicorn
    
    # Get configuration from environment
    environment = os.getenv("ENVIRONMENT", "development")
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = environment == "development"
    
    # Configure logging based on environment
    if environment == "production":
        configure_production_logging()
    else:
        configure_development_logging()
    
    # Create server logger
    server_logger = create_component_logger("server")
    
    server_logger.info("Starting server directly", operation="direct_start",
                      host=host, port=port, environment=environment, reload=reload)
    
    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None  # Use loguru instead of uvicorn's logging
    )


if __name__ == "__main__":
    main()