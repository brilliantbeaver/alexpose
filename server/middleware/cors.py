"""
CORS middleware configuration for AlexPose FastAPI application.

Provides flexible CORS configuration based on environment.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os


def setup_cors(
    app: FastAPI,
    origins: Optional[List[str]] = None,
    allow_credentials: bool = True,
    allow_methods: Optional[List[str]] = None,
    allow_headers: Optional[List[str]] = None
):
    """
    Configure CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        origins: List of allowed origins (defaults to environment-based)
        allow_credentials: Whether to allow credentials
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed headers
    """
    # Default origins based on environment
    if origins is None:
        environment = os.getenv("ENVIRONMENT", "development")
        
        if environment == "production":
            # Production origins (should be configured via environment variables)
            origins = os.getenv("CORS_ORIGINS", "").split(",")
            if not origins or origins == [""]:
                origins = ["https://alexpose.com", "https://www.alexpose.com"]
        else:
            # Development origins
            origins = [
                "http://localhost:3000",  # NextJS default
                "http://localhost:8080",  # Alternative frontend
                "http://localhost:5173",  # Vite default
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080",
                "http://127.0.0.1:5173",
            ]
    
    # Default methods
    if allow_methods is None:
        allow_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    
    # Default headers
    if allow_headers is None:
        allow_headers = [
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            "X-API-Key",
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )
