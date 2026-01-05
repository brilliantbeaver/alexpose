"""
Server middleware components for AlexPose FastAPI application.

This package contains middleware for authentication, logging, CORS, and other
cross-cutting concerns.
"""

from .auth import AuthMiddleware
from .logging import LoggingMiddleware
from .cors import setup_cors

__all__ = ["AuthMiddleware", "LoggingMiddleware", "setup_cors"]
