"""
Logging middleware for AlexPose FastAPI application.

Provides request/response logging with structured data and performance metrics.
"""

import time
import uuid
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    
    Logs request details, response status, and processing time with structured data.
    """
    
    def __init__(self, app, log_request_body: bool = False, log_response_body: bool = False):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request and log details.
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_host": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        # Log request body if enabled (be careful with sensitive data)
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                logger.debug(
                    "Request body",
                    extra={
                        "request_id": request_id,
                        "body_size": len(body),
                        # Don't log actual body content by default for security
                    }
                )
            except Exception as e:
                logger.warning(f"Could not read request body: {str(e)}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time * 1000, 2),
                }
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "process_time_ms": round(process_time * 1000, 2),
                }
            )
            raise
