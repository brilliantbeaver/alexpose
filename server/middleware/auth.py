"""
Authentication middleware for AlexPose FastAPI application.

Provides JWT-based authentication and authorization for API endpoints.
"""

from typing import Optional, Callable
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
from jose import jwt, JWTError
import os
from datetime import datetime, timedelta


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for JWT-based authentication.
    
    Validates JWT tokens and adds user information to request state.
    """
    
    def __init__(self, app, secret_key: Optional[str] = None, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
        self.exclude_paths = exclude_paths or [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register"
        ]
        self.algorithm = "HS256"
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request and validate authentication.
        """
        # Skip authentication for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            # Allow unauthenticated access for now (development mode)
            request.state.user = None
            return await call_next(request)
        
        try:
            # Parse Bearer token
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme"
                )
            
            # Decode and validate token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Add user information to request state
            request.state.user = {
                "user_id": payload.get("sub"),
                "email": payload.get("email"),
                "roles": payload.get("roles", [])
            }
            
            logger.debug(f"Authenticated user: {request.state.user['email']}")
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except JWTError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
        
        response = await call_next(request)
        return response


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    secret_key = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    secret_key = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
