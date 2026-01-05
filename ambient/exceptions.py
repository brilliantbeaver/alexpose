"""
Custom exceptions for the Ambient gait analysis system.

This module defines custom exceptions that provide meaningful error messages
and help with debugging and error handling throughout the system.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

from typing import Optional


class AmbientError(Exception):
    """Base exception for all Ambient system errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.details = details


class ConfigurationError(AmbientError):
    """Raised when there are configuration issues."""

    pass


class FileError(AmbientError):
    """Raised when there are file-related issues."""

    pass


class GeminiError(AmbientError):
    """Raised when there are Gemini API issues."""

    pass


class AnalysisError(AmbientError):
    """Raised when there are analysis-related issues."""

    pass


class ValidationError(AmbientError):
    """Raised when input validation fails."""

    pass


class OutputError(AmbientError):
    """Raised when there are output-related issues."""

    pass
