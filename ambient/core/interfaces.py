"""
Interface definitions for the Ambient gait analysis system.

This module defines the abstract interfaces that all concrete implementations
must follow, ensuring consistency and enabling dependency injection.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class IConfigurationManager(ABC):
    """Interface for configuration management."""

    @abstractmethod
    def get_config_value(
        self, key: str, default: Any = None, target_type: Optional[type] = None
    ) -> Any:
        """
        Get a configuration value by key with support for nested keys and type conversion.

        Args:
            key: Configuration key to retrieve (supports dot notation for nested keys)
            default: Default value if key not found
            target_type: Optional explicit type to convert the value to

        Returns:
            Configuration value or default, converted to appropriate type
        """
        pass

    @abstractmethod
    def get_videos_directory(self) -> Path:
        """Get the videos directory path."""
        pass

    @abstractmethod
    def get_openpose_directory(self) -> Path:
        """Get the OpenPose outputs directory path."""
        pass

    @abstractmethod
    def get_gemini_api_key(self) -> str:
        """Get the Gemini API key."""
        pass

    @abstractmethod
    def is_gemini_configured(self) -> bool:
        """Check if Gemini is properly configured."""
        pass

    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate all configuration settings."""
        pass


class IFileManager(ABC):
    """Interface for file management operations."""

    @abstractmethod
    def upload_video(self, video_path: str) -> Optional[Any]:
        """Upload a video file and return a reference."""
        pass

    @abstractmethod
    def upload_csv(self, csv_path: str) -> Optional[Any]:
        """Upload a CSV file and return a reference."""
        pass

    @abstractmethod
    def get_cached_reference(self, file_path: str) -> Optional[Any]:
        """Get a cached file reference if available."""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the file cache."""
        pass


class IAnalyzer(ABC):
    """Interface for analysis operations."""

    @abstractmethod
    def analyze_video(self, video_path: str, csv_paths: List[str]) -> tuple:
        """
        Analyze a video with associated CSV files.

        Returns:
            Tuple of (raw_response, generated_text) where:
            - raw_response: The complete response object from the AI model
            - generated_text: The extracted analysis text
        """
        pass

    @abstractmethod
    def get_analysis_prompt(self) -> str:
        """Get the analysis prompt template."""
        pass


class IOutputManager(ABC):
    """Interface for output management."""

    @abstractmethod
    def save_raw_response(self, record_id: str, response: Any) -> Path:
        """Save the raw response to a file."""
        pass

    @abstractmethod
    def save_analysis_text(self, record_id: str, text: str) -> Path:
        """Save the analysis text to a file."""
        pass

    @abstractmethod
    def get_output_directories(self) -> Dict[str, Path]:
        """Get the output directory paths."""
        pass
