"""
Configuration management for the Ambient gait analysis system.

This module provides configuration management functionality including environment
variable handling, YAML file loading, validation, and directory path management.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from .interfaces import IConfigurationManager


# Define ConfigurationError locally to avoid circular imports
class ConfigurationError(Exception):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


def _get_nested_value(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get nested value using dot notation.

    Args:
        data: Dictionary to search in
        key: Key with optional dot notation (e.g., "model.temperature")
        default: Default value if key not found

    Returns:
        Value from nested dictionary or default
    """
    keys = key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def _convert_value(value: Any, key: str, target_type: Optional[type] = None) -> Any:
    """
    Convert value to appropriate type based on key name or explicit type.

    Args:
        value: Value to convert
        key: Configuration key name (used for type inference)
        target_type: Optional explicit type to convert to

    Returns:
        Converted value
    """
    if value is None:
        return None

    # Use explicit type if provided
    if target_type:
        try:
            return target_type(value)
        except (ValueError, TypeError):
            return None

    # Infer type from key name patterns
    if key in ["temperature", "confidence", "score", "threshold"]:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    elif key in ["content_items", "items", "list", "tags"]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, dict)):
            return value
        else:
            return [str(value)]
    elif (
        key.endswith("_dir")
        or key.endswith("_path")
        or key in ["input_dir", "output_dir", "pose_dir"]
    ):
        try:
            return Path(value) if value else None
        except (ValueError, TypeError):
            return None
    elif key in ["enabled", "configured", "valid", "active"]:
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ["true", "1", "yes", "on"]
        else:
            return bool(value)
    elif key in ["port", "timeout", "max_retries"]:
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    return value


class YAMLConfigurationManager(IConfigurationManager):
    """
    Manages configuration from YAML files with fallback to environment variables.

    This class loads configuration from YAML files while maintaining backward
    compatibility with environment variables. It follows the SOLID principles
    and provides a robust, extensible configuration system.
    """

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize the YAML configuration manager.

        Args:
            config_file: Path to the YAML configuration file
        """
        # Load environment variables as fallback
        load_dotenv(find_dotenv())

        # Initialize configuration state
        self._config_data: Dict[str, Any] = {}
        self._videos_dir: Optional[Path] = None
        self._openpose_dir: Optional[Path] = None
        self._gemini_api_key: Optional[str] = None
        self._gemini_configured: bool = False

        # Load configuration
        self._load_configuration(config_file)

    def _load_configuration(self, config_file: Optional[Path] = None) -> None:
        """Load configuration from YAML file and environment variables."""
        if config_file:
            self._load_yaml_config(config_file)

        # Always load environment variables as fallback
        self._load_environment_config()

        # Validate and set up configuration
        self._setup_configuration()

    def _load_yaml_config(self, config_file: Path) -> None:
        """Load configuration from YAML file."""
        try:
            if not config_file.exists():
                raise ConfigurationError(
                    f"Configuration file does not exist: {config_file}",
                    f"Please check that the file exists and is accessible",
                )

            with open(config_file, "r", encoding="utf-8") as f:
                self._config_data = yaml.safe_load(f)

            if not isinstance(self._config_data, dict):
                raise ConfigurationError(
                    f"Invalid YAML configuration format in {config_file}",
                    "Configuration must be a dictionary",
                )

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML configuration file: {config_file}",
                f"YAML parsing error: {str(e)}",
            )
        except FileNotFoundError:
            raise ConfigurationError(
                f"Configuration file does not exist: {config_file}",
                f"Please check that the file exists and is accessible",
            )
        except OSError as e:
            if "No such file or directory" in str(e):
                raise ConfigurationError(
                    f"Configuration file does not exist: {config_file}",
                    f"Please check that the file exists and is accessible",
                )
            else:
                raise ConfigurationError(
                    f"Failed to load configuration file: {config_file}",
                    f"Error: {str(e)}",
                )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration file: {config_file}", f"Error: {str(e)}"
            )

    def _load_environment_config(self) -> None:
        """Load configuration from environment variables as fallback."""
        # Environment variables take precedence over YAML for critical settings
        self._gemini_api_key = os.getenv("GOOGLE_API_KEY")

        # Only set videos_dir and openpose_dir from env if not in YAML
        if "videos_dir" not in self._config_data:
            videos_dir = os.getenv("VIDEOS_DIR")
            if videos_dir:
                self._config_data["videos_dir"] = videos_dir.strip()

        if "openpose_dir" not in self._config_data:
            openpose_dir = os.getenv("OPENPOSE_OUTPUTS_DIR")
            if openpose_dir:
                self._config_data["openpose_dir"] = openpose_dir.strip()

    def _setup_configuration(self) -> None:
        """Set up and validate configuration settings."""
        self._setup_directories()
        self._setup_gemini_configuration()

    def _setup_directories(self) -> None:
        """Set up and validate directory paths."""
        # Videos directory
        videos_dir = self.get_config_value("videos_dir") or self.get_config_value(
            "input_dir"
        )
        if not videos_dir:
            raise ConfigurationError(
                "videos_dir or input_dir not configured in YAML or VIDEOS_DIR environment variable",
                "Please set videos_dir or input_dir in your config file or VIDEOS_DIR in your .env file",
            )

        self._videos_dir = (
            Path(videos_dir) if isinstance(videos_dir, str) else videos_dir
        )
        if not self._videos_dir.exists():
            raise ConfigurationError(
                f"Videos directory does not exist: {self._videos_dir}",
                f"Please check that the path exists and is accessible",
            )

        # OpenPose directory
        openpose_dir = self.get_config_value("openpose_dir") or self.get_config_value(
            "pose_dir"
        )
        if not openpose_dir:
            raise ConfigurationError(
                "openpose_dir or pose_dir not configured in YAML or OPENPOSE_OUTPUTS_DIR environment variable",
                "Please set openpose_dir or pose_dir in your config file or OPENPOSE_OUTPUTS_DIR in your .env file",
            )

        self._openpose_dir = (
            Path(openpose_dir) if isinstance(openpose_dir, str) else openpose_dir
        )
        if not self._openpose_dir.exists():
            raise ConfigurationError(
                f"OpenPose directory does not exist: {self._openpose_dir}",
                f"Please check that the path exists and is accessible",
            )

    def _setup_gemini_configuration(self) -> None:
        """Set up and validate Gemini API configuration."""
        if not self._gemini_api_key:
            logger.warning(
                "GOOGLE_API_KEY environment variable is not set!"
            )
            self._gemini_configured = False
            return

        # Check if API key looks valid (basic format check)
        if not self._gemini_api_key.startswith("AI"):
            print("WARNING: GOOGLE_API_KEY format appears invalid!", flush=True)
            self._gemini_configured = False
            return

        # Test the configuration
        try:
            import google.genai as genai

            genai.configure(api_key=self._gemini_api_key)

            # Test with a simple model creation
            model = genai.GenerativeModel(model_name="gemini-2.5-pro")
            print("Gemini API key configured successfully!", flush=True)
            self._gemini_configured = True
        except Exception as e:
            print(f"WARNING: Failed to configure Gemini API: {str(e)}", flush=True)
            self._gemini_configured = False

    def get_config_value(
        self, key: str, default: Any = None, target_type: Optional[type] = None
    ) -> Any:
        """
        Get a configuration value from the YAML config with nested key support and type conversion.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            target_type: Optional explicit type to convert the value to

        Returns:
            Configuration value or default, converted to appropriate type
        """
        value = _get_nested_value(self._config_data, key, None)
        if value is not None:
            return _convert_value(value, key, target_type)
        else:
            return default

    def get_videos_directory(self) -> Path:
        """Get the videos directory path."""
        if self._videos_dir is None:
            raise ConfigurationError("Videos directory not configured")
        return self._videos_dir

    def get_openpose_directory(self) -> Path:
        """Get the OpenPose outputs directory path."""
        if self._openpose_dir is None:
            raise ConfigurationError("OpenPose directory not configured")
        return self._openpose_dir

    def get_gemini_api_key(self) -> str:
        """Get the Gemini API key."""
        if not self._gemini_api_key:
            raise ConfigurationError("Gemini API key not configured")
        return self._gemini_api_key

    def is_gemini_configured(self) -> bool:
        """Check if Gemini is properly configured."""
        return self._gemini_configured

    def validate_configuration(self) -> bool:
        """Validate all configuration settings."""
        try:
            # Check that all required directories exist
            self.get_videos_directory()
            self.get_openpose_directory()

            # Check for MP4 files in videos directory
            mp4_files = list(self._videos_dir.glob("*.mp4"))
            if not mp4_files:
                print(f"WARNING: No MP4 files found in: {self._videos_dir}", flush=True)
            else:
                print(
                    f"Found {len(mp4_files)} MP4 file(s) in: {self._videos_dir}",
                    flush=True,
                )

            print(f"Using OpenPose path: {self._openpose_dir}", flush=True)

            return True
        except ConfigurationError as e:
            print(f"Configuration validation failed: {e.message}", flush=True)
            return False


class EnvConfigurationManager(IConfigurationManager):
    """
    Manages configuration for the Ambient gait analysis system using environment variables.

    This class handles environment variables, validates configuration settings,
    and provides access to configured paths and settings.

    Note: This class is maintained for backward compatibility. For new applications,
    consider using YAMLConfigurationManager for more flexible configuration.
    """

    def __init__(self):
        """Initialize the configuration manager."""
        # Load environment variables
        load_dotenv(find_dotenv())

        # Initialize configuration state
        self._videos_dir: Optional[Path] = None
        self._openpose_dir: Optional[Path] = None
        self._gemini_api_key: Optional[str] = None
        self._gemini_configured: bool = False

        # Load and validate configuration
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load and validate all configuration settings."""
        self._load_videos_directory()
        self._load_openpose_directory()
        self._load_gemini_configuration()

    def _load_videos_directory(self) -> None:
        """Load and validate the videos directory configuration."""
        videos_dir = self.get_config_value("videos_dir") or self.get_config_value(
            "input_dir"
        )
        if not videos_dir:
            raise ConfigurationError(
                "VIDEOS_DIR environment variable is not set!",
                "Please set VIDEOS_DIR in your .env file",
            )

        # Clean up any corrupted paths
        if isinstance(videos_dir, str):
            videos_dir = videos_dir.strip()
            self._videos_dir = Path(videos_dir)
        else:
            self._videos_dir = videos_dir

        if not self._videos_dir.exists():
            raise ConfigurationError(
                f"Videos directory does not exist: {self._videos_dir}",
                f"Please check that the path exists and is accessible",
            )

    def _load_openpose_directory(self) -> None:
        """Load and validate the OpenPose directory configuration."""
        openpose_dir = self.get_config_value("openpose_dir") or self.get_config_value(
            "pose_dir"
        )
        if not openpose_dir:
            raise ConfigurationError(
                "OPENPOSE_OUTPUTS_DIR environment variable is not set!",
                "Please set OPENPOSE_OUTPUTS_DIR in your .env file",
            )

        # Clean up any corrupted paths
        if isinstance(openpose_dir, str):
            openpose_dir = openpose_dir.strip()
            self._openpose_dir = Path(openpose_dir)
        else:
            self._openpose_dir = openpose_dir

        if not self._openpose_dir.exists():
            raise ConfigurationError(
                f"OpenPose directory does not exist: {self._openpose_dir}",
                f"Please check that the path exists and is accessible",
            )

    def _load_gemini_configuration(self) -> None:
        """Load and validate the Gemini API configuration."""
        self._gemini_api_key = os.getenv("GOOGLE_API_KEY")

        if not self._gemini_api_key:
            print(
                "WARNING: GOOGLE_API_KEY environment variable is not set!", flush=True
            )
            self._gemini_configured = False
            return

        # Check if API key looks valid (basic format check)
        if not self._gemini_api_key.startswith("AI"):
            print("WARNING: GOOGLE_API_KEY format appears invalid!", flush=True)
            self._gemini_configured = False
            return

        # Test the configuration
        try:
            import google.genai as genai

            genai.configure(api_key=self._gemini_api_key)

            # Test with a simple model creation
            model = genai.GenerativeModel(model_name="gemini-2.5-pro")
            print("Gemini API key configured successfully!", flush=True)
            self._gemini_configured = True
        except Exception as e:
            print(f"WARNING: Failed to configure Gemini API: {str(e)}", flush=True)
            self._gemini_configured = False

    def get_videos_directory(self) -> Path:
        """Get the videos directory path."""
        if self._videos_dir is None:
            raise ConfigurationError("Videos directory not configured")
        return self._videos_dir

    def get_openpose_directory(self) -> Path:
        """Get the OpenPose outputs directory path."""
        if self._openpose_dir is None:
            raise ConfigurationError("OpenPose directory not configured")
        return self._openpose_dir

    def get_gemini_api_key(self) -> str:
        """Get the Gemini API key."""
        if not self._gemini_api_key:
            raise ConfigurationError("Gemini API key not configured")
        return self._gemini_api_key

    def is_gemini_configured(self) -> bool:
        """Check if Gemini is properly configured."""
        return self._gemini_configured

    def get_config_value(
        self, key: str, default: Any = None, target_type: Optional[type] = None
    ) -> Any:
        """
        Get a configuration value from environment variables with type conversion.

        Args:
            key: Configuration key
            default: Default value if key not found
            target_type: Optional explicit type to convert the value to

        Returns:
            Configuration value or default, converted to appropriate type
        """
        # Map common configuration keys to environment variables
        env_mapping = {
            "videos_dir": "VIDEOS_DIR",
            "input_dir": "VIDEOS_DIR",
            "openpose_dir": "OPENPOSE_OUTPUTS_DIR",
            "pose_dir": "OPENPOSE_OUTPUTS_DIR",
            "gemini_api_key": "GOOGLE_API_KEY",
            "output_dir": "OUTPUT_DIR",
            "vlm_model": "VLM_MODEL",
            "temperature": "TEMPERATURE",
            "prompt": "PROMPT",
            "content_items": "CONTENT_ITEMS",
            "gemini_cache_config": "GEMINI_CACHE_CONFIG",
        }

        # Get the environment variable name for this key
        env_var = env_mapping.get(key, key.upper())

        # Try to get from environment variables
        value = os.getenv(env_var)

        # Convert the value to appropriate type
        if value is not None:
            return _convert_value(value, key, target_type)
        else:
            return default

    def validate_configuration(self) -> bool:
        """Validate all configuration settings."""
        try:
            # Check that all required directories exist
            self.get_videos_directory()
            self.get_openpose_directory()

            # Check for MP4 files in videos directory
            mp4_files = list(self._videos_dir.glob("*.mp4"))
            if not mp4_files:
                print(f"WARNING: No MP4 files found in: {self._videos_dir}", flush=True)
            else:
                print(
                    f"Found {len(mp4_files)} MP4 file(s) in: {self._videos_dir}",
                    flush=True,
                )

            print(f"Using OpenPose path: {self._openpose_dir}", flush=True)

            return True
        except ConfigurationError as e:
            print(f"Configuration validation failed: {e.message}", flush=True)
            return False
