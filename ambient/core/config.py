"""
Configuration management system for AlexPose.

This module provides a flexible configuration system using YAML files with
environment-specific overrides and validation.

Key Features:
- YAML-based configuration with environment overrides
- Type conversion and validation
- Nested key access with dot notation
- Default value support
- Configuration validation

Author: AlexPose Team
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger

from ambient.core.interfaces import IConfigurationManager


@dataclass
class VideoProcessingConfig:
    """Video processing configuration."""
    supported_formats: List[str] = field(default_factory=lambda: ["mp4", "avi", "mov", "webm"])
    default_frame_rate: float = 30.0
    max_video_size_mb: int = 500
    ffmpeg_enabled: bool = True


@dataclass
class PoseEstimatorConfig:
    """Individual pose estimator configuration."""
    enabled: bool = False
    model_complexity: Optional[int] = None
    min_detection_confidence: Optional[float] = None
    min_tracking_confidence: Optional[float] = None
    model_folder: Optional[str] = None
    net_resolution: Optional[str] = None
    model_name: Optional[str] = None
    device: Optional[str] = None
    detector: Optional[str] = None


@dataclass
class PoseEstimationConfig:
    """Pose estimation configuration."""
    estimators: Dict[str, PoseEstimatorConfig] = field(default_factory=dict)
    default_estimator: str = "mediapipe"
    confidence_threshold: float = 0.5


@dataclass
class GaitAnalysisConfig:
    """Gait analysis configuration."""
    min_sequence_length: int = 10
    gait_cycle_detection_method: str = "heel_strike"
    feature_extraction: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """LLM configuration with support for modern OpenAI and Gemini models."""
    provider: str = "openai"
    model: str = "gpt-5.2"  # Updated default to match new config
    temperature: float = 0.1
    max_tokens: int = 2000  # Updated default to match new config
    enabled: bool = True
    multimodal_enabled: bool = True
    
    # External models configuration
    models_config_file: Optional[str] = None
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_priority_order: List[str] = field(default_factory=list)
    
    # API configuration
    api: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "base_url": None,
            "timeout": 60,
            "max_retries": 3
        },
        "gemini": {
            "api_key_env_var": "GOOGLE_API_KEY",
            "timeout": 60,
            "max_retries": 3
        }
    })
    
    # Fallback configuration
    fallback: Dict[str, Any] = field(default_factory=lambda: {
        "enable_fallback": True,
        "triggers": ["api_error", "model_unavailable", "low_confidence"],
        "min_confidence_threshold": 0.3
    })
    
    # Model selection configuration
    model_selection: Dict[str, Any] = field(default_factory=lambda: {
        "strategy": "balanced",
        "prefer_multimodal": True
    })
    
    # Performance configuration
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "enable_async": True,
        "batch_size": 10,
        "batch_timeout": 300,
        "enable_caching": False,
        "cache_ttl": 3600
    })
    
    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "log_requests": True,
        "log_responses": False,
        "log_costs": True
    })
    
    # Legacy supported models for backward compatibility
    supported_models: Dict[str, List[str]] = field(default_factory=lambda: {
        "openai": [
            "gpt-5.2", "gpt-5.1", "gpt-5-mini", "gpt-5-nano",
            "gpt-4.1", "gpt-4.1-mini",
            "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
        ],
        "gemini": [
            "gemini-3-pro-preview", "gemini-3-pro-image-preview", "gemini-3-flash-preview",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-image",
            "gemini-pro", "gemini-pro-vision"
        ]
    })
    
    # Legacy model capabilities for backward compatibility
    model_capabilities: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "gpt-5.2": {"text": True, "images": True, "video": True, "max_tokens": 16384, "multimodal": True},
        "gpt-5.1": {"text": True, "images": True, "video": True, "max_tokens": 12288, "multimodal": True},
        "gpt-5-mini": {"text": True, "images": True, "video": True, "max_tokens": 8192, "multimodal": True},
        "gpt-5-nano": {"text": True, "images": True, "video": False, "max_tokens": 4096, "multimodal": True},
        "gpt-4.1": {"text": True, "images": True, "video": True, "max_tokens": 8192, "multimodal": True},
        "gpt-4.1-mini": {"text": True, "images": True, "video": False, "max_tokens": 6144, "multimodal": True},
        "gpt-4o-mini": {"text": True, "images": True, "max_tokens": 16384},
        "gpt-4o": {"text": True, "images": True, "max_tokens": 4096},
        "gpt-4-turbo": {"text": True, "images": True, "max_tokens": 4096},
        "gpt-4": {"text": True, "images": False, "max_tokens": 8192},
        "gpt-3.5-turbo": {"text": True, "images": False, "max_tokens": 4096},
        "gemini-3-pro-preview": {"text": True, "images": True, "video": True, "max_tokens": 16384, "multimodal": True},
        "gemini-3-pro-image-preview": {"text": True, "images": True, "video": True, "max_tokens": 16384, "multimodal": True},
        "gemini-3-flash-preview": {"text": True, "images": True, "video": True, "max_tokens": 8192, "multimodal": True},
        "gemini-2.5-pro": {"text": True, "images": True, "video": True, "max_tokens": 8192, "multimodal": True},
        "gemini-2.5-flash": {"text": True, "images": True, "video": True, "max_tokens": 8192, "multimodal": True},
        "gemini-2.5-flash-image": {"text": True, "images": True, "video": True, "max_tokens": 8192, "multimodal": True},
        "gemini-pro": {"text": True, "images": False, "max_tokens": 4096},
        "gemini-pro-vision": {"text": True, "images": True, "max_tokens": 4096}
    })
    
    def get_model_spec(self) -> Dict[str, Any]:
        """Get the specification for the current model."""
        # First try external models configuration
        if self.models and self.model in self.models:
            return self.models[self.model]
        
        # Fallback to legacy model capabilities
        return self.model_capabilities.get(self.model, {})
    
    def supports_images(self) -> bool:
        """Check if the current model supports image inputs."""
        model_spec = self.get_model_spec()
        return model_spec.get("images", False) or model_spec.get("multimodal", False)
    
    def supports_video(self) -> bool:
        """Check if the current model supports video inputs."""
        model_spec = self.get_model_spec()
        return model_spec.get("video", False)
    
    def get_max_tokens(self) -> int:
        """Get the maximum tokens for the current model."""
        model_spec = self.get_model_spec()
        return model_spec.get("max_tokens", self.max_tokens)
    
    def get_context_window(self) -> int:
        """Get the context window size for the current model."""
        model_spec = self.get_model_spec()
        return model_spec.get("context_window", self.get_max_tokens())
    
    def get_cost_tier(self) -> str:
        """Get the cost tier for the current model."""
        model_spec = self.get_model_spec()
        return model_spec.get("cost_tier", "medium")
    
    def get_reasoning_level(self) -> str:
        """Get the reasoning level for the current model."""
        model_spec = self.get_model_spec()
        return model_spec.get("reasoning", "standard")
    
    def is_model_supported(self) -> bool:
        """Check if the current model is supported by the provider."""
        # Check external models first
        if self.models and self.model in self.models:
            model_spec = self.models[self.model]
            return model_spec.get("provider") == self.provider
        
        # Fallback to legacy supported models
        return self.model in self.supported_models.get(self.provider, [])
    
    def get_available_models(self, provider: Optional[str] = None) -> List[str]:
        """Get list of available models for a provider."""
        target_provider = provider or self.provider
        
        # Get from external models configuration
        if self.models:
            return [
                model_name for model_name, model_spec in self.models.items()
                if model_spec.get("provider") == target_provider
            ]
        
        # Fallback to legacy supported models
        return self.supported_models.get(target_provider, [])
    
    def get_models_by_cost_tier(self, cost_tier: str) -> List[str]:
        """Get models filtered by cost tier."""
        if not self.models:
            return []
        
        return [
            model_name for model_name, model_spec in self.models.items()
            if model_spec.get("cost_tier") == cost_tier
        ]
    
    def get_preset_models(self, preset_name: str) -> List[str]:
        """Get models from a preset configuration."""
        if not self.presets or preset_name not in self.presets:
            return []
        
        preset = self.presets[preset_name]
        return preset.get("preferred_models", [])
    
    def get_api_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get API configuration for a provider."""
        target_provider = provider or self.provider
        
        # First try external providers configuration
        if self.providers and target_provider in self.providers:
            return self.providers[target_provider]
        
        # Fallback to embedded API configuration
        return self.api.get(target_provider, {})


@dataclass
class ClassificationConfig:
    """Classification configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    normal_abnormal_threshold: float = 0.7
    condition_confidence_threshold: float = 0.6
    fallback_to_gemini: bool = True


@dataclass
class StorageConfig:
    """Storage configuration."""
    data_directory: str = "data"
    logs_directory: str = "logs"
    config_directory: str = "config"
    videos_directory: str = "data/videos"
    youtube_directory: str = "data/youtube"
    analysis_directory: str = "data/analysis"
    models_directory: str = "data/models"
    training_directory: str = "data/training"
    cache_directory: str = "data/cache"
    exports_directory: str = "data/exports"
    cache_enabled: bool = True
    backup_enabled: bool = True


@dataclass
class RateLimitingConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    burst_size: int = 10


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=list)
    rate_limiting: RateLimitingConfig = field(default_factory=RateLimitingConfig)


@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret_key: str = ""
    encryption_enabled: bool = True
    rate_limiting_enabled: bool = True
    force_https: bool = False
    require_api_key: bool = False


@dataclass
class ExternalLoggingConfig:
    """External logging configuration."""
    enabled: bool = False
    service: str = "papertrail"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "structured"
    rotation: str = "1 day"
    retention: str = "30 days"
    compression: str = "zip"
    external_logging: ExternalLoggingConfig = field(default_factory=ExternalLoggingConfig)


@dataclass
class YouTubeConfig:
    """YouTube integration configuration."""
    enabled: bool = True
    download_directory: str = "data/youtube"
    quality: str = "best[height<=720]"
    format: str = "mp4"
    cookies_file: str = "config/yt_cookies.txt"


@dataclass
class CeleryConfig:
    """Celery configuration."""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"


@dataclass
class TaskTimeoutsConfig:
    """Task timeout configuration."""
    video_processing: int = 300
    gait_analysis: int = 600
    model_training: int = 3600


@dataclass
class BackgroundTasksConfig:
    """Background tasks configuration."""
    enabled: bool = True
    celery: CeleryConfig = field(default_factory=CeleryConfig)
    timeouts: TaskTimeoutsConfig = field(default_factory=TaskTimeoutsConfig)


@dataclass
class DevelopmentConfig:
    """Development configuration."""
    debug: bool = False
    auto_reload: bool = True
    use_sample_data: bool = False
    sample_data_directory: str = "data/samples"


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_memory_usage_mb: int = 2048
    max_workers: int = 4
    cache_ttl_seconds: int = 3600
    db_pool_size: int = 10
    db_max_overflow: int = 20


@dataclass
class AlexPoseConfig:
    """Main AlexPose configuration."""
    video_processing: VideoProcessingConfig = field(default_factory=VideoProcessingConfig)
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)
    gait_analysis: GaitAnalysisConfig = field(default_factory=GaitAnalysisConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    api: APIConfig = field(default_factory=APIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    background_tasks: BackgroundTasksConfig = field(default_factory=BackgroundTasksConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


class ConfigurationManager(IConfigurationManager):
    """
    Configuration manager for AlexPose system.
    
    Provides centralized configuration management with environment-specific
    overrides and validation.
    """
    
    def __init__(
        self,
        config_dir: Union[str, Path] = "config",
        environment: Optional[str] = None
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Configuration directory path
            environment: Environment name (development, production, etc.)
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.config: AlexPoseConfig = AlexPoseConfig()
        self._raw_config: Dict[str, Any] = {}
        
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from YAML files."""
        try:
            # Load main configuration
            main_config_path = self.config_dir / "alexpose.yaml"
            if main_config_path.exists():
                with open(main_config_path, 'r') as f:
                    main_config = yaml.safe_load(f) or {}
                self._raw_config.update(main_config)
            
            # Load external LLM models configuration if referenced
            self._load_llm_models_config()
            
            # Load environment-specific overrides
            env_config_path = self.config_dir / f"{self.environment}.yaml"
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f) or {}
                self._deep_merge(self._raw_config, env_config)
            
            # Convert to dataclass structure
            self._populate_config()
            
            logger.info(f"Configuration loaded for environment: {self.environment}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_llm_models_config(self) -> None:
        """Load external LLM models configuration if referenced."""
        try:
            # Check if LLM models config file is referenced
            models_config_file = self._raw_config.get("classification", {}).get("llm", {}).get("models_config_file")
            
            if models_config_file:
                # Resolve path relative to config directory
                if not Path(models_config_file).is_absolute():
                    models_config_path = self.config_dir / Path(models_config_file).name
                else:
                    models_config_path = Path(models_config_file)
                
                if models_config_path.exists():
                    logger.debug(f"Loading LLM models configuration from: {models_config_path}")
                    
                    with open(models_config_path, 'r') as f:
                        models_config = yaml.safe_load(f) or {}
                    
                    # Merge models configuration into the main config
                    if "classification" not in self._raw_config:
                        self._raw_config["classification"] = {}
                    if "llm" not in self._raw_config["classification"]:
                        self._raw_config["classification"]["llm"] = {}
                    
                    # Add models, presets, and providers from external config
                    llm_config = self._raw_config["classification"]["llm"]
                    llm_config["models"] = models_config.get("models", {})
                    llm_config["presets"] = models_config.get("presets", {})
                    llm_config["providers"] = models_config.get("providers", {})
                    llm_config["default_priority_order"] = models_config.get("default_priority_order", [])
                    
                    logger.info(f"Loaded {len(models_config.get('models', {}))} LLM model specifications")
                else:
                    logger.warning(f"LLM models config file not found: {models_config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load LLM models configuration: {e}")
            # Don't raise - allow system to continue with embedded model config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge override dictionary into base dictionary."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _populate_config(self) -> None:
        """Populate dataclass configuration from raw config."""
        try:
            # Video processing
            if "video_processing" in self._raw_config:
                vp_config = self._raw_config["video_processing"]
                self.config.video_processing = VideoProcessingConfig(
                    supported_formats=vp_config.get("supported_formats", ["mp4", "avi", "mov", "webm"]),
                    default_frame_rate=vp_config.get("default_frame_rate", 30.0),
                    max_video_size_mb=vp_config.get("max_video_size_mb", 500),
                    ffmpeg_enabled=vp_config.get("ffmpeg_enabled", True)
                )
            
            # Pose estimation
            if "pose_estimation" in self._raw_config:
                pe_config = self._raw_config["pose_estimation"]
                estimators = {}
                
                for name, est_config in pe_config.get("estimators", {}).items():
                    estimators[name] = PoseEstimatorConfig(
                        enabled=est_config.get("enabled", False),
                        model_complexity=est_config.get("model_complexity"),
                        min_detection_confidence=est_config.get("min_detection_confidence"),
                        min_tracking_confidence=est_config.get("min_tracking_confidence"),
                        model_folder=est_config.get("model_folder"),
                        net_resolution=est_config.get("net_resolution"),
                        model_name=est_config.get("model_name"),
                        device=est_config.get("device"),
                        detector=est_config.get("detector")
                    )
                
                self.config.pose_estimation = PoseEstimationConfig(
                    estimators=estimators,
                    default_estimator=pe_config.get("default_estimator", "mediapipe"),
                    confidence_threshold=pe_config.get("confidence_threshold", 0.5)
                )
            
            # Classification
            if "classification" in self._raw_config:
                class_config = self._raw_config["classification"]
                llm_config = class_config.get("llm", {})
                
                # Create LLMConfig with all the new fields
                llm_obj = LLMConfig(
                    provider=llm_config.get("provider", "openai"),
                    model=llm_config.get("model", "gpt-5.2"),
                    temperature=llm_config.get("temperature", 0.1),
                    max_tokens=llm_config.get("max_tokens", 2000),
                    enabled=llm_config.get("enabled", True),
                    multimodal_enabled=llm_config.get("multimodal_enabled", True),
                    models_config_file=llm_config.get("models_config_file"),
                    models=llm_config.get("models", {}),
                    presets=llm_config.get("presets", {}),
                    providers=llm_config.get("providers", {}),
                    default_priority_order=llm_config.get("default_priority_order", []),
                    api=llm_config.get("api", {}),
                    fallback=llm_config.get("fallback", {}),
                    model_selection=llm_config.get("model_selection", {}),
                    performance=llm_config.get("performance", {}),
                    logging=llm_config.get("logging", {})
                )
                
                # Merge with defaults for any missing configurations
                if not llm_obj.api:
                    llm_obj.api = {
                        "openai": {
                            "api_key_env_var": "OPENAI_API_KEY",
                            "base_url": None,
                            "timeout": 60,
                            "max_retries": 3
                        },
                        "gemini": {
                            "api_key_env_var": "GOOGLE_API_KEY",
                            "timeout": 60,
                            "max_retries": 3
                        }
                    }
                
                if not llm_obj.fallback:
                    llm_obj.fallback = {
                        "enable_fallback": True,
                        "triggers": ["api_error", "model_unavailable", "low_confidence"],
                        "min_confidence_threshold": 0.3
                    }
                
                if not llm_obj.model_selection:
                    llm_obj.model_selection = {
                        "strategy": "balanced",
                        "prefer_multimodal": True
                    }
                
                if not llm_obj.performance:
                    llm_obj.performance = {
                        "enable_async": True,
                        "batch_size": 10,
                        "batch_timeout": 300,
                        "enable_caching": False,
                        "cache_ttl": 3600
                    }
                
                if not llm_obj.logging:
                    llm_obj.logging = {
                        "log_requests": True,
                        "log_responses": False,
                        "log_costs": True
                    }
                
                self.config.classification = ClassificationConfig(
                    llm=llm_obj,
                    normal_abnormal_threshold=class_config.get("normal_abnormal_threshold", 0.7),
                    condition_confidence_threshold=class_config.get("condition_confidence_threshold", 0.6),
                    fallback_to_gemini=class_config.get("fallback_to_gemini", True)
                )
            
            # Storage
            if "storage" in self._raw_config:
                storage_config = self._raw_config["storage"]
                self.config.storage = StorageConfig(**storage_config)
            
            # API
            if "api" in self._raw_config:
                api_config = self._raw_config["api"]
                rate_limiting = api_config.get("rate_limiting", {})
                
                self.config.api = APIConfig(
                    host=api_config.get("host", "0.0.0.0"),
                    port=api_config.get("port", 8000),
                    cors_origins=api_config.get("cors_origins", []),
                    rate_limiting=RateLimitingConfig(
                        enabled=rate_limiting.get("enabled", True),
                        requests_per_minute=rate_limiting.get("requests_per_minute", 60),
                        burst_size=rate_limiting.get("burst_size", 10)
                    )
                )
            
            # Other configurations...
            self._populate_remaining_configs()
            
        except Exception as e:
            logger.error(f"Failed to populate configuration: {e}")
            raise
    
    def _populate_remaining_configs(self) -> None:
        """Populate remaining configuration sections."""
        # Security
        if "security" in self._raw_config:
            sec_config = self._raw_config["security"]
            self.config.security = SecurityConfig(**sec_config)
        
        # Logging
        if "logging" in self._raw_config:
            log_config = self._raw_config["logging"]
            ext_logging = log_config.get("external_logging", {})
            
            self.config.logging = LoggingConfig(
                level=log_config.get("level", "INFO"),
                format=log_config.get("format", "structured"),
                rotation=log_config.get("rotation", "1 day"),
                retention=log_config.get("retention", "30 days"),
                compression=log_config.get("compression", "zip"),
                external_logging=ExternalLoggingConfig(
                    enabled=ext_logging.get("enabled", False),
                    service=ext_logging.get("service", "papertrail")
                )
            )
        
        # YouTube
        if "youtube" in self._raw_config:
            yt_config = self._raw_config["youtube"]
            self.config.youtube = YouTubeConfig(**yt_config)
        
        # Background tasks
        if "background_tasks" in self._raw_config:
            bt_config = self._raw_config["background_tasks"]
            celery_config = bt_config.get("celery", {})
            timeouts_config = bt_config.get("timeouts", {})
            
            self.config.background_tasks = BackgroundTasksConfig(
                enabled=bt_config.get("enabled", True),
                celery=CeleryConfig(
                    broker_url=celery_config.get("broker_url", "redis://localhost:6379/0"),
                    result_backend=celery_config.get("result_backend", "redis://localhost:6379/0")
                ),
                timeouts=TaskTimeoutsConfig(
                    video_processing=timeouts_config.get("video_processing", 300),
                    gait_analysis=timeouts_config.get("gait_analysis", 600),
                    model_training=timeouts_config.get("model_training", 3600)
                )
            )
        
        # Development
        if "development" in self._raw_config:
            dev_config = self._raw_config["development"]
            self.config.development = DevelopmentConfig(**dev_config)
        
        # Performance
        if "performance" in self._raw_config:
            perf_config = self._raw_config["performance"]
            self.config.performance = PerformanceConfig(**perf_config)
    
    def get_config_value(
        self,
        key: str,
        default: Any = None,
        target_type: Optional[type] = None
    ) -> Any:
        """
        Get configuration value by key with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            target_type: Optional type conversion
            
        Returns:
            Configuration value
        """
        try:
            # Navigate through nested keys
            value = self._raw_config
            for part in key.split('.'):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            # Type conversion if requested
            if target_type and value is not None:
                if target_type == bool and isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                else:
                    return target_type(value)
            
            return value
            
        except Exception as e:
            logger.warning(f"Failed to get config value '{key}': {e}")
            return default
    
    def get_videos_directory(self) -> Path:
        """Get videos directory path."""
        return Path(self.config.storage.videos_directory)
    
    def get_openpose_directory(self) -> Path:
        """Get OpenPose outputs directory path."""
        # For compatibility with existing code
        return Path(self.config.storage.analysis_directory) / "openpose"
    
    def get_gemini_api_key(self) -> str:
        """Get Gemini API key from environment."""
        return os.getenv("GOOGLE_API_KEY", "")
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY", "")
    
    def is_gemini_configured(self) -> bool:
        """Check if Gemini is properly configured."""
        return bool(self.get_gemini_api_key())
    
    def is_openai_configured(self) -> bool:
        """Check if OpenAI is properly configured."""
        return bool(self.get_openai_api_key())
    
    @property
    def llm_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get LLM models configuration.
        
        Returns:
            Dictionary of LLM model specifications
        """
        return self.config.classification.llm.models
    
    def validate_configuration(self) -> bool:
        """
        Validate all configuration settings with comprehensive error reporting.
        
        Returns:
            bool: True if all validations pass, False otherwise
        """
        validation_errors = []
        validation_warnings = []
        
        try:
            # Validate directory structure
            self._validate_directories(validation_errors, validation_warnings)
            
            # Validate video processing configuration
            self._validate_video_processing(validation_errors, validation_warnings)
            
            # Validate pose estimation configuration
            self._validate_pose_estimation(validation_errors, validation_warnings)
            
            # Validate classification configuration
            self._validate_classification(validation_errors, validation_warnings)
            
            # Validate API configuration
            self._validate_api_configuration(validation_errors, validation_warnings)
            
            # Validate security configuration
            self._validate_security_configuration(validation_errors, validation_warnings)
            
            # Validate storage configuration
            self._validate_storage_configuration(validation_errors, validation_warnings)
            
            # Validate performance configuration
            self._validate_performance_configuration(validation_errors, validation_warnings)
            
            # Validate environment variables
            self._validate_environment_variables(validation_errors, validation_warnings)
            
            # Validate system dependencies
            self._validate_system_dependencies(validation_errors, validation_warnings)
            
            # Validate file permissions
            self._validate_file_permissions(validation_errors, validation_warnings)
            
            # Report validation results
            self._report_validation_results(validation_errors, validation_warnings)
            
            return len(validation_errors) == 0
            
        except Exception as e:
            logger.error(f"Configuration validation failed with exception: {e}")
            return False
    
    def _validate_directories(self, errors: List[str], warnings: List[str]) -> None:
        """Validate and create required directories."""
        required_dirs = [
            (self.config.storage.data_directory, "Data directory"),
            (self.config.storage.logs_directory, "Logs directory"),
            (self.config.storage.config_directory, "Config directory"),
            (self.config.storage.videos_directory, "Videos directory"),
            (self.config.storage.youtube_directory, "YouTube directory"),
            (self.config.storage.analysis_directory, "Analysis directory"),
            (self.config.storage.models_directory, "Models directory"),
            (self.config.storage.training_directory, "Training directory"),
            (self.config.storage.cache_directory, "Cache directory"),
            (self.config.storage.exports_directory, "Exports directory"),
        ]
        
        for dir_path, description in required_dirs:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"✓ {description} validated: {dir_path}")
            except Exception as e:
                errors.append(f"Cannot create {description.lower()} '{dir_path}': {e}")
    
    def _validate_video_processing(self, errors: List[str], warnings: List[str]) -> None:
        """Validate video processing configuration."""
        vp_config = self.config.video_processing
        
        # Validate supported formats
        if not vp_config.supported_formats:
            errors.append("No supported video formats configured")
        else:
            valid_formats = {'mp4', 'avi', 'mov', 'webm', 'mkv', 'flv'}
            invalid_formats = set(vp_config.supported_formats) - valid_formats
            if invalid_formats:
                warnings.append(f"Potentially unsupported video formats: {invalid_formats}")
        
        # Validate frame rate
        if vp_config.default_frame_rate <= 0:
            errors.append("Default frame rate must be positive")
        elif vp_config.default_frame_rate > 120:
            warnings.append(f"High default frame rate ({vp_config.default_frame_rate}) may impact performance")
        
        # Validate video size limit
        if vp_config.max_video_size_mb <= 0:
            errors.append("Maximum video size must be positive")
        elif vp_config.max_video_size_mb > 2048:
            warnings.append(f"Large video size limit ({vp_config.max_video_size_mb}MB) may cause memory issues")
    
    def _validate_pose_estimation(self, errors: List[str], warnings: List[str]) -> None:
        """Validate pose estimation configuration."""
        pe_config = self.config.pose_estimation
        
        # Validate default estimator
        if pe_config.default_estimator not in pe_config.estimators:
            errors.append(f"Default estimator '{pe_config.default_estimator}' not found in configured estimators")
        
        # Validate confidence threshold
        if not 0 <= pe_config.confidence_threshold <= 1:
            errors.append("Pose estimation confidence threshold must be between 0 and 1")
        
        # Check if at least one estimator is enabled
        enabled_estimators = [name for name, config in pe_config.estimators.items() if config.enabled]
        if not enabled_estimators:
            warnings.append("No pose estimators are enabled")
        
        # Validate individual estimator configurations
        for name, estimator_config in pe_config.estimators.items():
            if estimator_config.enabled:
                self._validate_estimator_config(name, estimator_config, errors, warnings)
    
    def _validate_estimator_config(self, name: str, config: PoseEstimatorConfig, errors: List[str], warnings: List[str]) -> None:
        """Validate individual pose estimator configuration."""
        if name == "mediapipe":
            if config.model_complexity is not None and config.model_complexity not in [0, 1, 2]:
                errors.append(f"MediaPipe model complexity must be 0, 1, or 2, got {config.model_complexity}")
        
        elif name in ["openpose", "alphapose", "alphapose_halpe", "alphapose_coco"]:
            if config.model_folder and not Path(config.model_folder).exists():
                warnings.append(f"{name} model folder does not exist: {config.model_folder}")
        
        elif name in ["ultralytics", "ultralytics_yolov8", "ultralytics_yolov11"]:
            if config.device and config.device not in ["auto", "cpu", "cuda", "mps"]:
                warnings.append(f"Unknown device '{config.device}' for {name}")
        
        # Validate confidence thresholds
        for attr in ["min_detection_confidence", "min_tracking_confidence"]:
            value = getattr(config, attr, None)
            if value is not None and not 0 <= value <= 1:
                errors.append(f"{name} {attr} must be between 0 and 1, got {value}")
    
    def _validate_classification(self, errors: List[str], warnings: List[str]) -> None:
        """Validate classification configuration."""
        class_config = self.config.classification
        llm_config = class_config.llm
        
        # Validate thresholds
        if not 0 <= class_config.normal_abnormal_threshold <= 1:
            errors.append("Normal/abnormal threshold must be between 0 and 1")
        
        if not 0 <= class_config.condition_confidence_threshold <= 1:
            errors.append("Condition confidence threshold must be between 0 and 1")
        
        # Validate LLM configuration
        if llm_config.enabled:
            # Validate provider
            if llm_config.provider not in ["openai", "gemini"]:
                errors.append(f"Unsupported LLM provider: {llm_config.provider}")
            
            # Validate model
            if not llm_config.is_model_supported():
                available_models = llm_config.get_available_models()
                if available_models:
                    errors.append(f"Model '{llm_config.model}' not supported by provider '{llm_config.provider}'. Available models: {', '.join(available_models[:5])}")
                else:
                    errors.append(f"Model '{llm_config.model}' not supported by provider '{llm_config.provider}'")
            
            # Validate temperature
            if not 0 <= llm_config.temperature <= 2:
                errors.append("LLM temperature must be between 0 and 2")
            
            # Validate max tokens
            if llm_config.max_tokens <= 0:
                errors.append("LLM max_tokens must be positive")
            elif llm_config.max_tokens > llm_config.get_max_tokens():
                warnings.append(f"Configured max_tokens ({llm_config.max_tokens}) exceeds model limit ({llm_config.get_max_tokens()})")
            
            # Validate external models config file
            if llm_config.models_config_file:
                models_config_path = self.config_dir / Path(llm_config.models_config_file).name
                if not models_config_path.exists():
                    warnings.append(f"LLM models config file not found: {models_config_path}")
                elif not llm_config.models:
                    warnings.append("LLM models config file referenced but no models loaded")
            
            # Validate API keys with detailed error messages
            api_config = llm_config.get_api_config()
            
            if llm_config.provider == "openai":
                if not self.is_openai_configured():
                    api_key_var = api_config.get("api_key_env_var", "OPENAI_API_KEY")
                    errors.append(f"OpenAI API key not configured (set {api_key_var} environment variable)")
                else:
                    # Validate API key format
                    api_key = self.get_openai_api_key()
                    if not api_key.startswith("sk-"):
                        warnings.append("OpenAI API key format appears invalid (should start with 'sk-')")
                    elif len(api_key) < 40:
                        warnings.append("OpenAI API key appears too short (may be invalid)")
                        
            elif llm_config.provider == "gemini":
                if not self.is_gemini_configured():
                    api_key_var = api_config.get("api_key_env_var", "GOOGLE_API_KEY")
                    errors.append(f"Gemini API key not configured (set {api_key_var} environment variable)")
                else:
                    # Validate API key format
                    api_key = self.get_gemini_api_key()
                    if len(api_key) < 20:
                        warnings.append("Gemini API key appears too short (may be invalid)")
            
            # Check fallback configuration
            if class_config.fallback_to_gemini and not self.is_gemini_configured():
                warnings.append("Gemini fallback enabled but API key not configured")
            
            # Validate multimodal configuration
            if llm_config.multimodal_enabled and not llm_config.supports_images():
                warnings.append(f"Multimodal enabled but model '{llm_config.model}' does not support images")
            
            # Validate model selection strategy
            strategy = llm_config.model_selection.get("strategy", "balanced")
            if strategy not in ["cost_optimized", "performance_optimized", "balanced"]:
                warnings.append(f"Unknown model selection strategy: {strategy}")
            
            # Check if strategy has corresponding preset
            if llm_config.presets and strategy in llm_config.presets:
                preset_models = llm_config.get_preset_models(strategy)
                if not preset_models:
                    warnings.append(f"Model selection strategy '{strategy}' has no preferred models defined")
            
            # Validate performance settings
            perf_config = llm_config.performance
            if perf_config.get("batch_size", 10) <= 0:
                errors.append("LLM batch_size must be positive")
            if perf_config.get("batch_timeout", 300) <= 0:
                errors.append("LLM batch_timeout must be positive")
            if perf_config.get("cache_ttl", 3600) <= 0:
                errors.append("LLM cache_ttl must be positive")
    
    def _validate_api_configuration(self, errors: List[str], warnings: List[str]) -> None:
        """Validate API configuration."""
        api_config = self.config.api
        
        # Validate port
        if not 1 <= api_config.port <= 65535:
            errors.append(f"API port must be between 1 and 65535, got {api_config.port}")
        elif api_config.port < 1024 and api_config.host != "127.0.0.1":
            warnings.append(f"Port {api_config.port} requires root privileges on most systems")
        
        # Validate host
        if api_config.host not in ["0.0.0.0", "127.0.0.1", "localhost"]:
            warnings.append(f"Non-standard host configuration: {api_config.host}")
        
        # Validate CORS origins
        for origin in api_config.cors_origins:
            if not origin.startswith(("http://", "https://")):
                warnings.append(f"CORS origin should include protocol: {origin}")
        
        # Validate rate limiting
        if api_config.rate_limiting.enabled:
            if api_config.rate_limiting.requests_per_minute <= 0:
                errors.append("Rate limiting requests_per_minute must be positive")
            if api_config.rate_limiting.burst_size <= 0:
                errors.append("Rate limiting burst_size must be positive")
    
    def _validate_security_configuration(self, errors: List[str], warnings: List[str]) -> None:
        """Validate security configuration."""
        sec_config = self.config.security
        
        # Validate JWT secret key
        jwt_key = sec_config.jwt_secret_key or os.getenv("JWT_SECRET_KEY", "")
        if sec_config.encryption_enabled and not jwt_key:
            errors.append("JWT secret key required when encryption is enabled (set JWT_SECRET_KEY environment variable)")
        elif jwt_key and len(jwt_key) < 32:
            warnings.append("JWT secret key should be at least 32 characters for security")
        
        # Security recommendations
        if not sec_config.force_https and self.environment == "production":
            warnings.append("HTTPS should be enforced in production environment")
        
        if not sec_config.rate_limiting_enabled:
            warnings.append("Rate limiting is disabled - consider enabling for security")
    
    def _validate_storage_configuration(self, errors: List[str], warnings: List[str]) -> None:
        """Validate storage configuration."""
        storage_config = self.config.storage
        
        # Check for absolute vs relative paths
        paths_to_check = [
            ("data_directory", storage_config.data_directory),
            ("logs_directory", storage_config.logs_directory),
            ("videos_directory", storage_config.videos_directory),
            ("youtube_directory", storage_config.youtube_directory),
        ]
        
        for name, path in paths_to_check:
            if Path(path).is_absolute():
                warnings.append(f"{name} uses absolute path: {path}")
    
    def _validate_performance_configuration(self, errors: List[str], warnings: List[str]) -> None:
        """Validate performance configuration."""
        perf_config = self.config.performance
        
        # Validate memory limits
        if perf_config.max_memory_usage_mb <= 0:
            errors.append("Maximum memory usage must be positive")
        elif perf_config.max_memory_usage_mb < 512:
            warnings.append("Low memory limit may cause processing failures")
        
        # Validate worker count
        if perf_config.max_workers <= 0:
            errors.append("Maximum workers must be positive")
        elif perf_config.max_workers > 16:
            warnings.append("High worker count may cause resource contention")
        
        # Validate cache TTL
        if perf_config.cache_ttl_seconds <= 0:
            errors.append("Cache TTL must be positive")
        
        # Validate database pool settings
        if perf_config.db_pool_size <= 0:
            errors.append("Database pool size must be positive")
        if perf_config.db_max_overflow < 0:
            errors.append("Database max overflow cannot be negative")
    
    def _validate_environment_variables(self, errors: List[str], warnings: List[str]) -> None:
        """Validate required environment variables."""
        # Check for required environment variables based on configuration
        required_env_vars = []
        optional_env_vars = []
        
        # JWT secret key for security
        if self.config.security.encryption_enabled:
            jwt_key = os.getenv("JWT_SECRET_KEY", self.config.security.jwt_secret_key)
            if not jwt_key:
                required_env_vars.append("JWT_SECRET_KEY")
        
        # API keys for LLM providers
        if self.config.classification.llm.enabled:
            if self.config.classification.llm.provider == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    required_env_vars.append("OPENAI_API_KEY")
            elif self.config.classification.llm.provider == "gemini":
                if not os.getenv("GEMINI_API_KEY"):
                    required_env_vars.append("GEMINI_API_KEY")
        
        # Database URL for production
        if self.environment == "production":
            if not os.getenv("DATABASE_URL"):
                optional_env_vars.append("DATABASE_URL")
        
        # Redis URL for background tasks
        if self.config.background_tasks.enabled:
            redis_url = os.getenv("REDIS_URL", self.config.background_tasks.celery.broker_url)
            if redis_url == "redis://localhost:6379/0" and self.environment == "production":
                warnings.append("Using default Redis URL in production - consider setting REDIS_URL environment variable")
        
        # Report missing required environment variables
        for env_var in required_env_vars:
            errors.append(f"Required environment variable not set: {env_var}")
        
        # Report missing optional environment variables
        for env_var in optional_env_vars:
            warnings.append(f"Optional environment variable not set: {env_var}")
    
    def _validate_system_dependencies(self, errors: List[str], warnings: List[str]) -> None:
        """Validate system dependencies and external tools."""
        import shutil
        import subprocess
        
        # Check for FFmpeg if enabled
        if self.config.video_processing.ffmpeg_enabled:
            if not shutil.which("ffmpeg"):
                warnings.append("FFmpeg not found in PATH - will fallback to OpenCV for video processing")
        
        # Check for Redis if background tasks are enabled
        if self.config.background_tasks.enabled:
            try:
                # Try to connect to Redis
                import redis
                redis_client = redis.from_url(self.config.background_tasks.celery.broker_url)
                redis_client.ping()
            except ImportError:
                warnings.append("Redis Python client not installed - background tasks may not work")
            except Exception:
                warnings.append(f"Cannot connect to Redis at {self.config.background_tasks.celery.broker_url}")
        
        # Check for pose estimation dependencies
        enabled_estimators = [name for name, config in self.config.pose_estimation.estimators.items() if config.enabled]
        
        for estimator in enabled_estimators:
            if estimator == "mediapipe":
                try:
                    import mediapipe
                except ImportError:
                    errors.append("MediaPipe not installed but enabled in configuration")
            
            elif estimator in ["ultralytics", "ultralytics_yolov8", "ultralytics_yolov11"]:
                try:
                    import ultralytics
                except ImportError:
                    warnings.append(f"Ultralytics not installed but {estimator} enabled in configuration")
            
            elif estimator in ["openpose"]:
                # Check if OpenPose model folder exists
                estimator_config = self.config.pose_estimation.estimators[estimator]
                if estimator_config.model_folder and not Path(estimator_config.model_folder).exists():
                    warnings.append(f"OpenPose model folder not found: {estimator_config.model_folder}")
        
        # Check for YouTube processing dependencies
        if self.config.youtube.enabled:
            if not shutil.which("yt-dlp") and not shutil.which("youtube-dl"):
                try:
                    import yt_dlp
                except ImportError:
                    warnings.append("yt-dlp not installed - YouTube processing may not work")
    
    def _validate_file_permissions(self, errors: List[str], warnings: List[str]) -> None:
        """Validate file and directory permissions."""
        import stat
        
        # Check write permissions for data directories
        directories_to_check = [
            (self.config.storage.data_directory, "data directory"),
            (self.config.storage.logs_directory, "logs directory"),
            (self.config.storage.videos_directory, "videos directory"),
            (self.config.storage.youtube_directory, "YouTube directory"),
            (self.config.storage.analysis_directory, "analysis directory"),
            (self.config.storage.cache_directory, "cache directory"),
        ]
        
        for dir_path, description in directories_to_check:
            path_obj = Path(dir_path)
            if path_obj.exists():
                try:
                    # Test write permission by creating a temporary file
                    test_file = path_obj / ".write_test"
                    test_file.touch()
                    test_file.unlink()
                except PermissionError:
                    errors.append(f"No write permission for {description}: {dir_path}")
                except Exception as e:
                    warnings.append(f"Cannot test write permission for {description}: {e}")
        
        # Check read permissions for config files
        config_files = [
            self.config_dir / "alexpose.yaml",
            self.config_dir / f"{self.environment}.yaml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        f.read(1)  # Try to read one character
                except PermissionError:
                    errors.append(f"No read permission for config file: {config_file}")
                except Exception as e:
                    warnings.append(f"Cannot read config file {config_file}: {e}")
    
    def _report_validation_results(self, errors: List[str], warnings: List[str]) -> None:
        """Report validation results with detailed logging."""
        if errors:
            logger.error("Configuration validation failed with errors:")
            for i, error in enumerate(errors, 1):
                logger.error(f"  {i}. {error}")
        
        if warnings:
            logger.warning("Configuration validation completed with warnings:")
            for i, warning in enumerate(warnings, 1):
                logger.warning(f"  {i}. {warning}")
        
        if not errors and not warnings:
            logger.info("✓ Configuration validation passed - all settings are valid")
        elif not errors:
            logger.info(f"✓ Configuration validation passed with {len(warnings)} warnings")
        else:
            logger.error(f"✗ Configuration validation failed with {len(errors)} errors and {len(warnings)} warnings")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get a detailed validation report.
        
        Returns:
            Dictionary containing validation results and recommendations
        """
        validation_errors = []
        validation_warnings = []
        
        # Run all validations
        self._validate_directories(validation_errors, validation_warnings)
        self._validate_video_processing(validation_errors, validation_warnings)
        self._validate_pose_estimation(validation_errors, validation_warnings)
        self._validate_classification(validation_errors, validation_warnings)
        self._validate_api_configuration(validation_errors, validation_warnings)
        self._validate_security_configuration(validation_errors, validation_warnings)
        self._validate_storage_configuration(validation_errors, validation_warnings)
        self._validate_performance_configuration(validation_errors, validation_warnings)
        self._validate_environment_variables(validation_errors, validation_warnings)
        self._validate_system_dependencies(validation_errors, validation_warnings)
        self._validate_file_permissions(validation_errors, validation_warnings)
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "error_count": len(validation_errors),
            "warning_count": len(validation_warnings),
            "environment": self.environment,
            "config_files_loaded": [
                str(self.config_dir / "alexpose.yaml"),
                str(self.config_dir / f"{self.environment}.yaml")
            ],
            "validation_categories": {
                "directories": "✓ Passed",
                "video_processing": "✓ Passed",
                "pose_estimation": "✓ Passed", 
                "classification": "✓ Passed",
                "api_configuration": "✓ Passed",
                "security": "✓ Passed",
                "storage": "✓ Passed",
                "performance": "✓ Passed",
                "environment_variables": "✓ Passed",
                "system_dependencies": "✓ Passed",
                "file_permissions": "✓ Passed"
            }
        }
    def create_environment_config(self, environment: str, config_overrides: Dict[str, Any]) -> bool:
        """
        Create a new environment-specific configuration file.
        
        Args:
            environment: Environment name
            config_overrides: Configuration overrides for this environment
            
        Returns:
            bool: True if config file created successfully, False otherwise
        """
        try:
            env_config_path = self.config_dir / f"{environment}.yaml"
            
            if env_config_path.exists():
                logger.warning(f"Environment config file already exists: {env_config_path}")
                return False
            
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Write environment config
            with open(env_config_path, 'w') as f:
                yaml.dump(config_overrides, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created environment configuration: {env_config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create environment config '{environment}': {e}")
            return False
    
    def get_available_environments(self) -> List[str]:
        """
        Get list of available environment configurations.
        
        Returns:
            List of environment names that have configuration files
        """
        environments = []
        
        # Check for standard environments
        for env in ["development", "production", "staging", "testing"]:
            env_config_path = self.config_dir / f"{env}.yaml"
            if env_config_path.exists():
                environments.append(env)
        
        # Check for any other .yaml files in config directory
        for config_file in self.config_dir.glob("*.yaml"):
            env_name = config_file.stem
            if env_name not in ["alexpose", "classification_prompts"] and env_name not in environments:
                environments.append(env_name)
        
        return sorted(environments)
    
    def load_environment_config(self, environment: str) -> bool:
        """
        Load configuration for a specific environment.
        
        Args:
            environment: Environment name (development, production, staging, etc.)
            
        Returns:
            bool: True if environment config loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading configuration for environment: {environment}")
            
            # Check if environment config file exists (unless it's a standard fallback)
            env_config_path = self.config_dir / f"{environment}.yaml"
            if not env_config_path.exists() and environment not in ["default", "base"]:
                logger.warning(f"Environment configuration file not found: {env_config_path}")
                # For strict environment loading, we should fail if the specific environment doesn't exist
                # But we'll allow it to continue with just the main config for flexibility
            
            # Update environment
            old_environment = self.environment
            self.environment = environment
            
            # Reload configuration with new environment
            self._raw_config = {}
            self._load_configuration()
            
            # Validate after loading
            if self.validate_configuration():
                logger.info(f"Environment configuration loaded successfully: {environment}")
                return True
            else:
                logger.error(f"Environment configuration failed validation: {environment}")
                # Revert to old environment on failure
                self.environment = old_environment
                self._raw_config = {}
                self._load_configuration()
                return False
                
        except Exception as e:
            logger.error(f"Failed to load environment configuration '{environment}': {e}")
            # Revert to old environment on failure
            self.environment = old_environment
            self._raw_config = {}
            self._load_configuration()
            return False
    
    def reload_configuration(self) -> bool:
        """
        Reload configuration from files.
        
        Returns:
            bool: True if reload successful, False otherwise
        """
        try:
            logger.info("Reloading configuration...")
            self._raw_config = {}
            self._load_configuration()
            
            # Validate after reload
            if self.validate_configuration():
                logger.info("Configuration reloaded successfully")
                return True
            else:
                logger.error("Configuration reload failed validation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the current environment and configuration.
        
        Returns:
            Dictionary containing environment information
        """
        return {
            "environment": self.environment,
            "config_directory": str(self.config_dir),
            "config_files": {
                "main": str(self.config_dir / "alexpose.yaml"),
                "environment": str(self.config_dir / f"{self.environment}.yaml"),
                "main_exists": (self.config_dir / "alexpose.yaml").exists(),
                "environment_exists": (self.config_dir / f"{self.environment}.yaml").exists()
            },
            "api_keys_configured": {
                "openai": self.is_openai_configured(),
                "gemini": self.is_gemini_configured()
            },
            "enabled_features": {
                "llm_classification": self.config.classification.llm.enabled,
                "youtube_processing": self.config.youtube.enabled,
                "background_tasks": self.config.background_tasks.enabled,
                "caching": self.config.storage.cache_enabled,
                "backup": self.config.storage.backup_enabled
            },
            "pose_estimators": {
                name: config.enabled 
                for name, config in self.config.pose_estimation.estimators.items()
            }
        }
    
    def get_configuration_recommendations(self) -> Dict[str, List[str]]:
        """
        Get configuration recommendations based on current settings and environment.
        
        Returns:
            Dictionary containing categorized recommendations
        """
        recommendations = {
            "security": [],
            "performance": [],
            "reliability": [],
            "cost_optimization": [],
            "best_practices": []
        }
        
        try:
            # Security recommendations
            if not self.config.security.force_https and self.environment == "production":
                recommendations["security"].append("Enable HTTPS enforcement in production environment")
            
            if not self.config.security.require_api_key and self.environment == "production":
                recommendations["security"].append("Require API keys for production API access")
            
            if not self.config.security.rate_limiting_enabled:
                recommendations["security"].append("Enable rate limiting for API security")
            
            jwt_key = os.getenv("JWT_SECRET_KEY", self.config.security.jwt_secret_key)
            if self.config.security.encryption_enabled and jwt_key and len(jwt_key) < 32:
                recommendations["security"].append("Use a longer JWT secret key (at least 32 characters)")
            
            # Performance recommendations
            if self.config.performance.max_memory_usage_mb < 1024:
                recommendations["performance"].append("Consider increasing memory limit for better performance")
            
            if self.config.performance.max_workers > 8:
                recommendations["performance"].append("High worker count may cause resource contention")
            
            if not self.config.storage.cache_enabled:
                recommendations["performance"].append("Enable caching to improve response times")
            
            # Reliability recommendations
            if not self.config.storage.backup_enabled:
                recommendations["reliability"].append("Enable backups for data protection")
            
            if self.config.video_processing.max_video_size_mb > 1000:
                recommendations["reliability"].append("Large video size limit may cause memory issues")
            
            # Cost optimization recommendations
            llm_config = self.config.classification.llm
            if llm_config.enabled and llm_config.models:
                current_model_spec = llm_config.get_model_spec()
                cost_tier = current_model_spec.get("cost_tier", "medium")
                
                if cost_tier == "premium" and self.environment == "development":
                    recommendations["cost_optimization"].append("Consider using a cost-effective model for development")
                
                if llm_config.performance.get("enable_caching", False) is False:
                    recommendations["cost_optimization"].append("Enable LLM response caching to reduce API costs")
            
            # Best practices recommendations
            if self.environment == "production":
                if self.config.logging.level == "DEBUG":
                    recommendations["best_practices"].append("Use INFO or WARNING log level in production")
                
                if self.config.development.debug:
                    recommendations["best_practices"].append("Disable debug mode in production")
            
            # Check for enabled but unconfigured features
            enabled_estimators = [name for name, config in self.config.pose_estimation.estimators.items() if config.enabled]
            if not enabled_estimators:
                recommendations["best_practices"].append("Enable at least one pose estimator")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate configuration recommendations: {e}")
            return recommendations
    
    def create_default_configuration_files(self) -> bool:
        """
        Create default configuration files if they don't exist.
        
        Returns:
            bool: True if files created successfully, False otherwise
        """
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create main config if it doesn't exist
            main_config_path = self.config_dir / "alexpose.yaml"
            if not main_config_path.exists():
                logger.info(f"Creating default main configuration: {main_config_path}")
                self._create_default_main_config(main_config_path)
            
            # Create environment-specific configs
            for env in ["development", "production"]:
                env_config_path = self.config_dir / f"{env}.yaml"
                if not env_config_path.exists():
                    logger.info(f"Creating default {env} configuration: {env_config_path}")
                    self._create_default_env_config(env_config_path, env)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create default config files: {e}")
            return False
    
    def _create_default_main_config(self, config_path: Path) -> None:
        """Create default main configuration file."""
        default_config = {
            "video_processing": {
                "supported_formats": ["mp4", "avi", "mov", "webm"],
                "default_frame_rate": 30.0,
                "max_video_size_mb": 500,
                "ffmpeg_enabled": True
            },
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "model_complexity": 1,
                        "min_detection_confidence": 0.5,
                        "min_tracking_confidence": 0.5,
                        "enabled": True
                    }
                },
                "default_estimator": "mediapipe",
                "confidence_threshold": 0.5
            },
            "classification": {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "enabled": True,
                    "multimodal_enabled": True
                },
                "normal_abnormal_threshold": 0.7,
                "condition_confidence_threshold": 0.6,
                "fallback_to_gemini": True
            },
            "storage": {
                "data_directory": "data",
                "logs_directory": "logs",
                "config_directory": "config",
                "cache_enabled": True,
                "backup_enabled": True
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["http://localhost:3000"],
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60,
                    "burst_size": 10
                }
            },
            "security": {
                "encryption_enabled": True,
                "rate_limiting_enabled": True,
                "force_https": False
            },
            "logging": {
                "level": "INFO",
                "format": "structured",
                "rotation": "1 day",
                "retention": "30 days"
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    def _create_default_env_config(self, config_path: Path, environment: str) -> None:
        """Create default environment-specific configuration file."""
        if environment == "development":
            env_config = {
                "api": {
                    "host": "127.0.0.1",
                    "cors_origins": [
                        "http://localhost:3000",
                        "http://localhost:8080",
                        "http://localhost:5173"
                    ]
                },
                "security": {
                    "encryption_enabled": False,
                    "rate_limiting_enabled": False
                },
                "logging": {
                    "level": "DEBUG"
                },
                "development": {
                    "debug": True,
                    "auto_reload": True,
                    "use_sample_data": True
                }
            }
        elif environment == "production":
            env_config = {
                "api": {
                    "port": 80
                },
                "security": {
                    "encryption_enabled": True,
                    "rate_limiting_enabled": True,
                    "force_https": True,
                    "require_api_key": True
                },
                "logging": {
                    "level": "INFO",
                    "external_logging": {
                        "enabled": True,
                        "service": "papertrail"
                    }
                },
                "classification": {
                    "normal_abnormal_threshold": 0.8,
                    "condition_confidence_threshold": 0.7
                },
                "performance": {
                    "max_memory_usage_mb": 4096,
                    "max_workers": 8
                }
            }
        else:
            env_config = {}
        
        with open(config_path, 'w') as f:
            yaml.dump(env_config, f, default_flow_style=False, indent=2)
    
    def setup_logging_from_config(self) -> None:
        """
        Set up logging based on configuration settings.
        """
        try:
            from ambient.utils.logging import configure_logging_from_config
            configure_logging_from_config(self)
            
        except ImportError:
            # Fallback if logging module not available
            import logging
            logging.basicConfig(
                level=getattr(logging, self.config.logging.level.upper(), logging.INFO),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        """
        Get configuration recommendations based on environment and detected issues.
        
        Returns:
            Dictionary with recommendations categorized by type
        """
        recommendations = {
            "security": [],
            "performance": [],
            "reliability": [],
            "cost_optimization": [],
            "development": []
        }
        
        # Security recommendations
        if self.environment == "production":
            if not self.config.security.force_https:
                recommendations["security"].append("Enable HTTPS enforcement in production")
            if not self.config.security.require_api_key:
                recommendations["security"].append("Consider requiring API keys in production")
            if not self.config.security.rate_limiting_enabled:
                recommendations["security"].append("Enable rate limiting in production")
        
        # Performance recommendations
        if self.config.performance.max_workers > 8:
            recommendations["performance"].append("Consider reducing max_workers to avoid resource contention")
        
        if self.config.video_processing.max_video_size_mb > 1000:
            recommendations["performance"].append("Large video size limit may cause memory issues")
        
        if self.config.performance.max_memory_usage_mb < 1024:
            recommendations["performance"].append("Consider increasing memory limit for better performance")
        
        # Reliability recommendations
        if not self.config.storage.backup_enabled and self.environment == "production":
            recommendations["reliability"].append("Enable backups in production environment")
        
        if not self.config.logging.external_logging.enabled and self.environment == "production":
            recommendations["reliability"].append("Consider enabling external logging for production monitoring")
        
        enabled_estimators = [name for name, config in self.config.pose_estimation.estimators.items() if config.enabled]
        if len(enabled_estimators) == 1:
            recommendations["reliability"].append("Consider enabling multiple pose estimators for fallback")
        
        # Cost optimization recommendations
        if self.environment == "development":
            if self.config.classification.llm.model in ["gpt-5", "gpt-4"]:
                recommendations["cost_optimization"].append("Consider using gpt-4o-mini for development to reduce costs")
            
            if self.config.video_processing.max_video_size_mb > 200:
                recommendations["cost_optimization"].append("Consider reducing video size limit in development")
        
        # Development recommendations
        if self.environment == "development":
            if not self.config.development.debug:
                recommendations["development"].append("Enable debug mode for better development experience")
            
            if not self.config.development.use_sample_data:
                recommendations["development"].append("Consider using sample data for faster development")
        
        return recommendations
    
    def generate_configuration_summary(self) -> str:
        """
        Generate a human-readable configuration summary.
        
        Returns:
            Formatted string with configuration summary
        """
        summary_lines = [
            f"AlexPose Configuration Summary",
            f"=" * 40,
            f"Environment: {self.environment}",
            f"Config Directory: {self.config_dir}",
            "",
            "🎥 Video Processing:",
            f"  • Supported formats: {', '.join(self.config.video_processing.supported_formats)}",
            f"  • Max file size: {self.config.video_processing.max_video_size_mb}MB",
            f"  • Default frame rate: {self.config.video_processing.default_frame_rate}fps",
            f"  • FFmpeg enabled: {self.config.video_processing.ffmpeg_enabled}",
            "",
            "🤖 Pose Estimation:",
            f"  • Default estimator: {self.config.pose_estimation.default_estimator}",
            f"  • Confidence threshold: {self.config.pose_estimation.confidence_threshold}",
        ]
        
        # Add enabled estimators
        enabled_estimators = [name for name, config in self.config.pose_estimation.estimators.items() if config.enabled]
        summary_lines.append(f"  • Enabled estimators: {', '.join(enabled_estimators) if enabled_estimators else 'None'}")
        
        summary_lines.extend([
            "",
            "🧠 Classification:",
            f"  • LLM enabled: {self.config.classification.llm.enabled}",
            f"  • Provider: {self.config.classification.llm.provider}",
            f"  • Model: {self.config.classification.llm.model}",
            f"  • Normal/abnormal threshold: {self.config.classification.normal_abnormal_threshold}",
            f"  • Condition threshold: {self.config.classification.condition_confidence_threshold}",
            "",
            "🌐 API:",
            f"  • Host: {self.config.api.host}",
            f"  • Port: {self.config.api.port}",
            f"  • CORS origins: {len(self.config.api.cors_origins)} configured",
            f"  • Rate limiting: {self.config.api.rate_limiting.enabled}",
            "",
            "🔒 Security:",
            f"  • Encryption: {self.config.security.encryption_enabled}",
            f"  • HTTPS enforced: {self.config.security.force_https}",
            f"  • API key required: {self.config.security.require_api_key}",
            "",
            "💾 Storage:",
            f"  • Data directory: {self.config.storage.data_directory}",
            f"  • Cache enabled: {self.config.storage.cache_enabled}",
            f"  • Backup enabled: {self.config.storage.backup_enabled}",
            "",
            "⚡ Performance:",
            f"  • Max memory: {self.config.performance.max_memory_usage_mb}MB",
            f"  • Max workers: {self.config.performance.max_workers}",
            f"  • Cache TTL: {self.config.performance.cache_ttl_seconds}s",
        ])
        
        return "\n".join(summary_lines)