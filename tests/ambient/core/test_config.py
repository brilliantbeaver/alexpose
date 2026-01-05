"""
Comprehensive unit tests for configuration management in ambient.core.config.

Tests the flexible configuration system including YAML loading, environment overrides,
validation, and all configuration data classes.
"""

import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from dataclasses import asdict

from tests.conftest import skip_if_no_ambient
from tests.utils.test_helpers import FileManager, PerformanceProfiler
from tests.utils.assertions import AssertionHelpers

try:
    from ambient.core.config import (
        VideoProcessingConfig, PoseEstimatorConfig, PoseEstimationConfig,
        GaitAnalysisConfig, LLMConfig, ClassificationConfig, StorageConfig,
        RateLimitingConfig, APIConfig, SecurityConfig, ExternalLoggingConfig,
        LoggingConfig, YouTubeConfig, CeleryConfig, TaskTimeoutsConfig,
        BackgroundTasksConfig, DevelopmentConfig, PerformanceConfig,
        AlexPoseConfig, ConfigurationManager
    )
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


@skip_if_no_ambient()
class TestVideoProcessingConfig:
    """Test VideoProcessingConfig data structure."""
    
    def test_video_processing_config_defaults(self):
        """Test VideoProcessingConfig default values."""
        config = VideoProcessingConfig()
        
        assert config.supported_formats == ["mp4", "avi", "mov", "webm"]
        assert config.default_frame_rate == 30.0
        assert config.max_video_size_mb == 500
        assert config.ffmpeg_enabled is True
    
    def test_video_processing_config_custom_values(self):
        """Test VideoProcessingConfig with custom values."""
        config = VideoProcessingConfig(
            supported_formats=["mp4", "mkv"],
            default_frame_rate=60.0,
            max_video_size_mb=1000,
            ffmpeg_enabled=False
        )
        
        assert config.supported_formats == ["mp4", "mkv"]
        assert config.default_frame_rate == 60.0
        assert config.max_video_size_mb == 1000
        assert config.ffmpeg_enabled is False


@skip_if_no_ambient()
class TestPoseEstimatorConfig:
    """Test PoseEstimatorConfig data structure."""
    
    def test_pose_estimator_config_defaults(self):
        """Test PoseEstimatorConfig default values."""
        config = PoseEstimatorConfig()
        
        assert config.enabled is False
        assert config.model_complexity is None
        assert config.min_detection_confidence is None
        assert config.min_tracking_confidence is None
        assert config.model_folder is None
        assert config.net_resolution is None
        assert config.model_name is None
        assert config.device is None
        assert config.detector is None
    
    def test_pose_estimator_config_mediapipe(self):
        """Test PoseEstimatorConfig for MediaPipe."""
        config = PoseEstimatorConfig(
            enabled=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        assert config.enabled is True
        assert config.model_complexity == 1
        assert config.min_detection_confidence == 0.5
        assert config.min_tracking_confidence == 0.5
    
    def test_pose_estimator_config_openpose(self):
        """Test PoseEstimatorConfig for OpenPose."""
        config = PoseEstimatorConfig(
            enabled=True,
            model_folder="/path/to/models",
            net_resolution="368x368",
            model_name="BODY_25"
        )
        
        assert config.enabled is True
        assert config.model_folder == "/path/to/models"
        assert config.net_resolution == "368x368"
        assert config.model_name == "BODY_25"


@skip_if_no_ambient()
class TestLLMConfig:
    """Test LLMConfig data structure and methods."""
    
    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        config = LLMConfig()
        
        assert config.provider == "openai"
        assert config.model == "gpt-5.2"
        assert config.temperature == 0.1
        assert config.max_tokens == 2000
        assert config.enabled is True
        assert config.multimodal_enabled is True
        assert config.models_config_file is None
        assert config.models == {}
        assert config.presets == {}
        assert config.providers == {}
        assert config.default_priority_order == []
    
    def test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-pro",
            temperature=0.3,
            max_tokens=1000,
            enabled=False,
            multimodal_enabled=False
        )
        
        assert config.provider == "gemini"
        assert config.model == "gemini-pro"
        assert config.temperature == 0.3
        assert config.max_tokens == 1000
        assert config.enabled is False
        assert config.multimodal_enabled is False
    
    def test_llm_config_get_model_spec_legacy(self):
        """Test get_model_spec with legacy model capabilities."""
        config = LLMConfig(model="gpt-4o")
        
        spec = config.get_model_spec()
        
        assert spec["text"] is True
        assert spec["images"] is True
        assert spec["max_tokens"] == 4096
    
    def test_llm_config_get_model_spec_external(self):
        """Test get_model_spec with external models configuration."""
        external_models = {
            "custom-model": {
                "provider": "openai",
                "text": True,
                "images": True,
                "video": True,
                "max_tokens": 8192,
                "cost_tier": "premium"
            }
        }
        
        config = LLMConfig(model="custom-model", models=external_models)
        
        spec = config.get_model_spec()
        
        assert spec["provider"] == "openai"
        assert spec["video"] is True
        assert spec["max_tokens"] == 8192
        assert spec["cost_tier"] == "premium"
    
    def test_llm_config_supports_images(self):
        """Test supports_images method."""
        # Model with explicit image support
        config1 = LLMConfig(model="gpt-4o")
        assert config1.supports_images() is True
        
        # Model with multimodal support
        config2 = LLMConfig(
            model="custom-multimodal",
            models={"custom-multimodal": {"multimodal": True}}
        )
        assert config2.supports_images() is True
        
        # Model without image support
        config3 = LLMConfig(model="gpt-3.5-turbo")
        assert config3.supports_images() is False
    
    def test_llm_config_supports_video(self):
        """Test supports_video method."""
        # Model with video support
        config1 = LLMConfig(model="gpt-5.2")
        assert config1.supports_video() is True
        
        # Model without video support
        config2 = LLMConfig(model="gpt-4")
        assert config2.supports_video() is False
    
    def test_llm_config_get_max_tokens(self):
        """Test get_max_tokens method."""
        config = LLMConfig(model="gpt-4o", max_tokens=2000)
        
        # Should return model's max tokens, not config max_tokens
        assert config.get_max_tokens() == 4096
    
    def test_llm_config_is_model_supported(self):
        """Test is_model_supported method."""
        # Supported model
        config1 = LLMConfig(provider="openai", model="gpt-4o")
        assert config1.is_model_supported() is True
        
        # Unsupported model
        config2 = LLMConfig(provider="openai", model="unsupported-model")
        assert config2.is_model_supported() is False
        
        # External model
        config3 = LLMConfig(
            provider="openai",
            model="custom-model",
            models={"custom-model": {"provider": "openai"}}
        )
        assert config3.is_model_supported() is True
    
    def test_llm_config_get_available_models(self):
        """Test get_available_models method."""
        config = LLMConfig(provider="openai")
        
        models = config.get_available_models()
        
        assert "gpt-4o" in models
        assert "gpt-3.5-turbo" in models
        assert len(models) > 5  # Should have multiple models
    
    def test_llm_config_get_models_by_cost_tier(self):
        """Test get_models_by_cost_tier method."""
        external_models = {
            "cheap-model": {"cost_tier": "low"},
            "expensive-model": {"cost_tier": "premium"},
            "medium-model": {"cost_tier": "medium"}
        }
        
        config = LLMConfig(models=external_models)
        
        low_cost_models = config.get_models_by_cost_tier("low")
        premium_models = config.get_models_by_cost_tier("premium")
        
        assert "cheap-model" in low_cost_models
        assert "expensive-model" in premium_models
        assert len(low_cost_models) == 1
        assert len(premium_models) == 1
    
    def test_llm_config_get_preset_models(self):
        """Test get_preset_models method."""
        presets = {
            "cost_optimized": {
                "preferred_models": ["gpt-4o-mini", "gemini-pro"]
            },
            "performance_optimized": {
                "preferred_models": ["gpt-5.2", "gemini-3-pro-preview"]
            }
        }
        
        config = LLMConfig(presets=presets)
        
        cost_models = config.get_preset_models("cost_optimized")
        perf_models = config.get_preset_models("performance_optimized")
        
        assert "gpt-4o-mini" in cost_models
        assert "gpt-5.2" in perf_models
        assert len(cost_models) == 2
        assert len(perf_models) == 2
    
    def test_llm_config_get_api_config(self):
        """Test get_api_config method."""
        config = LLMConfig(provider="openai")
        
        api_config = config.get_api_config()
        
        assert "api_key_env_var" in api_config
        assert api_config["api_key_env_var"] == "OPENAI_API_KEY"
        assert "timeout" in api_config
        assert "max_retries" in api_config


@skip_if_no_ambient()
class TestConfigurationManager:
    """Test ConfigurationManager functionality."""
    
    def test_configuration_manager_initialization(self, tmp_path):
        """Test ConfigurationManager initialization."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create minimal config file
        config_file = config_dir / "alexpose.yaml"
        config_data = {
            "video_processing": {
                "supported_formats": ["mp4", "avi"]
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir, environment="test")
        
        assert manager.config_dir == config_dir
        assert manager.environment == "test"
        assert isinstance(manager.config, AlexPoseConfig)
        assert manager.config.video_processing.supported_formats == ["mp4", "avi"]
    
    def test_configuration_manager_environment_override(self, tmp_path):
        """Test environment-specific configuration override."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create main config
        main_config = config_dir / "alexpose.yaml"
        main_data = {
            "api": {"port": 8000},
            "logging": {"level": "INFO"}
        }
        with open(main_config, 'w') as f:
            yaml.dump(main_data, f)
        
        # Create environment override
        env_config = config_dir / "test.yaml"
        env_data = {
            "api": {"port": 9000},
            "logging": {"level": "DEBUG"}
        }
        with open(env_config, 'w') as f:
            yaml.dump(env_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir, environment="test")
        
        assert manager.config.api.port == 9000  # Overridden
        assert manager.config.logging.level == "DEBUG"  # Overridden
    
    def test_configuration_manager_get_config_value(self, tmp_path):
        """Test get_config_value with dot notation."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        config_data = {
            "classification": {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o"
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir)
        
        # Test dot notation access
        assert manager.get_config_value("classification.llm.provider") == "openai"
        assert manager.get_config_value("classification.llm.model") == "gpt-4o"
        
        # Test default value
        assert manager.get_config_value("nonexistent.key", "default") == "default"
        
        # Test type conversion
        config_data["api"] = {"port": "9000"}
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        manager.reload_configuration()
        
        port = manager.get_config_value("api.port", target_type=int)
        assert port == 9000
        assert isinstance(port, int)
    
    def test_configuration_manager_api_key_methods(self):
        """Test API key retrieval methods."""
        manager = ConfigurationManager()
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123", "GOOGLE_API_KEY": "gemini-test456"}):
            assert manager.get_openai_api_key() == "sk-test123"
            assert manager.get_gemini_api_key() == "gemini-test456"
            assert manager.is_openai_configured() is True
            assert manager.is_gemini_configured() is True
        
        with patch.dict(os.environ, {}, clear=True):
            assert manager.get_openai_api_key() == ""
            assert manager.get_gemini_api_key() == ""
            assert manager.is_openai_configured() is False
            assert manager.is_gemini_configured() is False
    
    def test_configuration_manager_directory_methods(self, tmp_path):
        """Test directory path methods."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        config_data = {
            "storage": {
                "videos_directory": "data/videos",
                "analysis_directory": "data/analysis"
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir)
        
        videos_dir = manager.get_videos_directory()
        openpose_dir = manager.get_openpose_directory()
        
        assert videos_dir == Path("data/videos")
        assert openpose_dir == Path("data/analysis/openpose")
    
    def test_configuration_manager_reload(self, tmp_path):
        """Test configuration reload functionality."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        
        # Initial config with complete configuration to pass validation
        config_data = {
            "api": {"port": 8000},
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": True,
                        "model_complexity": 1,
                        "min_detection_confidence": 0.5,
                        "min_tracking_confidence": 0.5
                    }
                },
                "default_estimator": "mediapipe",
                "confidence_threshold": 0.5
            },
            "classification": {
                "llm": {
                    "enabled": False  # Disable to avoid API key validation
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir)
        assert manager.config.api.port == 8000
        
        # Update config file
        config_data["api"]["port"] = 9000
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Reload and verify
        success = manager.reload_configuration()
        assert success is True
        assert manager.config.api.port == 9000
    
    def test_configuration_manager_environment_switching(self, tmp_path):
        """Test switching between environments."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create main config with complete configuration
        main_config = config_dir / "alexpose.yaml"
        main_data = {
            "api": {"port": 8000},
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": True,
                        "model_complexity": 1,
                        "min_detection_confidence": 0.5,
                        "min_tracking_confidence": 0.5
                    }
                },
                "default_estimator": "mediapipe",
                "confidence_threshold": 0.5
            },
            "classification": {
                "llm": {
                    "enabled": False  # Disable to avoid API key validation
                }
            }
        }
        with open(main_config, 'w') as f:
            yaml.dump(main_data, f)
        
        # Create development config
        dev_config = config_dir / "development.yaml"
        dev_data = {"api": {"port": 3000}}
        with open(dev_config, 'w') as f:
            yaml.dump(dev_data, f)
        
        # Create production config
        prod_config = config_dir / "production.yaml"
        prod_data = {
            "api": {"port": 8080},  # Use 8080 instead of 80 to avoid privilege warnings
            "security": {
                "force_https": True,  # Enable HTTPS for production
                "require_api_key": True
            }
        }
        with open(prod_config, 'w') as f:
            yaml.dump(prod_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir, environment="development")
        assert manager.config.api.port == 3000
        
        # Switch to production
        success = manager.load_environment_config("production")
        assert success is True
        assert manager.environment == "production"
        assert manager.config.api.port == 8080
        
        # Switch back to development
        success = manager.load_environment_config("development")
        assert success is True
        assert manager.environment == "development"
        assert manager.config.api.port == 3000
    
    def test_configuration_manager_get_available_environments(self, tmp_path):
        """Test getting available environments."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create environment configs
        for env in ["development", "production", "staging", "custom"]:
            env_file = config_dir / f"{env}.yaml"
            with open(env_file, 'w') as f:
                yaml.dump({"api": {"port": 8000}}, f)
        
        # Create non-environment files
        (config_dir / "alexpose.yaml").touch()
        (config_dir / "classification_prompts.yaml").touch()
        
        manager = ConfigurationManager(config_dir=config_dir)
        environments = manager.get_available_environments()
        
        assert "development" in environments
        assert "production" in environments
        assert "staging" in environments
        assert "custom" in environments
        assert "alexpose" not in environments
        assert "classification_prompts" not in environments
    
    def test_configuration_manager_get_environment_info(self, tmp_path):
        """Test getting environment information."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create config files
        (config_dir / "alexpose.yaml").touch()
        (config_dir / "development.yaml").touch()
        
        # Explicitly control both API keys - set OpenAI and clear Gemini
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=False):
                manager = ConfigurationManager(config_dir=config_dir, environment="development")
                
                info = manager.get_environment_info()
                
                assert info["environment"] == "development"
                assert info["config_directory"] == str(config_dir)
                assert info["config_files"]["main_exists"] is True
                assert info["config_files"]["environment_exists"] is True
                assert info["api_keys_configured"]["openai"] is True
                assert info["api_keys_configured"]["gemini"] is False
                assert "llm_classification" in info["enabled_features"]
                assert "pose_estimators" in info


@skip_if_no_ambient()
class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def test_validate_video_processing_config(self, tmp_path):
        """Test video processing configuration validation."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Valid config with complete pose estimation configuration to avoid validation errors
        config_file = config_dir / "alexpose.yaml"
        valid_config = {
            "video_processing": {
                "supported_formats": ["mp4", "avi"],
                "default_frame_rate": 30.0,
                "max_video_size_mb": 500
            },
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": True,
                        "model_complexity": 1,
                        "min_detection_confidence": 0.5,
                        "min_tracking_confidence": 0.5
                    }
                },
                "default_estimator": "mediapipe",
                "confidence_threshold": 0.5
            },
            "classification": {
                "llm": {
                    "enabled": False  # Disable LLM to avoid API key validation
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        manager = ConfigurationManager(config_dir=config_dir)
        assert manager.validate_configuration() is True
        
        # Test invalid config - negative frame rate
        invalid_config = {
            "video_processing": {
                "supported_formats": ["mp4"],
                "default_frame_rate": -30.0,
                "max_video_size_mb": 500
            },
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": True,
                        "model_complexity": 1
                    }
                },
                "default_estimator": "mediapipe",
                "confidence_threshold": 0.5
            },
            "classification": {
                "llm": {
                    "enabled": False
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        manager.reload_configuration()
        assert manager.validate_configuration() is False
    
    def test_validate_pose_estimation_config(self, tmp_path):
        """Test pose estimation configuration validation."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        
        # Valid config
        valid_config = {
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": True,
                        "model_complexity": 1,
                        "min_detection_confidence": 0.5
                    }
                },
                "default_estimator": "mediapipe",
                "confidence_threshold": 0.7
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        manager = ConfigurationManager(config_dir=config_dir)
        assert manager.validate_configuration() is True
        
        # Invalid config - default estimator not in estimators
        invalid_config = {
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {"enabled": True}
                },
                "default_estimator": "nonexistent",
                "confidence_threshold": 0.7
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        manager.reload_configuration()
        assert manager.validate_configuration() is False
    
    def test_validate_llm_config(self, tmp_path):
        """Test LLM configuration validation."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        
        # Valid config with API key and complete pose estimation config
        valid_config = {
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": True,
                        "model_complexity": 1,
                        "min_detection_confidence": 0.5,
                        "min_tracking_confidence": 0.5
                    }
                },
                "default_estimator": "mediapipe",
                "confidence_threshold": 0.5
            },
            "classification": {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "enabled": True
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            manager = ConfigurationManager(config_dir=config_dir)
            assert manager.validate_configuration() is True
        
        # Invalid config - no API key
        with patch.dict(os.environ, {}, clear=True):
            manager.reload_configuration()
            assert manager.validate_configuration() is False
        
        # Invalid config - unsupported model
        invalid_config = {
            "classification": {
                "llm": {
                    "provider": "openai",
                    "model": "nonexistent-model",
                    "enabled": True
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            manager.reload_configuration()
            assert manager.validate_configuration() is False
    
    def test_validate_api_config(self, tmp_path):
        """Test API configuration validation."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        
        # Valid config with complete pose estimation config
        valid_config = {
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": True,
                        "model_complexity": 1,
                        "min_detection_confidence": 0.5,
                        "min_tracking_confidence": 0.5
                    }
                },
                "default_estimator": "mediapipe",
                "confidence_threshold": 0.5
            },
            "classification": {
                "llm": {
                    "enabled": False  # Disable to avoid API key validation
                }
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
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        manager = ConfigurationManager(config_dir=config_dir)
        assert manager.validate_configuration() is True
        
        # Invalid config - port out of range
        invalid_config = {
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": True,
                        "model_complexity": 1,
                        "min_detection_confidence": 0.5,
                        "min_tracking_confidence": 0.5
                    }
                },
                "default_estimator": "mediapipe",
                "confidence_threshold": 0.5
            },
            "classification": {
                "llm": {
                    "enabled": False
                }
            },
            "api": {
                "port": 70000,
                "rate_limiting": {
                    "requests_per_minute": -10
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        manager.reload_configuration()
        assert manager.validate_configuration() is False
    
    def test_get_validation_report(self, tmp_path):
        """Test getting detailed validation report."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        config_data = {
            "video_processing": {
                "default_frame_rate": -30.0  # Invalid
            },
            "api": {
                "port": 70000  # Invalid
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir)
        report = manager.get_validation_report()
        
        assert report["valid"] is False
        assert report["error_count"] > 0
        assert len(report["errors"]) > 0
        assert "environment" in report
        assert "config_files_loaded" in report
        assert "validation_categories" in report


@skip_if_no_ambient()
class TestConfigurationRecommendations:
    """Test configuration recommendations functionality."""
    
    def test_get_configuration_recommendations_development(self, tmp_path):
        """Test configuration recommendations for development environment."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        config_data = {
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": False  # Should be enabled for development
                    }
                },
                "default_estimator": "mediapipe"
            },
            "classification": {
                "llm": {
                    "model": "gpt-4",  # Expensive model
                    "enabled": False  # Disable to avoid API key validation
                }
            },
            "development": {
                "debug": False  # Should be True in development
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir, environment="development")
        recommendations = manager.get_configuration_recommendations()
        
        assert "best_practices" in recommendations
        assert len(recommendations["best_practices"]) > 0
    
    def test_get_configuration_recommendations_production(self, tmp_path):
        """Test configuration recommendations for production environment."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        config_data = {
            "security": {
                "force_https": False,  # Should be True in production
                "require_api_key": False  # Should be True in production
            },
            "storage": {
                "backup_enabled": False  # Should be True in production
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir, environment="production")
        recommendations = manager.get_configuration_recommendations()
        
        assert "security" in recommendations
        assert "reliability" in recommendations
        assert len(recommendations["security"]) > 0
        assert len(recommendations["reliability"]) > 0
    
    def test_generate_configuration_summary(self, tmp_path):
        """Test generating configuration summary."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        config_data = {
            "video_processing": {
                "supported_formats": ["mp4", "avi"],
                "max_video_size_mb": 1000
            },
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {"enabled": True},
                    "openpose": {"enabled": False}
                },
                "default_estimator": "mediapipe"
            },
            "classification": {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "enabled": True
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir, environment="test")
        summary = manager.generate_configuration_summary()
        
        assert "AlexPose Configuration Summary" in summary
        assert "Environment: test" in summary
        assert "🎥 Video Processing:" in summary
        assert "🤖 Pose Estimation:" in summary
        assert "🧠 Classification:" in summary
        assert "mp4, avi" in summary
        assert "mediapipe" in summary
        assert "gpt-4o" in summary


@skip_if_no_ambient()
class TestConfigurationFileManagement:
    """Test configuration file management functionality."""
    
    def test_create_default_configuration_files(self, tmp_path):
        """Test creating default configuration files."""
        config_dir = tmp_path / "config"
        
        manager = ConfigurationManager(config_dir=config_dir)
        success = manager.create_default_configuration_files()
        
        assert success is True
        assert (config_dir / "alexpose.yaml").exists()
        assert (config_dir / "development.yaml").exists()
        assert (config_dir / "production.yaml").exists()
        
        # Verify content of main config
        with open(config_dir / "alexpose.yaml", 'r') as f:
            main_config = yaml.safe_load(f)
        
        assert "video_processing" in main_config
        assert "pose_estimation" in main_config
        assert "classification" in main_config
        assert main_config["pose_estimation"]["estimators"]["mediapipe"]["enabled"] is True
    
    def test_create_environment_config(self, tmp_path):
        """Test creating environment-specific configuration."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = ConfigurationManager(config_dir=config_dir)
        
        custom_config = {
            "api": {"port": 9000},
            "logging": {"level": "WARNING"}
        }
        
        success = manager.create_environment_config("custom", custom_config)
        assert success is True
        
        custom_file = config_dir / "custom.yaml"
        assert custom_file.exists()
        
        with open(custom_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config["api"]["port"] == 9000
        assert loaded_config["logging"]["level"] == "WARNING"
        
        # Test creating duplicate environment config
        success = manager.create_environment_config("custom", custom_config)
        assert success is False  # Should fail because file already exists


@pytest.mark.performance
@skip_if_no_ambient()
class TestConfigurationPerformance:
    """Test configuration system performance."""
    
    def test_configuration_loading_performance(self, tmp_path):
        """Test configuration loading performance."""
        profiler = PerformanceProfiler()
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create large configuration file
        config_file = config_dir / "alexpose.yaml"
        large_config = {
            "pose_estimation": {
                "estimators": {
                    f"estimator_{i}": {
                        "enabled": i % 2 == 0,
                        "model_complexity": i % 3,
                        "parameters": {f"param_{j}": j * 0.1 for j in range(50)}
                    }
                    for i in range(100)
                }
            },
            "classification": {
                "llm": {
                    "models": {
                        f"model_{i}": {
                            "provider": "openai" if i % 2 == 0 else "gemini",
                            "max_tokens": 1000 + i * 100,
                            "capabilities": [f"cap_{j}" for j in range(10)]
                        }
                        for i in range(50)
                    }
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(large_config, f)
        
        with profiler.profile("load_large_config"):
            manager = ConfigurationManager(config_dir=config_dir)
        
        with profiler.profile("validate_large_config"):
            is_valid = manager.validate_configuration()
        
        with profiler.profile("reload_large_config"):
            manager.reload_configuration()
        
        load_time = profiler.get_metrics("load_large_config")["execution_time"]
        validate_time = profiler.get_metrics("validate_large_config")["execution_time"]
        reload_time = profiler.get_metrics("reload_large_config")["execution_time"]
        
        # Performance assertions - more realistic expectations for complex configs
        assert load_time < 10.0  # Should load in under 10 seconds
        assert validate_time < 5.0  # Should validate in under 5 seconds
        assert reload_time < 10.0  # Should reload in under 10 seconds
        assert is_valid is True
    
    def test_configuration_access_performance(self, tmp_path):
        """Test configuration value access performance."""
        profiler = PerformanceProfiler()
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "alexpose.yaml"
        config_data = {
            "deeply": {
                "nested": {
                    "configuration": {
                        "values": {
                            "test_value": "performance_test"
                        }
                    }
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir)
        
        # Test repeated access performance
        with profiler.profile("repeated_config_access"):
            for _ in range(1000):
                value = manager.get_config_value("deeply.nested.configuration.values.test_value")
                assert value == "performance_test"
        
        access_time = profiler.get_metrics("repeated_config_access")["execution_time"]
        
        # Should be able to access config values very quickly
        assert access_time < 0.1  # 1000 accesses in under 100ms
        
        # Test average access time
        avg_access_time = access_time / 1000
        assert avg_access_time < 0.0001  # Less than 0.1ms per access


@skip_if_no_ambient()
class TestConfigurationIntegration:
    """Test configuration system integration scenarios."""
    
    def test_external_llm_models_config_loading(self, tmp_path):
        """Test loading external LLM models configuration."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create main config referencing external models file
        main_config = config_dir / "alexpose.yaml"
        main_data = {
            "classification": {
                "llm": {
                    "provider": "openai",
                    "model": "custom-model",
                    "models_config_file": "llm_models.yaml"
                }
            }
        }
        with open(main_config, 'w') as f:
            yaml.dump(main_data, f)
        
        # Create external models config
        models_config = config_dir / "llm_models.yaml"
        models_data = {
            "models": {
                "custom-model": {
                    "provider": "openai",
                    "text": True,
                    "images": True,
                    "video": False,
                    "max_tokens": 4096,
                    "cost_tier": "medium"
                }
            },
            "presets": {
                "balanced": {
                    "preferred_models": ["custom-model"]
                }
            },
            "providers": {
                "openai": {
                    "api_key_env_var": "OPENAI_API_KEY",
                    "timeout": 60
                }
            }
        }
        with open(models_config, 'w') as f:
            yaml.dump(models_data, f)
        
        manager = ConfigurationManager(config_dir=config_dir)
        
        # Verify external models were loaded
        llm_config = manager.config.classification.llm
        assert "custom-model" in llm_config.models
        assert llm_config.models["custom-model"]["cost_tier"] == "medium"
        assert "balanced" in llm_config.presets
        assert "openai" in llm_config.providers
        
        # Test model spec retrieval
        spec = llm_config.get_model_spec()
        assert spec["provider"] == "openai"
        assert spec["max_tokens"] == 4096
        assert spec["cost_tier"] == "medium"
    
    def test_multi_environment_workflow(self, tmp_path):
        """Test complete multi-environment configuration workflow."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create base configuration
        base_config = {
            "video_processing": {
                "supported_formats": ["mp4", "avi"],
                "max_video_size_mb": 500
            },
            "classification": {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "enabled": True
                }
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "security": {
                "encryption_enabled": True,
                "force_https": False
            }
        }
        
        main_config = config_dir / "alexpose.yaml"
        with open(main_config, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create development overrides
        dev_config = config_dir / "development.yaml"
        dev_overrides = {
            "api": {"host": "127.0.0.1", "port": 3000},
            "security": {"encryption_enabled": False},
            "classification": {"llm": {"model": "gpt-4o-mini"}},  # Cheaper for dev
            "logging": {"level": "DEBUG"}
        }
        with open(dev_config, 'w') as f:
            yaml.dump(dev_overrides, f)
        
        # Create production overrides
        prod_config = config_dir / "production.yaml"
        prod_overrides = {
            "api": {"port": 80},
            "security": {"force_https": True, "require_api_key": True},
            "classification": {"llm": {"model": "gpt-4o"}},  # Better model for prod
            "logging": {"level": "INFO"},
            "performance": {"max_memory_usage_mb": 4096}
        }
        with open(prod_config, 'w') as f:
            yaml.dump(prod_overrides, f)
        
        # Test development environment
        dev_manager = ConfigurationManager(config_dir=config_dir, environment="development")
        assert dev_manager.config.api.host == "127.0.0.1"
        assert dev_manager.config.api.port == 3000
        assert dev_manager.config.security.encryption_enabled is False
        assert dev_manager.config.classification.llm.model == "gpt-4o-mini"
        assert dev_manager.config.logging.level == "DEBUG"
        
        # Test production environment
        prod_manager = ConfigurationManager(config_dir=config_dir, environment="production")
        assert prod_manager.config.api.host == "0.0.0.0"  # From base config
        assert prod_manager.config.api.port == 80
        assert prod_manager.config.security.force_https is True
        assert prod_manager.config.security.require_api_key is True
        assert prod_manager.config.classification.llm.model == "gpt-4o"
        assert prod_manager.config.logging.level == "INFO"
        assert prod_manager.config.performance.max_memory_usage_mb == 4096
        
        # Test environment switching
        success = dev_manager.load_environment_config("production")
        assert success is True
        assert dev_manager.environment == "production"
        assert dev_manager.config.api.port == 80
        
        # Test getting available environments
        environments = dev_manager.get_available_environments()
        assert "development" in environments
        assert "production" in environments
    
    def test_configuration_validation_integration(self, tmp_path):
        """Test comprehensive configuration validation integration."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create configuration with various validation scenarios
        config_file = config_dir / "alexpose.yaml"
        config_data = {
            "video_processing": {
                "supported_formats": ["mp4", "unknown_format"],  # Warning
                "default_frame_rate": 30.0,  # Valid
                "max_video_size_mb": 2000,  # Warning - large size
                "ffmpeg_enabled": True
            },
            "pose_estimation": {
                "estimators": {
                    "mediapipe": {
                        "enabled": True,
                        "model_complexity": 1,  # Valid
                        "min_detection_confidence": 0.5
                    },
                    "openpose": {
                        "enabled": False,
                        "model_folder": "/nonexistent/path"  # Warning
                    }
                },
                "default_estimator": "mediapipe",  # Valid
                "confidence_threshold": 0.7
            },
            "classification": {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "temperature": 0.1,  # Valid
                    "max_tokens": 1000,
                    "enabled": True
                },
                "normal_abnormal_threshold": 0.7,  # Valid
                "condition_confidence_threshold": 0.6  # Valid
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,  # Valid
                "cors_origins": ["localhost:3000"],  # Warning - no protocol
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60,  # Valid
                    "burst_size": 10
                }
            },
            "security": {
                "encryption_enabled": True,
                "force_https": False,  # Warning in production
                "rate_limiting_enabled": True
            },
            "performance": {
                "max_memory_usage_mb": 1024,  # Valid
                "max_workers": 4,  # Valid
                "cache_ttl_seconds": 3600
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test with API key configured
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            manager = ConfigurationManager(config_dir=config_dir, environment="production")
            
            # Get validation report
            report = manager.get_validation_report()
            
            # Should have warnings but no errors
            assert report["error_count"] == 0
            assert report["warning_count"] > 0
            assert report["valid"] is True
            
            # Check specific warnings
            warnings = report["warnings"]
            warning_text = " ".join(warnings)
            assert "unknown_format" in warning_text or "unsupported" in warning_text.lower()
            assert "large" in warning_text.lower() or "2000" in warning_text
            assert "protocol" in warning_text.lower() or "cors" in warning_text.lower()
        
        # Test without API key - should have errors
        with patch.dict(os.environ, {}, clear=True):
            manager.reload_configuration()
            report = manager.get_validation_report()
            
            assert report["error_count"] > 0
            assert report["valid"] is False
            
            errors = report["errors"]
            error_text = " ".join(errors)
            assert "api key" in error_text.lower() or "openai" in error_text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])