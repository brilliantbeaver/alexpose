"""
Unit tests for enhanced configuration validation with error reporting.
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
import yaml

from ambient.core.config import ConfigurationManager


class TestConfigurationValidation:
    """Test enhanced configuration validation functionality."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_dir = Path(temp_dir) / "config"
            temp_config_dir.mkdir()
            
            # Create main config
            main_config = {
                "video_processing": {
                    "supported_formats": ["mp4", "avi", "mov", "webm"],
                    "default_frame_rate": 30.0,
                    "max_video_size_mb": 500,
                    "ffmpeg_enabled": True
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
                "storage": {
                    "data_directory": "data",
                    "logs_directory": "logs",
                    "config_directory": "config",
                    "cache_enabled": True,
                    "backup_enabled": True
                },
                "performance": {
                    "max_memory_usage_mb": 2048,
                    "max_workers": 4,
                    "cache_ttl_seconds": 3600,
                    "db_pool_size": 10,
                    "db_max_overflow": 20
                }
            }
            with open(temp_config_dir / "alexpose.yaml", 'w') as f:
                yaml.dump(main_config, f)
            
            yield temp_config_dir
    
    def test_basic_validation_passes(self, temp_config_dir):
        """Test that basic valid configuration passes validation."""
        # Set a dummy API key to avoid validation errors
        os.environ["OPENAI_API_KEY"] = "sk-test-key-for-validation-testing-12345678901234567890"
        
        try:
            config_manager = ConfigurationManager(config_dir=temp_config_dir, environment="development")
            
            # Validation should pass
            is_valid = config_manager.validate_configuration()
            assert is_valid is True
            
            # Get validation report
            report = config_manager.get_validation_report()
            assert report["valid"] is True
            assert report["error_count"] == 0
            assert "validation_categories" in report
            
        finally:
            # Clean up environment variable
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
    
    def test_invalid_video_processing_config(self, temp_config_dir):
        """Test validation of invalid video processing configuration."""
        config_manager = ConfigurationManager(config_dir=temp_config_dir)
        
        # Set invalid values
        config_manager.config.video_processing.default_frame_rate = -10
        config_manager.config.video_processing.max_video_size_mb = 0
        config_manager.config.video_processing.supported_formats = []
        
        is_valid = config_manager.validate_configuration()
        assert is_valid is False
        
        report = config_manager.get_validation_report()
        assert report["error_count"] > 0
        assert any("frame rate must be positive" in error for error in report["errors"])
        assert any("video size must be positive" in error for error in report["errors"])
        assert any("No supported video formats" in error for error in report["errors"])
    
    def test_invalid_pose_estimation_config(self, temp_config_dir):
        """Test validation of invalid pose estimation configuration."""
        config_manager = ConfigurationManager(config_dir=temp_config_dir)
        
        # Set invalid values
        config_manager.config.pose_estimation.confidence_threshold = 1.5
        config_manager.config.pose_estimation.default_estimator = "nonexistent"
        
        is_valid = config_manager.validate_configuration()
        assert is_valid is False
        
        report = config_manager.get_validation_report()
        assert report["error_count"] > 0
        assert any("confidence threshold must be between 0 and 1" in error for error in report["errors"])
        assert any("Default estimator 'nonexistent' not found" in error for error in report["errors"])
    
    def test_invalid_classification_config(self, temp_config_dir):
        """Test validation of invalid classification configuration."""
        config_manager = ConfigurationManager(config_dir=temp_config_dir)
        
        # Set invalid values
        config_manager.config.classification.normal_abnormal_threshold = 1.5
        config_manager.config.classification.llm.provider = "invalid_provider"
        config_manager.config.classification.llm.temperature = 3.0
        config_manager.config.classification.llm.max_tokens = -100
        
        is_valid = config_manager.validate_configuration()
        assert is_valid is False
        
        report = config_manager.get_validation_report()
        assert report["error_count"] > 0
        assert any("threshold must be between 0 and 1" in error for error in report["errors"])
        assert any("Unsupported LLM provider" in error for error in report["errors"])
        assert any("temperature must be between 0 and 2" in error for error in report["errors"])
        assert any("max_tokens must be positive" in error for error in report["errors"])
    
    def test_invalid_api_config(self, temp_config_dir):
        """Test validation of invalid API configuration."""
        config_manager = ConfigurationManager(config_dir=temp_config_dir)
        
        # Set invalid values
        config_manager.config.api.port = 70000  # Invalid port
        config_manager.config.api.rate_limiting.requests_per_minute = -10
        config_manager.config.api.rate_limiting.burst_size = 0
        
        is_valid = config_manager.validate_configuration()
        assert is_valid is False
        
        report = config_manager.get_validation_report()
        assert report["error_count"] > 0
        assert any("port must be between 1 and 65535" in error for error in report["errors"])
        assert any("requests_per_minute must be positive" in error for error in report["errors"])
        assert any("burst_size must be positive" in error for error in report["errors"])
    
    def test_missing_api_key_validation(self, temp_config_dir):
        """Test validation of missing API keys."""
        # Ensure no API keys are set
        for key in ["OPENAI_API_KEY", "GEMINI_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
        
        config_manager = ConfigurationManager(config_dir=temp_config_dir)
        config_manager.config.classification.llm.enabled = True
        config_manager.config.classification.llm.provider = "openai"
        
        is_valid = config_manager.validate_configuration()
        assert is_valid is False
        
        report = config_manager.get_validation_report()
        assert any("OpenAI API key not configured" in error for error in report["errors"])
    
    def test_api_key_format_validation(self, temp_config_dir):
        """Test validation of API key formats."""
        # Set invalid OpenAI API key format
        os.environ["OPENAI_API_KEY"] = "invalid-key"
        
        try:
            config_manager = ConfigurationManager(config_dir=temp_config_dir)
            config_manager.config.classification.llm.enabled = True
            config_manager.config.classification.llm.provider = "openai"
            
            is_valid = config_manager.validate_configuration()
            # Should still be valid but with warnings
            
            report = config_manager.get_validation_report()
            assert any("API key format appears invalid" in warning for warning in report["warnings"])
            
        finally:
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
    
    def test_performance_config_validation(self, temp_config_dir):
        """Test validation of performance configuration."""
        config_manager = ConfigurationManager(config_dir=temp_config_dir)
        
        # Set invalid values
        config_manager.config.performance.max_memory_usage_mb = -100
        config_manager.config.performance.max_workers = 0
        config_manager.config.performance.cache_ttl_seconds = -10
        config_manager.config.performance.db_pool_size = -5
        config_manager.config.performance.db_max_overflow = -1
        
        is_valid = config_manager.validate_configuration()
        assert is_valid is False
        
        report = config_manager.get_validation_report()
        assert report["error_count"] >= 5  # Should have multiple errors
        assert any("memory usage must be positive" in error for error in report["errors"])
        assert any("workers must be positive" in error for error in report["errors"])
        assert any("Cache TTL must be positive" in error for error in report["errors"])
        assert any("pool size must be positive" in error for error in report["errors"])
        assert any("overflow cannot be negative" in error for error in report["errors"])
    
    def test_configuration_recommendations(self, temp_config_dir):
        """Test configuration recommendations generation."""
        config_manager = ConfigurationManager(config_dir=temp_config_dir, environment="production")
        
        # Set some suboptimal values for production
        config_manager.config.security.force_https = False
        config_manager.config.security.require_api_key = False
        config_manager.config.storage.backup_enabled = False
        
        recommendations = config_manager.get_configuration_recommendations()
        
        assert "security" in recommendations
        assert "reliability" in recommendations
        assert len(recommendations["security"]) > 0
        assert len(recommendations["reliability"]) > 0
        
        # Check specific recommendations
        security_recs = recommendations["security"]
        assert any("HTTPS enforcement" in rec for rec in security_recs)
        assert any("API keys" in rec for rec in security_recs)
        
        reliability_recs = recommendations["reliability"]
        assert any("backups" in rec for rec in reliability_recs)
    
    def test_configuration_summary_generation(self, temp_config_dir):
        """Test configuration summary generation."""
        config_manager = ConfigurationManager(config_dir=temp_config_dir)
        
        summary = config_manager.generate_configuration_summary()
        
        assert "AlexPose Configuration Summary" in summary
        assert f"Environment: {config_manager.environment}" in summary
        assert "Video Processing:" in summary
        assert "Pose Estimation:" in summary
        assert "Classification:" in summary
        assert "API:" in summary
        assert "Security:" in summary
        assert "Storage:" in summary
        assert "Performance:" in summary
    
    def test_validation_with_warnings_only(self, temp_config_dir):
        """Test validation that passes but generates warnings."""
        # Set a valid API key
        os.environ["OPENAI_API_KEY"] = "sk-test-key-for-validation-testing-12345678901234567890"
        
        try:
            config_manager = ConfigurationManager(config_dir=temp_config_dir)
            
            # Set values that generate warnings but not errors
            config_manager.config.video_processing.max_video_size_mb = 2500  # Large size
            config_manager.config.performance.max_workers = 20  # High worker count
            
            is_valid = config_manager.validate_configuration()
            assert is_valid is True  # Should still be valid
            
            report = config_manager.get_validation_report()
            assert report["valid"] is True
            assert report["error_count"] == 0
            assert report["warning_count"] > 0
            
        finally:
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
    
    def test_environment_specific_validation(self, temp_config_dir):
        """Test that validation behaves differently for different environments."""
        # Test development environment
        dev_config_manager = ConfigurationManager(config_dir=temp_config_dir, environment="development")
        dev_config_manager.config.security.force_https = False  # OK in development
        
        dev_report = dev_config_manager.get_validation_report()
        
        # Test production environment
        prod_config_manager = ConfigurationManager(config_dir=temp_config_dir, environment="production")
        prod_config_manager.config.security.force_https = False  # Should generate warning in production
        
        prod_report = prod_config_manager.get_validation_report()
        
        # Production should have more warnings about security
        assert prod_report["warning_count"] >= dev_report["warning_count"]