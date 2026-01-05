"""
Test models service with dependency injection and extensibility.

Tests the ModelsService and its providers with proper mocking and DI.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from server.services.models_service import (
    ModelsService,
    PoseEstimatorProvider,
    LLMModelProvider,
    ModelProvider
)
from ambient.core.config import ConfigurationManager
from ambient.pose.factory import PoseEstimatorFactory


class MockModelProvider:
    """Mock model provider for testing extensibility."""
    
    def get_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "mock_model_1",
                "display_name": "Mock Model 1",
                "type": "mock",
                "category": "testing",
                "available": True
            },
            {
                "name": "mock_model_2",
                "display_name": "Mock Model 2",
                "type": "mock",
                "category": "testing",
                "available": False
            }
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        if model_name not in ["mock_model_1", "mock_model_2"]:
            raise ValueError(f"Model {model_name} not found")
        
        return {
            "name": model_name,
            "display_name": f"Mock Model {model_name[-1]}",
            "type": "mock",
            "category": "testing",
            "available": model_name == "mock_model_1",
            "description": f"Test model {model_name}"
        }
    
    def get_provider_type(self) -> str:
        return "mock"


class TestPoseEstimatorProvider:
    """Test PoseEstimatorProvider."""
    
    def test_get_models(self):
        """Test getting list of pose estimators."""
        provider = PoseEstimatorProvider()
        models = provider.get_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Verify structure
        for model in models:
            assert "name" in model
            assert "display_name" in model
            assert "type" in model
            assert model["type"] == "pose_estimator"
            assert "category" in model
            assert model["category"] == "pose_estimation"
            assert "available" in model
    
    def test_get_model_info(self):
        """Test getting info for specific pose estimator."""
        provider = PoseEstimatorProvider()
        
        # Get info for mediapipe (should exist)
        info = provider.get_model_info("mediapipe")
        
        assert info["name"] == "mediapipe"
        assert "display_name" in info
        assert info["type"] == "pose_estimator"
        assert "description" in info
    
    def test_get_provider_type(self):
        """Test getting provider type."""
        provider = PoseEstimatorProvider()
        assert provider.get_provider_type() == "pose_estimator"
    
    def test_dependency_injection(self):
        """Test that factory can be injected."""
        mock_factory = Mock(spec=PoseEstimatorFactory)
        mock_factory.list_available_estimators.return_value = ["test_estimator"]
        mock_factory.get_estimator_info.return_value = {
            "class": "TestEstimator",
            "module": "test.module",
            "available": True
        }
        
        provider = PoseEstimatorProvider(factory=mock_factory)
        models = provider.get_models()
        
        assert len(models) == 1
        assert models[0]["name"] == "test_estimator"
        mock_factory.list_available_estimators.assert_called_once()


class TestLLMModelProvider:
    """Test LLMModelProvider."""
    
    def test_get_models(self):
        """Test getting list of LLM models."""
        # Create mock config with LLM models
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {
            "gpt-5.2": {
                "provider": "openai",
                "capability": "multimodal",
                "description": "Test GPT model",
                "cost_tier": "premium",
                "multimodal": True,
                "reasoning": "advanced"
            },
            "gemini-3-pro": {
                "provider": "gemini",
                "capability": "multimodal",
                "description": "Test Gemini model",
                "cost_tier": "medium",
                "multimodal": True,
                "reasoning": "advanced"
            }
        }
        
        provider = LLMModelProvider(config_manager)
        models = provider.get_models()
        
        assert len(models) == 2
        assert models[0]["name"] == "gpt-5.2"
        assert models[0]["type"] == "llm_model"
        assert models[0]["category"] == "classification"
        assert models[1]["name"] == "gemini-3-pro"
    
    def test_get_model_info(self):
        """Test getting info for specific LLM model."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {
            "gpt-5.2": {
                "provider": "openai",
                "capability": "multimodal",
                "description": "Test model",
                "max_tokens": 16384,
                "context_window": 200000,
                "cost_per_1k_tokens": {"input": 0.01, "output": 0.03},
                "supports_json_mode": True,
                "supports_video": True,
                "multimodal": True,
                "reasoning": "advanced",
                "cost_tier": "premium"
            }
        }
        
        provider = LLMModelProvider(config_manager)
        info = provider.get_model_info("gpt-5.2")
        
        assert info["name"] == "gpt-5.2"
        assert info["provider"] == "openai"
        assert info["max_tokens"] == 16384
        assert info["context_window"] == 200000
        assert info["supports_json_mode"] is True
    
    def test_get_model_info_not_found(self):
        """Test getting info for nonexistent model."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        provider = LLMModelProvider(config_manager)
        
        with pytest.raises(ValueError, match="No LLM models configured"):
            provider.get_model_info("nonexistent")
    
    def test_get_provider_type(self):
        """Test getting provider type."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        provider = LLMModelProvider(config_manager)
        assert provider.get_provider_type() == "llm_model"


class TestModelsService:
    """Test ModelsService with dependency injection."""
    
    def test_initialization(self):
        """Test service initialization with DI."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        service = ModelsService(config_manager)
        
        assert len(service.providers) == 2  # pose and llm
        assert "pose_estimator" in service.providers
        assert "llm_model" in service.providers
    
    def test_register_provider(self):
        """Test registering custom provider (extensibility)."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        service = ModelsService(config_manager)
        mock_provider = MockModelProvider()
        
        service.register_provider(mock_provider)
        
        assert "mock" in service.providers
        assert len(service.providers) == 3
    
    def test_get_all_models(self):
        """Test getting all models from all providers."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {
            "test-model": {
                "provider": "test",
                "capability": "text_only",
                "description": "Test",
                "cost_tier": "low"
            }
        }
        
        service = ModelsService(config_manager)
        result = service.get_all_models()
        
        assert "models" in result
        assert "by_type" in result
        assert "by_category" in result
        assert "summary" in result
        
        assert len(result["models"]) > 0
        assert "pose_estimator" in result["by_type"]
        assert "llm_model" in result["by_type"]
        assert result["summary"]["total_models"] > 0
    
    def test_get_models_by_type(self):
        """Test getting models by type."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        service = ModelsService(config_manager)
        
        pose_models = service.get_models_by_type("pose_estimator")
        assert isinstance(pose_models, list)
        assert all(m["type"] == "pose_estimator" for m in pose_models)
        
        llm_models = service.get_models_by_type("llm_model")
        assert isinstance(llm_models, list)
    
    def test_get_models_by_type_invalid(self):
        """Test getting models with invalid type."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        service = ModelsService(config_manager)
        
        with pytest.raises(ValueError, match="Unknown model type"):
            service.get_models_by_type("invalid_type")
    
    def test_get_model_info(self):
        """Test getting specific model info."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        service = ModelsService(config_manager)
        
        info = service.get_model_info("pose_estimator", "mediapipe")
        assert info["name"] == "mediapipe"
        assert info["type"] == "pose_estimator"
    
    def test_get_available_providers(self):
        """Test getting list of providers."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        service = ModelsService(config_manager)
        providers = service.get_available_providers()
        
        assert "pose_estimator" in providers
        assert "llm_model" in providers
    
    def test_search_models(self):
        """Test searching models."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {
            "gpt-5.2": {
                "provider": "openai",
                "capability": "multimodal",
                "description": "Advanced GPT model",
                "cost_tier": "premium"
            }
        }
        
        service = ModelsService(config_manager)
        
        # Search by name
        results = service.search_models("gpt")
        assert len(results) > 0
        assert any("gpt" in r["name"].lower() for r in results)
        
        # Search by description
        results = service.search_models("mediapipe")
        assert len(results) > 0
    
    def test_search_models_with_type_filter(self):
        """Test searching models with type filter."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        service = ModelsService(config_manager)
        
        results = service.search_models("media", model_type="pose_estimator")
        assert all(r["type"] == "pose_estimator" for r in results)
    
    def test_get_models_by_category(self):
        """Test getting models by category."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        service = ModelsService(config_manager)
        
        pose_models = service.get_models_by_category("pose_estimation")
        assert len(pose_models) > 0
        assert all(m["category"] == "pose_estimation" for m in pose_models)
    
    def test_get_statistics(self):
        """Test getting model statistics."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {
            "test-model": {
                "provider": "test",
                "capability": "text_only",
                "description": "Test",
                "cost_tier": "low"
            }
        }
        
        service = ModelsService(config_manager)
        stats = service.get_statistics()
        
        assert "total_models" in stats
        assert "by_type" in stats
        assert "by_category" in stats
        assert "available_count" in stats
        assert "unavailable_count" in stats
        assert "providers" in stats
        
        assert stats["total_models"] > 0
        assert len(stats["providers"]) == 2
    
    def test_extensibility_with_custom_provider(self):
        """Test that custom providers work seamlessly."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.llm_models = {}
        
        service = ModelsService(config_manager)
        mock_provider = MockModelProvider()
        service.register_provider(mock_provider)
        
        # Get all models should include mock models
        all_models = service.get_all_models()
        mock_models = [m for m in all_models["models"] if m["type"] == "mock"]
        assert len(mock_models) == 2
        
        # Get by type should work
        mock_models_by_type = service.get_models_by_type("mock")
        assert len(mock_models_by_type) == 2
        
        # Get model info should work
        info = service.get_model_info("mock", "mock_model_1")
        assert info["name"] == "mock_model_1"
        
        # Statistics should include mock models
        stats = service.get_statistics()
        assert "mock" in stats["by_type"]
        assert stats["by_type"]["mock"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
