"""
Unit tests for external LLM models configuration validation.

Tests the llm_models.yaml configuration file structure, model definitions,
presets, and provider configurations following SOLID principles.
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, List

from tests.conftest import skip_if_no_ambient
from tests.utils.test_helpers import FileManager
from tests.utils.assertions import AssertionHelpers


class TestLLMModelsConfigFile:
    """Test external LLM models configuration file structure and content."""
    
    @pytest.fixture
    def llm_models_config_path(self) -> Path:
        """Fixture providing path to LLM models configuration file."""
        return Path("config/llm_models.yaml")
    
    @pytest.fixture
    def llm_models_config(self, llm_models_config_path: Path) -> Dict:
        """Fixture providing loaded LLM models configuration."""
        if not llm_models_config_path.exists():
            pytest.skip(f"LLM models config file not found: {llm_models_config_path}")
        
        with open(llm_models_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_llm_models_config_file_exists(self, llm_models_config_path: Path):
        """Test that the LLM models configuration file exists."""
        assert llm_models_config_path.exists(), f"LLM models config file not found: {llm_models_config_path}"
    
    def test_llm_models_config_has_required_sections(self, llm_models_config: Dict):
        """Test that LLM models config has all required top-level sections."""
        required_sections = ["models", "presets", "default_priority_order", "providers"]
        
        for section in required_sections:
            assert section in llm_models_config, f"Missing required section: {section}"
    
    def test_models_section_structure(self, llm_models_config: Dict):
        """Test that models section has proper structure."""
        models = llm_models_config.get("models", {})
        assert isinstance(models, dict), "Models section must be a dictionary"
        assert len(models) > 0, "Models section cannot be empty"
        
        # Test that each model has required fields
        required_model_fields = ["provider", "max_tokens", "multimodal", "cost_tier"]
        
        for model_name, model_config in models.items():
            assert isinstance(model_config, dict), f"Model {model_name} config must be a dictionary"
            
            for field in required_model_fields:
                assert field in model_config, f"Model {model_name} missing required field: {field}"
    
    def test_presets_section_structure(self, llm_models_config: Dict):
        """Test that presets section has proper structure."""
        presets = llm_models_config.get("presets", {})
        assert isinstance(presets, dict), "Presets section must be a dictionary"
        
        expected_presets = ["cost_optimized", "performance_optimized", "balanced"]
        for preset in expected_presets:
            assert preset in presets, f"Missing expected preset: {preset}"
            preset_config = presets[preset]
            assert isinstance(preset_config, dict), f"Preset {preset} must be a dictionary"
            assert "description" in preset_config, f"Preset {preset} must have description"
            assert "preferred_models" in preset_config, f"Preset {preset} must have preferred_models"
            assert isinstance(preset_config["preferred_models"], list), f"Preset {preset} preferred_models must be a list"
    
    def test_providers_section_structure(self, llm_models_config: Dict):
        """Test that providers section has proper structure."""
        providers = llm_models_config.get("providers", {})
        assert isinstance(providers, dict), "Providers section must be a dictionary"
        
        expected_providers = ["openai", "gemini"]
        for provider in expected_providers:
            assert provider in providers, f"Missing expected provider: {provider}"


class TestOpenAIModelsConfiguration:
    """Test OpenAI models configuration specifically."""
    
    @pytest.fixture
    def llm_models_config(self) -> Dict:
        """Fixture providing loaded LLM models configuration."""
        config_path = Path("config/llm_models.yaml")
        if not config_path.exists():
            pytest.skip(f"LLM models config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def openai_models(self, llm_models_config: Dict) -> Dict:
        """Fixture providing OpenAI models from configuration."""
        models = llm_models_config.get("models", {})
        return {name: config for name, config in models.items() 
                if config.get("provider") == "openai"}
    
    def test_expected_openai_models_present(self, openai_models: Dict):
        """Test that all expected OpenAI models are present."""
        expected_models = [
            "gpt-5.2", "gpt-5.1", "gpt-5-mini", "gpt-5-nano", 
            "gpt-4.1", "gpt-4.1-mini"
        ]
        
        missing_models = [model for model in expected_models if model not in openai_models]
        assert not missing_models, f"Missing OpenAI models: {missing_models}"
    
    def test_openai_models_have_valid_configuration(self, openai_models: Dict):
        """Test that OpenAI models have valid configuration values."""
        for model_name, model_config in openai_models.items():
            # Test provider
            assert model_config["provider"] == "openai", f"Model {model_name} has incorrect provider"
            
            # Test max_tokens is positive integer
            assert isinstance(model_config["max_tokens"], int), f"Model {model_name} max_tokens must be integer"
            assert model_config["max_tokens"] > 0, f"Model {model_name} max_tokens must be positive"
            
            # Test multimodal is boolean
            assert isinstance(model_config["multimodal"], bool), f"Model {model_name} multimodal must be boolean"
            
            # Test cost_tier is valid
            valid_cost_tiers = ["low", "medium", "high", "premium"]
            assert model_config["cost_tier"] in valid_cost_tiers, f"Model {model_name} has invalid cost_tier"


class TestGeminiModelsConfiguration:
    """Test Gemini models configuration specifically."""
    
    @pytest.fixture
    def llm_models_config(self) -> Dict:
        """Fixture providing loaded LLM models configuration."""
        config_path = Path("config/llm_models.yaml")
        if not config_path.exists():
            pytest.skip(f"LLM models config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def gemini_models(self, llm_models_config: Dict) -> Dict:
        """Fixture providing Gemini models from configuration."""
        models = llm_models_config.get("models", {})
        return {name: config for name, config in models.items() 
                if config.get("provider") == "gemini"}
    
    def test_expected_gemini_models_present(self, gemini_models: Dict):
        """Test that all expected Gemini models are present."""
        expected_models = [
            "gemini-3-pro-preview", "gemini-3-pro-image-preview", 
            "gemini-3-flash-preview", "gemini-2.5-flash", 
            "gemini-2.5-flash-image", "gemini-2.5-pro"
        ]
        
        missing_models = [model for model in expected_models if model not in gemini_models]
        assert not missing_models, f"Missing Gemini models: {missing_models}"
    
    def test_gemini_models_have_valid_configuration(self, gemini_models: Dict):
        """Test that Gemini models have valid configuration values."""
        for model_name, model_config in gemini_models.items():
            # Test provider
            assert model_config["provider"] == "gemini", f"Model {model_name} has incorrect provider"
            
            # Test max_tokens is positive integer
            assert isinstance(model_config["max_tokens"], int), f"Model {model_name} max_tokens must be integer"
            assert model_config["max_tokens"] > 0, f"Model {model_name} max_tokens must be positive"
            
            # Test multimodal is boolean
            assert isinstance(model_config["multimodal"], bool), f"Model {model_name} multimodal must be boolean"
            
            # Test cost_tier is valid
            valid_cost_tiers = ["low", "medium", "high", "premium"]
            assert model_config["cost_tier"] in valid_cost_tiers, f"Model {model_name} has invalid cost_tier"


class TestModelPriorityConfiguration:
    """Test model priority and default ordering configuration."""
    
    @pytest.fixture
    def llm_models_config(self) -> Dict:
        """Fixture providing loaded LLM models configuration."""
        config_path = Path("config/llm_models.yaml")
        if not config_path.exists():
            pytest.skip(f"LLM models config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_default_priority_order_exists(self, llm_models_config: Dict):
        """Test that default priority order is defined."""
        priority_order = llm_models_config.get("default_priority_order", [])
        assert isinstance(priority_order, list), "Default priority order must be a list"
        assert len(priority_order) > 0, "Default priority order cannot be empty"
    
    def test_priority_order_prioritizes_new_models(self, llm_models_config: Dict):
        """Test that priority order correctly prioritizes new models."""
        priority_order = llm_models_config.get("default_priority_order", [])
        
        # Check that new models are at the top of priority
        expected_top_models = ["gpt-5.2", "gpt-5.1", "gpt-5-mini"]
        
        assert len(priority_order) >= len(expected_top_models), "Priority order too short"
        
        for i, expected_model in enumerate(expected_top_models):
            assert priority_order[i] == expected_model, f"Expected {expected_model} at position {i}, got {priority_order[i]}"
    
    def test_priority_order_models_exist_in_config(self, llm_models_config: Dict):
        """Test that all models in priority order exist in models configuration."""
        priority_order = llm_models_config.get("default_priority_order", [])
        models = llm_models_config.get("models", {})
        
        missing_models = [model for model in priority_order if model not in models]
        assert not missing_models, f"Priority order references non-existent models: {missing_models}"