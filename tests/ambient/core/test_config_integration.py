"""
Integration tests for configuration loading and external models integration.

Tests the integration between main configuration and external LLM models,
configuration manager functionality, and model capabilities.
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict

from tests.conftest import skip_if_no_ambient
from tests.utils.test_helpers import FileManager
from tests.utils.assertions import AssertionHelpers

try:
    from ambient.core.config import ConfigurationManager, LLMConfig
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


@skip_if_no_ambient()
class TestMainConfigExternalModelsIntegration:
    """Test integration between main config and external models configuration."""
    
    @pytest.fixture
    def alexpose_config_path(self) -> Path:
        """Fixture providing path to main alexpose configuration file."""
        return Path("config/alexpose.yaml")
    
    @pytest.fixture
    def llm_models_config_path(self) -> Path:
        """Fixture providing path to LLM models configuration file."""
        return Path("config/llm_models.yaml")
    
    @pytest.fixture
    def alexpose_config(self, alexpose_config_path: Path) -> Dict:
        """Fixture providing loaded alexpose configuration."""
        if not alexpose_config_path.exists():
            pytest.skip(f"Alexpose config file not found: {alexpose_config_path}")
        
        with open(alexpose_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_main_config_references_external_models(self, alexpose_config: Dict):
        """Test that main config properly references external models file."""
        llm_config = alexpose_config.get('classification', {}).get('llm', {})
        models_config_file = llm_config.get('models_config_file')
        
        assert models_config_file is not None, "Main config must reference external models file"
        assert 'llm_models.yaml' in models_config_file, "Must reference llm_models.yaml file"
    
    def test_main_config_has_no_embedded_models(self, alexpose_config: Dict):
        """Test that models are properly separated from main config."""
        llm_config = alexpose_config.get('classification', {}).get('llm', {})
        embedded_models = llm_config.get('models', {})
        
        assert not embedded_models, "Models should not be embedded in main config (should be in external file)"
    
    def test_main_config_has_default_model(self, alexpose_config: Dict):
        """Test that main config specifies a default model."""
        llm_config = alexpose_config.get('classification', {}).get('llm', {})
        default_model = llm_config.get('model')
        
        assert default_model is not None, "Main config must specify a default model"
        assert isinstance(default_model, str), "Default model must be a string"
        assert len(default_model) > 0, "Default model cannot be empty"
    
    def test_external_models_file_exists(self, alexpose_config: Dict, llm_models_config_path: Path):
        """Test that referenced external models file exists."""
        llm_config = alexpose_config.get('classification', {}).get('llm', {})
        models_config_file = llm_config.get('models_config_file')
        
        if models_config_file:
            # Handle relative path resolution
            if not Path(models_config_file).is_absolute():
                # If the path already starts with config/, use it as is
                if models_config_file.startswith("config/"):
                    models_path = Path(models_config_file)
                else:
                    models_path = Path("config") / models_config_file
            else:
                models_path = Path(models_config_file)
            
            assert models_path.exists(), f"Referenced external models file does not exist: {models_path}"


@skip_if_no_ambient()
class TestConfigurationManagerIntegration:
    """Test ConfigurationManager with external models loading."""
    
    @pytest.fixture
    def config_manager(self) -> ConfigurationManager:
        """Fixture providing ConfigurationManager instance."""
        return ConfigurationManager()
    
    def test_configuration_manager_loads_external_models(self, config_manager: ConfigurationManager):
        """Test that ConfigurationManager successfully loads external models."""
        llm_config = config_manager.config.classification.llm
        
        assert llm_config.models is not None, "External models should be loaded"
        assert len(llm_config.models) > 0, "Should load at least one model"
    
    def test_configuration_manager_loads_presets(self, config_manager: ConfigurationManager):
        """Test that ConfigurationManager loads model presets."""
        llm_config = config_manager.config.classification.llm
        
        assert llm_config.presets is not None, "Model presets should be loaded"
        assert len(llm_config.presets) > 0, "Should load at least one preset"
    
    def test_configuration_validation_passes(self, config_manager: ConfigurationManager):
        """Test that configuration validation passes with external models."""
        is_valid = config_manager.validate_configuration()
        assert is_valid, "Configuration validation should pass"
    
    def test_default_model_exists_in_external_models(self, config_manager: ConfigurationManager):
        """Test that the default model exists in the loaded external models."""
        llm_config = config_manager.config.classification.llm
        default_model = llm_config.model
        
        assert default_model in llm_config.models, f"Default model '{default_model}' not found in external models"


@skip_if_no_ambient()
class TestLLMConfigModelCapabilities:
    """Test LLMConfig model capability methods with external models."""
    
    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        """Fixture providing LLMConfig instance with loaded external models."""
        config_manager = ConfigurationManager()
        return config_manager.config.classification.llm
    
    def test_supports_images_method(self, llm_config: LLMConfig):
        """Test supports_images method functionality."""
        supports_images = llm_config.supports_images()
        assert isinstance(supports_images, bool), "supports_images should return boolean"
    
    def test_supports_video_method(self, llm_config: LLMConfig):
        """Test supports_video method functionality."""
        supports_video = llm_config.supports_video()
        assert isinstance(supports_video, bool), "supports_video should return boolean"
    
    def test_get_available_models_method(self, llm_config: LLMConfig):
        """Test get_available_models method functionality."""
        available_models = llm_config.get_available_models()
        
        assert isinstance(available_models, list), "get_available_models should return list"
        assert len(available_models) > 0, "Should return at least one available model"
    
    def test_get_preset_models_method(self, llm_config: LLMConfig):
        """Test get_preset_models method functionality."""
        balanced_models = llm_config.get_preset_models("balanced")
        
        assert isinstance(balanced_models, list), "get_preset_models should return list"
        # Note: May be empty if preset doesn't exist, which is valid
    
    def test_get_model_spec_method(self, llm_config: LLMConfig):
        """Test get_model_spec method functionality."""
        model_spec = llm_config.get_model_spec()
        
        if model_spec:  # Model exists in external config
            assert isinstance(model_spec, dict), "get_model_spec should return dict when model exists"
            # Check for either new format or legacy format fields
            has_provider = "provider" in model_spec
            has_max_tokens = "max_tokens" in model_spec
            has_legacy_max_tokens = "max_tokens" in model_spec
            
            assert has_provider or has_max_tokens or has_legacy_max_tokens, "Model spec should contain expected fields"


@skip_if_no_ambient()
class TestEnvironmentConfigurationOverrides:
    """Test environment-specific configuration overrides."""
    
    def test_development_config_structure(self):
        """Test development configuration file structure."""
        dev_config_path = Path("config/development.yaml")
        
        if dev_config_path.exists():
            with open(dev_config_path, 'r') as f:
                dev_config = yaml.safe_load(f)
            
            # Test that it's a valid YAML structure
            assert isinstance(dev_config, dict), "Development config must be a dictionary"
            
            # If it has LLM config, test structure
            if 'classification' in dev_config and 'llm' in dev_config['classification']:
                llm_config = dev_config['classification']['llm']
                if 'model' in llm_config:
                    assert isinstance(llm_config['model'], str), "Development model override must be string"
    
    def test_production_config_structure(self):
        """Test production configuration file structure."""
        prod_config_path = Path("config/production.yaml")
        
        if prod_config_path.exists():
            with open(prod_config_path, 'r') as f:
                prod_config = yaml.safe_load(f)
            
            # Test that it's a valid YAML structure
            assert isinstance(prod_config, dict), "Production config must be a dictionary"
            
            # If it has LLM config, test structure
            if 'classification' in prod_config and 'llm' in prod_config['classification']:
                llm_config = prod_config['classification']['llm']
                if 'model' in llm_config:
                    assert isinstance(llm_config['model'], str), "Production model override must be string"