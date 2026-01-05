# Configuration Files

This directory contains the configuration files for the AlexPose system.

## File Structure

- **`alexpose.yaml`** - Main system configuration
- **`llm_models.yaml`** - LLM model specifications and presets
- **`classification_prompts.yaml`** - Classification prompt templates
- **`development.yaml`** - Development environment overrides
- **`production.yaml`** - Production environment overrides
- **`heroku-production.yaml`** - Heroku-specific production settings

## Configuration Architecture

### Separation of Concerns

The configuration is organized following these principles:

1. **Main Configuration** (`alexpose.yaml`) - Core system settings, feature toggles, and operational parameters
2. **Model Specifications** (`llm_models.yaml`) - Reusable LLM model definitions, capabilities, and cost information
3. **Environment Overrides** - Environment-specific configurations that override base settings

### LLM Models Configuration

The LLM models are separated into their own file (`llm_models.yaml`) for several reasons:

- **Reusability**: Model specifications can be shared across different components
- **Maintainability**: Model updates and additions are centralized
- **Clarity**: Main configuration focuses on system behavior, not model details
- **Extensibility**: Easy to add new models or modify existing ones

### Usage Example

```python
import yaml
from pathlib import Path

def load_config():
    # Load main configuration
    with open('config/alexpose.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load LLM models if referenced
    models_config_file = config['classification']['llm'].get('models_config_file')
    if models_config_file:
        with open(models_config_file, 'r') as f:
            models_config = yaml.safe_load(f)
        
        # Merge or reference as needed
        config['classification']['llm']['models'] = models_config['models']
        config['classification']['llm']['presets'] = models_config['presets']
    
    return config
```

### Model Selection Strategy

The system supports three model selection strategies defined in `llm_models.yaml`:

- **`cost_optimized`** - Prioritizes cost-effective models
- **`performance_optimized`** - Prioritizes highest capability models  
- **`balanced`** - Balances cost and performance

### Environment Variables

API keys and sensitive configuration should be set via environment variables:

- `OPENAI_API_KEY` - OpenAI API key
- `GOOGLE_API_KEY` - Google Gemini API key

### Best Practices

1. **Never commit API keys** to version control
2. **Use environment-specific overrides** for different deployment environments
3. **Keep model specifications up to date** as new models are released
4. **Document configuration changes** that affect system behavior
5. **Validate configurations** before deployment