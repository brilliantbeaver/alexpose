# Installation Guide

This guide walks you through installing and setting up the AlexPose gait analysis system.

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Node.js**: 18 or higher (for frontend)
- **Git**: For cloning the repository
- **FFmpeg**: For video processing (optional but recommended)

### Hardware Requirements

- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB free space for installation
- **GPU**: Optional but recommended for pose estimation

## Installation Methods

### Method 1: Quick Install with uv (Recommended)

1. **Install uv** (Python package manager):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/alexpose.git
   cd alexpose
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Activate the environment**:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   .venv\Scripts\activate     # On Windows
   ```

### Method 2: Traditional pip Install

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/alexpose.git
   cd alexpose
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

## Configuration Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-api-key-here
GOOGLE_API_KEY=AIyour-google-api-key-here

# Optional Configuration
ENVIRONMENT=development
JWT_SECRET_KEY=your-super-secret-jwt-key-at-least-32-characters-long
```

### 2. Configuration Files

The system comes with default configuration files:

- `config/alexpose.yaml` - Main configuration
- `config/llm_models.yaml` - LLM model specifications
- `config/development.yaml` - Development overrides

For detailed configuration options, see the [Configuration Guide](configuration.md).

### 3. Validate Installation

Run the validation script to ensure everything is set up correctly:

```bash
python scripts/validate_config.py
```

## Optional Dependencies

### FFmpeg (Recommended)

For advanced video processing capabilities:

**macOS**:
```bash
brew install ffmpeg
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows**:
Download from [FFmpeg website](https://ffmpeg.org/download.html) or use chocolatey:
```bash
choco install ffmpeg
```

### GPU Support (Optional)

For faster pose estimation with GPU acceleration:

**CUDA (NVIDIA)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**MPS (Apple Silicon)**:
GPU support is automatically available on Apple Silicon Macs.

## Verification

### Test Basic Functionality

1. **Test configuration loading**:
   ```bash
   python test_new_models.py
   ```

2. **Test pose estimation**:
   ```bash
   python -c "
   from ambient.pose.mediapipe_estimator import MediaPipeEstimator
   estimator = MediaPipeEstimator()
   print('MediaPipe estimator initialized successfully')
   "
   ```

3. **Test LLM classification**:
   ```bash
   python -c "
   from ambient.classification.llm_classifier import LLMClassifier
   classifier = LLMClassifier(api_key='test')
   print('LLM classifier initialized successfully')
   "
   ```

### Run Test Suite

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/ambient/core/  # Core functionality
python -m pytest -m "not slow"       # Skip slow tests
```

## Development Setup

### Frontend Development (Optional)

If you plan to work on the web interface:

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

### Pre-commit Hooks

Set up code quality checks:

```bash
pip install pre-commit
pre-commit install
```

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Ensure you're in the virtual environment
which python  # Should point to .venv/bin/python

# Reinstall in development mode
pip install -e .
```

#### API Key Issues

```bash
# Check environment variables
env | grep API_KEY

# Test API key format
python -c "
import os
key = os.getenv('OPENAI_API_KEY', '')
print(f'Key length: {len(key)}')
print(f'Valid format: {key.startswith(\"sk-\")}')
"
```

#### Permission Errors

```bash
# Fix directory permissions
chmod -R 755 data/ logs/ config/

# Create missing directories
mkdir -p data/{videos,analysis,models,cache,exports}
mkdir -p logs
```

#### FFmpeg Not Found

```bash
# Check FFmpeg installation
ffmpeg -version

# Add to PATH if needed (Windows)
set PATH=%PATH%;C:\path\to\ffmpeg\bin
```

### Getting Help

1. **Check logs**: Look in `logs/` directory for detailed error messages
2. **Run validation**: `python scripts/validate_config.py`
3. **Review configuration**: See [Configuration Guide](configuration.md)
4. **Check documentation**: Browse the `docs/` directory

## Next Steps

After successful installation:

1. **Review Configuration**: Read the [Configuration Guide](configuration.md)
2. **Try Quick Start**: Follow the [Quick Start Guide](quickstart.md)
3. **Explore Examples**: Check the `examples/` directory
4. **Read Documentation**: Browse component documentation in `docs/`

## Deployment

### Local Development

```bash
# Start the API server
python -m ambient.server.main

# In another terminal, start the frontend (if installed)
cd frontend && npm run dev
```

### Production Deployment

For production deployment options, see:

- [Docker Deployment](deployment/docker.md)
- [Heroku Deployment](deployment/heroku.md)
- [Manual Deployment](deployment/manual.md)

## Updating

### Update Dependencies

```bash
# With uv
uv sync --upgrade

# With pip
pip install --upgrade -e .
```

### Update Configuration

When updating to a new version:

1. **Backup configuration**: `cp -r config/ config.backup/`
2. **Check for changes**: Review release notes for configuration changes
3. **Validate configuration**: `python scripts/validate_config.py`
4. **Test thoroughly**: Run test suite before deploying

This completes the installation process. You should now have a fully functional AlexPose system ready for gait analysis!