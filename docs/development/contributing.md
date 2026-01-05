# AlexPose Development Guide

This guide covers setting up the development environment and contributing to AlexPose.

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Installing uv

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or install via pip:
```bash
pip install uv
```

## Quick Setup

### Automated Setup (Recommended)

**Linux/macOS:**
```bash
./scripts/setup-dev.sh
```

**Windows:**
```cmd
scripts\setup-dev.bat
```

### Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alexpose/alexpose.git
   cd alexpose
   ```

2. **Create virtual environment:**
   ```bash
   uv venv
   ```

3. **Activate virtual environment:**
   ```bash
   # Linux/macOS
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   uv pip install -e ".[dev,all]"
   ```

5. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

6. **Create necessary directories:**
   ```bash
   mkdir -p data/{videos,youtube,analysis,models,training,cache,exports}
   mkdir -p logs config
   ```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure the following required variables:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Organization ID for OpenAI
OPENAI_ORG_ID=your_openai_org_id_here
```

### Configuration Files

AlexPose uses YAML configuration files in the `config/` directory:

- `config/alexpose.yaml` - Main configuration
- `config/development.yaml` - Development overrides
- `config/production.yaml` - Production overrides

## Development Workflow

### Running the Application

**Development server:**
```bash
make serve
# or
uvicorn server.main:app --reload
```

**CLI interface:**
```bash
alexpose --help
```

### Testing

**Run all tests:**
```bash
make test
# or
pytest
```

**Run fast tests only:**
```bash
make test-fast
# or
pytest -m "not slow"
```

**Run with coverage:**
```bash
make test-cov
# or
pytest --cov=ambient --cov=server --cov-report=html
```

### Code Quality

**Format code:**
```bash
make format
# or
black . && isort .
```

**Lint code:**
```bash
make lint
# or
flake8 ambient/ server/ tests/
mypy ambient/ server/
```

**Pre-commit hooks:**
```bash
pre-commit install  # Install hooks
pre-commit run --all-files  # Run on all files
```

## Project Structure

```
alexpose/
├── ambient/                 # Core Python package
│   ├── analysis/           # Gait analysis components
│   ├── classification/     # LLM-based classification
│   ├── cli/               # Command-line interface
│   ├── core/              # Core interfaces and data models
│   ├── data/              # Data management
│   ├── gavd/              # GAVD dataset processing
│   ├── pose/              # Pose estimation backends
│   ├── storage/           # Data persistence
│   ├── utils/             # Utility functions
│   └── video/             # Video processing
├── server/                 # FastAPI web server
├── tests/                  # Test suite
├── config/                 # Configuration files
├── data/                   # Data storage
├── docs/                   # Documentation
├── logs/                   # Log files
└── scripts/               # Development scripts
```

## Testing Guidelines

### Test Categories

- **Unit tests**: Test individual components in isolation
- **Property tests**: Test universal properties using Hypothesis
- **Integration tests**: Test component interactions
- **Slow tests**: Resource-intensive tests (marked with `@pytest.mark.slow`)

### Writing Tests

**Unit test example:**
```python
def test_frame_creation():
    frame = Frame.from_array(np.zeros((480, 640, 3)))
    assert frame.shape == (480, 640, 3)
```

**Property test example:**
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=1000))
def test_video_frame_count_property(frame_count):
    # Test that frame extraction preserves count
    pass
```

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.slow
def test_large_video_processing():
    pass

@pytest.mark.integration
def test_end_to_end_pipeline():
    pass

@pytest.mark.property
def test_gait_analysis_property():
    pass
```

## Contributing

### Code Style

- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Follow [PEP 8](https://pep8.org/) style guidelines
- Add type hints for all functions and methods
- Write docstrings for all public functions and classes

### Commit Guidelines

- Use conventional commit messages
- Keep commits focused and atomic
- Write clear, descriptive commit messages
- Reference issues in commit messages when applicable

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation if needed
7. Submit a pull request

## Troubleshooting

### Common Issues

**Import errors:**
- Ensure virtual environment is activated
- Reinstall dependencies: `uv pip install -e ".[dev,all]"`

**API key errors:**
- Check `.env` file configuration
- Verify API keys are valid and have proper permissions

**Test failures:**
- Run tests individually to isolate issues
- Check if external dependencies (FFmpeg, etc.) are installed

**Performance issues:**
- Use `pytest -m "not slow"` to skip resource-intensive tests
- Check system resources and close unnecessary applications

### Getting Help

- Check the [documentation](https://docs.alexpose.ai)
- Search existing [issues](https://github.com/alexpose/alexpose/issues)
- Create a new issue with detailed information
- Join our community discussions

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [uv Documentation](https://docs.astral.sh/uv/)