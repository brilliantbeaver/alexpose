#!/bin/bash
# Development environment setup script for AlexPose

set -e

echo "🚀 Setting up AlexPose development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.12"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    echo "❌ Python 3.12+ is required. Current version: $python_version"
    echo "   Please install Python 3.12 or higher"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -e ".[dev,all]"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/{videos,youtube,analysis,models,training,cache,exports}
mkdir -p logs
mkdir -p config

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys"
fi

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
pytest tests/ -v --tb=short || echo "⚠️  Some tests failed - this is expected for initial setup"

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Edit .env file and add your API keys (OPENAI_API_KEY, GOOGLE_API_KEY)"
echo "   2. Activate the virtual environment: source .venv/bin/activate"
echo "   3. Start the development server: uvicorn server.main:app --reload"
echo "   4. Run tests: pytest"
echo "   5. Format code: black . && isort ."
echo ""
echo "📚 Documentation: https://docs.alexpose.ai"
echo "🐛 Issues: https://github.com/alexpose/alexpose/issues"