@echo off
REM Development environment setup script for AlexPose (Windows)

echo 🚀 Setting up AlexPose development environment...

REM Check if uv is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ uv is not installed. Please install uv first:
    echo    https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

REM Check Python version
echo 📋 Checking Python version...
python --version 2>nul | findstr "3.12" >nul
if %errorlevel% neq 0 (
    echo ❌ Python 3.12+ is required
    echo    Please install Python 3.12 or higher
    exit /b 1
)

echo ✅ Python version check passed

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 🔧 Creating virtual environment...
    uv venv
)

REM Install dependencies
echo 📦 Installing dependencies...
uv pip install -e ".[dev,all]"

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "data" mkdir data
if not exist "data\videos" mkdir data\videos
if not exist "data\youtube" mkdir data\youtube
if not exist "data\analysis" mkdir data\analysis
if not exist "data\models" mkdir data\models
if not exist "data\training" mkdir data\training
if not exist "data\cache" mkdir data\cache
if not exist "data\exports" mkdir data\exports
if not exist "logs" mkdir logs
if not exist "config" mkdir config

REM Copy environment file if it doesn't exist
if not exist ".env" (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please edit .env file and add your API keys
)

echo.
echo ✅ Development environment setup complete!
echo.
echo 📋 Next steps:
echo    1. Edit .env file and add your API keys (OPENAI_API_KEY, GOOGLE_API_KEY)
echo    2. Activate the virtual environment: .venv\Scripts\activate
echo    3. Start the development server: uvicorn server.main:app --reload
echo    4. Run tests: pytest
echo    5. Format code: black . ^&^& isort .
echo.
echo 📚 Documentation: https://docs.alexpose.ai
echo 🐛 Issues: https://github.com/alexpose/alexpose/issues

pause