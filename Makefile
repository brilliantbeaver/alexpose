# AlexPose Development Makefile

.PHONY: help install install-dev test test-fast test-slow lint format clean docs serve

# Default target
help:
	@echo "AlexPose Development Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run fast tests only (exclude slow tests)"
	@echo "  test-slow    Run slow tests only"
	@echo "  test-cov     Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run linting (flake8, mypy)"
	@echo "  format       Format code (black, isort)"
	@echo "  format-check Check code formatting"
	@echo ""
	@echo "Development:"
	@echo "  serve        Start development server"
	@echo "  serve-prod   Start production server"
	@echo "  docs         Build documentation"
	@echo "  clean        Clean build artifacts"

# Installation
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,all]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-slow:
	pytest tests/ -v -m "slow"

test-cov:
	pytest tests/ --cov=ambient --cov=server --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 ambient/ server/ tests/
	mypy ambient/ server/

format:
	black ambient/ server/ tests/
	isort ambient/ server/ tests/

format-check:
	black --check ambient/ server/ tests/
	isort --check-only ambient/ server/ tests/

# Development
serve:
	uvicorn server.main:app --reload --host 127.0.0.1 --port 8000

serve-prod:
	uvicorn server.main:app --host 0.0.0.0 --port 8000

docs:
	@echo "Documentation generation not yet implemented"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/