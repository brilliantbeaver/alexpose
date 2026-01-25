"""
Tests for path utilities.
"""

import pytest
from pathlib import Path
from ambient.utils.path_utils import (
    get_project_root,
    get_data_dir,
    get_models_dir,
    get_config_dir,
)


def test_get_project_root():
    """Test that get_project_root finds the correct root."""
    project_root = get_project_root()
    
    # Should find pyproject.toml at root
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "README.md").exists()
    
    # Should be named 'alexpose'
    assert project_root.name == "alexpose"


def test_get_data_dir():
    """Test that get_data_dir returns correct path."""
    data_dir = get_data_dir()
    
    # Should be project_root/data
    assert data_dir.name == "data"
    assert data_dir.parent.name == "alexpose"


def test_get_models_dir():
    """Test that get_models_dir returns correct path."""
    models_dir = get_models_dir()
    
    # Should be project_root/data/models
    assert models_dir.name == "models"
    assert models_dir.parent.name == "data"


def test_get_config_dir():
    """Test that get_config_dir returns correct path."""
    config_dir = get_config_dir()
    
    # Should be project_root/config
    assert config_dir.name == "config"
    assert config_dir.parent.name == "alexpose"
    
    # Should contain config files
    assert (config_dir / "alexpose.yaml").exists()


def test_get_project_root_with_start_path():
    """Test that get_project_root works with custom start path."""
    # Start from a deep subdirectory
    start_path = Path(__file__).parent
    project_root = get_project_root(start_path=start_path)
    
    assert (project_root / "pyproject.toml").exists()


def test_get_project_root_fails_gracefully():
    """Test that get_project_root raises error when root not found."""
    # Start from root directory (no markers)
    with pytest.raises(RuntimeError, match="Could not find project root"):
        get_project_root(start_path=Path("/"))
