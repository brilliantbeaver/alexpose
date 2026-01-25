"""
Path Utilities Module

Provides utilities for working with file paths in the AlexPose project.

Author: AlexPose Team
"""

from pathlib import Path
from typing import Tuple, Optional
import os


def get_project_root(
    marker_files: Tuple[str, ...] = ('pyproject.toml', '.git'),
    start_path: Optional[Path] = None
) -> Path:
    """
    Get the AlexPose project root directory.
    
    Searches upward from the current file until it finds a marker file
    that indicates the project root (pyproject.toml, .git, etc.).
    
    Args:
        marker_files: Files that indicate project root
        start_path: Starting path for search (defaults to caller's location)
        
    Returns:
        Path to project root
        
    Raises:
        RuntimeError: If project root cannot be found
        
    Example:
        >>> from ambient.utils.path_utils import get_project_root
        >>> project_root = get_project_root()
        >>> print(project_root)
        /Users/pmui/dev/alex/alexpose
    """
    if start_path is None:
        # Try to get caller's file location
        try:
            import inspect
            frame = inspect.currentframe().f_back
            caller_file = frame.f_code.co_filename
            current = Path(caller_file).resolve().parent
        except:
            # Fallback to this file's location
            current = Path(__file__).resolve().parent
    else:
        current = Path(start_path).resolve()
    
    # Traverse up until we find a marker file
    max_depth = 10  # Prevent infinite loops
    for _ in range(max_depth):
        if current == current.parent:
            break
            
        if any((current / marker).exists() for marker in marker_files):
            return current
            
        current = current.parent
    
    raise RuntimeError(
        f"Could not find project root. Looked for: {marker_files}"
    )


def get_data_dir(project_root: Optional[Path] = None) -> Path:
    """
    Get the data directory path.
    
    Args:
        project_root: Project root path (auto-detected if None)
        
    Returns:
        Path to data directory
    """
    if project_root is None:
        project_root = get_project_root()
    
    return project_root / "data"


def get_models_dir(project_root: Optional[Path] = None) -> Path:
    """
    Get the models directory path.
    
    Args:
        project_root: Project root path (auto-detected if None)
        
    Returns:
        Path to models directory
    """
    data_dir = get_data_dir(project_root)
    return data_dir / "models"


def get_config_dir(project_root: Optional[Path] = None) -> Path:
    """
    Get the config directory path.
    
    Args:
        project_root: Project root path (auto-detected if None)
        
    Returns:
        Path to config directory
    """
    if project_root is None:
        project_root = get_project_root()
    
    return project_root / "config"
