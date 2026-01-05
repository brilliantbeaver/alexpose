"""
Model utilities for path resolution and management.

This module provides utilities for resolving model file paths with
intelligent fallback strategies to support both organized directory
structures and backward compatibility.

Author: AlexPose Team
"""

from pathlib import Path
from typing import Union, Optional
from loguru import logger


def resolve_yolo_model_path(
    model_name: str,
    models_directory: Optional[Union[str, Path]] = None,
    check_project_root: bool = True
) -> str:
    """
    Resolve YOLO model path with intelligent fallback strategy.
    
    This function implements a priority-based search for YOLO model files:
    1. If model_name is an absolute path that exists, use it
    2. If model_name is a relative path that exists, use it
    3. Check in models_directory (default: "data/models")
    4. Check in project root (for backward compatibility)
    5. Return original model_name (let Ultralytics download from hub)
    
    Args:
        model_name: Model name or path (e.g., "yolov8n-pose.pt" or "data/models/yolov8n-pose.pt")
        models_directory: Directory to search for models (default: "data/models")
        check_project_root: Whether to check project root for backward compatibility (default: True)
        
    Returns:
        Resolved model path as string, or original model_name if not found locally
        
    Examples:
        >>> # Model in data/models/
        >>> resolve_yolo_model_path("yolov8n-pose.pt")
        'data/models/yolov8n-pose.pt'
        
        >>> # Model in project root (backward compatibility)
        >>> resolve_yolo_model_path("yolov8n-pose.pt")
        'yolov8n-pose.pt'
        
        >>> # Absolute path
        >>> resolve_yolo_model_path("/opt/models/yolov8n-pose.pt")
        '/opt/models/yolov8n-pose.pt'
        
        >>> # Model not found locally (will be downloaded by Ultralytics)
        >>> resolve_yolo_model_path("yolov8x-pose.pt")
        'yolov8x-pose.pt'
    """
    if not model_name:
        raise ValueError("model_name cannot be empty")
    
    # Convert to Path for easier manipulation
    model_path = Path(model_name)
    
    # Priority 1: If it's an absolute path and exists, use it
    if model_path.is_absolute():
        if model_path.exists():
            logger.debug(f"Using absolute model path: {model_path}")
            return str(model_path)
        else:
            logger.warning(f"Absolute model path does not exist: {model_path}")
            return model_name  # Return as-is, let Ultralytics handle it
    
    # Priority 2: If it's a relative path and exists, use it
    if model_path.exists():
        resolved = model_path.resolve()
        logger.debug(f"Using existing relative model path: {resolved}")
        return str(model_path)  # Return relative path, not resolved
    
    # Priority 3: Check in models_directory
    if models_directory is None:
        models_directory = Path("data/models")
    else:
        models_directory = Path(models_directory)
    
    # Extract just the filename if model_name includes a path
    model_filename = model_path.name
    
    models_dir_path = models_directory / model_filename
    if models_dir_path.exists():
        logger.debug(f"Found model in models directory: {models_dir_path}")
        return str(models_dir_path)
    
    # Priority 4: Check in project root (backward compatibility)
    if check_project_root:
        root_path = Path(model_filename)
        if root_path.exists():
            logger.debug(f"Found model in project root (backward compatibility): {root_path}")
            logger.info(
                f"Model '{model_filename}' found in project root. "
                f"Consider moving to {models_directory} for better organization."
            )
            return str(root_path)
    
    # Priority 5: Model not found locally
    # Return original model_name - Ultralytics will attempt to download from hub
    logger.info(
        f"Model '{model_name}' not found locally. "
        f"Ultralytics will attempt to download from hub if needed."
    )
    return model_name


def get_model_info(model_path: Union[str, Path]) -> dict:
    """
    Get information about a model file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary containing model information
    """
    model_path = Path(model_path)
    
    info = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "name": model_path.name,
        "size_mb": None,
        "is_absolute": model_path.is_absolute()
    }
    
    if model_path.exists():
        try:
            size_bytes = model_path.stat().st_size
            info["size_mb"] = round(size_bytes / (1024 * 1024), 2)
        except Exception as e:
            logger.warning(f"Could not get model file size: {e}")
    
    return info


def list_available_yolo_models(models_directory: Optional[Union[str, Path]] = None) -> list:
    """
    List all available YOLO pose models in the models directory.
    
    Args:
        models_directory: Directory to search (default: "data/models")
        
    Returns:
        List of model filenames
    """
    if models_directory is None:
        models_directory = Path("data/models")
    else:
        models_directory = Path(models_directory)
    
    if not models_directory.exists():
        logger.warning(f"Models directory does not exist: {models_directory}")
        return []
    
    # Find all .pt files that look like YOLO pose models
    models = []
    for model_file in models_directory.glob("*.pt"):
        if "pose" in model_file.name.lower():
            models.append(model_file.name)
    
    return sorted(models)


def validate_model_path(model_path: Union[str, Path]) -> bool:
    """
    Validate that a model path exists and is accessible.
    
    Args:
        model_path: Path to model file
        
    Returns:
        True if model exists and is accessible, False otherwise
    """
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            return False
        
        # Try to read a few bytes to ensure it's accessible
        with open(model_path, 'rb') as f:
            f.read(1024)
        
        return True
    except Exception as e:
        logger.warning(f"Model validation failed for {model_path}: {e}")
        return False


def ensure_models_directory(models_directory: Optional[Union[str, Path]] = None) -> Path:
    """
    Ensure the models directory exists, creating it if necessary.
    
    Args:
        models_directory: Directory path (default: "data/models")
        
    Returns:
        Path object for the models directory
    """
    if models_directory is None:
        models_directory = Path("data/models")
    else:
        models_directory = Path(models_directory)
    
    try:
        models_directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Models directory ensured: {models_directory}")
    except Exception as e:
        logger.error(f"Failed to create models directory {models_directory}: {e}")
        raise
    
    return models_directory
