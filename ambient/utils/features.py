"""
Feature persistence utilities for gait analysis.

This module provides functions to save and load extracted gait features to/from disk
using Python's pickle format. Features are stored with metadata for tracking and validation.

Typical workflow:
    1. Extract features using FeatureExtractor
    2. Save features using save_features()
    3. Load features later using load_features()
    4. Train classifiers on loaded features

Example:
    >>> from ambient.utils.features import save_features, load_features
    >>> # After feature extraction
    >>> save_features(features, counts, filename="my_features.pkl", directory="data/")
    >>> # Later, in a different session
    >>> features, counts = load_features(filename="my_features.pkl", directory="data/")
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def save_features(
    all_features: List['GaitFeatureVector'],
    condition_counts: Dict[str, int],
    filename: str = "extracted_features.pkl",
    directory: Optional[Path] = None
) -> Path:
    """
    Save extracted gait features and condition counts to a pickle file.
    
    This function serializes feature vectors along with metadata for later use in
    classifier training or analysis. The saved file includes:
    - Feature vectors (GaitFeatureVector objects)
    - Condition counts (distribution of samples per condition)
    - Metadata (timestamp, counts, condition names)
    
    Args:
        all_features: List of GaitFeatureVector objects containing extracted features
                     from video analysis. Each vector represents one video/sequence.
        condition_counts: Dictionary mapping condition names (e.g., 'normal', 'parkinsons')
                         to the number of samples for that condition.
        filename: Name of the output pickle file. Should end with .pkl extension.
                 Default: "extracted_features.pkl"
        directory: Target directory for saving. If None, uses current working directory.
                  Directory will be created if it doesn't exist.
    
    Returns:
        Path: Absolute path to the saved pickle file.
    
    Raises:
        Exception: If file writing fails (permissions, disk space, etc.)
    
    Example:
        >>> features = [...]  # List of GaitFeatureVector objects
        >>> counts = {'normal': 50, 'parkinsons': 30}
        >>> filepath = save_features(features, counts, directory=Path("data/training"))
        ✓ Features saved successfully to: data/training/extracted_features.pkl
          Total features: 80
          Conditions: ['normal', 'parkinsons']
          File size: 245.67 KB
    
    Note:
        Uses pickle.HIGHEST_PROTOCOL for optimal performance and compatibility
        with recent Python versions (3.8+).
    """
    try:
        # Resolve directory path - use current working directory if not specified
        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)
        
        # Create directory structure if it doesn't exist
        # parents=True creates intermediate directories, exist_ok=True prevents errors if exists
        directory.mkdir(parents=True, exist_ok=True)
        
        # Construct full file path by joining directory and filename
        filepath = directory / filename
        
        # Package data with metadata for validation and tracking
        # Metadata helps verify file integrity and provides context when loading
        data = {
            'all_features': all_features,  # The actual feature vectors
            'condition_counts': condition_counts,  # Sample distribution
            'metadata': {
                'timestamp': datetime.now().isoformat(),  # When features were saved
                'n_features': len(all_features),  # Total number of feature vectors
                'n_conditions': len(condition_counts),  # Number of unique conditions
                'conditions': list(condition_counts.keys())  # Condition names for quick reference
            }
        }
        
        # Serialize to pickle file using highest protocol for efficiency
        # 'wb' = write binary mode (required for pickle)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Provide user feedback with key statistics
        print(f"✓ Features saved successfully to: {filepath}")
        print(f"  Total features: {len(all_features)}")
        print(f"  Conditions: {list(condition_counts.keys())}")
        print(f"  File size: {filepath.stat().st_size / 1024:.2f} KB")
        
        return filepath
        
    except Exception as e:
        # Re-raise with context for debugging
        print(f"✗ Error saving features: {e}")
        raise


def load_features(
    filename: str = "extracted_features.pkl",
    directory: Optional[Path] = None
) -> Tuple[List['GaitFeatureVector'], Dict[str, int]]:
    """
    Load previously saved gait features and condition counts from a pickle file.
    
    This function deserializes feature data saved by save_features(). It performs
    validation to ensure the file format is correct and provides helpful feedback
    about the loaded data.
    
    Args:
        filename: Name of the pickle file to load. Should match the filename used
                 when saving. Default: "extracted_features.pkl"
        directory: Directory containing the pickle file. If None, uses current
                  working directory. Must match the directory used when saving.
    
    Returns:
        Tuple containing:
            - all_features (List[GaitFeatureVector]): List of feature vectors
            - condition_counts (Dict[str, int]): Condition name to sample count mapping
    
    Raises:
        FileNotFoundError: If the specified file doesn't exist at the given path.
                          Check that filename and directory are correct.
        ValueError: If the file exists but has an invalid format (not a dict,
                   missing required keys, or corrupted).
        Exception: For other errors during file reading or deserialization.
    
    Example:
        >>> from ambient.utils.features import load_features
        >>> features, counts = load_features(
        ...     filename="my_features.pkl",
        ...     directory=Path("data/training")
        ... )
        ✓ Features loaded successfully from: data/training/my_features.pkl
          Saved on: 2026-01-21T10:30:45.123456
          Total features: 80
          Conditions: ['normal', 'parkinsons']
        >>> print(f"Loaded {len(features)} feature vectors")
        Loaded 80 feature vectors
    
    Note:
        - The function validates data integrity and warns about empty datasets
        - Metadata (if present) provides context about when features were extracted
        - Compatible with files saved using save_features() from this module
    """
    try:
        # Resolve directory path - use current working directory if not specified
        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)
        
        # Construct full file path
        filepath = directory / filename
        
        # Verify file exists before attempting to load
        # Provides helpful error message if file is missing
        if not filepath.exists():
            raise FileNotFoundError(
                f"Feature file not found: {filepath}\n"
                f"Please run feature extraction first or check the file path."
            )
        
        # Deserialize pickle file
        # 'rb' = read binary mode (required for pickle)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Validate file format - must be a dictionary
        if not isinstance(data, dict):
            raise ValueError("Invalid file format: expected dictionary")
        
        # Validate required keys are present
        # These keys are always included by save_features()
        if 'all_features' not in data or 'condition_counts' not in data:
            raise ValueError(
                "Invalid file format: missing 'all_features' or 'condition_counts'"
            )
        
        # Extract the actual feature data
        all_features = data['all_features']
        condition_counts = data['condition_counts']
        
        # Display metadata if available (added in newer versions)
        # Provides context about when and how features were extracted
        if 'metadata' in data:
            metadata = data['metadata']
            print(f"✓ Features loaded successfully from: {filepath}")
            print(f"  Saved on: {metadata.get('timestamp', 'unknown')}")
            print(f"  Total features: {metadata.get('n_features', len(all_features))}")
            print(f"  Conditions: {metadata.get('conditions', list(condition_counts.keys()))}")
        else:
            # Fallback for older files without metadata
            print(f"✓ Features loaded from: {filepath}")
            print(f"  Total features: {len(all_features)}")
        
        # Validate loaded data is not empty
        # Empty datasets might indicate extraction issues
        if not all_features:
            print("⚠️  Warning: No features found in loaded file")
        
        if not condition_counts:
            print("⚠️  Warning: No condition counts found in loaded file")
        
        return all_features, condition_counts
        
    except FileNotFoundError as e:
        # Re-raise with context for user
        print(f"✗ File not found: {e}")
        raise
    except Exception as e:
        # Catch-all for other errors (corrupted file, permission issues, etc.)
        print(f"✗ Error loading features: {e}")
        raise


def check_features_file(
    filename: str = "extracted_features.pkl",
    directory: Optional[Path] = None
) -> bool:
    """
    Check if a features file exists at the specified location.
    
    This is a lightweight utility function to verify file existence before
    attempting to load features. Useful for conditional logic in scripts
    and notebooks.
    
    Args:
        filename: Name of the pickle file to check for. Should match the
                 filename used when saving. Default: "extracted_features.pkl"
        directory: Directory to check in. If None, uses current working directory.
    
    Returns:
        bool: True if the file exists and is accessible, False otherwise.
    
    Example:
        >>> from ambient.utils.features import check_features_file, load_features
        >>> from pathlib import Path
        >>> 
        >>> # Check before loading to avoid exceptions
        >>> data_dir = Path("data/training")
        >>> if check_features_file(filename="my_features.pkl", directory=data_dir):
        ...     features, counts = load_features(filename="my_features.pkl", directory=data_dir)
        ...     print(f"Loaded {len(features)} features")
        ... else:
        ...     print("Features file not found - run extraction first")
        Features file not found - run extraction first
    
    Note:
        - This only checks existence, not file validity or format
        - Does not verify the file is a valid pickle or contains expected data
        - Use load_features() for full validation and error handling
    """
    # Resolve directory path - use current working directory if not specified
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory)
    
    # Construct full file path and check existence
    filepath = directory / filename
    return filepath.exists()
