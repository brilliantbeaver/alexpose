"""
Ambient Package

This package contains the core functionality for the "ambient" fall risk detection system.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

__version__ = "1.0.0"
__author__ = "Theodore Mui"

# Import modules carefully to avoid circular imports
try:
    from . import analysis, core, exceptions
    
    # Import existing packages
    from . import gavd, pose, utils, video
    
    # Import new packages (will be populated in later tasks)
    from . import cli, data, storage, classification

    # Import batch_upload after analysis to avoid circular imports
    from .analysis import batch_upload
except ImportError:
    # Core or analysis module not available
    pass
