"""
CLI commands for AlexPose Gait Analysis System.
"""

from .analyze import analyze
from .batch import batch
from .config import config_cmd
from .info import info

__all__ = ["analyze", "batch", "config_cmd", "info"]
