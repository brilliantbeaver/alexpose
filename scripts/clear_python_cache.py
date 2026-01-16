#!/usr/bin/env python3
"""
Clear Python bytecode cache files.

This script removes all .pyc files and __pycache__ directories
to resolve import issues caused by stale bytecode.

Usage:
    python scripts/clear_python_cache.py
"""

import os
import shutil
from pathlib import Path


def clear_pycache(root_dir: str = "."):
    """
    Clear all Python cache files and directories.
    
    Args:
        root_dir: Root directory to start search (default: current directory)
    """
    root_path = Path(root_dir).resolve()
    removed_files = 0
    removed_dirs = 0
    
    print(f"Clearing Python cache in: {root_path}")
    print("-" * 60)
    
    # Remove .pyc files
    for pyc_file in root_path.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            removed_files += 1
            print(f"[REMOVED FILE] {pyc_file.relative_to(root_path)}")
        except Exception as e:
            print(f"[ERROR] Could not remove {pyc_file}: {e}")
    
    # Remove __pycache__ directories
    for pycache_dir in root_path.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            removed_dirs += 1
            print(f"[REMOVED DIR]  {pycache_dir.relative_to(root_path)}")
        except Exception as e:
            print(f"[ERROR] Could not remove {pycache_dir}: {e}")
    
    print("-" * 60)
    print(f"Removed {removed_files} .pyc files")
    print(f"Removed {removed_dirs} __pycache__ directories")
    print("\n[OK] Python cache cleared successfully!")


if __name__ == "__main__":
    clear_pycache()
