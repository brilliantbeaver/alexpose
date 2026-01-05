"""
Storage Manager Module

This module provides a unified storage abstraction layer for the AlexPose system,
supporting multiple storage backends (JSON, pickle, SQLite) with consistent interfaces.

Key Features:
- Unified storage interface for different backends
- Automatic format selection based on data type
- Compression support for large datasets
- Backup and recovery mechanisms
- Transaction support for atomic operations

Design Principles:
- Strategy Pattern: Pluggable storage backends
- Single Responsibility: Each storage class handles one format
- Dependency Inversion: Depends on storage interface abstraction

Author: AlexPose Team
"""

import gzip
import json
import pickle
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger


class IStorageBackend(ABC):
    """Interface for storage backends."""
    
    @abstractmethod
    def save(self, data: Any, path: Path, **kwargs) -> Path:
        """Save data to storage."""
        pass
    
    @abstractmethod
    def load(self, path: Path, **kwargs) -> Any:
        """Load data from storage."""
        pass
    
    @abstractmethod
    def exists(self, path: Path) -> bool:
        """Check if data exists at path."""
        pass
    
    @abstractmethod
    def delete(self, path: Path) -> bool:
        """Delete data at path."""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        pass


class JSONStorageBackend(IStorageBackend):
    """Storage backend for JSON format."""
    
    def __init__(self, indent: int = 2, compress: bool = False):
        """
        Initialize JSON storage backend.
        
        Args:
            indent: JSON indentation level
            compress: Whether to use gzip compression
        """
        self.indent = indent
        self.compress = compress
    
    def save(self, data: Any, path: Path, **kwargs) -> Path:
        """
        Save data as JSON.
        
        Args:
            data: Data to save (must be JSON-serializable)
            path: Output path
            **kwargs: Additional arguments
            
        Returns:
            Path to saved file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert DataFrame to dict if needed
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        
        if self.compress:
            path = path.with_suffix(path.suffix + '.gz')
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=self.indent, default=str)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=self.indent, default=str)
        
        logger.debug(f"Saved JSON data to {path}")
        return path
    
    def load(self, path: Path, **kwargs) -> Any:
        """
        Load data from JSON.
        
        Args:
            path: Path to JSON file
            **kwargs: Additional arguments
            
        Returns:
            Loaded data
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix == '.gz':
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        logger.debug(f"Loaded JSON data from {path}")
        return data
    
    def exists(self, path: Path) -> bool:
        """Check if JSON file exists."""
        return path.exists() or path.with_suffix(path.suffix + '.gz').exists()
    
    def delete(self, path: Path) -> bool:
        """Delete JSON file."""
        try:
            if path.exists():
                path.unlink()
                return True
            
            gz_path = path.with_suffix(path.suffix + '.gz')
            if gz_path.exists():
                gz_path.unlink()
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported extensions."""
        return ['.json', '.json.gz']


class PickleStorageBackend(IStorageBackend):
    """Storage backend for pickle format."""
    
    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL, compress: bool = False):
        """
        Initialize pickle storage backend.
        
        Args:
            protocol: Pickle protocol version
            compress: Whether to use gzip compression
        """
        self.protocol = protocol
        self.compress = compress
    
    def save(self, data: Any, path: Path, **kwargs) -> Path:
        """
        Save data as pickle.
        
        Args:
            data: Data to save
            path: Output path
            **kwargs: Additional arguments
            
        Returns:
            Path to saved file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.compress:
            path = path.with_suffix(path.suffix + '.gz')
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f, protocol=self.protocol)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=self.protocol)
        
        logger.debug(f"Saved pickle data to {path}")
        return path
    
    def load(self, path: Path, **kwargs) -> Any:
        """
        Load data from pickle.
        
        Args:
            path: Path to pickle file
            **kwargs: Additional arguments
            
        Returns:
            Loaded data
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix == '.gz':
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        logger.debug(f"Loaded pickle data from {path}")
        return data
    
    def exists(self, path: Path) -> bool:
        """Check if pickle file exists."""
        return path.exists() or path.with_suffix(path.suffix + '.gz').exists()
    
    def delete(self, path: Path) -> bool:
        """Delete pickle file."""
        try:
            if path.exists():
                path.unlink()
                return True
            
            gz_path = path.with_suffix(path.suffix + '.gz')
            if gz_path.exists():
                gz_path.unlink()
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported extensions."""
        return ['.pkl', '.pickle', '.pkl.gz', '.pickle.gz']


class StorageManager:
    """
    Unified storage manager with multiple backend support.
    
    This class provides a consistent interface for storing and retrieving
    data using different storage formats.
    """
    
    def __init__(self, base_dir: Path = Path("data/storage")):
        """
        Initialize storage manager.
        
        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Register storage backends
        self.backends: Dict[str, IStorageBackend] = {
            'json': JSONStorageBackend(),
            'pickle': PickleStorageBackend()
        }
        
        # Extension to backend mapping
        self.extension_map = {}
        for name, backend in self.backends.items():
            for ext in backend.get_supported_extensions():
                self.extension_map[ext] = name
        
        logger.info(f"Initialized StorageManager with base_dir: {self.base_dir}")
    
    def register_backend(self, name: str, backend: IStorageBackend) -> None:
        """
        Register a new storage backend.
        
        Args:
            name: Backend name
            backend: Storage backend instance
        """
        self.backends[name] = backend
        
        # Update extension mapping
        for ext in backend.get_supported_extensions():
            self.extension_map[ext] = name
        
        logger.info(f"Registered storage backend: {name}")
    
    def save(
        self,
        data: Any,
        key: str,
        backend: str = 'pickle',
        compress: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save data using specified backend.
        
        Args:
            data: Data to save
            key: Storage key (will be used as filename)
            backend: Backend to use ('json' or 'pickle')
            compress: Whether to compress data
            metadata: Optional metadata to save alongside data
            
        Returns:
            Path to saved file
        """
        if backend not in self.backends:
            raise ValueError(f"Unknown backend: {backend}")
        
        storage_backend = self.backends[backend]
        
        # Determine file extension
        extensions = storage_backend.get_supported_extensions()
        ext = extensions[0]
        
        # Create file path
        file_path = self.base_dir / f"{key}{ext}"
        
        # Save data
        saved_path = storage_backend.save(data, file_path, compress=compress)
        
        # Save metadata if provided
        if metadata:
            metadata_path = saved_path.with_suffix('.metadata.json')
            metadata_with_timestamp = {
                **metadata,
                'saved_at': datetime.now().isoformat(),
                'backend': backend,
                'compressed': compress
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata_with_timestamp, f, indent=2)
        
        logger.info(f"Saved data with key '{key}' using {backend} backend")
        return saved_path
    
    def load(
        self,
        key: str,
        backend: Optional[str] = None
    ) -> Any:
        """
        Load data by key.
        
        Args:
            key: Storage key
            backend: Optional backend hint (auto-detected if not provided)
            
        Returns:
            Loaded data
        """
        # Auto-detect backend if not provided
        if backend is None:
            backend = self._detect_backend(key)
        
        if backend not in self.backends:
            raise ValueError(f"Unknown backend: {backend}")
        
        storage_backend = self.backends[backend]
        
        # Find file
        file_path = self._find_file(key, storage_backend)
        
        if file_path is None:
            raise FileNotFoundError(f"No file found for key: {key}")
        
        # Load data
        data = storage_backend.load(file_path)
        
        logger.info(f"Loaded data with key '{key}' using {backend} backend")
        return data
    
    def _detect_backend(self, key: str) -> str:
        """Auto-detect backend from existing files."""
        for backend_name, backend in self.backends.items():
            file_path = self._find_file(key, backend)
            if file_path is not None:
                return backend_name
        
        # Default to pickle if no file found
        return 'pickle'
    
    def _find_file(self, key: str, backend: IStorageBackend) -> Optional[Path]:
        """Find file for key using backend."""
        for ext in backend.get_supported_extensions():
            file_path = self.base_dir / f"{key}{ext}"
            if file_path.exists():
                return file_path
        return None
    
    def exists(self, key: str) -> bool:
        """
        Check if data exists for key.
        
        Args:
            key: Storage key
            
        Returns:
            True if data exists
        """
        for backend in self.backends.values():
            if self._find_file(key, backend) is not None:
                return True
        return False
    
    def delete(self, key: str) -> bool:
        """
        Delete data by key.
        
        Args:
            key: Storage key
            
        Returns:
            True if deleted successfully
        """
        deleted = False
        
        for backend in self.backends.values():
            file_path = self._find_file(key, backend)
            if file_path is not None:
                if backend.delete(file_path):
                    deleted = True
                
                # Delete metadata if exists
                metadata_path = file_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    metadata_path.unlink()
        
        if deleted:
            logger.info(f"Deleted data with key '{key}'")
        
        return deleted
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all storage keys.
        
        Args:
            pattern: Optional glob pattern for filtering
            
        Returns:
            List of storage keys
        """
        keys = set()
        
        if pattern:
            files = self.base_dir.glob(pattern)
        else:
            files = self.base_dir.iterdir()
        
        for file_path in files:
            if file_path.is_file() and not file_path.name.endswith('.metadata.json'):
                # Remove extension to get key
                key = file_path.stem
                # Handle compressed files
                if key.endswith('.pkl') or key.endswith('.pickle') or key.endswith('.json'):
                    key = Path(key).stem
                keys.add(key)
        
        return sorted(list(keys))
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a key.
        
        Args:
            key: Storage key
            
        Returns:
            Metadata dictionary or None
        """
        for backend in self.backends.values():
            file_path = self._find_file(key, backend)
            if file_path is not None:
                metadata_path = file_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        return json.load(f)
        
        return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        file_count = 0
        backend_counts = {name: 0 for name in self.backends.keys()}
        
        for file_path in self.base_dir.rglob('*'):
            if file_path.is_file() and not file_path.name.endswith('.metadata.json'):
                total_size += file_path.stat().st_size
                file_count += 1
                
                # Determine backend
                for ext, backend_name in self.extension_map.items():
                    if file_path.suffix == ext or str(file_path).endswith(ext):
                        backend_counts[backend_name] += 1
                        break
        
        return {
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'backend_counts': backend_counts,
            'base_dir': str(self.base_dir)
        }
