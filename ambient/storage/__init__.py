"""
Storage Package

This package provides comprehensive storage capabilities for the AlexPose system,
including multiple storage backends, SQLite database, and backup management.

Modules:
- storage_manager: Unified storage interface with multiple backends
- sqlite_storage: SQLite database for structured data
- backup_manager: Backup and recovery management
"""

from ambient.storage.backup_manager import BackupManager
from ambient.storage.sqlite_storage import SQLiteStorage
from ambient.storage.storage_manager import (
    IStorageBackend,
    JSONStorageBackend,
    PickleStorageBackend,
    StorageManager,
)

__all__ = [
    # Storage Manager
    "StorageManager",
    "IStorageBackend",
    "JSONStorageBackend",
    "PickleStorageBackend",
    # SQLite Storage
    "SQLiteStorage",
    # Backup Manager
    "BackupManager",
]
