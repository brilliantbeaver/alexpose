"""
Backup and Recovery Manager Module

This module provides comprehensive backup and recovery capabilities for the AlexPose system,
ensuring data safety and enabling disaster recovery.

Key Features:
- Automated backup scheduling
- Incremental and full backup support
- Backup verification and integrity checking
- Point-in-time recovery
- Backup rotation and retention policies

Design Principles:
- Single Responsibility: Each class handles one aspect of backup/recovery
- Fail-safe: Multiple verification steps to ensure backup integrity
- Configurable: Flexible backup policies and retention rules

Author: AlexPose Team
"""

import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class BackupManager:
    """
    Comprehensive backup and recovery manager.
    
    This class handles backup creation, verification, and restoration
    for the AlexPose system data.
    """
    
    def __init__(
        self,
        backup_dir: Path = Path("data/backups"),
        source_dirs: Optional[List[Path]] = None
    ):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory for storing backups
            source_dirs: List of directories to backup
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.source_dirs = source_dirs or [
            Path("data/storage"),
            Path("data/training"),
            Path("data/analysis")
        ]
        
        self.manifest_file = self.backup_dir / "backup_manifest.json"
        self.manifest = self._load_manifest()
        
        logger.info(f"Initialized BackupManager with backup_dir: {self.backup_dir}")
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load backup manifest from disk."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load backup manifest: {e}")
        
        return {
            "backups": [],
            "last_backup": None,
            "retention_days": 30
        }
    
    def _save_manifest(self) -> None:
        """Save backup manifest to disk."""
        try:
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup manifest: {e}")
    
    def create_backup(
        self,
        backup_name: Optional[str] = None,
        backup_type: str = "full",
        compress: bool = True,
        verify: bool = True
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Create a backup of specified directories.
        
        Args:
            backup_name: Optional backup name (auto-generated if not provided)
            backup_type: Type of backup ("full" or "incremental")
            compress: Whether to compress the backup
            verify: Whether to verify backup integrity
            
        Returns:
            Tuple of (backup path, backup metadata)
        """
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_path = self.backup_dir / backup_name
        
        if compress:
            backup_path = backup_path.with_suffix('.tar.gz')
        
        logger.info(f"Creating {backup_type} backup: {backup_name}")
        
        # Create backup metadata
        metadata = {
            "backup_name": backup_name,
            "backup_type": backup_type,
            "created_at": datetime.now().isoformat(),
            "compressed": compress,
            "source_dirs": [str(d) for d in self.source_dirs],
            "files_backed_up": 0,
            "total_size_bytes": 0,
            "checksum": ""
        }
        
        try:
            if compress:
                # Create compressed tar archive
                with tarfile.open(backup_path, 'w:gz') as tar:
                    for source_dir in self.source_dirs:
                        if source_dir.exists():
                            tar.add(source_dir, arcname=source_dir.name)
                            
                            # Count files and size
                            for file_path in source_dir.rglob('*'):
                                if file_path.is_file():
                                    metadata["files_backed_up"] += 1
                                    metadata["total_size_bytes"] += file_path.stat().st_size
            else:
                # Create uncompressed backup directory
                backup_path.mkdir(parents=True, exist_ok=True)
                
                for source_dir in self.source_dirs:
                    if source_dir.exists():
                        dest_dir = backup_path / source_dir.name
                        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
                        
                        # Count files and size
                        for file_path in dest_dir.rglob('*'):
                            if file_path.is_file():
                                metadata["files_backed_up"] += 1
                                metadata["total_size_bytes"] += file_path.stat().st_size
            
            # Calculate checksum
            metadata["checksum"] = self._calculate_backup_checksum(backup_path)
            
            # Verify backup if requested
            if verify:
                verification = self.verify_backup(backup_path, metadata)
                metadata["verified"] = verification["valid"]
                metadata["verification_details"] = verification
            
            # Update manifest
            self.manifest["backups"].append(metadata)
            self.manifest["last_backup"] = metadata["created_at"]
            self._save_manifest()
            
            logger.info(f"Backup created successfully: {backup_path}")
            logger.info(f"Backed up {metadata['files_backed_up']} files, "
                       f"total size: {metadata['total_size_bytes'] / (1024**2):.2f} MB")
            
            return backup_path, metadata
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def _calculate_backup_checksum(self, backup_path: Path) -> str:
        """
        Calculate checksum for backup.
        
        Args:
            backup_path: Path to backup file or directory
            
        Returns:
            SHA256 checksum string
        """
        hasher = hashlib.sha256()
        
        if backup_path.is_file():
            # Single file (compressed backup)
            with open(backup_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
        else:
            # Directory (uncompressed backup)
            for file_path in sorted(backup_path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b''):
                            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def verify_backup(
        self,
        backup_path: Path,
        expected_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify backup integrity.
        
        Args:
            backup_path: Path to backup
            expected_metadata: Optional expected metadata for verification
            
        Returns:
            Verification results dictionary
        """
        verification = {
            "valid": True,
            "checks": {},
            "errors": []
        }
        
        # Check if backup exists
        if not backup_path.exists():
            verification["valid"] = False
            verification["errors"].append("Backup file/directory not found")
            return verification
        
        verification["checks"]["exists"] = True
        
        # Verify checksum if metadata provided
        if expected_metadata and "checksum" in expected_metadata:
            current_checksum = self._calculate_backup_checksum(backup_path)
            checksum_match = current_checksum == expected_metadata["checksum"]
            verification["checks"]["checksum_match"] = checksum_match
            
            if not checksum_match:
                verification["valid"] = False
                verification["errors"].append("Checksum mismatch")
        
        # Verify file count if metadata provided
        if expected_metadata and "files_backed_up" in expected_metadata:
            if backup_path.is_file():
                # Compressed backup - would need to extract to count
                verification["checks"]["file_count_verified"] = "skipped_compressed"
            else:
                actual_count = sum(1 for _ in backup_path.rglob('*') if _.is_file())
                count_match = actual_count == expected_metadata["files_backed_up"]
                verification["checks"]["file_count_match"] = count_match
                
                if not count_match:
                    verification["valid"] = False
                    verification["errors"].append(
                        f"File count mismatch: expected {expected_metadata['files_backed_up']}, "
                        f"found {actual_count}"
                    )
        
        if verification["valid"]:
            logger.info(f"Backup verification passed: {backup_path}")
        else:
            logger.warning(f"Backup verification failed: {backup_path}")
            logger.warning(f"Errors: {verification['errors']}")
        
        return verification
    
    def restore_backup(
        self,
        backup_path: Path,
        restore_dir: Optional[Path] = None,
        verify_before_restore: bool = True
    ) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_path: Path to backup to restore
            restore_dir: Optional custom restore directory
            verify_before_restore: Whether to verify backup before restoring
            
        Returns:
            True if restored successfully
        """
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        # Find backup metadata
        backup_metadata = None
        for backup in self.manifest["backups"]:
            if backup_path.name.startswith(backup["backup_name"]):
                backup_metadata = backup
                break
        
        # Verify backup if requested
        if verify_before_restore and backup_metadata:
            verification = self.verify_backup(backup_path, backup_metadata)
            if not verification["valid"]:
                logger.error("Backup verification failed, aborting restore")
                return False
        
        logger.info(f"Restoring backup from: {backup_path}")
        
        try:
            if backup_path.is_file() and backup_path.suffix == '.gz':
                # Extract compressed backup
                with tarfile.open(backup_path, 'r:gz') as tar:
                    if restore_dir:
                        tar.extractall(restore_dir)
                    else:
                        # Extract to parent of source dirs
                        tar.extractall(Path("data"))
            else:
                # Copy uncompressed backup
                if restore_dir is None:
                    restore_dir = Path("data")
                
                for item in backup_path.iterdir():
                    dest = restore_dir / item.name
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            
            logger.info("Backup restored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup metadata dictionaries
        """
        return sorted(
            self.manifest["backups"],
            key=lambda x: x["created_at"],
            reverse=True
        )
    
    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_name: Name of backup to delete
            
        Returns:
            True if deleted successfully
        """
        # Find backup in manifest
        backup_metadata = None
        for i, backup in enumerate(self.manifest["backups"]):
            if backup["backup_name"] == backup_name:
                backup_metadata = backup
                backup_index = i
                break
        
        if not backup_metadata:
            logger.warning(f"Backup not found in manifest: {backup_name}")
            return False
        
        # Find and delete backup file/directory
        deleted = False
        for backup_path in self.backup_dir.glob(f"{backup_name}*"):
            try:
                if backup_path.is_file():
                    backup_path.unlink()
                elif backup_path.is_dir():
                    shutil.rmtree(backup_path)
                deleted = True
                logger.info(f"Deleted backup: {backup_path}")
            except Exception as e:
                logger.error(f"Failed to delete backup {backup_path}: {e}")
        
        # Remove from manifest
        if deleted:
            self.manifest["backups"].pop(backup_index)
            self._save_manifest()
        
        return deleted
    
    def apply_retention_policy(self, retention_days: int = 30) -> List[str]:
        """
        Apply retention policy to delete old backups.
        
        Args:
            retention_days: Number of days to retain backups
            
        Returns:
            List of deleted backup names
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted_backups = []
        
        for backup in self.manifest["backups"][:]:  # Copy list to modify during iteration
            backup_date = datetime.fromisoformat(backup["created_at"])
            
            if backup_date < cutoff_date:
                if self.delete_backup(backup["backup_name"]):
                    deleted_backups.append(backup["backup_name"])
        
        if deleted_backups:
            logger.info(f"Deleted {len(deleted_backups)} old backups per retention policy")
        
        # Update retention policy in manifest
        self.manifest["retention_days"] = retention_days
        self._save_manifest()
        
        return deleted_backups
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """
        Get backup statistics.
        
        Returns:
            Dictionary with backup statistics
        """
        total_backups = len(self.manifest["backups"])
        total_size = 0
        
        for backup_path in self.backup_dir.glob('*'):
            if backup_path.is_file() and not backup_path.name.endswith('.json'):
                total_size += backup_path.stat().st_size
            elif backup_path.is_dir():
                total_size += sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
        
        return {
            "total_backups": total_backups,
            "total_size_mb": total_size / (1024 ** 2),
            "last_backup": self.manifest.get("last_backup"),
            "retention_days": self.manifest.get("retention_days", 30),
            "backup_dir": str(self.backup_dir)
        }
