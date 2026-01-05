"""
Tests for Backup Manager

This module provides comprehensive tests for backup and recovery capabilities.

Feature: gavd-gait-analysis
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from ambient.storage.backup_manager import BackupManager


class TestBackupManager:
    """Tests for backup manager."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            backup_dir = base / "backups"
            source_dir = base / "source"
            source_dir.mkdir()
            
            # Create some test files
            (source_dir / "test1.txt").write_text("content1")
            (source_dir / "test2.txt").write_text("content2")
            
            yield backup_dir, [source_dir]
    
    @pytest.fixture
    def manager(self, temp_dirs):
        """Create backup manager."""
        backup_dir, source_dirs = temp_dirs
        return BackupManager(backup_dir=backup_dir, source_dirs=source_dirs)
    
    def test_initialization(self, manager, temp_dirs):
        """Test manager initialization."""
        backup_dir, _ = temp_dirs
        assert manager is not None
        assert manager.backup_dir == backup_dir
        assert backup_dir.exists()
    
    def test_create_backup_uncompressed(self, manager):
        """Test creating uncompressed backup."""
        backup_path, metadata = manager.create_backup(
            backup_name="test_backup",
            compress=False,
            verify=False
        )
        
        assert backup_path.exists()
        assert backup_path.is_dir()
        assert metadata["backup_name"] == "test_backup"
        assert metadata["files_backed_up"] > 0
    
    def test_create_backup_compressed(self, manager):
        """Test creating compressed backup."""
        backup_path, metadata = manager.create_backup(
            backup_name="test_backup",
            compress=True,
            verify=False
        )
        
        assert backup_path.exists()
        assert backup_path.is_file()
        assert backup_path.suffix == ".gz"
        assert metadata["compressed"] is True
    
    def test_verify_backup(self, manager):
        """Test backup verification."""
        backup_path, metadata = manager.create_backup(
            backup_name="test_backup",
            compress=False,
            verify=False
        )
        
        verification = manager.verify_backup(backup_path, metadata)
        
        assert verification["valid"] is True
        assert "checks" in verification
    
    def test_list_backups(self, manager):
        """Test listing backups."""
        manager.create_backup("backup1", compress=False, verify=False)
        manager.create_backup("backup2", compress=False, verify=False)
        
        backups = manager.list_backups()
        
        assert len(backups) >= 2
        assert any(b["backup_name"] == "backup1" for b in backups)
        assert any(b["backup_name"] == "backup2" for b in backups)
    
    def test_delete_backup(self, manager):
        """Test deleting backup."""
        manager.create_backup("test_backup", compress=False, verify=False)
        
        backups_before = manager.list_backups()
        assert any(b["backup_name"] == "test_backup" for b in backups_before)
        
        success = manager.delete_backup("test_backup")
        assert success is True
        
        backups_after = manager.list_backups()
        assert not any(b["backup_name"] == "test_backup" for b in backups_after)
    
    def test_restore_backup_uncompressed(self, manager, temp_dirs):
        """Test restoring uncompressed backup."""
        backup_path, _ = manager.create_backup(
            "test_backup",
            compress=False,
            verify=False
        )
        
        # Create restore directory
        _, source_dirs = temp_dirs
        restore_dir = source_dirs[0].parent / "restored"
        restore_dir.mkdir()
        
        success = manager.restore_backup(backup_path, restore_dir, verify_before_restore=False)
        assert success is True
    
    def test_get_backup_stats(self, manager):
        """Test getting backup statistics."""
        manager.create_backup("backup1", compress=False, verify=False)
        
        stats = manager.get_backup_stats()
        
        assert stats["total_backups"] >= 1
        assert "total_size_mb" in stats
        assert "last_backup" in stats


# Property-Based Tests

@given(
    backup_count=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=10, deadline=None)
def test_multiple_backups_property(backup_count):
    """
    Feature: gavd-gait-analysis, Property 11: Data Management Integrity
    
    For any number of backups, the system should create and track them correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        backup_dir = base / "backups"
        source_dir = base / "source"
        source_dir.mkdir()
        
        # Create test files
        (source_dir / "test.txt").write_text("content")
        
        manager = BackupManager(backup_dir=backup_dir, source_dirs=[source_dir])
        
        # Create multiple backups
        for i in range(backup_count):
            manager.create_backup(
                backup_name=f"backup_{i}",
                compress=False,
                verify=False
            )
        
        # Verify all backups are listed
        backups = manager.list_backups()
        assert len(backups) == backup_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
