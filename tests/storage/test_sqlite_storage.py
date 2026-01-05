"""
Tests for SQLite Storage

This module provides comprehensive tests for SQLite storage capabilities.

Feature: gavd-gait-analysis
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from ambient.storage.sqlite_storage import SQLiteStorage


class TestSQLiteStorage:
    """Tests for SQLite storage."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test.db"
    
    @pytest.fixture
    def storage(self, temp_db_path):
        """Create SQLite storage instance."""
        return SQLiteStorage(db_path=temp_db_path)
    
    def test_initialization(self, storage, temp_db_path):
        """Test storage initialization."""
        assert storage is not None
        assert storage.db_path == temp_db_path
        assert temp_db_path.exists()
    
    def test_save_video_analysis(self, storage):
        """Test saving video analysis record."""
        analysis_id = storage.save_video_analysis(
            analysis_id="test_analysis_1",
            video_path="/path/to/video.mp4",
            status="pending",
            metadata={"duration": 30.0}
        )
        
        assert analysis_id == "test_analysis_1"
    
    def test_get_video_analysis(self, storage):
        """Test retrieving video analysis."""
        storage.save_video_analysis(
            analysis_id="test_analysis_1",
            video_path="/path/to/video.mp4",
            status="pending"
        )
        
        analysis = storage.get_video_analysis("test_analysis_1")
        
        assert analysis is not None
        assert analysis["id"] == "test_analysis_1"
        assert analysis["video_path"] == "/path/to/video.mp4"
        assert analysis["status"] == "pending"
    
    def test_update_video_analysis_status(self, storage):
        """Test updating analysis status."""
        storage.save_video_analysis(
            analysis_id="test_analysis_1",
            video_path="/path/to/video.mp4",
            status="pending"
        )
        
        success = storage.update_video_analysis_status(
            "test_analysis_1",
            "completed"
        )
        
        assert success is True
        
        analysis = storage.get_video_analysis("test_analysis_1")
        assert analysis["status"] == "completed"
    
    def test_list_video_analyses(self, storage):
        """Test listing video analyses."""
        storage.save_video_analysis("analysis_1", "/path/1.mp4", "pending")
        storage.save_video_analysis("analysis_2", "/path/2.mp4", "completed")
        storage.save_video_analysis("analysis_3", "/path/3.mp4", "pending")
        
        all_analyses = storage.list_video_analyses()
        assert len(all_analyses) == 3
        
        pending_analyses = storage.list_video_analyses(status="pending")
        assert len(pending_analyses) == 2
    
    def test_save_classification_result(self, storage):
        """Test saving classification result."""
        storage.save_video_analysis("analysis_1", "/path/1.mp4", "completed")
        
        result_id = storage.save_classification_result(
            result_id="result_1",
            analysis_id="analysis_1",
            is_normal=False,
            confidence=0.85,
            conditions=[{"name": "Parkinson's", "confidence": 0.85}],
            explanation="Detected tremor patterns"
        )
        
        assert result_id == "result_1"
    
    def test_get_classification_result(self, storage):
        """Test retrieving classification result."""
        storage.save_video_analysis("analysis_1", "/path/1.mp4", "completed")
        storage.save_classification_result(
            result_id="result_1",
            analysis_id="analysis_1",
            is_normal=False,
            confidence=0.85
        )
        
        result = storage.get_classification_result("analysis_1")
        
        assert result is not None
        assert result["analysis_id"] == "analysis_1"
        assert result["is_normal"] is False
        assert result["confidence"] == 0.85
    
    def test_save_training_dataset(self, storage):
        """Test saving training dataset record."""
        dataset_id = storage.save_training_dataset(
            dataset_id="dataset_1",
            dataset_name="GAVD",
            source_path="/path/to/gavd.csv",
            metadata={"version": "1.0"}
        )
        
        assert dataset_id == "dataset_1"
    
    def test_save_training_sample(self, storage):
        """Test saving training sample."""
        storage.save_training_dataset(
            dataset_id="dataset_1",
            dataset_name="GAVD",
            source_path="/path/to/gavd.csv"
        )
        
        sample_id = storage.save_training_sample(
            sample_id="sample_1",
            dataset_id="dataset_1",
            condition_label="Normal",
            features={"stride_length": 1.2, "cadence": 120}
        )
        
        assert sample_id == "sample_1"
    
    def test_get_training_samples_by_condition(self, storage):
        """Test retrieving samples by condition."""
        storage.save_training_dataset("dataset_1", "GAVD", "/path/to/gavd.csv")
        
        storage.save_training_sample(
            "sample_1", "dataset_1", "Normal",
            {"stride_length": 1.2}
        )
        storage.save_training_sample(
            "sample_2", "dataset_1", "Normal",
            {"stride_length": 1.3}
        )
        storage.save_training_sample(
            "sample_3", "dataset_1", "Abnormal",
            {"stride_length": 0.8}
        )
        
        normal_samples = storage.get_training_samples_by_condition("Normal")
        assert len(normal_samples) == 2
    
    def test_get_database_stats(self, storage):
        """Test getting database statistics."""
        storage.save_video_analysis("analysis_1", "/path/1.mp4", "pending")
        storage.save_training_dataset("dataset_1", "GAVD", "/path/to/gavd.csv")
        
        stats = storage.get_database_stats()
        
        assert stats["video_analysis_count"] >= 1
        assert stats["training_dataset_count"] >= 1
        assert "database_size_mb" in stats
    
    def test_backup(self, storage, temp_db_path):
        """Test database backup."""
        storage.save_video_analysis("analysis_1", "/path/1.mp4", "pending")
        
        backup_path = temp_db_path.parent / "backup.db"
        result_path = storage.backup(backup_path)
        
        assert result_path.exists()
        assert result_path == backup_path


# Property-Based Tests

@given(
    analysis_count=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=20, deadline=None)
def test_multiple_analyses_property(analysis_count):
    """
    Feature: gavd-gait-analysis, Property 11: Data Management Integrity
    
    For any number of analysis records, the system should store and retrieve
    them correctly with proper indexing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SQLiteStorage(db_path=Path(tmpdir) / "test.db")
        
        # Save multiple analyses
        for i in range(analysis_count):
            storage.save_video_analysis(
                analysis_id=f"analysis_{i}",
                video_path=f"/path/{i}.mp4",
                status="pending"
            )
        
        # Verify all can be retrieved
        all_analyses = storage.list_video_analyses(limit=100)
        assert len(all_analyses) == analysis_count
        
        # Verify each can be retrieved individually
        for i in range(analysis_count):
            analysis = storage.get_video_analysis(f"analysis_{i}")
            assert analysis is not None
            assert analysis["id"] == f"analysis_{i}"


@given(
    status=st.sampled_from(["pending", "processing", "completed", "failed"])
)
@settings(max_examples=10, deadline=None)
def test_status_filtering_property(status):
    """
    Feature: gavd-gait-analysis, Property 11: Data Management Integrity
    
    For any status filter, the system should return only analyses with that status.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SQLiteStorage(db_path=Path(tmpdir) / "test.db")
        
        # Create analyses with different statuses
        statuses = ["pending", "processing", "completed", "failed"]
        for i, s in enumerate(statuses):
            storage.save_video_analysis(
                analysis_id=f"analysis_{i}",
                video_path=f"/path/{i}.mp4",
                status=s
            )
        
        # Filter by status
        filtered = storage.list_video_analyses(status=status)
        
        # Verify all returned analyses have the correct status
        for analysis in filtered:
            assert analysis["status"] == status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
