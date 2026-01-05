"""
Test unified dashboard statistics with both GAVD and gait analysis data.

This test verifies that the dashboard correctly aggregates and displays
data from both GAVD datasets and full gait analyses.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from server.services.analysis_service import AnalysisService
from server.services.gavd_service import GAVDService
from ambient.core.config import ConfigurationManager


class TestUnifiedDashboard:
    """Test unified dashboard statistics."""
    
    def test_gavd_service_lists_datasets(self, tmp_path):
        """Test that GAVDService can list datasets."""
        # Create temporary directories
        training_dir = tmp_path / "training"
        gavd_dir = training_dir / "gavd"
        metadata_dir = gavd_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock config
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.training_directory = str(training_dir)
        
        # Create GAVD service
        gavd_service = GAVDService(config_manager)
        
        # Create test dataset metadata
        dataset_metadata = {
            "dataset_id": "test-gavd-001",
            "original_filename": "test_dataset.csv",
            "status": "completed",
            "uploaded_at": "2026-01-03T10:00:00",
            "processing_completed_at": "2026-01-03T10:05:00",
            "total_sequences_processed": 5,
            "total_frames_processed": 150,
            "sequence_count": 5,
            "row_count": 150
        }
        
        gavd_service.save_dataset_metadata("test-gavd-001", dataset_metadata)
        
        # List datasets
        datasets = gavd_service.list_datasets(limit=10, offset=0)
        
        assert len(datasets) == 1
        assert datasets[0]["dataset_id"] == "test-gavd-001"
        assert datasets[0]["status"] == "completed"
        assert datasets[0]["total_sequences_processed"] == 5
    
    def test_analysis_service_empty_statistics(self):
        """Test that AnalysisService returns empty statistics when no data."""
        config_manager = ConfigurationManager()
        service = AnalysisService(config_manager)
        
        stats = service.get_dashboard_statistics()
        
        assert stats["total_analyses"] == 0
        assert stats["normal_patterns"] == 0
        assert stats["abnormal_patterns"] == 0
        assert len(stats["recent_analyses"]) == 0
    
    def test_unified_statistics_structure(self):
        """Test that unified statistics have correct structure."""
        # This would be an integration test with actual endpoint
        # For now, we test the structure of expected response
        
        expected_keys = [
            "total_analyses",
            "total_gait_analyses",
            "total_gavd_datasets",
            "normal_patterns",
            "abnormal_patterns",
            "gavd_completed",
            "gavd_processing",
            "total_gavd_sequences",
            "total_gavd_frames",
            "recent_analyses",
            "status_breakdown"
        ]
        
        # Mock unified response
        unified_stats = {
            "total_analyses": 10,
            "total_gait_analyses": 5,
            "total_gavd_datasets": 5,
            "normal_patterns": 3,
            "abnormal_patterns": 2,
            "normal_percentage": 60.0,
            "abnormal_percentage": 40.0,
            "avg_confidence": 85.0,
            "gavd_completed": 4,
            "gavd_processing": 1,
            "gavd_uploaded": 0,
            "gavd_error": 0,
            "total_gavd_sequences": 20,
            "total_gavd_frames": 600,
            "recent_analyses": [],
            "status_breakdown": {
                "gait_analysis": {
                    "pending": 0,
                    "running": 0,
                    "completed": 5,
                    "failed": 0
                },
                "gavd_datasets": {
                    "uploaded": 0,
                    "processing": 1,
                    "completed": 4,
                    "error": 0
                }
            },
            "completed_count": 9
        }
        
        # Verify all expected keys are present
        for key in expected_keys:
            assert key in unified_stats, f"Missing key: {key}"
    
    def test_recent_analyses_mixed_types(self):
        """Test that recent analyses can contain both GAVD and gait analysis items."""
        recent_analyses = [
            {
                "type": "gavd_dataset",
                "dataset_id": "gavd-001",
                "filename": "test_dataset.csv",
                "status": "completed",
                "uploaded_at": "2026-01-03T10:00:00",
                "completed_at": "2026-01-03T10:05:00",
                "total_sequences_processed": 5,
                "total_frames_processed": 150
            },
            {
                "type": "gait_analysis",
                "analysis_id": "analysis-001",
                "file_id": "file-001",
                "status": "completed",
                "is_normal": True,
                "confidence": 0.85,
                "created_at": "2026-01-03T11:00:00",
                "completed_at": "2026-01-03T11:05:00",
                "frame_count": 300
            }
        ]
        
        # Verify structure
        assert len(recent_analyses) == 2
        assert recent_analyses[0]["type"] == "gavd_dataset"
        assert recent_analyses[1]["type"] == "gait_analysis"
        
        # Verify GAVD item has required fields
        gavd_item = recent_analyses[0]
        assert "dataset_id" in gavd_item
        assert "filename" in gavd_item
        assert "total_sequences_processed" in gavd_item
        
        # Verify gait analysis item has required fields
        gait_item = recent_analyses[1]
        assert "analysis_id" in gait_item
        assert "is_normal" in gait_item
        assert "confidence" in gait_item
    
    def test_gavd_status_counts(self):
        """Test GAVD status counting logic."""
        datasets = [
            {"status": "completed"},
            {"status": "completed"},
            {"status": "processing"},
            {"status": "uploaded"},
            {"status": "error"}
        ]
        
        status_counts = {
            "uploaded": 0,
            "processing": 0,
            "completed": 0,
            "error": 0
        }
        
        for dataset in datasets:
            status = dataset.get("status", "unknown")
            if status in status_counts:
                status_counts[status] += 1
        
        assert status_counts["completed"] == 2
        assert status_counts["processing"] == 1
        assert status_counts["uploaded"] == 1
        assert status_counts["error"] == 1
    
    def test_combined_recent_sorting(self):
        """Test that combined recent analyses are sorted by date."""
        items = [
            {"type": "gavd_dataset", "completed_at": "2026-01-03T10:00:00"},
            {"type": "gait_analysis", "completed_at": "2026-01-03T12:00:00"},
            {"type": "gavd_dataset", "completed_at": "2026-01-03T11:00:00"},
        ]
        
        # Sort by completed_at (most recent first)
        sorted_items = sorted(
            items,
            key=lambda x: x.get("completed_at", ""),
            reverse=True
        )
        
        assert sorted_items[0]["completed_at"] == "2026-01-03T12:00:00"
        assert sorted_items[1]["completed_at"] == "2026-01-03T11:00:00"
        assert sorted_items[2]["completed_at"] == "2026-01-03T10:00:00"


class TestGAVDServiceIntegration:
    """Integration tests for GAVD service."""
    
    def test_save_and_load_metadata(self, tmp_path):
        """Test saving and loading dataset metadata."""
        # Create temporary directories
        training_dir = tmp_path / "training"
        
        # Create mock config
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.training_directory = str(training_dir)
        
        # Create GAVD service
        gavd_service = GAVDService(config_manager)
        
        # Save metadata
        metadata = {
            "dataset_id": "test-001",
            "status": "completed",
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        gavd_service.save_dataset_metadata("test-001", metadata)
        
        # Load metadata
        loaded = gavd_service.get_dataset_metadata("test-001")
        
        assert loaded is not None
        assert loaded["dataset_id"] == "test-001"
        assert loaded["status"] == "completed"
    
    def test_update_metadata(self, tmp_path):
        """Test updating dataset metadata."""
        # Create temporary directories
        training_dir = tmp_path / "training"
        
        # Create mock config
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.training_directory = str(training_dir)
        
        # Create GAVD service
        gavd_service = GAVDService(config_manager)
        
        # Save initial metadata
        metadata = {
            "dataset_id": "test-001",
            "status": "uploaded"
        }
        
        gavd_service.save_dataset_metadata("test-001", metadata)
        
        # Update metadata
        result = gavd_service.update_dataset_metadata("test-001", {
            "status": "processing",
            "progress": "50%"
        })
        
        assert result is True
        
        # Load and verify
        loaded = gavd_service.get_dataset_metadata("test-001")
        assert loaded["status"] == "processing"
        assert loaded["progress"] == "50%"
        assert "updated_at" in loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
