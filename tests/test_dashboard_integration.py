"""
Integration test for unified dashboard endpoint.

Tests the actual /api/v1/analysis/statistics endpoint with real data.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from server.main import app
from ambient.core.config import ConfigurationManager


@pytest.fixture
def test_client_with_data(tmp_path):
    """Create test client with mock data directories."""
    # Create temporary directories
    analysis_dir = tmp_path / "analysis"
    training_dir = tmp_path / "training"
    gavd_dir = training_dir / "gavd"
    
    analysis_metadata_dir = analysis_dir / "metadata"
    gavd_metadata_dir = gavd_dir / "metadata"
    
    analysis_metadata_dir.mkdir(parents=True, exist_ok=True)
    gavd_metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test GAVD dataset metadata
    gavd_metadata = {
        "dataset_id": "gavd-test-001",
        "original_filename": "test_dataset.csv",
        "file_path": str(gavd_dir / "gavd-test-001.csv"),
        "status": "completed",
        "uploaded_at": "2026-01-03T10:00:00",
        "processing_completed_at": "2026-01-03T10:05:00",
        "total_sequences_processed": 5,
        "total_frames_processed": 150,
        "sequence_count": 5,
        "row_count": 150
    }
    
    with open(gavd_metadata_dir / "gavd-test-001.json", 'w') as f:
        json.dump(gavd_metadata, f, indent=2)
    
    # Patch the config to use temporary directories
    with patch.object(ConfigurationManager, '__init__', lambda self: None):
        config_manager = ConfigurationManager()
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.analysis_directory = str(analysis_dir)
        config_manager.config.storage.training_directory = str(training_dir)
        
        # Set up app state
        app.state.config = config_manager
        
        client = TestClient(app)
        yield client


class TestDashboardIntegration:
    """Integration tests for dashboard endpoint."""
    
    def test_statistics_endpoint_with_gavd_data(self, test_client_with_data):
        """Test statistics endpoint returns GAVD data."""
        response = test_client_with_data.get("/api/v1/analysis/statistics")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "statistics" in data
        
        stats = data["statistics"]
        
        # Verify GAVD data is included
        assert "total_gavd_datasets" in stats
        assert stats["total_gavd_datasets"] >= 1
        
        # Verify recent analyses includes GAVD items
        assert "recent_analyses" in stats
        recent = stats["recent_analyses"]
        
        # Should have at least one GAVD dataset
        gavd_items = [item for item in recent if item.get("type") == "gavd_dataset"]
        assert len(gavd_items) >= 1
        
        # Verify GAVD item structure
        if gavd_items:
            gavd_item = gavd_items[0]
            assert "dataset_id" in gavd_item
            assert "filename" in gavd_item
            assert "status" in gavd_item
            assert gavd_item["status"] == "completed"
    
    def test_statistics_response_structure(self, test_client_with_data):
        """Test that statistics response has all required fields."""
        response = test_client_with_data.get("/api/v1/analysis/statistics")
        
        assert response.status_code == 200
        data = response.json()
        stats = data["statistics"]
        
        required_fields = [
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
        
        for field in required_fields:
            assert field in stats, f"Missing required field: {field}"
    
    def test_status_breakdown_structure(self, test_client_with_data):
        """Test that status breakdown has correct structure."""
        response = test_client_with_data.get("/api/v1/analysis/statistics")
        
        assert response.status_code == 200
        data = response.json()
        stats = data["statistics"]
        
        assert "status_breakdown" in stats
        breakdown = stats["status_breakdown"]
        
        # Should have both gait_analysis and gavd_datasets
        assert "gait_analysis" in breakdown
        assert "gavd_datasets" in breakdown
        
        # Verify gait_analysis status keys
        gait_status = breakdown["gait_analysis"]
        assert "pending" in gait_status
        assert "running" in gait_status
        assert "completed" in gait_status
        assert "failed" in gait_status
        
        # Verify gavd_datasets status keys
        gavd_status = breakdown["gavd_datasets"]
        assert "uploaded" in gavd_status
        assert "processing" in gavd_status
        assert "completed" in gavd_status
        assert "error" in gavd_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
