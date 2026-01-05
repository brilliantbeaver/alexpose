"""
Test GAVD detail page endpoints and functionality.

Verifies that the GAVD detail page can load dataset metadata and sequences.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from server.main import app
from server.services.gavd_service import GAVDService
from ambient.core.config import ConfigurationManager


@pytest.fixture
def test_client_with_gavd_data(tmp_path):
    """Create test client with GAVD dataset data."""
    # Create temporary directories
    training_dir = tmp_path / "training"
    gavd_dir = training_dir / "gavd"
    metadata_dir = gavd_dir / "metadata"
    results_dir = gavd_dir / "results"
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test GAVD dataset metadata
    dataset_id = "test-gavd-001"
    gavd_metadata = {
        "dataset_id": dataset_id,
        "original_filename": "test_dataset.csv",
        "file_path": str(gavd_dir / f"{dataset_id}.csv"),
        "file_size": 208179,
        "description": "Test GAVD dataset",
        "uploaded_at": "2026-01-03T10:00:00",
        "status": "completed",
        "validation": {
            "valid": True,
            "row_count": 727,
            "sequence_count": 2,
            "headers": ["seq", "frame_num", "cam_view", "gait_event", "dataset", "gait_pat", "bbox", "vid_info", "id", "url"]
        },
        "row_count": 727,
        "sequence_count": 2,
        "processing_started_at": "2026-01-03T10:00:10",
        "processing_completed_at": "2026-01-03T10:05:00",
        "total_sequences_processed": 2,
        "total_frames_processed": 727,
        "average_frames_per_sequence": 363.5,
        "progress": "Completed"
    }
    
    with open(metadata_dir / f"{dataset_id}.json", 'w') as f:
        json.dump(gavd_metadata, f, indent=2)
    
    # Create test results
    results_data = {
        "total_sequences": 2,
        "summary": {
            "total_frames": 727,
            "average_frames_per_sequence": 363.5
        },
        "sequences": {
            "seq_001": {
                "frame_count": 364,
                "has_pose_data": True
            },
            "seq_002": {
                "frame_count": 363,
                "has_pose_data": True
            }
        }
    }
    
    with open(results_dir / f"{dataset_id}_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Patch the config
    with patch.object(ConfigurationManager, '__init__', lambda self: None):
        config_manager = ConfigurationManager()
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.training_directory = str(training_dir)
        
        app.state.config = config_manager
        
        client = TestClient(app)
        yield client, dataset_id


class TestGAVDDetailEndpoints:
    """Test GAVD detail page endpoints."""
    
    def test_get_dataset_status(self, test_client_with_gavd_data):
        """Test getting dataset status/metadata."""
        client, dataset_id = test_client_with_gavd_data
        
        response = client.get(f"/api/v1/gavd/status/{dataset_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "metadata" in data
        
        metadata = data["metadata"]
        assert metadata["dataset_id"] == dataset_id
        assert metadata["status"] == "completed"
        assert metadata["original_filename"] == "test_dataset.csv"
        assert metadata["total_sequences_processed"] == 2
        assert metadata["total_frames_processed"] == 727
    
    def test_get_dataset_sequences(self, test_client_with_gavd_data):
        """Test getting dataset sequences."""
        client, dataset_id = test_client_with_gavd_data
        
        response = client.get(f"/api/v1/gavd/sequences/{dataset_id}?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "sequences" in data
        
        sequences = data["sequences"]
        assert len(sequences) == 2
        
        # Verify sequence structure
        seq = sequences[0]
        assert "sequence_id" in seq
        assert "frame_count" in seq
        assert "has_pose_data" in seq
    
    def test_get_nonexistent_dataset(self, test_client_with_gavd_data):
        """Test getting nonexistent dataset returns 404."""
        client, _ = test_client_with_gavd_data
        
        response = client.get("/api/v1/gavd/status/nonexistent-id")
        
        assert response.status_code == 404
    
    def test_dataset_metadata_structure(self, test_client_with_gavd_data):
        """Test that dataset metadata has all required fields."""
        client, dataset_id = test_client_with_gavd_data
        
        response = client.get(f"/api/v1/gavd/status/{dataset_id}")
        data = response.json()
        metadata = data["metadata"]
        
        required_fields = [
            "dataset_id",
            "original_filename",
            "file_path",
            "file_size",
            "uploaded_at",
            "status",
            "validation",
            "row_count",
            "sequence_count"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"
    
    def test_validation_info_structure(self, test_client_with_gavd_data):
        """Test that validation info has correct structure."""
        client, dataset_id = test_client_with_gavd_data
        
        response = client.get(f"/api/v1/gavd/status/{dataset_id}")
        data = response.json()
        validation = data["metadata"]["validation"]
        
        assert "valid" in validation
        assert "row_count" in validation
        assert "sequence_count" in validation
        assert "headers" in validation
        
        assert validation["valid"] is True
        assert validation["row_count"] == 727
        assert validation["sequence_count"] == 2
        assert isinstance(validation["headers"], list)


class TestGAVDServiceMethods:
    """Test GAVDService methods used by detail page."""
    
    def test_get_dataset_metadata(self, tmp_path):
        """Test getting dataset metadata."""
        training_dir = tmp_path / "training"
        
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.training_directory = str(training_dir)
        
        gavd_service = GAVDService(config_manager)
        
        # Save metadata
        metadata = {
            "dataset_id": "test-001",
            "status": "completed",
            "uploaded_at": datetime.utcnow().isoformat()
        }
        gavd_service.save_dataset_metadata("test-001", metadata)
        
        # Retrieve metadata
        loaded = gavd_service.get_dataset_metadata("test-001")
        
        assert loaded is not None
        assert loaded["dataset_id"] == "test-001"
        assert loaded["status"] == "completed"
    
    def test_get_dataset_sequences(self, tmp_path):
        """Test getting dataset sequences."""
        training_dir = tmp_path / "training"
        results_dir = training_dir / "gavd" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.training_directory = str(training_dir)
        
        gavd_service = GAVDService(config_manager)
        
        # Create results file
        dataset_id = "test-001"
        results = {
            "total_sequences": 2,
            "sequences": {
                "seq_001": {"frame_count": 100, "has_pose_data": True},
                "seq_002": {"frame_count": 150, "has_pose_data": True}
            }
        }
        
        with open(results_dir / f"{dataset_id}_results.json", 'w') as f:
            json.dump(results, f)
        
        # Get sequences
        sequences = gavd_service.get_dataset_sequences(dataset_id, limit=10, offset=0)
        
        assert sequences is not None
        assert len(sequences) == 2
        assert sequences[0]["sequence_id"] == "seq_001"
        assert sequences[0]["frame_count"] == 100
        assert sequences[1]["sequence_id"] == "seq_002"
        assert sequences[1]["frame_count"] == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
