"""
Test GAVD sequence viewer endpoints and functionality.

Verifies that the sequence viewer can load frames and pose data correctly.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import pandas as pd

from server.main import app
from server.services.gavd_service import GAVDService
from ambient.core.config import ConfigurationManager


@pytest.fixture
def test_client_with_sequence_data(tmp_path):
    """Create test client with sequence frame data."""
    # Create temporary directories
    training_dir = tmp_path / "training"
    gavd_dir = training_dir / "gavd"
    metadata_dir = gavd_dir / "metadata"
    results_dir = gavd_dir / "results"
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_id = "test-gavd-001"
    sequence_id = "seq_001"
    
    # Create test CSV file with sequence data
    csv_data = {
        "seq": [sequence_id] * 3,
        "frame_num": [1, 2, 3],
        "cam_view": ["bottom", "bottom", "bottom"],
        "gait_event": ["heel_strike", "mid_stance", "toe_off"],
        "dataset": ["test"] * 3,
        "gait_pat": ["normal"] * 3,
        "bbox": [
            '{"left": 100, "top": 50, "width": 200, "height": 300}',
            '{"left": 105, "top": 52, "width": 198, "height": 298}',
            '{"left": 110, "top": 54, "width": 196, "height": 296}'
        ],
        "vid_info": [
            '{"width": 1280, "height": 720, "fps": 30}',
            '{"width": 1280, "height": 720, "fps": 30}',
            '{"width": 1280, "height": 720, "fps": 30}'
        ],
        "id": ["test_id"] * 3,
        "url": ["https://youtube.com/watch?v=test"] * 3
    }
    
    df = pd.DataFrame(csv_data)
    csv_file = gavd_dir / f"{dataset_id}.csv"
    df.to_csv(csv_file, index=False)
    
    # Create metadata
    gavd_metadata = {
        "dataset_id": dataset_id,
        "original_filename": "test_dataset.csv",
        "file_path": str(csv_file),
        "file_size": csv_file.stat().st_size,
        "uploaded_at": "2026-01-03T10:00:00",
        "status": "completed",
        "validation": {
            "valid": True,
            "row_count": 3,
            "sequence_count": 1
        },
        "row_count": 3,
        "sequence_count": 1,
        "total_sequences_processed": 1,
        "total_frames_processed": 3
    }
    
    with open(metadata_dir / f"{dataset_id}.json", 'w') as f:
        json.dump(gavd_metadata, f, indent=2)
    
    # Create results
    results_data = {
        "total_sequences": 1,
        "sequences": {
            sequence_id: {
                "frame_count": 3,
                "has_pose_data": True
            }
        }
    }
    
    with open(results_dir / f"{dataset_id}_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Create pose data
    pose_data = {
        sequence_id: {
            "1": {
                "keypoints": [
                    {"x": 0.5, "y": 0.5, "confidence": 0.9, "id": 0},
                    {"x": 0.6, "y": 0.6, "confidence": 0.85, "id": 1}
                ],
                "source_width": 640,
                "source_height": 360
            },
            "2": {
                "keypoints": [
                    {"x": 0.51, "y": 0.51, "confidence": 0.88, "id": 0},
                    {"x": 0.61, "y": 0.61, "confidence": 0.83, "id": 1}
                ],
                "source_width": 640,
                "source_height": 360
            },
            "3": {
                "keypoints": [
                    {"x": 0.52, "y": 0.52, "confidence": 0.87, "id": 0},
                    {"x": 0.62, "y": 0.62, "confidence": 0.82, "id": 1}
                ],
                "source_width": 640,
                "source_height": 360
            }
        }
    }
    
    with open(results_dir / f"{dataset_id}_pose_data.json", 'w') as f:
        json.dump(pose_data, f, indent=2)
    
    # Patch the config
    with patch.object(ConfigurationManager, '__init__', lambda self: None):
        config_manager = ConfigurationManager()
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.training_directory = str(training_dir)
        
        app.state.config = config_manager
        
        client = TestClient(app)
        yield client, dataset_id, sequence_id


class TestSequenceViewerEndpoints:
    """Test sequence viewer endpoints."""
    
    def test_get_sequence_frames(self, test_client_with_sequence_data):
        """Test getting sequence frames."""
        client, dataset_id, sequence_id = test_client_with_sequence_data
        
        response = client.get(f"/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frames")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "frames" in data
        assert data["frame_count"] == 3
        
        frames = data["frames"]
        assert len(frames) == 3
        
        # Verify frame structure
        frame = frames[0]
        assert "frame_num" in frame
        assert "bbox" in frame
        assert "vid_info" in frame
        assert "url" in frame
        assert "gait_event" in frame
        assert "cam_view" in frame
    
    def test_get_frame_pose_data(self, test_client_with_sequence_data):
        """Test getting pose data for a specific frame."""
        client, dataset_id, sequence_id = test_client_with_sequence_data
        
        response = client.get(f"/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frame/1/pose")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "pose_keypoints" in data
        assert "source_video_width" in data
        assert "source_video_height" in data
        
        keypoints = data["pose_keypoints"]
        assert len(keypoints) == 2
        assert keypoints[0]["x"] == 0.5
        assert keypoints[0]["y"] == 0.5
        assert keypoints[0]["confidence"] == 0.9
    
    def test_get_nonexistent_sequence(self, test_client_with_sequence_data):
        """Test getting nonexistent sequence returns 404."""
        client, dataset_id, _ = test_client_with_sequence_data
        
        response = client.get(f"/api/v1/gavd/sequence/{dataset_id}/nonexistent/frames")
        
        assert response.status_code == 404
    
    def test_get_nonexistent_frame_pose(self, test_client_with_sequence_data):
        """Test getting pose data for nonexistent frame."""
        client, dataset_id, sequence_id = test_client_with_sequence_data
        
        response = client.get(f"/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frame/999/pose")
        
        # Should return 404 or empty data
        assert response.status_code in [404, 200]
    
    def test_frame_data_structure(self, test_client_with_sequence_data):
        """Test that frame data has all required fields."""
        client, dataset_id, sequence_id = test_client_with_sequence_data
        
        response = client.get(f"/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frames")
        data = response.json()
        frames = data["frames"]
        
        required_fields = [
            "frame_num",
            "bbox",
            "vid_info",
            "url",
            "gait_event",
            "cam_view",
            "gait_pat",
            "dataset"
        ]
        
        for frame in frames:
            for field in required_fields:
                assert field in frame, f"Missing required field: {field}"
    
    def test_bbox_structure(self, test_client_with_sequence_data):
        """Test that bounding box has correct structure."""
        client, dataset_id, sequence_id = test_client_with_sequence_data
        
        response = client.get(f"/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frames")
        data = response.json()
        bbox = data["frames"][0]["bbox"]
        
        assert "left" in bbox
        assert "top" in bbox
        assert "width" in bbox
        assert "height" in bbox
        
        assert isinstance(bbox["left"], (int, float))
        assert isinstance(bbox["top"], (int, float))
        assert isinstance(bbox["width"], (int, float))
        assert isinstance(bbox["height"], (int, float))
    
    def test_vid_info_structure(self, test_client_with_sequence_data):
        """Test that video info has correct structure."""
        client, dataset_id, sequence_id = test_client_with_sequence_data
        
        response = client.get(f"/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frames")
        data = response.json()
        vid_info = data["frames"][0]["vid_info"]
        
        assert "width" in vid_info
        assert "height" in vid_info
        assert "fps" in vid_info
        
        assert vid_info["width"] == 1280
        assert vid_info["height"] == 720
        assert vid_info["fps"] == 30
    
    def test_pose_keypoints_structure(self, test_client_with_sequence_data):
        """Test that pose keypoints have correct structure."""
        client, dataset_id, sequence_id = test_client_with_sequence_data
        
        response = client.get(f"/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frame/1/pose")
        data = response.json()
        keypoints = data["pose_keypoints"]
        
        for kp in keypoints:
            assert "x" in kp
            assert "y" in kp
            assert "confidence" in kp
            assert "id" in kp
            
            assert 0 <= kp["x"] <= 1
            assert 0 <= kp["y"] <= 1
            assert 0 <= kp["confidence"] <= 1


class TestGAVDServiceSequenceMethods:
    """Test GAVDService methods for sequence viewing."""
    
    def test_get_sequence_frames(self, tmp_path):
        """Test getting sequence frames from service."""
        training_dir = tmp_path / "training"
        gavd_dir = training_dir / "gavd"
        gavd_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_id = "test-001"
        sequence_id = "seq_001"
        
        # Create CSV file
        csv_data = {
            "seq": [sequence_id] * 2,
            "frame_num": [1, 2],
            "cam_view": ["bottom", "bottom"],
            "gait_event": ["heel_strike", "toe_off"],
            "dataset": ["test"] * 2,
            "gait_pat": ["normal"] * 2,
            "bbox": [
                '{"left": 100, "top": 50, "width": 200, "height": 300}',
                '{"left": 105, "top": 52, "width": 198, "height": 298}'
            ],
            "vid_info": [
                '{"width": 1280, "height": 720, "fps": 30}',
                '{"width": 1280, "height": 720, "fps": 30}'
            ],
            "id": ["test_id"] * 2,
            "url": ["https://youtube.com/watch?v=test"] * 2
        }
        
        df = pd.DataFrame(csv_data)
        csv_file = gavd_dir / f"{dataset_id}.csv"
        df.to_csv(csv_file, index=False)
        
        # Create metadata
        metadata_dir = gavd_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "dataset_id": dataset_id,
            "file_path": str(csv_file),
            "status": "completed"
        }
        
        with open(metadata_dir / f"{dataset_id}.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create service
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.training_directory = str(training_dir)
        
        gavd_service = GAVDService(config_manager)
        
        # Get frames
        frames = gavd_service.get_sequence_frames(dataset_id, sequence_id)
        
        assert frames is not None
        assert len(frames) == 2
        assert frames[0]["frame_num"] == 1
        assert frames[1]["frame_num"] == 2
        assert frames[0]["gait_event"] == "heel_strike"
        assert frames[1]["gait_event"] == "toe_off"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
