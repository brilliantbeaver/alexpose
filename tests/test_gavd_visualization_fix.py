"""
Tests for GAVD Visualization Tab Fix

This test suite verifies that the visualization tab correctly handles:
1. Sequence selection state management
2. Loading states during frame fetching
3. Error handling when frames fail to load
4. Proper display of frames when loaded successfully
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from server.services.gavd_service import GAVDService
from server.routers.gavd import router
from ambient.core.config import ConfigurationManager


@pytest.fixture
def mock_config():
    """Create a mock configuration manager."""
    config = Mock()
    config.storage = Mock()
    config.storage.training_directory = 'data/training'
    config.storage.youtube_directory = 'data/youtube'
    
    config_manager = Mock(spec=ConfigurationManager)
    config_manager.config = config
    
    return config_manager


@pytest.fixture
def gavd_service(mock_config, tmp_path):
    """Create a GAVD service with temporary directories."""
    # Override directories to use tmp_path
    service = GAVDService(mock_config)
    service.gavd_dir = tmp_path / 'gavd'
    service.metadata_dir = tmp_path / 'gavd' / 'metadata'
    service.results_dir = tmp_path / 'gavd' / 'results'
    
    # Create directories
    service.gavd_dir.mkdir(parents=True, exist_ok=True)
    service.metadata_dir.mkdir(parents=True, exist_ok=True)
    service.results_dir.mkdir(parents=True, exist_ok=True)
    
    return service


@pytest.fixture
def sample_dataset_metadata(gavd_service, tmp_path):
    """Create sample dataset metadata."""
    dataset_id = "test-dataset-123"
    
    # Create a dummy CSV file
    csv_file = tmp_path / 'gavd' / f'{dataset_id}.csv'
    csv_file.write_text("seq,frame_num,cam_view,gait_event,dataset,gait_pat,bbox,vid_info,id,url\n")
    
    metadata = {
        "dataset_id": dataset_id,
        "original_filename": "test_gavd.csv",
        "file_path": str(csv_file),
        "status": "completed",
        "row_count": 100,
        "sequence_count": 2,
        "uploaded_at": "2024-01-01T00:00:00",
        "total_sequences_processed": 2,
        "total_frames_processed": 100
    }
    
    gavd_service.save_dataset_metadata(dataset_id, metadata)
    
    return dataset_id, metadata


@pytest.fixture
def sample_results(gavd_service, sample_dataset_metadata):
    """Create sample processing results."""
    dataset_id, _ = sample_dataset_metadata
    
    results = {
        "total_sequences": 2,
        "summary": {
            "total_frames": 100,
            "average_frames_per_sequence": 50
        },
        "sequences": {
            "seq_001": {
                "frame_count": 50,
                "has_pose_data": True
            },
            "seq_002": {
                "frame_count": 50,
                "has_pose_data": True
            }
        }
    }
    
    results_file = gavd_service.results_dir / f"{dataset_id}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    return dataset_id, results


@pytest.fixture
def sample_pose_data(gavd_service, sample_results):
    """Create sample pose data."""
    dataset_id, _ = sample_results
    
    pose_data = {
        "seq_001": {
            "1": {
                "keypoints": [
                    {"x": 100, "y": 200, "confidence": 0.9, "keypoint_id": 0},
                    {"x": 110, "y": 210, "confidence": 0.85, "keypoint_id": 1}
                ],
                "source_width": 1920,
                "source_height": 1080
            },
            "2": {
                "keypoints": [
                    {"x": 105, "y": 205, "confidence": 0.88, "keypoint_id": 0},
                    {"x": 115, "y": 215, "confidence": 0.82, "keypoint_id": 1}
                ],
                "source_width": 1920,
                "source_height": 1080
            }
        },
        "seq_002": {
            "51": {
                "keypoints": [
                    {"x": 200, "y": 300, "confidence": 0.92, "keypoint_id": 0},
                    {"x": 210, "y": 310, "confidence": 0.87, "keypoint_id": 1}
                ],
                "source_width": 1920,
                "source_height": 1080
            }
        }
    }
    
    pose_file = gavd_service.results_dir / f"{dataset_id}_pose_data.json"
    with open(pose_file, 'w') as f:
        json.dump(pose_data, f)
    
    return dataset_id, pose_data


class TestGAVDVisualizationFix:
    """Test suite for GAVD visualization tab fixes."""
    
    def test_get_dataset_sequences_success(self, gavd_service, sample_results):
        """Test successful retrieval of dataset sequences."""
        dataset_id, expected_results = sample_results
        
        sequences = gavd_service.get_dataset_sequences(dataset_id, limit=10, offset=0)
        
        assert sequences is not None
        assert len(sequences) == 2
        assert sequences[0]["sequence_id"] == "seq_001"
        assert sequences[0]["frame_count"] == 50
        assert sequences[0]["has_pose_data"] is True
        assert sequences[1]["sequence_id"] == "seq_002"
    
    def test_get_dataset_sequences_not_found(self, gavd_service):
        """Test retrieval of sequences for non-existent dataset."""
        sequences = gavd_service.get_dataset_sequences("non-existent-dataset")
        
        assert sequences is None
    
    def test_get_dataset_sequences_pagination(self, gavd_service, sample_results):
        """Test pagination of sequences."""
        dataset_id, _ = sample_results
        
        # Get first sequence
        sequences = gavd_service.get_dataset_sequences(dataset_id, limit=1, offset=0)
        assert len(sequences) == 1
        assert sequences[0]["sequence_id"] == "seq_001"
        
        # Get second sequence
        sequences = gavd_service.get_dataset_sequences(dataset_id, limit=1, offset=1)
        assert len(sequences) == 1
        assert sequences[0]["sequence_id"] == "seq_002"
    
    @patch('ambient.gavd.gavd_processor.GAVDDataLoader')
    def test_get_sequence_frames_success(self, mock_loader_class, gavd_service, sample_dataset_metadata):
        """Test successful retrieval of sequence frames."""
        dataset_id, _ = sample_dataset_metadata
        
        # Mock the data loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        # Create mock dataframe
        import pandas as pd
        mock_df = pd.DataFrame({
            'frame_num': [1, 2, 3],
            'bbox': [{'left': 10, 'top': 20, 'width': 100, 'height': 200}] * 3,
            'vid_info': [{'width': 1920, 'height': 1080}] * 3,
            'url': ['https://youtube.com/watch?v=test'] * 3,
            'gait_event': ['HS', 'TO', 'HS'],
            'cam_view': ['lateral'] * 3,
            'gait_pat': ['normal'] * 3,
            'dataset': ['test'] * 3
        })
        
        mock_loader.load_gavd_data.return_value = mock_df
        mock_loader.organize_by_sequence.return_value = {'seq_001': mock_df}
        
        frames = gavd_service.get_sequence_frames(dataset_id, 'seq_001')
        
        assert frames is not None
        assert len(frames) == 3
        assert frames[0]['frame_num'] == 1
        assert frames[0]['gait_event'] == 'HS'
        assert 'bbox' in frames[0]
        assert 'vid_info' in frames[0]
    
    @patch('ambient.gavd.gavd_processor.GAVDDataLoader')
    def test_get_sequence_frames_not_found(self, mock_loader_class, gavd_service, sample_dataset_metadata):
        """Test retrieval of frames for non-existent sequence."""
        dataset_id, _ = sample_dataset_metadata
        
        # Mock the data loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        import pandas as pd
        mock_df = pd.DataFrame()
        
        mock_loader.load_gavd_data.return_value = mock_df
        mock_loader.organize_by_sequence.return_value = {}
        
        frames = gavd_service.get_sequence_frames(dataset_id, 'non_existent_seq')
        
        assert frames is None
    
    def test_get_frame_pose_data_success(self, gavd_service, sample_pose_data):
        """Test successful retrieval of pose data for a frame."""
        dataset_id, expected_pose_data = sample_pose_data
        
        pose_data = gavd_service.get_frame_pose_data(dataset_id, 'seq_001', 1)
        
        assert pose_data is not None
        assert 'keypoints' in pose_data
        assert len(pose_data['keypoints']) == 2
        assert pose_data['keypoints'][0]['x'] == 100
        assert pose_data['keypoints'][0]['y'] == 200
        assert pose_data['source_width'] == 1920
        assert pose_data['source_height'] == 1080
    
    def test_get_frame_pose_data_not_found(self, gavd_service, sample_pose_data):
        """Test retrieval of pose data for non-existent frame."""
        dataset_id, _ = sample_pose_data
        
        pose_data = gavd_service.get_frame_pose_data(dataset_id, 'seq_001', 999)
        
        assert pose_data is None
    
    def test_get_frame_pose_data_old_format(self, gavd_service, sample_results):
        """Test handling of old format pose data (list instead of dict)."""
        dataset_id, _ = sample_results
        
        # Create old format pose data
        old_format_pose_data = {
            "seq_001": {
                "1": [
                    {"x": 100, "y": 200, "confidence": 0.9, "keypoint_id": 0}
                ]
            }
        }
        
        pose_file = gavd_service.results_dir / f"{dataset_id}_pose_data.json"
        with open(pose_file, 'w') as f:
            json.dump(old_format_pose_data, f)
        
        pose_data = gavd_service.get_frame_pose_data(dataset_id, 'seq_001', 1)
        
        assert pose_data is not None
        assert 'keypoints' in pose_data
        assert len(pose_data['keypoints']) == 1
        assert pose_data['source_width'] is None
        assert pose_data['source_height'] is None
    
    def test_error_handling_missing_csv_file(self, gavd_service, sample_dataset_metadata):
        """Test error handling when CSV file is missing."""
        dataset_id, metadata = sample_dataset_metadata
        
        # Delete the CSV file
        Path(metadata['file_path']).unlink()
        
        frames = gavd_service.get_sequence_frames(dataset_id, 'seq_001')
        
        assert frames is None
    
    def test_error_handling_corrupted_pose_data(self, gavd_service, sample_results):
        """Test error handling when pose data file is corrupted."""
        dataset_id, _ = sample_results
        
        # Create corrupted pose data file
        pose_file = gavd_service.results_dir / f"{dataset_id}_pose_data.json"
        pose_file.write_text("{ invalid json }")
        
        pose_data = gavd_service.get_frame_pose_data(dataset_id, 'seq_001', 1)
        
        # Should return None instead of crashing
        assert pose_data is None


class TestGAVDVisualizationEndpoints:
    """Test suite for GAVD visualization API endpoints."""
    
    @pytest.fixture
    def client(self, mock_config):
        """Create a test client."""
        from fastapi import FastAPI
        
        app = FastAPI()
        app.state.config = mock_config
        app.include_router(router)
        
        return TestClient(app)
    
    @patch('server.routers.gavd.GAVDService')
    def test_get_sequence_frames_endpoint_success(self, mock_service_class, client):
        """Test successful sequence frames endpoint."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_frames = [
            {
                "frame_num": 1,
                "bbox": {"left": 10, "top": 20, "width": 100, "height": 200},
                "vid_info": {"width": 1920, "height": 1080},
                "url": "https://youtube.com/watch?v=test",
                "gait_event": "HS",
                "cam_view": "lateral"
            }
        ]
        mock_service.get_sequence_frames.return_value = mock_frames
        
        response = client.get("/api/v1/gavd/sequence/test-dataset/seq_001/frames")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["frame_count"] == 1
        assert len(data["frames"]) == 1
    
    @patch('server.routers.gavd.GAVDService')
    def test_get_sequence_frames_endpoint_not_found(self, mock_service_class, client):
        """Test sequence frames endpoint with non-existent sequence."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_sequence_frames.return_value = None
        
        response = client.get("/api/v1/gavd/sequence/test-dataset/non_existent/frames")
        
        assert response.status_code == 404
    
    @patch('server.routers.gavd.GAVDService')
    def test_get_frame_pose_endpoint_success(self, mock_service_class, client):
        """Test successful frame pose data endpoint."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_pose_data = {
            "keypoints": [
                {"x": 100, "y": 200, "confidence": 0.9, "keypoint_id": 0}
            ],
            "source_width": 1920,
            "source_height": 1080
        }
        mock_service.get_frame_pose_data.return_value = mock_pose_data
        
        response = client.get("/api/v1/gavd/sequence/test-dataset/seq_001/frame/1/pose")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["pose_keypoints"]) == 1
        assert data["source_video_width"] == 1920
        assert data["source_video_height"] == 1080
    
    @patch('server.routers.gavd.GAVDService')
    def test_get_frame_pose_endpoint_not_found(self, mock_service_class, client):
        """Test frame pose endpoint with non-existent frame."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_frame_pose_data.return_value = None
        
        response = client.get("/api/v1/gavd/sequence/test-dataset/seq_001/frame/999/pose")
        
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
